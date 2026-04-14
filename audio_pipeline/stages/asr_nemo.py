import logging
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from audio_pipeline.models import ChunkRecord, TranscribedSegment

logger = logging.getLogger(__name__)


def _safe_percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[position])


class NeMoTranscriber:
    def __init__(self, model_name: str, device: str) -> None:
        import nemo.collections.asr as nemo_asr

        logger.info("Loading NeMo model: %s", model_name)
        model_obj = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model: Any = model_obj

        if hasattr(self.model, "to"):
            self.model = self.model.to(torch.device(device))

    def transcribe_batch(
        self,
        audio_paths: list[Path],
        batch_size: int,
        pretokenize: bool,
    ) -> list[str]:
        if not audio_paths:
            return []

        # NeMo can emit this deprecation warning repeatedly for each chunk on current torch builds.
        warnings.filterwarnings(
            "ignore",
            message=r"An output with one or more elements was resized since it had shape \[\]",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"'pin_memory' argument is set as true but not supported on MPS now",
            category=UserWarning,
        )

        audio_list = [str(path) for path in audio_paths]

        transcribe_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "use_lhotse": False,
            "num_workers": 0,
            "verbose": False,
            "override_config": {"pretokenize": bool(pretokenize)},
        }

        try:
            output = self.model.transcribe(audio_list, **transcribe_kwargs)
        except TypeError:
            output = self.model.transcribe(audio_list)
        except Exception:  # noqa: BLE001
            # Some NeMo versions reject override_config shape. Fall back to the same fast path without it.
            transcribe_kwargs.pop("override_config", None)
            output = self.model.transcribe(audio_list, **transcribe_kwargs)

        if not isinstance(output, list):
            output = [output]

        texts: list[str] = []
        for item in output:
            if hasattr(item, "text"):
                texts.append(str(item.text).strip())
            else:
                texts.append(str(item).strip())

        if len(texts) < len(audio_paths):
            texts.extend([""] * (len(audio_paths) - len(texts)))

        return texts[: len(audio_paths)]


def transcribe_chunks(
    chunks: list[ChunkRecord],
    model_name: str,
    device: str,
    batch_size: int = 16,
    orchestration_batch_size: int = 64,
    min_chunk_duration_sec: float = 0.2,
    pretokenize: bool = False,
) -> tuple[list[TranscribedSegment], dict[str, Any]]:
    transcriber = NeMoTranscriber(model_name=model_name, device=device)
    text_by_index: dict[int, str] = {}
    status_by_index: dict[int, str] = {}
    effective_chunks: list[ChunkRecord] = []

    total_audio_sec = sum(max(0.0, float(item.end - item.start)) for item in chunks)
    eligible_audio_sec = 0.0

    min_duration = max(0.0, float(min_chunk_duration_sec))
    skipped_short = 0
    failed_chunks = 0
    processed_chunks = 0
    for chunk in chunks:
        duration = max(0.0, float(chunk.end - chunk.start))
        if duration < min_duration:
            text_by_index[chunk.index] = ""
            status_by_index[chunk.index] = "skipped_short"
            skipped_short += 1
            continue
        effective_chunks.append(chunk)
        eligible_audio_sec += duration

    if skipped_short > 0:
        logger.info(
            "ASR short-chunk skip: %s/%s chunks shorter than %.3fs",
            skipped_short,
            len(chunks),
            min_duration,
        )

    if not effective_chunks:
        transcripts = [
            TranscribedSegment(
                index=chunk.index,
                raw_text=text_by_index.get(chunk.index, ""),
                status=status_by_index.get(chunk.index, "skipped_short"),
            )
            for chunk in chunks
        ]
        report: dict[str, Any] = {
            "total_chunks": len(chunks),
            "eligible_chunks": 0,
            "processed_chunks": 0,
            "skipped_short_chunks": skipped_short,
            "failed_chunks": 0,
            "asr_batch_size": max(1, int(batch_size)),
            "asr_orchestration_batch_size": max(1, int(orchestration_batch_size)),
            "asr_pretokenize": bool(pretokenize),
            "asr_latency_total_ms": 0.0,
            "asr_latency_per_min_audio_ms": 0.0,
            "asr_audio_coverage_ratio": 0.0,
            "chars_per_second_of_audio": 0.0,
            "empty_asr_output_ratio": 0.0,
            "orchestration_batches_total": 0,
            "actual_orchestration_batch_sizes": [],
            "orchestration_batch_latency_ms_p50": 0.0,
            "orchestration_batch_latency_ms_p90": 0.0,
            "asr_throughput_audio_sec_per_wall_sec": 0.0,
        }
        return transcripts, report

    safe_batch_size = max(1, int(batch_size))
    safe_orchestration_batch_size = max(safe_batch_size, int(orchestration_batch_size))
    total = len(effective_chunks)
    orchestration_batch_sizes: list[int] = []
    orchestration_batch_latencies_ms: list[float] = []

    for offset in range(0, total, safe_orchestration_batch_size):
        batch = effective_chunks[offset : offset + safe_orchestration_batch_size]
        batch_paths = [Path(chunk.chunk_path) for chunk in batch]
        batch_audio_sec = sum(max(0.0, float(item.end - item.start)) for item in batch)

        batch_failed = False
        call_start = perf_counter()
        try:
            batch_texts = transcriber.transcribe_batch(
                batch_paths,
                batch_size=safe_batch_size,
                pretokenize=pretokenize,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ASR batch failed (%s..%s): %s", offset, offset + len(batch), exc, exc_info=True)
            batch_texts = [""] * len(batch)
            batch_failed = True

        call_elapsed_ms = (perf_counter() - call_start) * 1000.0
        orchestration_batch_sizes.append(len(batch))
        orchestration_batch_latencies_ms.append(call_elapsed_ms)

        throughput = batch_audio_sec / max(1e-6, call_elapsed_ms / 1000.0)
        logger.info(
            "ASR orchestration batch: %s chunks, latency %.1fms, throughput %.2f audio-sec/s",
            len(batch),
            call_elapsed_ms,
            throughput,
        )

        for item_index, chunk in enumerate(batch):
            raw_text = batch_texts[item_index] if item_index < len(batch_texts) else ""
            text_by_index[chunk.index] = raw_text
            if batch_failed or item_index >= len(batch_texts):
                status_by_index[chunk.index] = "error"
                failed_chunks += 1
            else:
                status_by_index[chunk.index] = "ok"
                processed_chunks += 1

        logger.info("ASR progress: %s/%s chunks", min(offset + len(batch), total), total)

    asr_latency_total_ms = float(sum(orchestration_batch_latencies_ms))
    audio_minutes = total_audio_sec / 60.0 if total_audio_sec > 0 else 0.0
    chars_total = sum(
        len(text_by_index.get(chunk.index, "").strip())
        for chunk in effective_chunks
        if status_by_index.get(chunk.index) == "ok"
    )
    empty_outputs = sum(
        1
        for chunk in effective_chunks
        if status_by_index.get(chunk.index) == "ok" and not text_by_index.get(chunk.index, "").strip()
    )

    transcripts = [
        TranscribedSegment(
            index=chunk.index,
            raw_text=text_by_index.get(chunk.index, ""),
            status=status_by_index.get(chunk.index, "error"),
        )
        for chunk in chunks
    ]

    report = {
        "total_chunks": len(chunks),
        "eligible_chunks": len(effective_chunks),
        "processed_chunks": processed_chunks,
        "skipped_short_chunks": skipped_short,
        "failed_chunks": failed_chunks,
        "asr_batch_size": safe_batch_size,
        "asr_orchestration_batch_size": safe_orchestration_batch_size,
        "asr_pretokenize": bool(pretokenize),
        "asr_latency_total_ms": asr_latency_total_ms,
        "asr_latency_per_min_audio_ms": (asr_latency_total_ms / audio_minutes) if audio_minutes else 0.0,
        "asr_audio_coverage_ratio": (eligible_audio_sec / total_audio_sec) if total_audio_sec > 0 else 0.0,
        "chars_per_second_of_audio": chars_total / max(1e-6, eligible_audio_sec),
        "empty_asr_output_ratio": empty_outputs / max(1, processed_chunks),
        "orchestration_batches_total": len(orchestration_batch_sizes),
        "actual_orchestration_batch_sizes": orchestration_batch_sizes,
        "orchestration_batch_latency_ms_p50": _safe_percentile(orchestration_batch_latencies_ms, 0.50),
        "orchestration_batch_latency_ms_p90": _safe_percentile(orchestration_batch_latencies_ms, 0.90),
        "asr_throughput_audio_sec_per_wall_sec": eligible_audio_sec / max(1e-6, asr_latency_total_ms / 1000.0),
    }
    return transcripts, report
