import re
from pathlib import Path

import torchaudio

from audio_pipeline.models import ChunkRecord, DiarizationSegment


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    return cleaned or "speaker"


def build_chunks(
    normalized_audio_path: Path,
    segments: list[DiarizationSegment],
    chunks_dir: Path,
    sample_rate: int,
) -> list[ChunkRecord]:
    waveform, loaded_sample_rate = torchaudio.load(str(normalized_audio_path))
    if loaded_sample_rate != sample_rate:
        raise ValueError(
            f"Chunking expects {sample_rate}Hz audio, got {loaded_sample_rate}Hz"
        )

    chunks_dir.mkdir(parents=True, exist_ok=True)
    total_frames = waveform.shape[1]

    chunk_records: list[ChunkRecord] = []
    for segment in segments:
        start_frame = max(0, int(segment.start * sample_rate))
        end_frame = min(total_frames, int(segment.end * sample_rate))

        if end_frame <= start_frame:
            continue

        chunk = waveform[:, start_frame:end_frame]
        filename = f"chunk_{segment.index:04d}_{_safe_label(segment.speaker)}.wav"
        chunk_path = chunks_dir / filename
        torchaudio.save(str(chunk_path), chunk, sample_rate)

        chunk_records.append(
            ChunkRecord(
                index=segment.index,
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                chunk_path=str(chunk_path),
            )
        )

    return chunk_records
