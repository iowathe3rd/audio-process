from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from app.config import PipelineConfig
from app.io_utils import read_json, write_json
from app.models import ChunkRecord, DiarizationSegment, PipelineSegment, TranscribedSegment
from app.pipeline.artifact_manager import ArtifactManager
from app.pipeline.factory import StageFactory
from app.pipeline.stages.text.metrics import language_switching_ratio, punctuation_density

T = TypeVar("T")


@dataclass(slots=True)
class StageContext:
    config: PipelineConfig
    artifacts: ArtifactManager
    factory: StageFactory
    logger: logging.Logger
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult(Generic[T]):
    value: T
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False


class PipelineStage(ABC, Generic[T]):
    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    artifact_keys: tuple[str, ...] = ()

    @abstractmethod
    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[T]:
        ...


def build_fingerprint(config: PipelineConfig) -> dict[str, Any]:
    return {
        "input_file": str(config.input_path.expanduser().resolve()),
        "target_sample_rate": config.target_sample_rate,
        "enable_audio_enhancement": config.enable_audio_enhancement,
        "audio_denoise_strength": config.audio_denoise_strength,
        "audio_noise_quantile": config.audio_noise_quantile,
        "audio_highpass_hz": config.audio_highpass_hz,
        "audio_target_rms_dbfs": config.audio_target_rms_dbfs,
        "audio_target_peak_dbfs": config.audio_target_peak_dbfs,
        "segment_min_duration_ms": config.segment_min_duration_ms,
        "segment_merge_gap_ms": config.segment_merge_gap_ms,
        "segment_padding_ms": config.segment_padding_ms,
        "segment_absorb_short_gap_ms": config.segment_absorb_short_gap_ms,
        "asr_batch_size": config.asr_batch_size,
        "asr_orchestration_batch_size": config.asr_orchestration_batch_size,
        "asr_min_chunk_duration_sec": config.asr_min_chunk_duration_sec,
        "asr_pretokenize": config.asr_pretokenize,
        "asr_provider": config.asr_provider,
        "asr_fallback_provider": config.asr_fallback_provider,
        "cleanup_min_duration_ms": config.cleanup_min_duration_ms,
        "cleanup_duplicate_window_ms": config.cleanup_duplicate_window_ms,
        "llm_window_max_chars": config.llm_window_max_chars,
        "llm_window_max_duration_sec": config.llm_window_max_duration_sec,
        "llm_window_max_gap_sec": config.llm_window_max_gap_sec,
        "llm_window_max_speaker_switches": config.llm_window_max_speaker_switches,
        "text_enhancement_mode": config.text_enhancement_mode,
        "enhancement_skip_low_confidence": config.enhancement_skip_low_confidence,
        "low_confidence_min_cps": config.low_confidence_min_cps,
        "low_confidence_max_cps": config.low_confidence_max_cps,
        "device": config.device,
    }


def low_confidence_positions(
    segments: list[PipelineSegment],
    min_cps: float,
    max_cps: float,
) -> set[int]:
    flagged: set[int] = set()
    for index, segment in enumerate(segments):
        duration = max(1e-6, float(segment.end - segment.start))
        cps = len(segment.raw_text.strip()) / duration
        if (
            segment.asr_status != "ok"
            or not segment.raw_text.strip()
            or cps < float(min_cps)
            or cps > float(max_cps)
        ):
            flagged.add(index)
    return flagged


class NormalizeAudioStage(PipelineStage[dict[str, Any]]):
    name = "normalized_audio"
    outputs = ("normalize_meta", "normalized_audio_path")
    artifact_keys = ("normalized.wav", "normalize.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("normalized.wav")
        meta_path = context.artifacts.get_artifact_path("normalize.json")
        cache_hit = context.artifacts.is_fresh(output_path, [context.config.input_path])
        if cache_hit:
            meta = read_json(meta_path)
        else:
            context.logger.info("Stage: normalize")
            normalizer = context.factory.create_normalizer(context.config)
            meta = normalizer.normalize(
                context.config.input_path,
                output_path,
                context.config.target_sample_rate,
            )
            write_json(meta_path, meta)
        return StageResult(
            value={"normalize_meta": meta, "normalized_audio_path": output_path},
            artifacts={"audio": output_path, "metadata": meta_path},
            metrics={"duration_sec": float(meta.get("duration_sec", 0.0))},
            cache_hit=cache_hit,
        )


class EnhanceAudioStage(PipelineStage[dict[str, Any]]):
    name = "enhanced_audio"
    inputs = ("normalized_audio_path",)
    outputs = ("audio_path", "enhancement_meta")
    artifact_keys = ("enhanced.wav", "audio_enhancement.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        normalized_audio_path = inputs["normalized_audio_path"]
        if not context.config.enable_audio_enhancement:
            return StageResult(
                value={"audio_path": normalized_audio_path, "enhancement_meta": {}},
                artifacts={"audio": normalized_audio_path},
                metrics={"enabled": False},
                cache_hit=True,
            )

        output_path = context.artifacts.get_artifact_path("enhanced.wav")
        meta_path = context.artifacts.get_artifact_path("audio_enhancement.json")
        cache_hit = context.artifacts.is_fresh(output_path, [normalized_audio_path])
        if cache_hit:
            meta = read_json(meta_path)
        else:
            context.logger.info("Stage: audio enhancement")
            enhancer = context.factory.create_enhancer(context.config)
            meta = enhancer.enhance(
                normalized_audio_path,
                output_path,
                context.config.target_sample_rate,
                context.config.audio_denoise_strength,
                context.config.audio_noise_quantile,
                context.config.audio_highpass_hz,
                context.config.audio_target_rms_dbfs,
                context.config.audio_target_peak_dbfs,
            )
            write_json(meta_path, meta)
        return StageResult(
            value={"audio_path": output_path, "enhancement_meta": meta},
            artifacts={"audio": output_path, "metadata": meta_path},
            metrics={"enabled": True},
            cache_hit=cache_hit,
        )


class DiarizationStage(PipelineStage[dict[str, Any]]):
    name = "diarization"
    inputs = ("audio_path",)
    outputs = ("diarization_segments",)
    artifact_keys = ("diarization.json",)

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        audio_path = inputs["audio_path"]
        output_path = context.artifacts.get_artifact_path("diarization.json")
        cache_hit = context.artifacts.is_fresh(output_path, [audio_path])
        if cache_hit:
            segments = [DiarizationSegment.from_dict(s) for s in read_json(output_path)]
        else:
            context.logger.info("Stage: diarization")
            diarizer = context.factory.create_diarizer(context.config)
            segments = diarizer.diarize(audio_path)
            write_json(output_path, [s.to_dict() for s in segments])
        return StageResult(
            value={"diarization_segments": segments},
            artifacts={"segments": output_path},
            metrics={"segment_count": len(segments)},
            cache_hit=cache_hit,
        )


class PostprocessSegmentsStage(PipelineStage[dict[str, Any]]):
    name = "postprocessed_segments"
    inputs = ("diarization_segments", "normalize_meta")
    outputs = ("processed_segments", "postprocess_report", "segment_groups")
    artifact_keys = (
        "diarization_postprocessed.json",
        "segment_postprocess_report.json",
        "segment_merge_groups.json",
    )

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("diarization_postprocessed.json")
        report_path = context.artifacts.get_artifact_path("segment_postprocess_report.json")
        groups_path = context.artifacts.get_artifact_path("segment_merge_groups.json")
        diarization_path = context.artifacts.get_artifact_path("diarization.json")
        cache_hit = context.artifacts.is_fresh(output_path, [diarization_path])
        if cache_hit:
            segments = [DiarizationSegment.from_dict(s) for s in read_json(output_path)]
            report = read_json(report_path)
            groups = read_json(groups_path)
        else:
            context.logger.info("Stage: segment post-processing")
            postprocessor = context.factory.create_segment_postprocessor(context.config)
            segments, report, groups = postprocessor.postprocess(
                inputs["diarization_segments"],
                float(inputs["normalize_meta"].get("duration_sec", 0.0)),
                context.config.segment_min_duration_ms,
                context.config.segment_merge_gap_ms,
                context.config.segment_padding_ms,
                context.config.segment_absorb_short_gap_ms,
            )
            write_json(output_path, [s.to_dict() for s in segments])
            write_json(report_path, report)
            write_json(groups_path, groups)
        return StageResult(
            value={
                "processed_segments": segments,
                "postprocess_report": report,
                "segment_groups": groups,
            },
            artifacts={"segments": output_path, "report": report_path, "groups": groups_path},
            metrics={
                "input_segments": int(report.get("input_segments", len(inputs["diarization_segments"]))),
                "output_segments": int(report.get("output_segments", len(segments))),
            },
            cache_hit=cache_hit,
        )


class AudioChunksStage(PipelineStage[dict[str, Any]]):
    name = "audio_chunks"
    inputs = ("audio_path", "processed_segments")
    outputs = ("chunk_records",)
    artifact_keys = ("segments_manifest.json", "chunks")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        manifest_path = context.artifacts.get_artifact_path("segments_manifest.json")
        chunks_dir = context.artifacts.get_artifact_path("chunks")
        postprocessed_path = context.artifacts.get_artifact_path("diarization_postprocessed.json")
        cache_hit = context.artifacts.is_fresh(manifest_path, [inputs["audio_path"], postprocessed_path])
        if cache_hit:
            records = [ChunkRecord.from_dict(c) for c in read_json(manifest_path)]
        else:
            context.logger.info("Stage: segmentation")
            chunk_builder = context.factory.create_chunk_builder(context.config)
            records = chunk_builder.build_chunks(
                inputs["audio_path"],
                inputs["processed_segments"],
                chunks_dir,
                context.config.target_sample_rate,
            )
            write_json(manifest_path, [c.to_dict() for c in records])
        return StageResult(
            value={"chunk_records": records},
            artifacts={"manifest": manifest_path, "chunks_dir": chunks_dir},
            metrics={"chunk_count": len(records)},
            cache_hit=cache_hit,
        )


class ASRTranscriptsStage(PipelineStage[dict[str, Any]]):
    name = "asr_transcripts"
    inputs = ("chunk_records",)
    outputs = ("transcripts", "asr_report")
    artifact_keys = ("asr_raw.json", "asr_report.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("asr_raw.json")
        report_path = context.artifacts.get_artifact_path("asr_report.json")
        chunks_manifest_path = context.artifacts.get_artifact_path("segments_manifest.json")
        cache_hit = context.artifacts.is_fresh(output_path, [chunks_manifest_path])
        if cache_hit:
            transcripts = [TranscribedSegment.from_dict(t) for t in read_json(output_path)]
            report = read_json(report_path)
        else:
            context.logger.info("Stage: ASR")
            adapter = context.factory.create_asr_adapter(context.config)
            from app.pipeline.stages.asr.transcribe import transcribe_chunks

            transcripts, report = transcribe_chunks(
                chunks=inputs["chunk_records"],
                adapter=adapter,
                batch_size=context.config.asr_batch_size,
                orchestration_batch_size=context.config.asr_orchestration_batch_size,
                min_chunk_duration_sec=context.config.asr_min_chunk_duration_sec,
                pretokenize=context.config.asr_pretokenize,
            )
            write_json(output_path, [t.to_dict() for t in transcripts])
            write_json(report_path, report)
        return StageResult(
            value={"transcripts": transcripts, "asr_report": report},
            artifacts={"transcripts": output_path, "report": report_path},
            metrics={
                "total_chunks": int(report.get("total_chunks", len(inputs["chunk_records"]))),
                "provider": report.get("asr_provider_effective", context.config.asr_provider),
                "model": report.get("asr_model_effective", context.config.nemo_model_name),
            },
            cache_hit=cache_hit,
        )


class MergeTranscriptStage(PipelineStage[dict[str, Any]]):
    name = "merged_transcript"
    inputs = ("chunk_records", "transcripts")
    outputs = ("merged_segments",)
    artifact_keys = ("merged_raw.json",)

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("merged_raw.json")
        chunks_manifest_path = context.artifacts.get_artifact_path("segments_manifest.json")
        asr_raw_path = context.artifacts.get_artifact_path("asr_raw.json")
        cache_hit = context.artifacts.is_fresh(output_path, [chunks_manifest_path, asr_raw_path])
        if cache_hit:
            segments = [PipelineSegment.from_dict(s) for s in read_json(output_path)]
        else:
            context.logger.info("Stage: merge")
            merger = context.factory.create_segment_merger(context.config)
            segments = merger.merge(inputs["chunk_records"], inputs["transcripts"])
            write_json(output_path, [s.to_dict() for s in segments])
        return StageResult(
            value={"merged_segments": segments},
            artifacts={"segments": output_path},
            metrics={"segment_count": len(segments)},
            cache_hit=cache_hit,
        )


class CleanTranscriptStage(PipelineStage[dict[str, Any]]):
    name = "cleaned_transcript"
    inputs = ("merged_segments",)
    outputs = ("cleaned_segments", "cleanup_report")
    artifact_keys = ("merged_clean.json", "cleanup_report.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("merged_clean.json")
        report_path = context.artifacts.get_artifact_path("cleanup_report.json")
        merged_path = context.artifacts.get_artifact_path("merged_raw.json")
        cache_hit = context.artifacts.is_fresh(output_path, [merged_path])
        if cache_hit:
            segments = [PipelineSegment.from_dict(s) for s in read_json(output_path)]
            report = read_json(report_path)
        else:
            context.logger.info("Stage: cleanup")
            cleaner = context.factory.create_transcript_cleaner(context.config)
            segments, report = cleaner.cleanup(
                inputs["merged_segments"],
                context.config.cleanup_min_duration_ms,
                context.config.cleanup_duplicate_window_ms,
            )
            write_json(output_path, [s.to_dict() for s in segments])
            write_json(report_path, report)
        return StageResult(
            value={"cleaned_segments": segments, "cleanup_report": report},
            artifacts={"segments": output_path, "report": report_path},
            metrics={"segment_count": len(segments)},
            cache_hit=cache_hit,
        )


class SemanticWindowsStage(PipelineStage[dict[str, Any]]):
    name = "semantic_windows"
    inputs = ("cleaned_segments",)
    outputs = ("semantic_windows", "semantic_windows_report")
    artifact_keys = ("semantic_windows.json", "semantic_windows_report.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("semantic_windows.json")
        report_path = context.artifacts.get_artifact_path("semantic_windows_report.json")
        cleaned_path = context.artifacts.get_artifact_path("merged_clean.json")
        cache_hit = context.artifacts.is_fresh(output_path, [cleaned_path])
        if cache_hit:
            windows = read_json(output_path)
            report = read_json(report_path)
        else:
            context.logger.info("Stage: semantic windows")
            window_builder = context.factory.create_semantic_window_builder(context.config)
            windows, report = window_builder.build_windows(
                inputs["cleaned_segments"],
                context.config.llm_window_max_chars,
                context.config.llm_window_max_duration_sec,
                context.config.llm_window_max_gap_sec,
                context.config.llm_window_max_speaker_switches,
            )
            write_json(output_path, windows)
            write_json(report_path, report)
        return StageResult(
            value={"semantic_windows": windows, "semantic_windows_report": report},
            artifacts={"windows": output_path, "report": report_path},
            metrics={"window_count": len(windows)},
            cache_hit=cache_hit,
        )


class AnonymizeTranscriptStage(PipelineStage[dict[str, Any]]):
    name = "anonymized_transcript"
    inputs = ("cleaned_segments", "semantic_windows")
    outputs = ("anonymized_segments", "anonymize_report")
    artifact_keys = ("anonymized.json", "anonymize_report.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("anonymized.json")
        report_path = context.artifacts.get_artifact_path("anonymize_report.json")
        cleaned_path = context.artifacts.get_artifact_path("merged_clean.json")
        windows_path = context.artifacts.get_artifact_path("semantic_windows.json")
        cache_hit = context.artifacts.is_fresh(output_path, [cleaned_path, windows_path])
        if cache_hit:
            segments = [PipelineSegment.from_dict(s) for s in read_json(output_path)]
            report = read_json(report_path)
        else:
            context.logger.info("Stage: text anonymization")
            text_processor = context.factory.create_text_processor(context.config)
            segments, report = text_processor.anonymize(
                inputs["cleaned_segments"],
                inputs["semantic_windows"],
            )
            write_json(output_path, [s.to_dict() for s in segments])
            write_json(report_path, report)
        return StageResult(
            value={"anonymized_segments": segments, "anonymize_report": report},
            artifacts={"segments": output_path, "report": report_path},
            metrics={"llm_calls": int(report.get("llm_calls_total", 0))},
            cache_hit=cache_hit,
        )


class EnhanceTranscriptStage(PipelineStage[dict[str, Any]]):
    name = "enhanced_transcript"
    inputs = ("anonymized_segments", "semantic_windows")
    outputs = ("final_segments", "enhancement_report")
    artifact_keys = ("enhanced.json", "enhancement_report.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("enhanced.json")
        report_path = context.artifacts.get_artifact_path("enhancement_report.json")
        anonymized_path = context.artifacts.get_artifact_path("anonymized.json")
        cache_hit = context.artifacts.is_fresh(output_path, [anonymized_path])
        if cache_hit:
            segments = [PipelineSegment.from_dict(s) for s in read_json(output_path)]
            report = read_json(report_path)
        else:
            context.logger.info("Stage: text enhancement")
            text_processor = context.factory.create_text_processor(context.config)
            low_conf = low_confidence_positions(
                inputs["anonymized_segments"],
                context.config.low_confidence_min_cps,
                context.config.low_confidence_max_cps,
            )
            segments, report = text_processor.enhance(
                inputs["anonymized_segments"],
                inputs["semantic_windows"],
                context.config.text_enhancement_mode,
                low_conf,
                context.config.enhancement_skip_low_confidence,
            )
            write_json(output_path, [s.to_dict() for s in segments])
            write_json(report_path, report)
        return StageResult(
            value={"final_segments": segments, "enhancement_report": report},
            artifacts={"segments": output_path, "report": report_path},
            metrics={"llm_calls": int(report.get("llm_calls_total", 0))},
            cache_hit=cache_hit,
        )


class QualityReportStage(PipelineStage[dict[str, Any]]):
    name = "quality_report"
    inputs = ("chunk_records", "transcripts", "final_segments")
    outputs = ("quality_rows", "quality_report")
    artifact_keys = ("chunk_quality.json", "chunk_quality_report.json")

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        output_path = context.artifacts.get_artifact_path("chunk_quality.json")
        report_path = context.artifacts.get_artifact_path("chunk_quality_report.json")
        chunks_manifest_path = context.artifacts.get_artifact_path("segments_manifest.json")
        asr_raw_path = context.artifacts.get_artifact_path("asr_raw.json")
        enhanced_path = context.artifacts.get_artifact_path("enhanced.json")
        cache_hit = context.artifacts.is_fresh(output_path, [chunks_manifest_path, asr_raw_path, enhanced_path])
        if cache_hit:
            rows = read_json(output_path)
            report = read_json(report_path)
        else:
            context.logger.info("Stage: quality analytics")
            analyzer = context.factory.create_chunk_quality_analyzer(context.config)
            rows, report = analyzer.analyze(
                inputs["chunk_records"],
                inputs["transcripts"],
                inputs["final_segments"],
                context.config.low_confidence_min_cps,
                context.config.low_confidence_max_cps,
            )
            write_json(output_path, rows)
            write_json(report_path, report)
        return StageResult(
            value={"quality_rows": rows, "quality_report": report},
            artifacts={"rows": output_path, "report": report_path},
            metrics={
                "row_count": len(rows),
                "low_confidence_chunk_ratio": float(report.get("low_confidence_chunk_ratio", 0.0)),
            },
            cache_hit=cache_hit,
        )


class ProcessedAudioResultStage(PipelineStage[dict[str, Any]]):
    name = "processed_audio_result"
    inputs = (
        "normalize_meta",
        "diarization_segments",
        "processed_segments",
        "postprocess_report",
        "chunk_records",
        "asr_report",
        "anonymize_report",
        "enhancement_report",
        "quality_report",
        "final_segments",
    )
    outputs = ("result",)
    artifact_keys = ("result.json",)

    def run(self, context: StageContext, inputs: dict[str, Any]) -> StageResult[dict[str, Any]]:
        audio_duration_sec = float(inputs["normalize_meta"].get("duration_sec", 0.0))
        audio_minutes = (audio_duration_sec / 60.0) if audio_duration_sec > 0 else 0.0
        postprocess_report = inputs["postprocess_report"]
        asr_report = inputs["asr_report"]
        anonymize_report = inputs["anonymize_report"]
        enhancement_report = inputs["enhancement_report"]
        quality_report = inputs["quality_report"]
        final_segments = inputs["final_segments"]

        anonymize_calls = int(anonymize_report.get("llm_calls_total", 0))
        enhancement_calls = int(enhancement_report.get("llm_calls_total", 0))
        enhanced_texts = [item.enhanced_text for item in final_segments if item.enhanced_text.strip()]
        metrics = {
            "total_segments_before_postprocess": int(
                postprocess_report.get("input_segments", len(inputs["diarization_segments"]))
            ),
            "total_segments_after_postprocess": int(
                postprocess_report.get("output_segments", len(inputs["processed_segments"]))
            ),
            "asr_chunks_total": int(asr_report.get("total_chunks", len(inputs["chunk_records"]))),
            "asr_latency_total_ms": float(asr_report.get("asr_latency_total_ms", 0.0)),
            "llm_calls_total": anonymize_calls + enhancement_calls,
            "llm_latency_total_ms": float(anonymize_report.get("llm_latency_total_ms", 0.0))
            + float(enhancement_report.get("llm_latency_total_ms", 0.0)),
            "language_switching_ratio": language_switching_ratio(enhanced_texts),
            "punctuation_density_after_enhancement": punctuation_density(enhanced_texts),
            "semantic_drift_flags_total": int(enhancement_report.get("semantic_drift_flags_total", 0)),
            "low_confidence_chunk_ratio": float(quality_report.get("low_confidence_chunk_ratio", 0.0)),
        }
        result = {
            "input_file": str(context.config.input_path),
            "artifacts_dir": str(context.config.run_dir),
            "metrics": metrics,
            "segments": [s.to_dict() for s in final_segments],
        }
        output_path = context.artifacts.get_artifact_path("result.json")
        write_json(output_path, result)
        return StageResult(
            value={"result": result},
            artifacts={"result": output_path},
            metrics={**metrics, "audio_minutes": audio_minutes},
            cache_hit=False,
        )


PIPELINE_STAGES: tuple[PipelineStage[Any], ...] = (
    NormalizeAudioStage(),
    EnhanceAudioStage(),
    DiarizationStage(),
    PostprocessSegmentsStage(),
    AudioChunksStage(),
    ASRTranscriptsStage(),
    MergeTranscriptStage(),
    CleanTranscriptStage(),
    SemanticWindowsStage(),
    AnonymizeTranscriptStage(),
    EnhanceTranscriptStage(),
    QualityReportStage(),
    ProcessedAudioResultStage(),
)


STAGES_BY_NAME: dict[str, PipelineStage[Any]] = {stage.name: stage for stage in PIPELINE_STAGES}

