import logging
from pathlib import Path
from statistics import median
from typing import Any

from audio_pipeline.config import PipelineConfig
from audio_pipeline.io_utils import ensure_dir, read_json, write_json
from audio_pipeline.models import ChunkRecord, DiarizationSegment, PipelineSegment, TranscribedSegment
from audio_pipeline.stages.asr_adapters import make_asr_adapter
from audio_pipeline.stages.asr_transcribe import transcribe_chunks
from audio_pipeline.stages.chunk_quality import build_chunk_quality_analytics
from audio_pipeline.stages.cleanup import cleanup_transcript_segments
from audio_pipeline.stages.diarization import run_diarization
from audio_pipeline.stages.enhance_audio import enhance_audio
from audio_pipeline.stages.merge import merge_segments
from audio_pipeline.stages.normalize import normalize_audio
from audio_pipeline.stages.postprocess_segments import normalize_speaker_segments
from audio_pipeline.stages.semantic_windows import build_semantic_windows
from audio_pipeline.stages.segmentation import build_chunks
from audio_pipeline.stages.text_metrics import language_switching_ratio, punctuation_density
from audio_pipeline.stages.vertex_text import (
    VertexTextProcessor,
    anonymize_segments,
    enhance_segments_deterministic,
    enhance_segments,
)

logger = logging.getLogger(__name__)


SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def _load_diarization(path: Path) -> list[DiarizationSegment]:
    return [DiarizationSegment.from_dict(item) for item in read_json(path)]


def _load_chunks(path: Path) -> list[ChunkRecord]:
    return [ChunkRecord.from_dict(item) for item in read_json(path)]


def _load_transcripts(path: Path) -> list[TranscribedSegment]:
    return [TranscribedSegment.from_dict(item) for item in read_json(path)]


def _load_pipeline_segments(path: Path) -> list[PipelineSegment]:
    return [PipelineSegment.from_dict(item) for item in read_json(path)]


def _all_chunk_files_exist(chunks: list[ChunkRecord]) -> bool:
    return all(Path(item.chunk_path).exists() for item in chunks)


def _artifact_is_fresh(
    output_path: Path,
    dependencies: list[Path],
    invalidate_cache: bool,
) -> bool:
    if invalidate_cache or not output_path.exists():
        return False

    output_mtime = output_path.stat().st_mtime
    for dependency in dependencies:
        if not dependency.exists():
            return False
        if dependency.stat().st_mtime > output_mtime:
            return False

    return True


def _config_fingerprint(config: PipelineConfig) -> dict[str, Any]:
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
        "chirp_model_name": config.chirp_model_name,
        "chirp_language_code": config.chirp_language_code,
        "chirp_project": config.chirp_project,
        "chirp_location": config.chirp_location,
        "chirp_recognizer": config.chirp_recognizer,
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
        "nemo_model_name": config.nemo_model_name,
        "has_google_api_key": bool(config.google_api_key),
        "vertex_project": config.vertex_project,
        "vertex_location": config.vertex_location,
        "vertex_model_name": config.vertex_model_name,
        "use_vertex": config.use_vertex,
        "device": config.device,
    }


def _result_segments(segments: list[PipelineSegment]) -> list[dict[str, Any]]:
    return [
        {
            "speaker": item.speaker,
            "start": item.start,
            "end": item.end,
            "raw_text": item.raw_text,
            "anonymized_text": item.anonymized_text,
            "enhanced_text": item.enhanced_text,
            "asr_status": item.asr_status,
        }
        for item in segments
    ]


def _safe_percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[position])


def _duration_stats(segments: list[DiarizationSegment]) -> dict[str, float]:
    durations = [max(0.0, float(item.end - item.start)) for item in segments]
    if not durations:
        return {
            "avg_segment_duration_sec": 0.0,
            "median_segment_duration_sec": 0.0,
            "p10_segment_duration_sec": 0.0,
            "p90_segment_duration_sec": 0.0,
        }

    return {
        "avg_segment_duration_sec": float(sum(durations) / len(durations)),
        "median_segment_duration_sec": float(median(durations)),
        "p10_segment_duration_sec": _safe_percentile(durations, 0.10),
        "p90_segment_duration_sec": _safe_percentile(durations, 0.90),
    }


def _punctuation_density(texts: list[str]) -> float:
    return punctuation_density(texts)


def _language_switching_ratio(texts: list[str]) -> float:
    return language_switching_ratio(texts)


def _segment_chars_per_sec(segment: PipelineSegment) -> float:
    duration = max(1e-6, float(segment.end - segment.start))
    return len(segment.raw_text.strip()) / duration


def _low_confidence_positions(
    segments: list[PipelineSegment],
    min_cps: float,
    max_cps: float,
) -> set[int]:
    flagged: set[int] = set()
    for index, segment in enumerate(segments):
        text = segment.raw_text.strip()
        cps = _segment_chars_per_sec(segment)
        if (
            segment.asr_status != "ok"
            or not text
            or cps < float(min_cps)
            or cps > float(max_cps)
        ):
            flagged.add(index)
    return flagged


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    if not config.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {config.input_path}")

    if config.input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file format: {config.input_path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if config.use_vertex:
        has_vertex_settings = bool(config.vertex_project and config.vertex_location)
        has_api_key = bool(config.google_api_key)
        if not has_vertex_settings and not has_api_key:
            raise ValueError(
                "Text processing is enabled but no Google auth configured. "
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY), or set GOOGLE_CLOUD_PROJECT and "
                "GOOGLE_CLOUD_LOCATION, or run with --no-vertex."
            )

    if config.asr_batch_size < 1:
        raise ValueError("--asr-batch-size must be >= 1")

    if config.segment_min_duration_ms < 0:
        raise ValueError("segment_min_duration_ms must be >= 0")

    if config.segment_merge_gap_ms < 0:
        raise ValueError("segment_merge_gap_ms must be >= 0")

    if config.segment_padding_ms < 0:
        raise ValueError("segment_padding_ms must be >= 0")

    if config.segment_absorb_short_gap_ms < 0:
        raise ValueError("segment_absorb_short_gap_ms must be >= 0")

    if config.asr_orchestration_batch_size < 1:
        raise ValueError("asr_orchestration_batch_size must be >= 1")

    if config.asr_provider not in {"chirp", "nemo"}:
        raise ValueError("asr_provider must be one of: chirp, nemo")

    if config.asr_fallback_provider not in {"none", "chirp", "nemo"}:
        raise ValueError("asr_fallback_provider must be one of: none, chirp, nemo")

    if config.cleanup_min_duration_ms < 0:
        raise ValueError("cleanup_min_duration_ms must be >= 0")

    if config.cleanup_duplicate_window_ms < 0:
        raise ValueError("cleanup_duplicate_window_ms must be >= 0")

    if config.llm_window_max_chars < 80:
        raise ValueError("llm_window_max_chars must be >= 80")

    if config.llm_window_max_duration_sec <= 0:
        raise ValueError("llm_window_max_duration_sec must be > 0")

    if config.llm_window_max_gap_sec < 0:
        raise ValueError("llm_window_max_gap_sec must be >= 0")

    if config.llm_window_max_speaker_switches < 0:
        raise ValueError("llm_window_max_speaker_switches must be >= 0")

    if config.low_confidence_min_cps < 0:
        raise ValueError("low_confidence_min_cps must be >= 0")

    if config.low_confidence_max_cps <= 0:
        raise ValueError("low_confidence_max_cps must be > 0")

    if config.low_confidence_max_cps <= config.low_confidence_min_cps:
        raise ValueError("low_confidence_max_cps must be > low_confidence_min_cps")

    if config.text_enhancement_mode not in {"deterministic", "llm"}:
        raise ValueError("text_enhancement_mode must be 'deterministic' or 'llm'")

    logger.info("Using device: %s", config.device)

    run_dir = config.run_dir
    ensure_dir(run_dir)

    state_path = run_dir / "run_state.json"
    current_state = _config_fingerprint(config)
    invalidate_cache = bool(config.force)
    if state_path.exists():
        previous_state = read_json(state_path)
        if previous_state != current_state:
            logger.info("Configuration/input changed since previous run, invalidating cache")
            invalidate_cache = True
    normalize_meta_path = run_dir / "normalize.json"
    normalized_audio_path = run_dir / "normalized.wav"
    enhancement_meta_path = run_dir / "audio_enhancement.json"
    enhanced_audio_path = run_dir / "enhanced.wav"
    diarization_path = run_dir / "diarization.json"
    diarization_postprocessed_path = run_dir / "diarization_postprocessed.json"
    segment_postprocess_report_path = run_dir / "segment_postprocess_report.json"
    segment_merge_groups_path = run_dir / "segment_merge_groups.json"
    chunks_manifest_path = run_dir / "segments_manifest.json"
    asr_path = run_dir / "asr_raw.json"
    asr_report_path = run_dir / "asr_report.json"
    merged_path = run_dir / "merged_raw.json"
    merged_clean_path = run_dir / "merged_clean.json"
    cleanup_report_path = run_dir / "cleanup_report.json"
    semantic_windows_path = run_dir / "semantic_windows.json"
    semantic_windows_report_path = run_dir / "semantic_windows_report.json"
    anonymized_path = run_dir / "anonymized.json"
    anonymize_report_path = run_dir / "anonymize_report.json"
    enhanced_path = run_dir / "enhanced.json"
    enhancement_report_path = run_dir / "enhancement_report.json"
    chunk_quality_path = run_dir / "chunk_quality.json"
    chunk_quality_report_path = run_dir / "chunk_quality_report.json"
    metrics_path = run_dir / "metrics.json"
    result_path = run_dir / "result.json"

    normalize_cached = _artifact_is_fresh(
        normalize_meta_path,
        [config.input_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        normalized_audio_path,
        [config.input_path],
        invalidate_cache,
    )

    if not normalize_cached:
        logger.info("Stage: normalize")
        normalize_meta = normalize_audio(
            input_path=config.input_path,
            output_path=normalized_audio_path,
            target_sample_rate=config.target_sample_rate,
        )
        write_json(normalize_meta_path, normalize_meta)
    else:
        logger.info("Stage: normalize skipped (using cache)")
        normalize_meta = read_json(normalize_meta_path)

    pipeline_audio_path = normalized_audio_path
    if config.enable_audio_enhancement:
        audio_enhancement_cached = _artifact_is_fresh(
            enhancement_meta_path,
            [normalize_meta_path, normalized_audio_path],
            invalidate_cache,
        ) and _artifact_is_fresh(
            enhanced_audio_path,
            [normalize_meta_path, normalized_audio_path],
            invalidate_cache,
        )

        if not audio_enhancement_cached:
            logger.info("Stage: audio enhancement")
            enhancement_meta = enhance_audio(
                input_path=normalized_audio_path,
                output_path=enhanced_audio_path,
                sample_rate=config.target_sample_rate,
                denoise_strength=config.audio_denoise_strength,
                noise_quantile=config.audio_noise_quantile,
                highpass_hz=config.audio_highpass_hz,
                target_rms_dbfs=config.audio_target_rms_dbfs,
                target_peak_dbfs=config.audio_target_peak_dbfs,
            )
            write_json(enhancement_meta_path, enhancement_meta)
        else:
            logger.info("Stage: audio enhancement skipped (using cache)")
            enhancement_meta = read_json(enhancement_meta_path)

        if not enhanced_audio_path.exists():
            raise FileNotFoundError(f"Enhanced audio not found: {enhanced_audio_path}")

        pipeline_audio_path = enhanced_audio_path
    else:
        logger.info("Stage: audio enhancement disabled")
        enhancement_meta = {}

    diarization_cached = _artifact_is_fresh(
        diarization_path,
        [normalize_meta_path, pipeline_audio_path],
        invalidate_cache,
    )

    if not diarization_cached:
        logger.info("Stage: diarization")
        if not config.hf_token:
            raise ValueError(
                "HF token is required for diarization stage. "
                "Set HF_TOKEN or pass --hf-token."
            )
        diarization_segments = run_diarization(
            audio_path=pipeline_audio_path,
            hf_token=config.hf_token,
            device=config.device,
        )
        write_json(diarization_path, [item.to_dict() for item in diarization_segments])
    else:
        logger.info("Stage: diarization skipped (using cache)")
        diarization_segments = _load_diarization(diarization_path)

    segment_postprocess_cached = _artifact_is_fresh(
        diarization_postprocessed_path,
        [diarization_path, pipeline_audio_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        segment_postprocess_report_path,
        [diarization_path, pipeline_audio_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        segment_merge_groups_path,
        [diarization_path, pipeline_audio_path],
        invalidate_cache,
    )

    if not segment_postprocess_cached:
        logger.info("Stage: segment post-processing")
        diarization_segments_for_asr, segment_postprocess_report, segment_merge_groups = normalize_speaker_segments(
            segments=diarization_segments,
            audio_duration_sec=float(normalize_meta.get("duration_sec", 0.0)),
            min_segment_duration_ms=config.segment_min_duration_ms,
            merge_gap_ms=config.segment_merge_gap_ms,
            padding_ms=config.segment_padding_ms,
            absorb_short_gap_ms=config.segment_absorb_short_gap_ms,
        )
        write_json(
            diarization_postprocessed_path,
            [item.to_dict() for item in diarization_segments_for_asr],
        )
        write_json(segment_postprocess_report_path, segment_postprocess_report)
        write_json(segment_merge_groups_path, segment_merge_groups)
    else:
        logger.info("Stage: segment post-processing skipped (using cache)")
        diarization_segments_for_asr = _load_diarization(diarization_postprocessed_path)
        segment_postprocess_report = read_json(segment_postprocess_report_path)
        segment_merge_groups = read_json(segment_merge_groups_path)

    chunks_dir = run_dir / "chunks"
    segmentation_cached = _artifact_is_fresh(
        chunks_manifest_path,
        [pipeline_audio_path, diarization_postprocessed_path],
        invalidate_cache,
    )

    if segmentation_cached:
        cached_chunks = _load_chunks(chunks_manifest_path)
        if not _all_chunk_files_exist(cached_chunks):
            logger.info("Chunk files missing, rebuilding segmentation artifacts")
            segmentation_cached = False

    if not segmentation_cached:
        logger.info("Stage: segmentation")
        chunk_records = build_chunks(
            normalized_audio_path=pipeline_audio_path,
            segments=diarization_segments_for_asr,
            chunks_dir=chunks_dir,
            sample_rate=config.target_sample_rate,
        )
        write_json(chunks_manifest_path, [item.to_dict() for item in chunk_records])
    else:
        logger.info("Stage: segmentation skipped (using cache)")
        chunk_records = cached_chunks

    asr_dependencies = [chunks_manifest_path]
    asr_dependencies.extend(Path(item.chunk_path) for item in chunk_records)
    asr_cached = _artifact_is_fresh(
        asr_path,
        asr_dependencies,
        invalidate_cache,
    ) and _artifact_is_fresh(
        asr_report_path,
        asr_dependencies,
        invalidate_cache,
    )

    if not asr_cached:
        logger.info("Stage: asr (%s)", config.asr_provider)
        primary_model_name = (
            config.chirp_model_name if config.asr_provider == "chirp" else config.nemo_model_name
        )
        fallback_model_name = (
            config.nemo_model_name
            if config.asr_fallback_provider == "nemo"
            else config.chirp_model_name
        )
        asr_adapter = make_asr_adapter(
            provider_name=config.asr_provider,
            model_name=primary_model_name,
            device=config.device,
            fallback_provider_name=config.asr_fallback_provider,
            fallback_model_name=fallback_model_name,
            google_api_key=config.google_api_key,
            chirp_language_code=config.chirp_language_code,
            chirp_project=config.chirp_project,
            chirp_location=config.chirp_location,
            chirp_recognizer=config.chirp_recognizer,
        )
        transcripts, asr_report = transcribe_chunks(
            chunks=chunk_records,
            adapter=asr_adapter,
            batch_size=config.asr_batch_size,
            orchestration_batch_size=config.asr_orchestration_batch_size,
            min_chunk_duration_sec=config.asr_min_chunk_duration_sec,
            pretokenize=config.asr_pretokenize,
        )
        write_json(asr_path, [item.to_dict() for item in transcripts])
        write_json(asr_report_path, asr_report)
    else:
        logger.info("Stage: asr skipped (using cache)")
        transcripts = _load_transcripts(asr_path)
        asr_report = read_json(asr_report_path)

    merge_cached = _artifact_is_fresh(
        merged_path,
        [chunks_manifest_path, asr_path],
        invalidate_cache,
    )

    if not merge_cached:
        logger.info("Stage: merge")
        merged_segments = merge_segments(chunks=chunk_records, transcripts=transcripts)
        write_json(merged_path, [item.to_dict() for item in merged_segments])
    else:
        logger.info("Stage: merge skipped (using cache)")
        merged_segments = _load_pipeline_segments(merged_path)

    cleanup_cached = _artifact_is_fresh(
        merged_clean_path,
        [merged_path, asr_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        cleanup_report_path,
        [merged_path, asr_path],
        invalidate_cache,
    )

    if not cleanup_cached:
        logger.info("Stage: cleanup")
        cleaned_segments, cleanup_report = cleanup_transcript_segments(
            segments=merged_segments,
            min_duration_ms=config.cleanup_min_duration_ms,
            duplicate_window_ms=config.cleanup_duplicate_window_ms,
        )
        write_json(merged_clean_path, [item.to_dict() for item in cleaned_segments])
        write_json(cleanup_report_path, cleanup_report)
    else:
        logger.info("Stage: cleanup skipped (using cache)")
        cleaned_segments = _load_pipeline_segments(merged_clean_path)
        cleanup_report = read_json(cleanup_report_path)

    semantic_windows_cached = _artifact_is_fresh(
        semantic_windows_path,
        [merged_clean_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        semantic_windows_report_path,
        [merged_clean_path],
        invalidate_cache,
    )

    if not semantic_windows_cached:
        logger.info("Stage: semantic windows")
        semantic_windows, semantic_windows_report = build_semantic_windows(
            segments=cleaned_segments,
            max_chars=config.llm_window_max_chars,
            max_duration_sec=config.llm_window_max_duration_sec,
            max_gap_sec=config.llm_window_max_gap_sec,
            max_speaker_switches=config.llm_window_max_speaker_switches,
        )
        write_json(semantic_windows_path, semantic_windows)
        write_json(semantic_windows_report_path, semantic_windows_report)
    else:
        logger.info("Stage: semantic windows skipped (using cache)")
        semantic_windows = read_json(semantic_windows_path)
        semantic_windows_report = read_json(semantic_windows_report_path)

    anonymization_cached = _artifact_is_fresh(
        anonymized_path,
        [merged_clean_path, semantic_windows_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        anonymize_report_path,
        [merged_clean_path, semantic_windows_path],
        invalidate_cache,
    )

    if not config.use_vertex:
        logger.info("Vertex is disabled. Building pass-through anonymized/enhanced outputs")
        applied_text_enhancement_mode = "deterministic"
        anonymized_segments = []
        for segment in cleaned_segments:
            anonymized_segments.append(
                PipelineSegment(
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                    raw_text=segment.raw_text,
                    anonymized_text=segment.raw_text,
                    enhanced_text="",
                    chunk_path=segment.chunk_path,
                    asr_status=segment.asr_status,
                )
            )

        final_segments, enhancement_report = enhance_segments_deterministic(anonymized_segments)

        write_json(anonymized_path, [item.to_dict() for item in anonymized_segments])
        write_json(enhanced_path, [item.to_dict() for item in final_segments])
        anonymize_report = {
            "llm_calls_total": 0,
            "llm_latency_total_ms": 0.0,
            "avg_chars_per_llm_call": 0.0,
            "avg_segments_per_llm_call": 0.0,
            "entity_mask_counts": {},
            "windows_total": len(semantic_windows),
        }
        write_json(anonymize_report_path, anonymize_report)
        write_json(enhancement_report_path, enhancement_report)
    else:
        logger.info("Stage: Vertex AI anonymization")
        applied_text_enhancement_mode = config.text_enhancement_mode
        text_processor = VertexTextProcessor(
            api_key=config.google_api_key,
            project=config.vertex_project,
            location=config.vertex_location,
            model_name=config.vertex_model_name,
            enabled=config.use_vertex,
            strict=True,
        )

        if not anonymization_cached:
            anonymized_segments, anonymize_report = anonymize_segments(
                cleaned_segments,
                semantic_windows,
                text_processor,
            )
            write_json(anonymized_path, [item.to_dict() for item in anonymized_segments])
            write_json(anonymize_report_path, anonymize_report)
        else:
            logger.info("Stage: anonymization skipped (using cache)")
            anonymized_segments = _load_pipeline_segments(anonymized_path)
            anonymize_report = read_json(anonymize_report_path)

        logger.info("Stage: Vertex AI enhancement")
        text_enhancement_cached = _artifact_is_fresh(
            enhanced_path,
            [anonymized_path, semantic_windows_path],
            invalidate_cache,
        ) and _artifact_is_fresh(
            enhancement_report_path,
            [anonymized_path, semantic_windows_path],
            invalidate_cache,
        )
        if not text_enhancement_cached:
            low_confidence_positions = _low_confidence_positions(
                anonymized_segments,
                min_cps=config.low_confidence_min_cps,
                max_cps=config.low_confidence_max_cps,
            )

            if config.text_enhancement_mode == "deterministic":
                final_segments, enhancement_report = enhance_segments_deterministic(anonymized_segments)
            else:
                final_segments, enhancement_report = enhance_segments(
                    anonymized_segments,
                    semantic_windows,
                    text_processor,
                    low_confidence_positions=low_confidence_positions,
                    skip_low_confidence=config.enhancement_skip_low_confidence,
                )
            write_json(enhanced_path, [item.to_dict() for item in final_segments])
            write_json(enhancement_report_path, enhancement_report)
        else:
            logger.info("Stage: enhancement skipped (using cache)")
            final_segments = _load_pipeline_segments(enhanced_path)
            enhancement_report = read_json(enhancement_report_path)

    chunk_quality_cached = _artifact_is_fresh(
        chunk_quality_path,
        [chunks_manifest_path, asr_path, enhanced_path],
        invalidate_cache,
    ) and _artifact_is_fresh(
        chunk_quality_report_path,
        [chunks_manifest_path, asr_path, enhanced_path],
        invalidate_cache,
    )

    if not chunk_quality_cached:
        logger.info("Stage: chunk quality analytics")
        chunk_quality_rows, chunk_quality_report = build_chunk_quality_analytics(
            chunks=chunk_records,
            transcripts=transcripts,
            final_segments=final_segments,
            low_confidence_min_cps=config.low_confidence_min_cps,
            low_confidence_max_cps=config.low_confidence_max_cps,
        )
        write_json(chunk_quality_path, chunk_quality_rows)
        write_json(chunk_quality_report_path, chunk_quality_report)
    else:
        logger.info("Stage: chunk quality analytics skipped (using cache)")
        chunk_quality_rows = read_json(chunk_quality_path)
        chunk_quality_report = read_json(chunk_quality_report_path)

    audio_duration_sec = max(
        0.0,
        float(normalize_meta.get("duration_sec", enhancement_meta.get("duration_sec", 0.0))),
    )
    audio_minutes = (audio_duration_sec / 60.0) if audio_duration_sec > 0 else 0.0

    total_segments_before = int(segment_postprocess_report.get("input_segments", len(diarization_segments)))
    total_segments_after = int(segment_postprocess_report.get("output_segments", len(diarization_segments_for_asr)))
    short_after = int(segment_postprocess_report.get("short_segments_after", 0))
    duration_stats = _duration_stats(diarization_segments_for_asr)

    asr_latency_total_ms = float(asr_report.get("asr_latency_total_ms", 0.0))
    asr_latency_per_min_audio_ms = float(asr_report.get("asr_latency_per_min_audio_ms", 0.0))
    anonymize_calls = int(anonymize_report.get("llm_calls_total", 0))
    enhancement_calls = int(enhancement_report.get("llm_calls_total", 0))
    llm_calls_total = anonymize_calls + enhancement_calls
    llm_latency_total_ms = float(anonymize_report.get("llm_latency_total_ms", 0.0)) + float(
        enhancement_report.get("llm_latency_total_ms", 0.0)
    )
    llm_chars_total = (
        float(anonymize_report.get("avg_chars_per_llm_call", 0.0)) * anonymize_calls
        + float(enhancement_report.get("avg_chars_per_llm_call", 0.0)) * enhancement_calls
    )
    llm_segments_total = (
        float(anonymize_report.get("avg_segments_per_llm_call", 0.0)) * anonymize_calls
        + float(enhancement_report.get("avg_segments_per_llm_call", 0.0)) * enhancement_calls
    )

    cleanup_removed_segments = int(cleanup_report.get("input_segments", 0)) - int(
        cleanup_report.get("output_segments", 0)
    )
    empty_transcript_ratio = (
        sum(1 for item in final_segments if not item.raw_text.strip()) / max(1, len(final_segments))
    )

    enhanced_texts = [item.enhanced_text for item in final_segments if item.enhanced_text.strip()]
    metrics = {
        "total_segments_before_postprocess": total_segments_before,
        "total_segments_after_postprocess": total_segments_after,
        "avg_segment_duration_sec": duration_stats["avg_segment_duration_sec"],
        "median_segment_duration_sec": duration_stats["median_segment_duration_sec"],
        "p10_segment_duration_sec": duration_stats["p10_segment_duration_sec"],
        "p90_segment_duration_sec": duration_stats["p90_segment_duration_sec"],
        "short_segment_ratio": (short_after / max(1, total_segments_after)),
        "merged_segment_count": int(segment_postprocess_report.get("merged_segments_count", 0)),
        "asr_chunks_total": int(asr_report.get("total_chunks", len(chunk_records))),
        "asr_chunks_skipped": int(asr_report.get("skipped_short_chunks", 0)),
        "asr_latency_total_ms": asr_latency_total_ms,
        "asr_latency_per_min_audio_ms": asr_latency_per_min_audio_ms,
        "llm_calls_total": llm_calls_total,
        "llm_calls_per_min_audio": (llm_calls_total / audio_minutes) if audio_minutes else 0.0,
        "llm_latency_total_ms": llm_latency_total_ms,
        "llm_latency_per_min_audio_ms": (llm_latency_total_ms / audio_minutes) if audio_minutes else 0.0,
        "empty_transcript_ratio": empty_transcript_ratio,
        "cleanup_removed_segments": cleanup_removed_segments,
        "audio_coverage_ratio": float(asr_report.get("asr_audio_coverage_ratio", 0.0)),
        "asr_provider_requested": str(asr_report.get("asr_provider_requested", config.asr_provider)),
        "asr_provider_effective": str(asr_report.get("asr_provider_effective", config.asr_provider)),
        "asr_fallback_used": bool(asr_report.get("asr_fallback_used", False)),
        "avg_chars_per_llm_call": (llm_chars_total / llm_calls_total) if llm_calls_total else 0.0,
        "avg_segments_per_llm_call": (llm_segments_total / llm_calls_total) if llm_calls_total else 0.0,
        "language_switching_ratio": _language_switching_ratio(enhanced_texts),
        "punctuation_density_after_enhancement": _punctuation_density(enhanced_texts),
        "semantic_drift_flags_total": int(enhancement_report.get("semantic_drift_flags_total", 0)),
        "anonymization_recall_proxy": float(anonymize_report.get("anonymization_recall_proxy", 0.0)),
        "low_confidence_chunk_ratio": float(chunk_quality_report.get("low_confidence_chunk_ratio", 0.0)),
        "code_switch_chunk_ratio": float(chunk_quality_report.get("code_switch_chunk_ratio", 0.0)),
        "suspicious_substitution_chunks": int(chunk_quality_report.get("suspicious_substitution_chunks", 0)),
        "enhancement_mode": applied_text_enhancement_mode,
    }
    write_json(metrics_path, metrics)

    overlap_conflicts_count = int(segment_postprocess_report.get("overlap_conflicts_count", 0)) + int(
        cleanup_report.get("overlap_conflicts_count", 0)
    )

    quality_debug = {
        "total_diarization_segments": len(diarization_segments),
        "total_asr_chunks": int(asr_report.get("total_chunks", len(chunk_records))),
        "skipped_short_chunks": int(asr_report.get("skipped_short_chunks", 0)),
        "empty_segments_removed": int(cleanup_report.get("empty_segments_removed", 0)),
        "merged_segments_count": int(segment_postprocess_report.get("merged_segments_count", 0)),
        "overlap_conflicts_count": overlap_conflicts_count,
        "audio_preprocessing_mode": "enhanced" if config.enable_audio_enhancement else "normalized",
        "semantic_windows_total": int(semantic_windows_report.get("windows_total", 0)),
        "llm_calls_total": llm_calls_total,
        "llm_latency_total_ms": llm_latency_total_ms,
        "enhancement_mode": applied_text_enhancement_mode,
        "asr_provider_requested": str(asr_report.get("asr_provider_requested", config.asr_provider)),
        "asr_provider_effective": str(asr_report.get("asr_provider_effective", config.asr_provider)),
        "asr_fallback_used": bool(asr_report.get("asr_fallback_used", False)),
        "asr_fallback_reason": str(asr_report.get("asr_fallback_reason", "")),
        "segment_postprocess_report": segment_postprocess_report,
        "segment_merge_groups_count": len(segment_merge_groups),
        "segment_merge_groups_path": str(segment_merge_groups_path),
        "asr_report": asr_report,
        "cleanup_report": cleanup_report,
        "semantic_windows_report": semantic_windows_report,
        "anonymize_report": anonymize_report,
        "enhancement_report": enhancement_report,
        "chunk_quality_report": chunk_quality_report,
        "chunk_quality_path": str(chunk_quality_path),
        "metrics_path": str(metrics_path),
    }

    result = {
        "input_file": str(config.input_path),
        "normalized_audio": str(normalized_audio_path),
        "enhanced_audio": str(pipeline_audio_path),
        "audio_enhancement": enhancement_meta,
        "artifacts_dir": str(run_dir),
        "result_json_path": str(result_path),
        "quality_debug": quality_debug,
        "metrics": metrics,
        "segments": _result_segments(final_segments),
    }

    write_json(state_path, current_state)
    write_json(result_path, result)
    return result
