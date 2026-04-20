from __future__ import annotations
import logging
from typing import Any

from app.config import PipelineConfig
from app.io_utils import read_json, write_json
from app.models import ChunkRecord, DiarizationSegment, PipelineSegment, TranscribedSegment
from app.pipeline.artifact_manager import ArtifactManager
from app.pipeline.factory import StageFactory
from app.stages.text_metrics import language_switching_ratio, punctuation_density

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the audio processing pipeline stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.artifacts = ArtifactManager(config)
        self.factory = StageFactory()

    def _get_fingerprint(self) -> dict[str, Any]:
        """Generate a configuration fingerprint for cache validation."""
        return {
            "input_file": str(self.config.input_path.expanduser().resolve()),
            "target_sample_rate": self.config.target_sample_rate,
            "enable_audio_enhancement": self.config.enable_audio_enhancement,
            "audio_denoise_strength": self.config.audio_denoise_strength,
            "audio_noise_quantile": self.config.audio_noise_quantile,
            "audio_highpass_hz": self.config.audio_highpass_hz,
            "audio_target_rms_dbfs": self.config.audio_target_rms_dbfs,
            "audio_target_peak_dbfs": self.config.audio_target_peak_dbfs,
            "segment_min_duration_ms": self.config.segment_min_duration_ms,
            "segment_merge_gap_ms": self.config.segment_merge_gap_ms,
            "segment_padding_ms": self.config.segment_padding_ms,
            "segment_absorb_short_gap_ms": self.config.segment_absorb_short_gap_ms,
            "asr_batch_size": self.config.asr_batch_size,
            "asr_orchestration_batch_size": self.config.asr_orchestration_batch_size,
            "asr_min_chunk_duration_sec": self.config.asr_min_chunk_duration_sec,
            "asr_pretokenize": self.config.asr_pretokenize,
            "asr_provider": self.config.asr_provider,
            "asr_fallback_provider": self.config.asr_fallback_provider,
            "cleanup_min_duration_ms": self.config.cleanup_min_duration_ms,
            "cleanup_duplicate_window_ms": self.config.cleanup_duplicate_window_ms,
            "llm_window_max_chars": self.config.llm_window_max_chars,
            "llm_window_max_duration_sec": self.config.llm_window_max_duration_sec,
            "llm_window_max_gap_sec": self.config.llm_window_max_gap_sec,
            "llm_window_max_speaker_switches": self.config.llm_window_max_speaker_switches,
            "text_enhancement_mode": self.config.text_enhancement_mode,
            "enhancement_skip_low_confidence": self.config.enhancement_skip_low_confidence,
            "low_confidence_min_cps": self.config.low_confidence_min_cps,
            "low_confidence_max_cps": self.config.low_confidence_max_cps,
            "device": self.config.device,
        }

    def run(self) -> dict[str, Any]:
        """Run the full pipeline."""
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_path}")

        self.artifacts.ensure_run_dir()
        fingerprint = self._get_fingerprint()
        if self.artifacts.check_cache_invalidation(fingerprint):
            logger.info("Cache invalidated or missing")

        # 1. Normalize
        normalized_audio_path = self.artifacts.get_artifact_path("normalized.wav")
        normalize_meta_path = self.artifacts.get_artifact_path("normalize.json")
        
        if not self.artifacts.is_fresh(normalized_audio_path, [self.config.input_path]):
            logger.info("Stage: normalize")
            normalizer = self.factory.create_normalizer(self.config)
            normalize_meta = normalizer.normalize(
                self.config.input_path, normalized_audio_path, self.config.target_sample_rate
            )
            write_json(normalize_meta_path, normalize_meta)
        else:
            logger.info("Stage: normalize skipped (cache)")
            normalize_meta = read_json(normalize_meta_path)

        pipeline_audio_path = normalized_audio_path
        enhancement_meta = {}

        # 2. Enhance
        if self.config.enable_audio_enhancement:
            enhanced_audio_path = self.artifacts.get_artifact_path("enhanced.wav")
            enhancement_meta_path = self.artifacts.get_artifact_path("audio_enhancement.json")
            
            if not self.artifacts.is_fresh(enhanced_audio_path, [normalized_audio_path]):
                logger.info("Stage: audio enhancement")
                enhancer = self.factory.create_enhancer(self.config)
                enhancement_meta = enhancer.enhance(
                    normalized_audio_path,
                    enhanced_audio_path,
                    self.config.target_sample_rate,
                    self.config.audio_denoise_strength,
                    self.config.audio_noise_quantile,
                    self.config.audio_highpass_hz,
                    self.config.audio_target_rms_dbfs,
                    self.config.audio_target_peak_dbfs,
                )
                write_json(enhancement_meta_path, enhancement_meta)
            else:
                logger.info("Stage: audio enhancement skipped (cache)")
                enhancement_meta = read_json(enhancement_meta_path)
            pipeline_audio_path = enhanced_audio_path

        # 3. Diarization
        diarization_path = self.artifacts.get_artifact_path("diarization.json")
        if not self.artifacts.is_fresh(diarization_path, [pipeline_audio_path]):
            logger.info("Stage: diarization")
            diarizer = self.factory.create_diarizer(self.config)
            diarization_segments = diarizer.diarize(pipeline_audio_path)
            write_json(diarization_path, [s.to_dict() for s in diarization_segments])
        else:
            logger.info("Stage: diarization skipped (cache)")
            diarization_segments = [DiarizationSegment.from_dict(s) for s in read_json(diarization_path)]

        # 4. Post-process segments
        postprocessed_path = self.artifacts.get_artifact_path("diarization_postprocessed.json")
        report_path = self.artifacts.get_artifact_path("segment_postprocess_report.json")
        groups_path = self.artifacts.get_artifact_path("segment_merge_groups.json")
        
        if not self.artifacts.is_fresh(postprocessed_path, [diarization_path]):
            logger.info("Stage: segment post-processing")
            postprocessor = self.factory.create_segment_postprocessor(self.config)
            processed_segments, report, groups = postprocessor.postprocess(
                diarization_segments,
                float(normalize_meta.get("duration_sec", 0.0)),
                self.config.segment_min_duration_ms,
                self.config.segment_merge_gap_ms,
                self.config.segment_padding_ms,
                self.config.segment_absorb_short_gap_ms,
            )
            write_json(postprocessed_path, [s.to_dict() for s in processed_segments])
            write_json(report_path, report)
            write_json(groups_path, groups)
        else:
            logger.info("Stage: segment post-processing skipped (cache)")
            processed_segments = [DiarizationSegment.from_dict(s) for s in read_json(postprocessed_path)]
            report = read_json(report_path)
            groups = read_json(groups_path)

        # 5. Segmentation (Chunks)
        chunks_manifest_path = self.artifacts.get_artifact_path("segments_manifest.json")
        chunks_dir = self.artifacts.get_artifact_path("chunks")
        
        if not self.artifacts.is_fresh(chunks_manifest_path, [pipeline_audio_path, postprocessed_path]):
            logger.info("Stage: segmentation")
            chunk_builder = self.factory.create_chunk_builder(self.config)
            chunk_records = chunk_builder.build_chunks(
                pipeline_audio_path,
                processed_segments,
                chunks_dir,
                self.config.target_sample_rate,
            )
            write_json(chunks_manifest_path, [c.to_dict() for c in chunk_records])
        else:
            logger.info("Stage: segmentation skipped (cache)")
            chunk_records = [ChunkRecord.from_dict(c) for c in read_json(chunks_manifest_path)]

        # 6. ASR
        asr_raw_path = self.artifacts.get_artifact_path("asr_raw.json")
        asr_report_path = self.artifacts.get_artifact_path("asr_report.json")
        
        if not self.artifacts.is_fresh(asr_raw_path, [chunks_manifest_path]):
            logger.info("Stage: ASR")
            adapter = self.factory.create_asr_adapter(self.config)
            from app.stages.asr_transcribe import transcribe_chunks
            transcripts, asr_report = transcribe_chunks(
                chunks=chunk_records,
                adapter=adapter,
                batch_size=self.config.asr_batch_size,
                orchestration_batch_size=self.config.asr_orchestration_batch_size,
                min_chunk_duration_sec=self.config.asr_min_chunk_duration_sec,
                pretokenize=self.config.asr_pretokenize,
            )
            write_json(asr_raw_path, [t.to_dict() for t in transcripts])
            write_json(asr_report_path, asr_report)
        else:
            logger.info("Stage: ASR skipped (cache)")
            transcripts = [TranscribedSegment.from_dict(t) for t in read_json(asr_raw_path)]
            asr_report = read_json(asr_report_path)

        # 7. Merge
        merged_path = self.artifacts.get_artifact_path("merged_raw.json")
        if not self.artifacts.is_fresh(merged_path, [chunks_manifest_path, asr_raw_path]):
            logger.info("Stage: merge")
            merger = self.factory.create_segment_merger(self.config)
            merged_segments = merger.merge(chunk_records, transcripts)
            write_json(merged_path, [s.to_dict() for s in merged_segments])
        else:
            logger.info("Stage: merge skipped (cache)")
            merged_segments = [PipelineSegment.from_dict(s) for s in read_json(merged_path)]

        # 8. Cleanup
        cleaned_path = self.artifacts.get_artifact_path("merged_clean.json")
        cleanup_report_path = self.artifacts.get_artifact_path("cleanup_report.json")
        if not self.artifacts.is_fresh(cleaned_path, [merged_path]):
            logger.info("Stage: cleanup")
            cleaner = self.factory.create_transcript_cleaner(self.config)
            cleaned_segments, cleanup_report = cleaner.cleanup(merged_segments, self.config.cleanup_min_duration_ms, self.config.cleanup_duplicate_window_ms)
            write_json(cleaned_path, [s.to_dict() for s in cleaned_segments])
            write_json(cleanup_report_path, cleanup_report)
        else:
            logger.info("Stage: cleanup skipped (cache)")
            cleaned_segments = [PipelineSegment.from_dict(s) for s in read_json(cleaned_path)]
            cleanup_report = read_json(cleanup_report_path)

        # 9. Semantic Windows
        windows_path = self.artifacts.get_artifact_path("semantic_windows.json")
        windows_report_path = self.artifacts.get_artifact_path("semantic_windows_report.json")
        if not self.artifacts.is_fresh(windows_path, [cleaned_path]):
            logger.info("Stage: semantic windows")
            window_builder = self.factory.create_semantic_window_builder(self.config)
            windows, windows_report = window_builder.build_windows(
                cleaned_segments,
                self.config.llm_window_max_chars,
                self.config.llm_window_max_duration_sec,
                self.config.llm_window_max_gap_sec,
                self.config.llm_window_max_speaker_switches,
            )
            write_json(windows_path, windows)
            write_json(windows_report_path, windows_report)
        else:
            logger.info("Stage: semantic windows skipped (cache)")
            windows = read_json(windows_path)
            windows_report = read_json(windows_report_path)

        # 10. Text Processing (Anonymization & Enhancement)
        text_processor = self.factory.create_text_processor(self.config)
        anonymized_path = self.artifacts.get_artifact_path("anonymized.json")
        anonymize_report_path = self.artifacts.get_artifact_path("anonymize_report.json")
        
        if not self.artifacts.is_fresh(anonymized_path, [cleaned_path, windows_path]):
            logger.info("Stage: text anonymization")
            anonymized_segments, anonymize_report = text_processor.anonymize(cleaned_segments, windows)
            write_json(anonymized_path, [s.to_dict() for s in anonymized_segments])
            write_json(anonymize_report_path, anonymize_report)
        else:
            logger.info("Stage: text anonymization skipped (cache)")
            anonymized_segments = [PipelineSegment.from_dict(s) for s in read_json(anonymized_path)]
            anonymize_report = read_json(anonymize_report_path)

        enhanced_path = self.artifacts.get_artifact_path("enhanced.json")
        enhancement_report_path = self.artifacts.get_artifact_path("enhancement_report.json")
        
        if not self.artifacts.is_fresh(enhanced_path, [anonymized_path]):
            logger.info("Stage: text enhancement")
            # Logic for low confidence positions...
            from app.pipeline.orchestrator import _low_confidence_positions
            low_conf = _low_confidence_positions(
                anonymized_segments, self.config.low_confidence_min_cps, self.config.low_confidence_max_cps
            )
            final_segments, enhancement_report = text_processor.enhance(
                anonymized_segments,
                windows,
                self.config.text_enhancement_mode,
                low_conf,
                self.config.enhancement_skip_low_confidence,
            )
            write_json(enhanced_path, [s.to_dict() for s in final_segments])
            write_json(enhancement_report_path, enhancement_report)
        else:
            logger.info("Stage: text enhancement skipped (cache)")
            final_segments = [PipelineSegment.from_dict(s) for s in read_json(enhanced_path)]
            enhancement_report = read_json(enhancement_report_path)

        # 11. Quality Analytics
        quality_path = self.artifacts.get_artifact_path("chunk_quality.json")
        quality_report_path = self.artifacts.get_artifact_path("chunk_quality_report.json")
        if not self.artifacts.is_fresh(quality_path, [chunks_manifest_path, asr_raw_path, enhanced_path]):
            logger.info("Stage: quality analytics")
            analyzer = self.factory.create_chunk_quality_analyzer(self.config)
            quality_rows, quality_report = analyzer.analyze(
                chunk_records, transcripts, final_segments, self.config.low_confidence_min_cps, self.config.low_confidence_max_cps
            )
            write_json(quality_path, quality_rows)
            write_json(quality_report_path, quality_report)
        else:
            logger.info("Stage: quality analytics skipped (cache)")
            quality_rows = read_json(quality_path)
            quality_report = read_json(quality_report_path)

        # 12. Metrics & Final Result
        audio_duration_sec = float(normalize_meta.get("duration_sec", 0.0))
        audio_minutes = (audio_duration_sec / 60.0) if audio_duration_sec > 0 else 0.0

        total_segments_before = int(report.get("input_segments", len(diarization_segments)))
        total_segments_after = int(report.get("output_segments", len(processed_segments)))
        
        asr_latency_total_ms = float(asr_report.get("asr_latency_total_ms", 0.0))
        anonymize_calls = int(anonymize_report.get("llm_calls_total", 0))
        enhancement_calls = int(enhancement_report.get("llm_calls_total", 0))
        llm_calls_total = anonymize_calls + enhancement_calls
        llm_latency_total_ms = float(anonymize_report.get("llm_latency_total_ms", 0.0)) + float(
            enhancement_report.get("llm_latency_total_ms", 0.0)
        )

        enhanced_texts = [item.enhanced_text for item in final_segments if item.enhanced_text.strip()]
        metrics = {
            "total_segments_before_postprocess": total_segments_before,
            "total_segments_after_postprocess": total_segments_after,
            "asr_chunks_total": int(asr_report.get("total_chunks", len(chunk_records))),
            "asr_latency_total_ms": asr_latency_total_ms,
            "llm_calls_total": llm_calls_total,
            "llm_latency_total_ms": llm_latency_total_ms,
            "language_switching_ratio": language_switching_ratio(enhanced_texts),
            "punctuation_density_after_enhancement": punctuation_density(enhanced_texts),
            "semantic_drift_flags_total": int(enhancement_report.get("semantic_drift_flags_total", 0)),
            "low_confidence_chunk_ratio": float(quality_report.get("low_confidence_chunk_ratio", 0.0)),
        }

        result = {
            "input_file": str(self.config.input_path),
            "artifacts_dir": str(self.config.run_dir),
            "metrics": metrics,
            "segments": [s.to_dict() for s in final_segments],
        }
        
        self.artifacts.save_state(fingerprint)
        write_json(self.artifacts.get_artifact_path("result.json"), result)
        return result

def _low_confidence_positions(
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
