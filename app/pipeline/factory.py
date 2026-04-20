"""Factory for creating pipeline stage instances.

Provides centralized dependency injection and stage instantiation.
"""

import logging
from typing import Any

from app.config import PipelineConfig
from app.domain.contracts import (
    ASRAdapter,
    AudioEnhancer,
    AudioNormalizer,
    ChunkBuilder,
    ChunkQualityAnalyzer,
    Diarizer,
    SemanticWindowBuilder,
    SegmentMerger,
    SegmentPostprocessor,
    TextProcessor,
    TranscriptCleaner,
)

logger = logging.getLogger(__name__)


class StageFactory:
    """Factory for creating pipeline stage implementations.

    This factory centralizes the creation of stage instances based on
    configuration, enabling easy substitution of implementations and
    simplifying testing through dependency injection.
    """

    @staticmethod
    def create_normalizer(_config: PipelineConfig) -> AudioNormalizer:
        """Create audio normalizer instance."""
        from app.stages.normalize import SoXNormalizer

        return SoXNormalizer()

    @staticmethod
    def create_enhancer(config: PipelineConfig) -> AudioEnhancer:
        """Create audio enhancer instance."""
        from app.stages.enhance_audio import SpectralEnhancer

        return SpectralEnhancer()

    @staticmethod
    def create_diarizer(config: PipelineConfig) -> Diarizer:
        """Create diarizer instance."""
        from app.stages.diarization import PyannoteDiarizer

        if not config.hf_token:
            raise ValueError(
                "HF token is required for diarization. "
                "Set HF_TOKEN or pass --hf-token."
            )

        return PyannoteDiarizer(
            hf_token=config.hf_token,
            device=config.device,
        )

    @staticmethod
    def create_segment_postprocessor(_config: PipelineConfig) -> SegmentPostprocessor:
        """Create segment postprocessor instance."""
        from app.stages.postprocess_segments import SegmentNormalizer

        return SegmentNormalizer()

    @staticmethod
    def create_chunk_builder(_config: PipelineConfig) -> ChunkBuilder:
        """Create chunk builder instance."""
        from app.stages.segmentation import WavChunkBuilder

        return WavChunkBuilder()

    @staticmethod
    def create_asr_adapter(config: PipelineConfig) -> ASRAdapter:
        """Create ASR adapter with fallback support."""
        from app.stages.asr_adapters import make_asr_adapter

        primary_model_name = (
            config.chirp_model_name
            if config.asr_provider == "chirp"
            else config.nemo_model_name
        )
        fallback_model_name = (
            config.nemo_model_name
            if config.asr_fallback_provider == "nemo"
            else config.chirp_model_name
        )

        return make_asr_adapter(
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

    @staticmethod
    def create_segment_merger(_config: PipelineConfig) -> SegmentMerger:
        """Create segment merger instance."""
        from app.stages.merge import ChunkTranscriptMerger

        return ChunkTranscriptMerger()

    @staticmethod
    def create_transcript_cleaner(config: PipelineConfig) -> TranscriptCleaner:
        """Create transcript cleaner instance."""
        from app.stages.cleanup import SegmentCleaner

        return SegmentCleaner(
            min_duration_ms=config.cleanup_min_duration_ms,
            duplicate_window_ms=config.cleanup_duplicate_window_ms,
        )

    @staticmethod
    def create_semantic_window_builder(config: PipelineConfig) -> SemanticWindowBuilder:
        """Create semantic window builder instance."""
        from app.stages.semantic_windows import SemanticWindowGrouper

        return SemanticWindowGrouper(
            max_chars=config.llm_window_max_chars,
            max_duration_sec=config.llm_window_max_duration_sec,
            max_gap_sec=config.llm_window_max_gap_sec,
            max_speaker_switches=config.llm_window_max_speaker_switches,
        )

    @staticmethod
    def create_text_processor(config: PipelineConfig) -> TextProcessor:
        """Create text processor instance."""
        from app.stages.vertex_text import VertexTextProcessor

        return VertexTextProcessor(
            api_key=config.google_api_key,
            project=config.vertex_project,
            location=config.vertex_location,
            model_name=config.vertex_model_name,
            enabled=config.use_vertex,
            strict=True,
        )

    @staticmethod
    def create_chunk_quality_analyzer(config: PipelineConfig) -> ChunkQualityAnalyzer:
        """Create chunk quality analyzer instance."""
        from app.stages.chunk_quality import QualityAnalyzer

        return QualityAnalyzer(
            low_confidence_min_cps=config.low_confidence_min_cps,
            low_confidence_max_cps=config.low_confidence_max_cps,
        )

    @classmethod
    def create_all_stages(
        cls, config: PipelineConfig
    ) -> dict[str, Any]:
        """Create all pipeline stage instances.

        Returns a dict mapping stage names to instantiated classes.
        Useful for debugging and introspection.
        """
        return {
            "normalizer": cls.create_normalizer(config),
            "enhancer": cls.create_enhancer(config),
            "diarizer": cls.create_diarizer(config),
            "segment_postprocessor": cls.create_segment_postprocessor(config),
            "chunk_builder": cls.create_chunk_builder(config),
            "asr_adapter": cls.create_asr_adapter(config),
            "segment_merger": cls.create_segment_merger(config),
            "transcript_cleaner": cls.create_transcript_cleaner(config),
            "semantic_window_builder": cls.create_semantic_window_builder(config),
            "text_processor": cls.create_text_processor(config),
            "chunk_quality_analyzer": cls.create_chunk_quality_analyzer(config),
        }
