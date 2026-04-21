"""Compatibility imports for legacy ``app.stages`` users.

New code should import from ``app.pipeline.stages``.
"""

from app.pipeline.stages.asr.adapters import (
    FallbackASRAdapter,
    NeMoASRAdapter,
    build_single_asr_adapter,
    make_asr_adapter,
)
from app.pipeline.stages.quality.chunk_quality import QualityAnalyzer
from app.pipeline.stages.transcript.cleanup import SegmentCleaner
from app.pipeline.stages.diarization.diarization import PyannoteDiarizer
from app.pipeline.stages.audio.enhance_audio import SpectralEnhancer
from app.pipeline.stages.transcript.merge import ChunkTranscriptMerger
from app.pipeline.stages.audio.normalize import SoXNormalizer
from app.pipeline.stages.diarization.postprocess_segments import SegmentNormalizer
from app.pipeline.stages.chunking.segmentation import WavChunkBuilder
from app.pipeline.stages.transcript.semantic_windows import SemanticWindowGrouper
from app.pipeline.stages.text.vertex_text import VertexTextProcessor

__all__ = [
    "FallbackASRAdapter",
    "NeMoASRAdapter",
    "build_single_asr_adapter",
    "make_asr_adapter",
    "QualityAnalyzer",
    "SegmentCleaner",
    "PyannoteDiarizer",
    "SpectralEnhancer",
    "ChunkTranscriptMerger",
    "SoXNormalizer",
    "SegmentNormalizer",
    "WavChunkBuilder",
    "SemanticWindowGrouper",
    "VertexTextProcessor",
]
