"""Stage implementations for the audio pipeline."""

from app.stages.asr_adapters import (
    FallbackASRAdapter,
    GoogleChirpASRAdapter,
    NeMoASRAdapter,
)
from app.stages.chunk_quality import QualityAnalyzer
from app.stages.cleanup import SegmentCleaner
from app.stages.diarization import PyannoteDiarizer
from app.stages.enhance_audio import SpectralEnhancer
from app.stages.merge import ChunkTranscriptMerger
from app.stages.normalize import SoXNormalizer
from app.stages.postprocess_segments import SegmentNormalizer
from app.stages.segmentation import WavChunkBuilder
from app.stages.semantic_windows import SemanticWindowGrouper
from app.stages.vertex_text import VertexTextProcessor

__all__ = [
    "FallbackASRAdapter",
    "GoogleChirpASRAdapter",
    "NeMoASRAdapter",
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
