"""Abstract contracts for audio pipeline stages.

All pipeline stage implementations must inherit from these ABCs.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.models import (
    ChunkRecord,
    DiarizationSegment,
    PipelineSegment,
    TranscribedSegment,
)


class AudioNormalizer(ABC):
    """Normalizes input audio to target sample rate and format."""

    @abstractmethod
    def normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
    ) -> dict[str, Any]:
        ...


class AudioEnhancer(ABC):
    """Enhances audio quality through denoising, filtering, and normalization."""

    @abstractmethod
    def enhance(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        denoise_strength: float,
        noise_quantile: float,
        highpass_hz: int,
        target_rms_dbfs: float,
        target_peak_dbfs: float,
    ) -> dict[str, Any]:
        ...


class Diarizer(ABC):
    """Performs speaker diarization to identify who spoke when."""

    @abstractmethod
    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        ...


class SegmentPostprocessor(ABC):
    """Post-processes diarization segments for ASR readiness."""

    @abstractmethod
    def postprocess(
        self,
        segments: list[DiarizationSegment],
        audio_duration_sec: float,
        min_duration_ms: int,
        merge_gap_ms: int,
        padding_ms: int,
        absorb_short_gap_ms: int,
    ) -> tuple[list[DiarizationSegment], dict[str, Any], list[Any]]:
        ...


class ChunkBuilder(ABC):
    """Builds audio chunks from diarization segments."""

    @abstractmethod
    def build_chunks(
        self,
        normalized_audio_path: Path,
        segments: list[DiarizationSegment],
        chunks_dir: Path,
        sample_rate: int,
    ) -> list[ChunkRecord]:
        ...


class ASRAdapter(ABC):
    """Abstract adapter for Automatic Speech Recognition providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def transcribe_batch(
        self,
        audio_paths: list[Path],
        batch_size: int,
        pretokenize: bool,
    ) -> list[str]:
        ...


class SegmentMerger(ABC):
    """Merges chunk records with transcribed segments."""

    @abstractmethod
    def merge(
        self,
        chunks: list[ChunkRecord],
        transcripts: list[TranscribedSegment],
    ) -> list[PipelineSegment]:
        ...


class TranscriptCleaner(ABC):
    """Cleans up transcript segments (duplicates, short segments)."""

    @abstractmethod
    def cleanup(
        self,
        segments: list[PipelineSegment],
        min_duration_ms: int,
        duplicate_window_ms: int,
    ) -> tuple[list[PipelineSegment], dict[str, Any]]:
        ...


class SemanticWindowBuilder(ABC):
    """Builds semantic windows for LLM processing."""

    @abstractmethod
    def build_windows(
        self,
        segments: list[PipelineSegment],
        max_chars: int,
        max_duration_sec: float,
        max_gap_sec: float,
        max_speaker_switches: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...


class TextProcessor(ABC):
    """Processes text through LLM for anonymization and enhancement."""

    @abstractmethod
    def anonymize(
        self,
        segments: list[PipelineSegment],
        semantic_windows: list[dict[str, Any]],
    ) -> tuple[list[PipelineSegment], dict[str, Any]]:
        ...

    @abstractmethod
    def enhance(
        self,
        segments: list[PipelineSegment],
        semantic_windows: list[dict[str, Any]] | None,
        mode: str,
        low_confidence_positions: set[int] | None,
        skip_low_confidence: bool,
    ) -> tuple[list[PipelineSegment], dict[str, Any]]:
        ...


class ChunkQualityAnalyzer(ABC):
    """Analyzes quality of chunks and transcriptions."""

    @abstractmethod
    def analyze(
        self,
        chunks: list[ChunkRecord],
        transcripts: list[TranscribedSegment],
        final_segments: list[PipelineSegment],
        low_confidence_min_cps: float,
        low_confidence_max_cps: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...
