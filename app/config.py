from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field


class PipelineConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    input_path: Path
    artifacts_root: Path = Field(default=Path(".runtime"))
    target_sample_rate: int = 16000
    enable_audio_enhancement: bool = True
    audio_denoise_strength: float = 1.0
    audio_noise_quantile: float = 0.15
    audio_highpass_hz: int = 60
    audio_target_rms_dbfs: float = -22.0
    audio_target_peak_dbfs: float = -1.0
    segment_min_duration_ms: int = 450
    segment_merge_gap_ms: int = 300
    segment_padding_ms: int = 60
    segment_absorb_short_gap_ms: int = 220
    asr_batch_size: int = 8
    asr_orchestration_batch_size: int = 64
    asr_min_chunk_duration_sec: float = 0.25
    asr_pretokenize: bool = False
    asr_provider: str = "chirp"
    asr_fallback_provider: str = "nemo"
    chirp_model_name: str = "chirp_2"
    chirp_language_code: str = "ru-RU"
    chirp_project: str = ""
    chirp_location: str = "global"
    chirp_recognizer: str = ""
    cleanup_min_duration_ms: int = 350
    cleanup_duplicate_window_ms: int = 280
    llm_window_max_chars: int = 1000
    llm_window_max_duration_sec: float = 25.0
    llm_window_max_gap_sec: float = 1.2
    llm_window_max_speaker_switches: int = 6
    text_enhancement_mode: str = "deterministic"
    enhancement_skip_low_confidence: bool = True
    low_confidence_min_cps: float = 1.5
    low_confidence_max_cps: float = 28.0
    hf_token: str = ""
    nemo_model_name: str = "nvidia/stt_kk_ru_fastconformer_hybrid_large"
    google_api_key: str = ""
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_model_name: str = "gemini-2.5-flash"
    use_vertex: bool = True
    force: bool = False
    device: str = Field(default_factory=lambda: PipelineConfig.detect_device())

    @staticmethod
    def detect_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def run_dir(self) -> Path:
        resolved_input = str(self.input_path.expanduser().resolve())
        digest = hashlib.sha1(resolved_input.encode("utf-8")).hexdigest()[:10]
        safe_stem = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in self.input_path.stem
        ).strip("_") or "audio"
        return self.artifacts_root / f"{safe_stem}_{digest}"

    @classmethod
    def create(cls, **kwargs) -> PipelineConfig:
        return cls(**kwargs)
