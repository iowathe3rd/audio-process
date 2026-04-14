from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch


@dataclass
class PipelineConfig:
    input_path: Path
    artifacts_root: Path
    target_sample_rate: int
    enable_audio_enhancement: bool
    audio_denoise_strength: float
    audio_noise_quantile: float
    audio_highpass_hz: int
    audio_target_rms_dbfs: float
    audio_target_peak_dbfs: float
    segment_min_duration_ms: int
    segment_merge_gap_ms: int
    segment_padding_ms: int
    segment_absorb_short_gap_ms: int
    asr_batch_size: int
    asr_orchestration_batch_size: int
    asr_min_chunk_duration_sec: float
    asr_pretokenize: bool
    asr_provider: str
    asr_fallback_provider: str
    chirp_model_name: str
    chirp_language_code: str
    chirp_project: str
    chirp_location: str
    chirp_recognizer: str
    cleanup_min_duration_ms: int
    cleanup_duplicate_window_ms: int
    llm_window_max_chars: int
    llm_window_max_duration_sec: float
    llm_window_max_gap_sec: float
    llm_window_max_speaker_switches: int
    text_enhancement_mode: str
    enhancement_skip_low_confidence: bool
    low_confidence_min_cps: float
    low_confidence_max_cps: float
    hf_token: str
    nemo_model_name: str
    google_api_key: str
    vertex_project: str
    vertex_location: str
    vertex_model_name: str
    use_vertex: bool
    force: bool
    device: str

    @classmethod
    def detect_device(cls) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def create(
        cls,
        input_path: str,
        artifacts_root: str,
        target_sample_rate: int,
        enable_audio_enhancement: bool,
        audio_denoise_strength: float,
        audio_noise_quantile: float,
        audio_highpass_hz: int,
        audio_target_rms_dbfs: float,
        audio_target_peak_dbfs: float,
        segment_min_duration_ms: int,
        segment_merge_gap_ms: int,
        segment_padding_ms: int,
        segment_absorb_short_gap_ms: int,
        asr_batch_size: int,
        asr_orchestration_batch_size: int,
        asr_min_chunk_duration_sec: float,
        asr_pretokenize: bool,
        asr_provider: str,
        asr_fallback_provider: str,
        chirp_model_name: str,
        chirp_language_code: str,
        chirp_project: str,
        chirp_location: str,
        chirp_recognizer: str,
        cleanup_min_duration_ms: int,
        cleanup_duplicate_window_ms: int,
        llm_window_max_chars: int,
        llm_window_max_duration_sec: float,
        llm_window_max_gap_sec: float,
        llm_window_max_speaker_switches: int,
        text_enhancement_mode: str,
        enhancement_skip_low_confidence: bool,
        low_confidence_min_cps: float,
        low_confidence_max_cps: float,
        hf_token: str,
        nemo_model_name: str,
        google_api_key: str,
        vertex_project: str,
        vertex_location: str,
        vertex_model_name: str,
        use_vertex: bool,
        force: bool,
        device: str | None = None,
    ) -> "PipelineConfig":
        resolved_device = device or cls.detect_device()
        return cls(
            input_path=Path(input_path),
            artifacts_root=Path(artifacts_root),
            target_sample_rate=target_sample_rate,
            enable_audio_enhancement=enable_audio_enhancement,
            audio_denoise_strength=audio_denoise_strength,
            audio_noise_quantile=audio_noise_quantile,
            audio_highpass_hz=audio_highpass_hz,
            audio_target_rms_dbfs=audio_target_rms_dbfs,
            audio_target_peak_dbfs=audio_target_peak_dbfs,
            segment_min_duration_ms=segment_min_duration_ms,
            segment_merge_gap_ms=segment_merge_gap_ms,
            segment_padding_ms=segment_padding_ms,
            segment_absorb_short_gap_ms=segment_absorb_short_gap_ms,
            asr_batch_size=asr_batch_size,
            asr_orchestration_batch_size=asr_orchestration_batch_size,
            asr_min_chunk_duration_sec=asr_min_chunk_duration_sec,
            asr_pretokenize=asr_pretokenize,
            asr_provider=asr_provider,
            asr_fallback_provider=asr_fallback_provider,
            chirp_model_name=chirp_model_name,
            chirp_language_code=chirp_language_code,
            chirp_project=chirp_project,
            chirp_location=chirp_location,
            chirp_recognizer=chirp_recognizer,
            cleanup_min_duration_ms=cleanup_min_duration_ms,
            cleanup_duplicate_window_ms=cleanup_duplicate_window_ms,
            llm_window_max_chars=llm_window_max_chars,
            llm_window_max_duration_sec=llm_window_max_duration_sec,
            llm_window_max_gap_sec=llm_window_max_gap_sec,
            llm_window_max_speaker_switches=llm_window_max_speaker_switches,
            text_enhancement_mode=text_enhancement_mode,
            enhancement_skip_low_confidence=enhancement_skip_low_confidence,
            low_confidence_min_cps=low_confidence_min_cps,
            low_confidence_max_cps=low_confidence_max_cps,
            hf_token=hf_token,
            nemo_model_name=nemo_model_name,
            google_api_key=google_api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_model_name=vertex_model_name,
            use_vertex=use_vertex,
            force=force,
            device=resolved_device,
        )

    @property
    def run_dir(self) -> Path:
        resolved_input = str(self.input_path.expanduser().resolve())
        digest = hashlib.sha1(resolved_input.encode("utf-8")).hexdigest()[:10]
        safe_stem = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in self.input_path.stem
        ).strip("_") or "audio"
        return self.artifacts_root / f"{safe_stem}_{digest}"
