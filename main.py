import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from audio_pipeline import PipelineConfig, run_pipeline
from audio_pipeline.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-ready audio processing pipeline",
    )
    parser.add_argument(
        "--input",
        default="audio.wav",
        help="Input audio file (wav/mp3/m4a)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for intermediate artifacts and final JSON",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Normalization target sample rate",
    )
    parser.add_argument(
        "--disable-audio-enhancement",
        action="store_true",
        help="Disable speech-focused audio enhancement stage",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=1.0,
        help="Spectral denoise strength (0 disables denoising)",
    )
    parser.add_argument(
        "--noise-quantile",
        type=float,
        default=0.15,
        help="Quantile used to estimate stationary noise floor (0.01-0.5)",
    )
    parser.add_argument(
        "--highpass-hz",
        type=int,
        default=60,
        help="High-pass cutoff in Hz to suppress low-frequency rumble",
    )
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=-22.0,
        help="Post-enhancement RMS loudness target in dBFS",
    )
    parser.add_argument(
        "--target-peak-dbfs",
        type=float,
        default=-1.0,
        help="Post-enhancement peak target in dBFS",
    )
    parser.add_argument(
        "--min-segment-duration-ms",
        type=int,
        default=450,
        help="Minimum segment duration for diarization post-processing",
    )
    parser.add_argument(
        "--segment-merge-gap-ms",
        type=int,
        default=300,
        help="Merge adjacent same-speaker segments when gap is below this threshold",
    )
    parser.add_argument(
        "--segment-padding-ms",
        type=int,
        default=60,
        help="Padding added on both sides of post-processed diarization segments",
    )
    parser.add_argument(
        "--segment-absorb-short-gap-ms",
        type=int,
        default=220,
        help="Absorb remaining short diarization segments into adjacent context when gap is small",
    )
    parser.add_argument(
        "--asr-batch-size",
        type=int,
        default=8,
        help="Batch size for ASR providers that support internal batching (NeMo)",
    )
    parser.add_argument(
        "--asr-orchestration-batch-size",
        type=int,
        default=64,
        help="Max chunks per adapter call to reduce orchestration overhead",
    )
    parser.add_argument(
        "--asr-min-chunk-duration",
        type=float,
        default=0.25,
        help="Skip ASR for chunks shorter than this duration (seconds)",
    )
    parser.add_argument(
        "--asr-pretokenize",
        action="store_true",
        help="Enable NeMo pretokenize mode (disabled by default)",
    )
    parser.add_argument(
        "--asr-provider",
        choices=["chirp", "nemo"],
        default="chirp",
        help="ASR backend provider. Default is chirp with automatic NeMo fallback.",
    )
    parser.add_argument(
        "--asr-fallback-provider",
        choices=["none", "nemo", "chirp"],
        default="nemo",
        help="Fallback ASR provider when primary provider fails",
    )
    parser.add_argument(
        "--cleanup-min-duration-ms",
        type=int,
        default=350,
        help="Cleanup threshold for short segments",
    )
    parser.add_argument(
        "--cleanup-duplicate-window-ms",
        type=int,
        default=280,
        help="Time window for near-duplicate cleanup",
    )
    parser.add_argument(
        "--llm-window-max-chars",
        type=int,
        default=1000,
        help="Maximum characters in one semantic LLM window",
    )
    parser.add_argument(
        "--llm-window-max-duration",
        type=float,
        default=25.0,
        help="Maximum duration in seconds for one semantic LLM window",
    )
    parser.add_argument(
        "--llm-window-max-gap",
        type=float,
        default=1.2,
        help="Maximum temporal gap between neighboring ASR segments inside one semantic LLM window",
    )
    parser.add_argument(
        "--llm-window-max-speaker-switches",
        type=int,
        default=6,
        help="Maximum speaker switches allowed inside one semantic LLM window",
    )
    parser.add_argument(
        "--enhancement-mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Text enhancement mode: deterministic (safe local cleanup) or llm",
    )
    parser.add_argument(
        "--enhance-low-confidence",
        action="store_true",
        help="Allow LLM enhancement on low-confidence ASR chunks (disabled by default)",
    )
    parser.add_argument(
        "--low-confidence-min-cps",
        type=float,
        default=1.5,
        help="Low-confidence threshold: minimum chars/sec",
    )
    parser.add_argument(
        "--low-confidence-max-cps",
        type=float,
        default=28.0,
        help="Low-confidence threshold: maximum chars/sec",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Hugging Face token for pyannote model access",
    )
    parser.add_argument(
        "--nemo-model",
        default="nvidia/stt_kk_ru_fastconformer_hybrid_large",
        help="NeMo model name",
    )
    parser.add_argument(
        "--chirp-model",
        default="chirp_2",
        help="Google Chirp model name",
    )
    parser.add_argument(
        "--chirp-language-code",
        default="ru-RU",
        help="BCP-47 language code for Chirp recognition",
    )
    parser.add_argument(
        "--chirp-project",
        default="",
        help="Google Cloud project for Chirp recognizer path",
    )
    parser.add_argument(
        "--chirp-location",
        default="global",
        help="Google Cloud location for Chirp recognizer path",
    )
    parser.add_argument(
        "--chirp-recognizer",
        default="",
        help="Full Chirp recognizer resource path. Overrides --chirp-project/--chirp-location when set",
    )
    parser.add_argument(
        "--google-api-key",
        default="",
        help="Google GenAI API key (alternative to Vertex project/location auth)",
    )
    parser.add_argument(
        "--vertex-project",
        default="",
        help="Google Cloud project for Vertex AI",
    )
    parser.add_argument(
        "--vertex-location",
        default="us-central1",
        help="Google Cloud location for Vertex AI",
    )
    parser.add_argument(
        "--vertex-model",
        default="gemini-2.5-flash",
        help="Vertex model name",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Execution device: mps/cuda/cpu. Auto-detected when omitted",
    )
    parser.add_argument(
        "--no-vertex",
        action="store_true",
        help="Disable Vertex AI anonymization and enhancement",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all stages even when cached artifacts exist",
    )
    parser.add_argument(
        "--print-result-json",
        action="store_true",
        help="Print final result JSON to stdout (disabled by default)",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Path for system logs. Defaults to <run_dir>/pipeline.log",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="System log level",
    )
    parser.add_argument(
        "--ab-compare-preprocessing",
        action="store_true",
        help="Run A/B comparison: normalized audio vs enhanced audio",
    )
    return parser.parse_args()


def _build_config(
    args: argparse.Namespace,
    artifacts_dir_override: str | None = None,
    enable_audio_enhancement_override: bool | None = None,
    use_vertex_override: bool | None = None,
) -> PipelineConfig:
    return PipelineConfig.create(
        input_path=args.input,
        artifacts_root=artifacts_dir_override or args.artifacts_dir,
        target_sample_rate=args.target_sample_rate,
        enable_audio_enhancement=(
            enable_audio_enhancement_override
            if enable_audio_enhancement_override is not None
            else not args.disable_audio_enhancement
        ),
        audio_denoise_strength=args.denoise_strength,
        audio_noise_quantile=args.noise_quantile,
        audio_highpass_hz=args.highpass_hz,
        audio_target_rms_dbfs=args.target_rms_dbfs,
        audio_target_peak_dbfs=args.target_peak_dbfs,
        segment_min_duration_ms=args.min_segment_duration_ms,
        segment_merge_gap_ms=args.segment_merge_gap_ms,
        segment_padding_ms=args.segment_padding_ms,
        segment_absorb_short_gap_ms=args.segment_absorb_short_gap_ms,
        asr_batch_size=args.asr_batch_size,
        asr_orchestration_batch_size=args.asr_orchestration_batch_size,
        asr_min_chunk_duration_sec=args.asr_min_chunk_duration,
        asr_pretokenize=args.asr_pretokenize,
        asr_provider=args.asr_provider,
        asr_fallback_provider=args.asr_fallback_provider,
        chirp_model_name=args.chirp_model,
        chirp_language_code=args.chirp_language_code,
        chirp_project=args.chirp_project,
        chirp_location=args.chirp_location,
        chirp_recognizer=args.chirp_recognizer,
        cleanup_min_duration_ms=args.cleanup_min_duration_ms,
        cleanup_duplicate_window_ms=args.cleanup_duplicate_window_ms,
        llm_window_max_chars=args.llm_window_max_chars,
        llm_window_max_duration_sec=args.llm_window_max_duration,
        llm_window_max_gap_sec=args.llm_window_max_gap,
        llm_window_max_speaker_switches=args.llm_window_max_speaker_switches,
        text_enhancement_mode=args.enhancement_mode,
        enhancement_skip_low_confidence=not args.enhance_low_confidence,
        low_confidence_min_cps=args.low_confidence_min_cps,
        low_confidence_max_cps=args.low_confidence_max_cps,
        hf_token=args.hf_token,
        nemo_model_name=args.nemo_model,
        google_api_key=args.google_api_key,
        vertex_project=args.vertex_project,
        vertex_location=args.vertex_location,
        vertex_model_name=args.vertex_model,
        use_vertex=(
            use_vertex_override
            if use_vertex_override is not None
            else not args.no_vertex
        ),
        force=args.force,
        device=args.device,
    )


def _configure_logging(level_name: str, log_file: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Suppress known noisy warnings that do not affect business output quality.
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\]",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"'pin_memory' argument is set as true but not supported on MPS now",
        category=UserWarning,
    )

    for noisy_logger in [
        "numexpr",
        "pytorch_lightning",
        "lightning",
        "nemo",
        "urllib3.connectionpool",
        "httpx",
        "httpcore",
        "google.api_core",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _ab_metrics(result: dict) -> dict[str, int | str]:
    quality = result.get("quality_debug", {})
    metrics = result.get("metrics", {})
    return {
        "audio_preprocessing_mode": str(
            quality.get("audio_preprocessing_mode", "unknown")
        ),
        "total_diarization_segments": int(quality.get("total_diarization_segments", 0)),
        "total_asr_chunks": int(quality.get("total_asr_chunks", 0)),
        "skipped_short_chunks": int(quality.get("skipped_short_chunks", 0)),
        "empty_segments_removed": int(quality.get("empty_segments_removed", 0)),
        "llm_calls_total": int(metrics.get("llm_calls_total", 0)),
        "final_segments_count": int(len(result.get("segments", []))),
    }


def _run_ab_compare(args: argparse.Namespace) -> dict:
    normalized_root = str(Path(args.artifacts_dir) / "ab_normalized")
    enhanced_root = str(Path(args.artifacts_dir) / "ab_enhanced")

    normalized_config = _build_config(
        args,
        artifacts_dir_override=normalized_root,
        enable_audio_enhancement_override=False,
        use_vertex_override=False,
    )
    enhanced_config = _build_config(
        args,
        artifacts_dir_override=enhanced_root,
        enable_audio_enhancement_override=True,
        use_vertex_override=False,
    )

    logging.info("A/B: running normalized-only path")
    normalized_result = run_pipeline(normalized_config)
    logging.info("A/B: running enhanced-audio path")
    enhanced_result = run_pipeline(enhanced_config)

    report = {
        "input_file": args.input,
        "normalized": _ab_metrics(normalized_result),
        "enhanced": _ab_metrics(enhanced_result),
        "notes": (
            "This A/B report compares operational quality metrics. "
            "WER cannot be computed automatically without a ground-truth transcript."
        ),
        "normalized_result_json": normalized_result.get("result_json_path", ""),
        "enhanced_result_json": enhanced_result.get("result_json_path", ""),
    }
    comparison_path = Path(args.artifacts_dir) / "ab_comparison.json"
    write_json(comparison_path, report)
    logging.info("A/B comparison saved to %s", comparison_path)
    return report


def main() -> None:
    args = parse_args()
    config = _build_config(args)

    log_file = Path(args.log_file) if args.log_file else config.run_dir / "pipeline.log"
    _configure_logging(args.log_level, log_file)

    logging.info("Running pipeline")
    result = run_pipeline(config)

    if args.ab_compare_preprocessing:
        _run_ab_compare(args)

    if args.print_result_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logging.info(
            "Pipeline finished. Result JSON: %s", result.get("result_json_path", "")
        )


if __name__ == "__main__":
    main()
