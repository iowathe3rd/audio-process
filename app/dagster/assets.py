from __future__ import annotations
from pathlib import Path
from typing import Any
import os
import dagster as dg

from app.config import PipelineConfig
from app.pipeline.orchestrator import PipelineOrchestrator


@dg.asset
def pipeline_config() -> PipelineConfig:
    """Create PipelineConfig for the run from environment variables (safe default).

    Using env vars avoids relying on an API surface that may not be present
    across Dagster versions (e.g. `dg.Config`).
    """
    return PipelineConfig(
        input_path=Path(os.getenv("INPUT_PATH", "audio.wav")),
        artifacts_root=Path(os.getenv("ARTIFACTS_ROOT", "artifacts")),
        force=os.getenv("FORCE", "false").lower() in ("1", "true"),
        hf_token=os.getenv("HF_TOKEN", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    )


@dg.asset
def processed_audio_result(pipeline_config: PipelineConfig) -> dict[str, Any]:
    """Runs the full audio processing pipeline as a single Dagster asset."""
    orchestrator = PipelineOrchestrator(pipeline_config)
    return orchestrator.run()
