from __future__ import annotations
from dagster import asset, Config, ResourceParam
from pathlib import Path
from typing import Any

from app.config import PipelineConfig
from app.pipeline.orchestrator import PipelineOrchestrator
from app.models import DiarizationSegment, ChunkRecord, TranscribedSegment, PipelineSegment

class PipelineAssetConfig(Config):
    input_path: str = "audio.wav"
    artifacts_root: str = "artifacts"
    force: bool = False

@asset
def pipeline_config(config: PipelineAssetConfig) -> PipelineConfig:
    import os
    return PipelineConfig(
        input_path=Path(config.input_path),
        artifacts_root=Path(config.artifacts_root),
        force=config.force,
        hf_token=os.getenv("HF_TOKEN", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    )

@asset
def processed_audio_result(pipeline_config: PipelineConfig) -> dict[str, Any]:
    """Runs the full audio processing pipeline as a single Dagster asset."""
    orchestrator = PipelineOrchestrator(pipeline_config)
    return orchestrator.run()
