from __future__ import annotations

import logging
from typing import Any

from app.config import PipelineConfig
from app.pipeline.artifact_manager import ArtifactManager
from app.pipeline.factory import StageFactory
from app.pipeline.stage_graph import PIPELINE_STAGES, StageContext, build_fingerprint

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Executes the audio processing stage graph."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.artifacts = ArtifactManager(config)
        self.factory = StageFactory()

    def run(self) -> dict[str, Any]:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_path}")

        self.artifacts.ensure_run_dir()
        fingerprint = build_fingerprint(self.config)
        if self.artifacts.check_cache_invalidation(fingerprint):
            logger.info("Cache invalidated or missing")

        context = StageContext(
            config=self.config,
            artifacts=self.artifacts,
            factory=self.factory,
            logger=logger,
        )
        values: dict[str, Any] = {}
        for stage in PIPELINE_STAGES:
            missing = [key for key in stage.inputs if key not in values]
            if missing:
                raise RuntimeError(f"Stage '{stage.name}' missing inputs: {missing}")

            result = stage.run(context, values)
            values.update(result.value)
            context.run_metadata[stage.name] = {
                "cache_hit": result.cache_hit,
                "artifacts": {key: str(path) for key, path in result.artifacts.items()},
                "metrics": result.metrics,
            }
            logger.info(
                "Stage %s complete (%s)",
                stage.name,
                "cache" if result.cache_hit else "executed",
            )

        self.artifacts.save_state(fingerprint)
        return values["result"]

