from __future__ import annotations

import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.config import PipelineConfig
from app.pipeline.artifact_manager import ArtifactManager
from app.pipeline.factory import StageFactory
from app.pipeline.stage_graph import PIPELINE_STAGES, StageContext, StageResult, build_fingerprint


class StageGraphContractTests(unittest.TestCase):
    def test_stage_names_are_unique_and_ordered(self) -> None:
        names = [stage.name for stage in PIPELINE_STAGES]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "normalized_audio")
        self.assertEqual(names[-1], "processed_audio_result")

    def test_stage_context_and_result_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.wav"
            input_path.write_bytes(b"")
            config = PipelineConfig(input_path=input_path, artifacts_root=Path(tmp) / "artifacts")
            context = StageContext(
                config=config,
                artifacts=ArtifactManager(config),
                factory=StageFactory(),
                logger=logging.getLogger("test-stage-contract"),
            )
            result: StageResult[dict[str, Any]] = StageResult(
                value={"ok": True},
                artifacts={"result": context.artifacts.get_artifact_path("result.json")},
                metrics={"rows": 1},
                cache_hit=True,
            )

            self.assertTrue(result.value["ok"])
            self.assertEqual(result.metrics["rows"], 1)
            self.assertTrue(result.cache_hit)
            self.assertIn("input_file", build_fingerprint(config))
