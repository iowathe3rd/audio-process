from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dagster import materialize

from app.dagster import assets
from app.dagster.definitions import defs


class DagsterAssetsTests(unittest.TestCase):
    def test_definitions_expose_one_asset_per_pipeline_stage(self) -> None:
        asset_graph = defs.resolve_asset_graph()
        asset_names = {key.to_user_string() for key in asset_graph.get_all_asset_keys()}

        self.assertEqual(
            {
                "pipeline_config",
                "normalized_audio",
                "enhanced_audio",
                "diarization",
                "postprocessed_segments",
                "audio_chunks",
                "asr_transcripts",
                "merged_transcript",
                "cleaned_transcript",
                "semantic_windows",
                "anonymized_transcript",
                "enhanced_transcript",
                "quality_report",
                "processed_audio_result",
            },
            asset_names,
        )

    def test_can_materialize_subset_to_cleaned_transcript(self) -> None:
        original_run_stage = assets._run_stage

        def fake_run_stage(
            name: str,
            _config: Any,
            _inputs: dict[str, Any],
            _context: Any,
        ) -> dict[str, Any]:
            path = Path("artifact.wav")
            outputs: dict[str, dict[str, Any]] = {
                "normalized_audio": {"normalize_meta": {"duration_sec": 1.0}, "normalized_audio_path": path},
                "enhanced_audio": {"audio_path": path, "enhancement_meta": {}},
                "diarization": {"diarization_segments": []},
                "postprocessed_segments": {
                    "processed_segments": [],
                    "postprocess_report": {"input_segments": 0, "output_segments": 0},
                    "segment_groups": [],
                },
                "audio_chunks": {"chunk_records": []},
                "asr_transcripts": {"transcripts": [], "asr_report": {"total_chunks": 0}},
                "merged_transcript": {"merged_segments": []},
                "cleaned_transcript": {"cleaned_segments": [], "cleanup_report": {}},
            }
            return outputs[name]

        with TemporaryDirectory() as tmp:
            try:
                assets._run_stage = fake_run_stage
                result = materialize(
                    [
                        assets.pipeline_config,
                        assets.normalized_audio,
                        assets.enhanced_audio,
                        assets.diarization,
                        assets.postprocessed_segments,
                        assets.audio_chunks,
                        assets.asr_transcripts,
                        assets.merged_transcript,
                        assets.cleaned_transcript,
                    ],
                    resources={
                        "pipeline_settings": assets.PipelineConfigResource(
                            input_path=str(Path(tmp) / "input.wav"),
                            artifacts_root=str(Path(tmp) / "artifacts"),
                        )
                    },
                )
            finally:
                assets._run_stage = original_run_stage

        self.assertTrue(result.success)
