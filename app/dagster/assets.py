from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dagster as dg

from app.config import PipelineConfig
from app.pipeline.artifact_manager import ArtifactManager
from app.pipeline.factory import StageFactory
from app.pipeline.stage_graph import STAGES_BY_NAME, StageContext, StageResult, build_fingerprint


class PipelineConfigResource(dg.ConfigurableResource):
    input_path: str = "audio.wav"
    artifacts_root: str = "artifacts"
    force: bool = False

    def build_config(self) -> PipelineConfig:
        return PipelineConfig(
            input_path=Path(self.input_path),
            artifacts_root=Path(self.artifacts_root),
            force=self.force,
            hf_token=os.getenv("HF_TOKEN", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )


def _context(config: PipelineConfig, logger: Any) -> StageContext:
    artifacts = ArtifactManager(config)
    artifacts.ensure_run_dir()
    artifacts.check_cache_invalidation(build_fingerprint(config))
    return StageContext(
        config=config,
        artifacts=artifacts,
        factory=StageFactory(),
        logger=logger,
    )


def _merge_inputs(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        merged.update(item)
    return merged


def _metadata(result: StageResult[Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "cache_hit": result.cache_hit,
        **result.metrics,
    }
    for key, path in result.artifacts.items():
        metadata[f"artifact_{key}"] = dg.MetadataValue.path(path)
    return metadata


def _run_stage(
    name: str,
    config: PipelineConfig,
    inputs: dict[str, Any],
    context,
) -> dict[str, Any]:
    stage_context = _context(config, context.log)
    result = STAGES_BY_NAME[name].run(stage_context, inputs)
    context.add_output_metadata(_metadata(result))
    if name == "processed_audio_result":
        stage_context.artifacts.save_state(build_fingerprint(config))
    return result.value


@dg.asset
def pipeline_config(pipeline_settings: PipelineConfigResource) -> PipelineConfig:
    return pipeline_settings.build_config()


@dg.asset
def normalized_audio(
    context,
    pipeline_config: PipelineConfig,
) -> dict[str, Any]:
    return _run_stage("normalized_audio", pipeline_config, {}, context)


@dg.asset
def enhanced_audio(
    context,
    pipeline_config: PipelineConfig,
    normalized_audio: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage("enhanced_audio", pipeline_config, normalized_audio, context)


@dg.asset
def diarization(
    context,
    pipeline_config: PipelineConfig,
    enhanced_audio: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage("diarization", pipeline_config, enhanced_audio, context)


@dg.asset
def postprocessed_segments(
    context,
    pipeline_config: PipelineConfig,
    normalized_audio: dict[str, Any],
    diarization: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "postprocessed_segments",
        pipeline_config,
        _merge_inputs(normalized_audio, diarization),
        context,
    )


@dg.asset
def audio_chunks(
    context,
    pipeline_config: PipelineConfig,
    enhanced_audio: dict[str, Any],
    postprocessed_segments: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "audio_chunks",
        pipeline_config,
        _merge_inputs(enhanced_audio, postprocessed_segments),
        context,
    )


@dg.asset
def asr_transcripts(
    context,
    pipeline_config: PipelineConfig,
    audio_chunks: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage("asr_transcripts", pipeline_config, audio_chunks, context)


@dg.asset
def merged_transcript(
    context,
    pipeline_config: PipelineConfig,
    audio_chunks: dict[str, Any],
    asr_transcripts: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "merged_transcript",
        pipeline_config,
        _merge_inputs(audio_chunks, asr_transcripts),
        context,
    )


@dg.asset
def cleaned_transcript(
    context,
    pipeline_config: PipelineConfig,
    merged_transcript: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage("cleaned_transcript", pipeline_config, merged_transcript, context)


@dg.asset
def semantic_windows(
    context,
    pipeline_config: PipelineConfig,
    cleaned_transcript: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage("semantic_windows", pipeline_config, cleaned_transcript, context)


@dg.asset
def anonymized_transcript(
    context,
    pipeline_config: PipelineConfig,
    cleaned_transcript: dict[str, Any],
    semantic_windows: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "anonymized_transcript",
        pipeline_config,
        _merge_inputs(cleaned_transcript, semantic_windows),
        context,
    )


@dg.asset
def enhanced_transcript(
    context,
    pipeline_config: PipelineConfig,
    anonymized_transcript: dict[str, Any],
    semantic_windows: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "enhanced_transcript",
        pipeline_config,
        _merge_inputs(anonymized_transcript, semantic_windows),
        context,
    )


@dg.asset
def quality_report(
    context,
    pipeline_config: PipelineConfig,
    audio_chunks: dict[str, Any],
    asr_transcripts: dict[str, Any],
    enhanced_transcript: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "quality_report",
        pipeline_config,
        _merge_inputs(audio_chunks, asr_transcripts, enhanced_transcript),
        context,
    )


@dg.asset
def processed_audio_result(
    context,
    pipeline_config: PipelineConfig,
    normalized_audio: dict[str, Any],
    diarization: dict[str, Any],
    postprocessed_segments: dict[str, Any],
    audio_chunks: dict[str, Any],
    asr_transcripts: dict[str, Any],
    anonymized_transcript: dict[str, Any],
    enhanced_transcript: dict[str, Any],
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    return _run_stage(
        "processed_audio_result",
        pipeline_config,
        _merge_inputs(
            normalized_audio,
            diarization,
            postprocessed_segments,
            audio_chunks,
            asr_transcripts,
            anonymized_transcript,
            enhanced_transcript,
            quality_report,
        ),
        context,
    )["result"]
