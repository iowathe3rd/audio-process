from typing import TYPE_CHECKING

from .config import PipelineConfig

if TYPE_CHECKING:
    from .pipeline import run_pipeline as run_pipeline


def run_pipeline(config: PipelineConfig) -> dict:
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(config)


__all__ = ["PipelineConfig", "run_pipeline"]
