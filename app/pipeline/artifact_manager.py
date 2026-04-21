from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from app.config import PipelineConfig
from app.io_utils import read_json, write_json

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages pipeline artifacts and caching state."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_dir = config.run_dir
        self.state_path = self.run_dir / "run_state.json"
        self._invalidate_cache = bool(config.force)

    def ensure_run_dir(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def check_cache_invalidation(self, current_fingerprint: dict[str, Any]) -> bool:
        if self._invalidate_cache:
            return True

        if not self.state_path.exists():
            self._invalidate_cache = True
            return True

        try:
            previous_state = read_json(self.state_path)
            if previous_state != current_fingerprint:
                logger.info("Configuration/input changed, invalidating cache")
                self._invalidate_cache = True
                return True
        except Exception:
            self._invalidate_cache = True
            return True

        return False

    def is_fresh(self, output_path: Path, dependencies: list[Path]) -> bool:
        if self._invalidate_cache or not output_path.exists():
            return False

        try:
            output_mtime = output_path.stat().st_mtime
            for dependency in dependencies:
                if not dependency.exists() or dependency.stat().st_mtime > output_mtime:
                    return False
        except Exception:
            return False

        return True

    def save_state(self, fingerprint: dict[str, Any]):
        write_json(self.state_path, fingerprint)

    def get_artifact_path(self, filename: str) -> Path:
        return self.run_dir / filename
