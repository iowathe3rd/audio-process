import logging
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from app.domain.contracts import Diarizer
from app.models import DiarizationSegment

logger = logging.getLogger(__name__)


class PyannoteDiarizer(Diarizer):
    """Speaker diarization using pyannote.audio."""

    def __init__(self, hf_token: str, device: str) -> None:
        self._hf_token = hf_token
        self._device = device

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        if not self._hf_token:
            raise ValueError("HF_TOKEN is required for pyannote diarization model access")

        logger.info("Loading diarization model...")
        pipeline_obj = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self._hf_token,
        )
        if pipeline_obj is None:
            raise RuntimeError("Failed to load pyannote diarization pipeline")

        pipeline: Any = pipeline_obj
        pipeline.to(torch.device(self._device))

        logger.info("Running diarization...")
        with ProgressHook() as hook:
            annotation = pipeline(str(audio_path), hook=hook)

        segments: list[DiarizationSegment] = []

        if hasattr(annotation, "speaker_diarization"):
            iterator = annotation.speaker_diarization
            for index, (turn, speaker) in enumerate(iterator):
                start = float(turn.start)
                end = float(turn.end)
                if end <= start:
                    continue
                segments.append(
                    DiarizationSegment(
                        index=index,
                        speaker=str(speaker),
                        start=start,
                        end=end,
                    )
                )
        else:
            for index, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
                start = float(turn.start)
                end = float(turn.end)
                if end <= start:
                    continue
                segments.append(
                    DiarizationSegment(
                        index=index,
                        speaker=str(speaker),
                        start=start,
                        end=end,
                    )
                )

        segments.sort(key=lambda item: (item.start, item.end))
        for index, segment in enumerate(segments):
            segment.index = index

        return segments
