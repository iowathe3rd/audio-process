import logging
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio

from app.domain.contracts import AudioNormalizer

logger = logging.getLogger(__name__)


class SoXNormalizer(AudioNormalizer):
    """Normalizes audio to WAV 16-bit mono at target sample rate."""

    def normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
    ) -> dict[str, Any]:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        waveform, sample_rate = torchaudio.load(str(input_path))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        peak = waveform.abs().max().item()
        if peak > 0:
            waveform = waveform / peak * 0.95

        sf.write(
            str(output_path),
            waveform.squeeze(0).numpy(),
            sample_rate,
            subtype="PCM_16",
        )

        duration_sec = waveform.shape[1] / sample_rate
        meta: dict[str, Any] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "channels": 1,
            "format": "wav",
        }
        logger.info(
            "Normalized %s → %s (%.1fs, %dHz)",
            input_path.name,
            output_path.name,
            duration_sec,
            sample_rate,
        )
        return meta
