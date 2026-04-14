import logging
from pathlib import Path

import librosa
import torch
import torchaudio

logger = logging.getLogger(__name__)


def _load_audio(input_path: Path) -> tuple[torch.Tensor, int]:
    try:
        waveform, sample_rate = torchaudio.load(str(input_path))
        return waveform, sample_rate
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "torchaudio.load failed for %s, falling back to librosa (%s)",
            input_path,
            exc,
        )
        samples, sample_rate = librosa.load(str(input_path), sr=None, mono=False)
        if getattr(samples, "ndim", 1) == 1:
            waveform = torch.from_numpy(samples).unsqueeze(0)
        else:
            waveform = torch.from_numpy(samples)
        return waveform.float(), int(sample_rate)


def normalize_audio(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
) -> dict[str, float | int | str]:
    waveform, original_sample_rate = _load_audio(input_path)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, target_sample_rate)

    duration_sec = float(waveform.shape[1] / target_sample_rate)
    return {
        "input_path": str(input_path),
        "normalized_path": str(output_path),
        "original_sample_rate": int(original_sample_rate),
        "target_sample_rate": int(target_sample_rate),
        "channels": int(waveform.shape[0]),
        "duration_sec": duration_sec,
    }
