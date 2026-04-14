import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _dbfs(value: float) -> float:
    safe = max(value, _EPS)
    return float(20.0 * torch.log10(torch.tensor(safe)).item())


def _rms(waveform: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(waveform)) + _EPS).item())


def _peak(waveform: torch.Tensor) -> float:
    return float(torch.max(torch.abs(waveform)).item())


def _spectral_denoise(
    waveform: torch.Tensor,
    sample_rate: int,
    denoise_strength: float,
    noise_quantile: float,
) -> torch.Tensor:
    _ = sample_rate
    if denoise_strength <= 0:
        return waveform

    n_fft = 512
    hop_length = 128
    win_length = 512

    mono = waveform.squeeze(0)
    original_len = mono.shape[0]
    window = torch.hann_window(win_length, dtype=mono.dtype, device=mono.device)

    stft = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    magnitude = torch.abs(stft)
    phase = stft / (magnitude + _EPS)

    q = min(max(noise_quantile, 0.01), 0.5)
    noise_floor = torch.quantile(magnitude, q=q, dim=1, keepdim=True)
    cleaned_magnitude = torch.clamp(magnitude - denoise_strength * noise_floor, min=0.0)

    enhanced_stft = cleaned_magnitude * phase
    restored = torch.istft(
        enhanced_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=original_len,
    )

    return restored.unsqueeze(0)


def enhance_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    denoise_strength: float,
    noise_quantile: float,
    highpass_hz: int,
    target_rms_dbfs: float,
    target_peak_dbfs: float,
) -> dict[str, float | int | str]:
    waveform, loaded_sample_rate = torchaudio.load(str(input_path))

    if loaded_sample_rate != sample_rate:
        raise ValueError(
            f"Audio enhancement expects {sample_rate}Hz input, got {loaded_sample_rate}Hz"
        )

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.float()

    input_rms = _rms(waveform)
    input_peak = _peak(waveform)

    # Remove DC offset before filtering and denoising.
    waveform = waveform - waveform.mean(dim=1, keepdim=True)

    if highpass_hz > 0:
        waveform = F.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=highpass_hz)

    waveform = _spectral_denoise(
        waveform=waveform,
        sample_rate=sample_rate,
        denoise_strength=denoise_strength,
        noise_quantile=noise_quantile,
    )

    target_rms = 10.0 ** (target_rms_dbfs / 20.0)
    current_rms = _rms(waveform)
    if current_rms > _EPS:
        waveform = waveform * (target_rms / current_rms)

    target_peak = 10.0 ** (target_peak_dbfs / 20.0)
    current_peak = _peak(waveform)
    if current_peak > target_peak:
        waveform = waveform * (target_peak / (current_peak + _EPS))

    waveform = torch.clamp(waveform, min=-0.999, max=0.999)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, sample_rate)

    output_rms = _rms(waveform)
    output_peak = _peak(waveform)
    duration_sec = float(waveform.shape[1] / sample_rate)

    logger.info(
        "Audio enhancement done: rms %.2f dBFS -> %.2f dBFS, peak %.2f dBFS -> %.2f dBFS",
        _dbfs(input_rms),
        _dbfs(output_rms),
        _dbfs(input_peak),
        _dbfs(output_peak),
    )

    return {
        "input_path": str(input_path),
        "enhanced_path": str(output_path),
        "sample_rate": int(sample_rate),
        "duration_sec": duration_sec,
        "denoise_strength": float(denoise_strength),
        "noise_quantile": float(noise_quantile),
        "highpass_hz": int(highpass_hz),
        "target_rms_dbfs": float(target_rms_dbfs),
        "target_peak_dbfs": float(target_peak_dbfs),
        "input_rms_dbfs": _dbfs(input_rms),
        "output_rms_dbfs": _dbfs(output_rms),
        "input_peak_dbfs": _dbfs(input_peak),
        "output_peak_dbfs": _dbfs(output_peak),
    }
