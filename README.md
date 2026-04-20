# Production Audio Processing Pipeline

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, production-ready pipeline for end-to-end audio processing. It handles everything from raw audio normalization to diarization, ASR, and LLM-powered text enhancement with robust caching and restartability.

---

## Overview

This pipeline transforms raw audio calls into structured, anonymized, and enhanced transcripts. It is designed for high reliability and follows a sequential ETL pattern.

### Pipeline Stages

1.  **Normalization:** Converts input (WAV, MP3, M4A, etc.) to 16kHz mono WAV using `torchaudio`.
2.  **Audio Enhancement:** Applies spectral denoising, high-pass filtering (60Hz), and RMS/Peak normalization.
3.  **Diarization:** Speaker separation using `pyannote/speaker-diarization-3.1`.
4.  **Segment Post-processing:** Merges small segments, adds padding, and absorbs short gaps for better ASR context.
5.  **Segmentation:** Chops the enhanced audio into speaker-specific chunks.
6.  **ASR (Automatic Speech Recognition):** Supports **Google Chirp** (primary) with automatic fallback to **NVIDIA NeMo** (FastConformer).
7.  **Merge & Cleanup:** Recombines ASR results with diarization metadata and removes low-quality/duplicate segments.
8.  **Semantic Windowing:** Groups segments into LLM-ready context windows (~1000 chars) while respecting speaker switches.
9.  **Anonymization:** Detects and masks PII (Names, Phones, Emails) using Vertex AI / Google GenAI.
10. **Text Enhancement:** Fixes punctuation, casing, and minor disfluencies using LLM or deterministic rules.
11. **Quality Analytics:** Computes confidence metrics and identifies potential semantic drift.

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd audio-process

# Install dependencies and create venv
uv sync
```

**External Dependencies:**
- `ffmpeg` or `libsndfile` (for `torchaudio` and `soundfile` backends).

---

## Authentication

The pipeline requires several API keys and tokens to function fully.

### 1. Hugging Face (Diarization)
Required for `pyannote` models.
- Set `HF_TOKEN` environment variable.
- Ensure you have accepted the user conditions for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face.

### 2. Google Cloud / GenAI (ASR & Text Processing)
The pipeline supports two ways to authenticate for Google services (Chirp ASR, Gemini LLM):

**Option A: API Key (Simplest)**
- Set `GOOGLE_API_KEY` in your `.env` file or environment.

**Option B: Vertex AI (Production)**
- Use Application Default Credentials (ADC): `gcloud auth application-default login`.
- Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (default: `us-central1`).

---

## Usage

### Basic Run
```bash
uv run main.py --input call_recording.wav --artifacts-dir ./runs
```

### Advanced Configuration
While the current `main.py` is minimal, you can configure the pipeline via environment variables or by modifying `PipelineConfig`.

| Category | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **Audio** | `target_sample_rate` | `16000` | Sample rate for internal processing |
| | `enable_audio_enhancement` | `True` | Toggle spectral enhancement |
| **Diarization** | `segment_min_duration_ms` | `450` | Min duration of a speaker segment |
| | `segment_merge_gap_ms` | `300` | Gap threshold for merging same-speaker segments |
| **ASR** | `asr_provider` | `chirp` | Primary ASR (`chirp` or `nemo`) |
| | `asr_fallback_provider` | `nemo` | Fallback if primary fails |
| | `chirp_language_code` | `ru-RU` | Language for recognition |
| **LLM** | `text_enhancement_mode` | `deterministic` | `deterministic` or `llm` |
| | `llm_window_max_chars` | `1000` | Target size for LLM context windows |

---

## Artifacts & Output

The pipeline saves everything in a unique run directory under `artifacts/` based on a hash of the input file path.

```text
artifacts/<input_name>_<hash>/
├── normalized.wav       # Cleaned 16kHz mono audio
├── enhanced.wav         # Denoised/Normalized audio
├── diarization.json     # Raw speaker timestamps
├── chunks/              # Individual .wav files per segment
├── asr_raw.json         # Raw transcription output
├── merged_clean.json    # Final text after cleanup
├── result.json          # <--- FINAL BUSINESS OUTPUT
└── pipeline.log         # Detailed execution logs
```

### `result.json` Structure
```json
{
  "input_file": "audio.wav",
  "artifacts_dir": "artifacts/audio_8f2a1b",
  "metrics": {
    "total_segments_after_postprocess": 42,
    "asr_latency_total_ms": 12450,
    "llm_calls_total": 8,
    "language_switching_ratio": 0.02
  },
  "segments": [
    {
      "speaker": "SPEAKER_01",
      "start": 1.2,
      "end": 4.5,
      "raw_text": "алло здравствуйте",
      "anonymized_text": "алло здравствуйте",
      "enhanced_text": "Алло, здравствуйте."
    }
  ]
}
```

---

## Restartability

The pipeline is **restart-safe**. Each stage checks for existing artifacts and validates them against a configuration fingerprint.
- If you stop the pipeline and restart it, it will resume from the last successful stage.
- Use the `--force` flag to ignore cache and re-run all stages.

---

## Evaluation & Quality

A specialized dataset template and benchmark script are provided in `evaluation/` to track:
- **ASR Fidelity:** Compare results against ground truth.
- **Diarization Accuracy:** Check speaker switch points.
- **Enhancement Safety:** Ensure LLM doesn't hallucinate or lose meaning.

Run ASR benchmark:
```bash
uv run evaluation/run_asr_benchmark.py --dataset my_test_set.csv --backends nemo,chirp
```

---

## Developer Guide

### Running Tests
```bash
uv run pytest tests/
```

### Dagster Integration
The pipeline is compatible with [Dagster](https://dagster.io/). You can load it as an asset:
```bash
uv run dagster dev -f app/dagster/definitions.py
```
