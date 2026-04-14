## Production Audio Processing Pipeline

This project provides an end-to-end audio call processing pipeline:

1. normalize
2. audio enhancement (denoise + high-pass + loudness/peak normalization)
3. diarization (pyannote)
4. diarization segment post-processing (merge + padding + min-duration)
5. segmentation
6. ASR (provider adapters: Google Chirp by default, NeMo fallback)
7. merge (speaker + timestamps + raw text)
8. cleanup (remove skipped/empty/duplicate transcript entries)
9. semantic windows (aggregate ASR chunks into LLM-ready context windows)
10. anonymization (Vertex AI / Google GenAI)
11. enhancement (deterministic safe cleanup by default, optional LLM strict mode)
12. result JSON + metrics JSON

## Input / Output

- Input: wav/mp3/m4a (also supports ogg/flac)
- Output: structured JSON with per-segment fields:
  - `speaker`
  - `start`
  - `end`
  - `raw_text`
  - `anonymized_text`
  - `enhanced_text`

## Pipeline Architecture


Stages are separated into modules under `audio_pipeline/stages`:

- `normalize.py`
- `enhance_audio.py`
- `diarization.py`
- `postprocess_segments.py`
- `segmentation.py`
- `asr_adapters.py`
- `asr_transcribe.py`
- `asr_nemo.py`
- `merge.py`
- `cleanup.py`
- `semantic_windows.py`
- `vertex_text.py`

Orchestration is in `audio_pipeline/pipeline.py`.

## Required Credentials

### Hugging Face token (pyannote)

Set one of:

- env: `HF_TOKEN`
- CLI flag: `--hf-token`

Note: token is required when diarization must run. If artifacts are already cached and valid, rerun can proceed without a token.

### Text processing auth (Google GenAI)

Option A: API key (no project/location required)

Set one of:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`
- CLI flag: `--google-api-key`

### Vertex AI access (optional alternative)

Use ADC (Application Default Credentials):

```bash
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
```

Set:

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION` (default: `us-central1`)
- optional: `VERTEX_MODEL_NAME` (default: `gemini-2.5-flash`)

If Vertex is temporarily unavailable, run with `--no-vertex` to keep the rest of the pipeline operational.

If text processing is enabled but neither API key nor project/location are set, the pipeline fails fast before heavy diarization/ASR compute.

### Chirp ASR auth

Chirp adapter uses the same `GOOGLE_API_KEY` by default. Additionally, set either:

- `--chirp-recognizer` (full recognizer path), or
- `--chirp-project` (and optional `--chirp-location`) to build recognizer path automatically.

## Run

```bash
uv run main.py \
  --input audio.wav \
  --artifacts-dir artifacts
```

Full run with API key auth:

```bash
HF_TOKEN="<your_hf_token>" GOOGLE_API_KEY="<your_google_api_key>" \
uv run main.py --input audio.wav --artifacts-dir artifacts --force
```

Chirp-first ASR test run (with automatic NeMo fallback):

```bash
HF_TOKEN="<your_hf_token>" GOOGLE_API_KEY="<your_google_api_key>" \
uv run main.py \
  --input audio.wav \
  --artifacts-dir artifacts \
  --asr-provider chirp \
  --asr-fallback-provider nemo \
  --chirp-model chirp_2 \
  --chirp-language-code ru-RU \
  --force
```

Useful options:

- `--force` - recompute all stages
- `--device mps|cuda|cpu` - explicit device selection
- `--no-vertex` - skip anonymization/enhancement calls to Vertex
- `--target-sample-rate 16000` - normalization sample rate
- `--disable-audio-enhancement` - disable speech enhancement stage
- `--denoise-strength 1.0` - denoise intensity (set `0` to disable denoise only)
- `--noise-quantile 0.15` - quantile used to estimate stationary noise floor
- `--highpass-hz 60` - low-cut for hum/rumble suppression
- `--target-rms-dbfs -22` - output loudness target
- `--target-peak-dbfs -1` - output peak limiter target
- `--min-segment-duration-ms 450` - minimum diarization segment length after post-processing
- `--segment-merge-gap-ms 300` - merge neighboring same-speaker segments when gap is small
- `--segment-padding-ms 60` - add context around diarization boundaries
- `--segment-absorb-short-gap-ms 220` - absorb remaining very short segments into nearby context
- `--asr-provider chirp|nemo` - select ASR backend (default: `chirp`)
- `--asr-fallback-provider none|nemo|chirp` - automatic fallback backend (default: `nemo`)
- `--asr-batch-size 8` - chunk batch size for providers that support internal batching (NeMo)
- `--asr-orchestration-batch-size 64` - max chunks per adapter call
- `--asr-min-chunk-duration 0.25` - skip ASR for very short chunks
- `--asr-pretokenize` - enable NeMo pretokenize mode (off by default)
- `--chirp-model chirp_2` - Google Chirp model name
- `--chirp-language-code ru-RU` - Chirp recognition language code
- `--chirp-project <project>` - Chirp project for recognizer path
- `--chirp-location <location>` - Chirp location for recognizer path
- `--chirp-recognizer <resource>` - explicit recognizer resource path
- `--cleanup-min-duration-ms 350` - cleanup threshold for short transcript segments
- `--cleanup-duplicate-window-ms 280` - window for near-duplicate segment removal
- `--llm-window-max-chars 1000` - target semantic window size for LLM stages
- `--llm-window-max-duration 25` - max semantic window duration in seconds
- `--llm-window-max-gap 1.2` - max gap between neighboring segments in one semantic window
- `--llm-window-max-speaker-switches 6` - cap speaker switch count in one semantic window
- `--enhancement-mode deterministic|llm` - deterministic mode is safer and cheaper; `llm` is optional
- `--enhance-low-confidence` - allow LLM enhancement for low-confidence chunks (off by default)
- `--low-confidence-min-cps 1.5` - min chars/sec threshold for low-confidence marker
- `--low-confidence-max-cps 28` - max chars/sec threshold for low-confidence marker
- `--ab-compare-preprocessing` - run A/B comparison: normalized audio vs enhanced audio
- `--log-file <path>` - write system logs to a dedicated file
- `--print-result-json` - optional stdout output for final JSON (disabled by default)

By default the pipeline writes business output only to artifacts (`result.json`) and keeps system logs in `pipeline.log`.

## Artifacts

Artifacts are stored per input file path hash (to avoid collisions for same file names in different folders):

```text
artifacts/<input_stem>_<path_hash>/
  normalize.json
  normalized.wav
  audio_enhancement.json
  enhanced.wav
  diarization.json
  diarization_postprocessed.json
  segment_postprocess_report.json
  segment_merge_groups.json
  chunks/
    chunk_0000_SPEAKER_00.wav
    ...
  segments_manifest.json
  asr_raw.json
  asr_report.json
  merged_raw.json
  merged_clean.json
  cleanup_report.json
  semantic_windows.json
  semantic_windows_report.json
  anonymized.json
  anonymize_report.json
  enhanced.json
  enhancement_report.json
  chunk_quality.json
  chunk_quality_report.json
  metrics.json
  pipeline.log
  run_state.json
  result.json
  ab_comparison.json
```

## Restartability

Pipeline is restart-safe:

- If stage artifacts exist, the stage is skipped and cache is reused.
- Use `--force` to invalidate cache and rerun all stages.

## Quality Review Dataset

A manual evaluation baseline template is included in:

- `evaluation/review_dataset_template.csv`
- `evaluation/README.md`
- `evaluation/asr_eval_dataset_template.csv`
- `evaluation/domain_lexicon_template.csv`

Use this set (20-30 representative fragments) to track diarization adequacy, transcript faithfulness, anonymization correctness, and enhancement safety over time.

For ASR fidelity benchmarking and baseline comparison:

```bash
uv run evaluation/run_asr_benchmark.py \
  --dataset evaluation/asr_eval_dataset_template.csv \
  --backends nemo,faster-whisper \
  --output evaluation/reports/asr_benchmark.json
```

## Final Result Format

`result.json`:

```json
{
  "input_file": "audio.wav",
  "normalized_audio": "artifacts/audio_<hash>/normalized.wav",
  "enhanced_audio": "artifacts/audio_<hash>/enhanced.wav",
  "result_json_path": "artifacts/audio_<hash>/result.json",
  "audio_enhancement": {
    "denoise_strength": 1.0,
    "highpass_hz": 60,
    "target_rms_dbfs": -22.0,
    "target_peak_dbfs": -1.0
  },
  "artifacts_dir": "artifacts/audio_<hash>",
  "quality_debug": {
    "total_diarization_segments": 284,
    "total_asr_chunks": 163,
    "skipped_short_chunks": 12,
    "empty_segments_removed": 9,
    "merged_segments_count": 121,
    "overlap_conflicts_count": 4,
    "audio_preprocessing_mode": "enhanced",
    "semantic_windows_total": 31,
    "llm_calls_total": 62,
    "llm_latency_total_ms": 154322.3
  },
  "metrics": {
    "total_segments_before_postprocess": 288,
    "total_segments_after_postprocess": 201,
    "avg_segment_duration_sec": 1.74,
    "median_segment_duration_sec": 1.36,
    "short_segment_ratio": 0.07,
    "asr_chunks_total": 201,
    "asr_chunks_skipped": 9,
    "asr_latency_total_ms": 24128.2,
    "llm_calls_total": 62,
    "llm_calls_per_min_audio": 7.6,
    "llm_latency_total_ms": 154322.3,
    "empty_transcript_ratio": 0.0,
    "cleanup_removed_segments": 17,
    "audio_coverage_ratio": 0.96,
    "semantic_drift_flags_total": 5
  },
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.6,
      "end": 2.1,
      "raw_text": "...",
      "anonymized_text": "...",
      "enhanced_text": "..."
    }
  ]
}
```
