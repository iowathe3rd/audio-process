# Quality Review Dataset

This folder stores a manual evaluation set for transcript quality reviews.

## Goal

Build a stable 20-30 fragment benchmark with representative call conditions:

- Russian speech
- Kazakh speech
- mixed-language speech
- noisy segments
- overlap-heavy segments

## Files

- review_dataset_template.csv: template table for fragment-level annotation.
- asr_eval_dataset_template.csv: template for gold references used in WER benchmarking.
- domain_lexicon_template.csv: domain/slang/entity keyword list for keyword accuracy tracking.
- run_asr_benchmark.py: computes WER/keyword metrics and compares multiple ASR backends.
- build_error_taxonomy.py: extracts frequent substitution pairs from benchmark outputs.

## Recommended workflow

1. Copy selected call fragments (10-30 seconds each) into a dedicated audio folder.
2. Fill metadata columns in review_dataset_template.csv.
3. Run the pipeline and attach artifact/result paths in notes.
4. Manually score each fragment on 1-5 scale:
   - diarization_adequacy_score
   - transcript_faithfulness_score
   - anonymization_correctness_score
   - enhancement_safety_score
5. Track regressions after parameter or prompt changes.

## Acceptance baseline

Use this dataset to compare:

- normalized only vs normalized + enhanced audio
- old vs new segmentation parameters
- different semantic window settings for LLM stages

## Benchmark Commands

Run multi-backend benchmark on the gold subset:

```bash
uv run evaluation/run_asr_benchmark.py \
   --dataset evaluation/asr_eval_dataset_template.csv \
   --backends nemo,faster-whisper \
   --device cpu \
   --output evaluation/reports/asr_benchmark.json
```

Build error taxonomy from benchmark substitutions:

```bash
uv run evaluation/build_error_taxonomy.py \
   --benchmark evaluation/reports/asr_benchmark.json \
   --backend nemo \
   --output evaluation/reports/error_taxonomy.json
```
