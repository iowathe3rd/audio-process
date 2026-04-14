# Запуск аудио-пайплайна

## Быстрый старт

```bash
uv run main.py \
  --input audio.wav \
  --artifacts-dir artifacts
```

Полный запуск с API-ключами:

```bash
HF_TOKEN="<ваш_hf_token>" GOOGLE_API_KEY="<ваш_google_api_key>" \
uv run main.py --input audio.wav --artifacts-dir artifacts --force
```

Тестовый запуск с Chirp ASR (с автоматическим fallback на NeMo):

```bash
HF_TOKEN="<ваш_hf_token>" GOOGLE_API_KEY="<ваш_google_api_key>" \
uv run main.py \
  --input audio.wav \
  --artifacts-dir artifacts \
  --asr-provider chirp \
  --asr-fallback-provider nemo \
  --chirp-model chirp_2 \
  --chirp-language-code ru-RU \
  --force
```

## Основные опции

- `--input` — входной аудиофайл (wav/mp3/m4a/ogg/flac)
- `--artifacts-dir` — директория для артефактов
- `--force` — пересчитать все стадии заново
- `--device mps|cuda|cpu` — выбор устройства
- `--no-vertex` — пропустить анонимизацию/enhancement через Vertex
- `--target-sample-rate 16000` — частота нормализации

## Аудио-улучшение

- `--disable-audio-enhancement` — отключить улучшение аудио
- `--denoise-strength 1.0` — интенсивность шумоподавления (0 — отключить)
- `--highpass-hz 60` — низкочастотный фильтр
- `--target-rms-dbfs -22` — целевая громкость
- `--target-peak-dbfs -1` — пиковый лимитер

## Сегментация и диаризация

- `--min-segment-duration-ms 450` — минимальная длина сегмента
- `--segment-merge-gap-ms 300` — объединение при маленьком промежутке
- `--segment-padding-ms 60` — доп. контекст вокруг границ

## ASR

- `--asr-provider chirp|nemo` — провайдер (по умолчанию: chirp)
- `--asr-fallback-provider none|nemo|chirp` — fallback провайдер
- `--asr-batch-size 8` — размер батча
- `--chirp-model chirp_2` — модель Chirp
- `--chirp-language-code ru-RU` — язык

## Очистка и LLM

- `--cleanup-min-duration-ms 350` — порог очистки коротких сегментов
- `--llm-window-max-chars 1000` — размер семантического окна
- `--enhancement-mode deterministic|llm` — режим улучшения текста
- `--print-result-json` — вывести result.json в stdout

## Credentials

Создайте `.env.local` из `.env.example` и заполните нужные переменные:

```bash
cp .env.example .env.local
# отредактируйте .env.local
```

Для Vertex AI дополнительно нужно настроить ADC:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## Артефакты

Результаты сохраняются в `artifacts/<input_stem>_<hash>/`:

```
normalize.json
normalized.wav
enhanced.wav
diarization.json
chunks/
asr_raw.json
merged_clean.json
semantic_windows.json
anonymized.json
enhanced.json
result.json
metrics.json
pipeline.log
```

## Перезапуск

Пайплайн перезапускаемый — если артефакт exists, стадия пропуск��ется.
Используйте `--force` для принудительного пересчёта всех стадий.