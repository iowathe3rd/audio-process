#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiwer import wer


@dataclass
class EvalRow:
    fragment_id: str
    audio_path: Path
    reference_text: str
    language_profile: str


def _normalize_language(value: str) -> str:
    text = (value or "").strip().lower()
    if text in {"ru", "russian"}:
        return "ru"
    if text in {"kk", "kazakh"}:
        return "kk"
    if "mix" in text:
        return "mixed"
    if text in {"ru-kk", "kk-ru", "mixed-ru-kk"}:
        return "mixed"
    return text or "unknown"


def load_dataset(dataset_path: Path, audio_column: str, reference_column: str, language_column: str) -> list[EvalRow]:
    rows: list[EvalRow] = []
    dataset_root = dataset_path.parent
    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, item in enumerate(reader):
            audio_raw = str(item.get(audio_column, "")).strip()
            ref = str(item.get(reference_column, "")).strip()
            if not audio_raw or not ref:
                continue
            audio_path = Path(audio_raw)
            if not audio_path.is_absolute():
                audio_path = (dataset_root / audio_path).resolve()
            fragment_id = str(item.get("fragment_id", "")).strip() or f"row_{index:04d}"
            rows.append(
                EvalRow(
                    fragment_id=fragment_id,
                    audio_path=audio_path,
                    reference_text=ref,
                    language_profile=_normalize_language(str(item.get(language_column, ""))),
                )
            )
    return rows


def load_lexicon(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            keyword = str(item.get("keyword", "")).strip().lower()
            category = str(item.get("category", "")).strip().lower() or "domain"
            if keyword:
                rows.append({"keyword": keyword, "category": category})
    return rows


class NemoBackend:
    def __init__(self, model_name: str, device: str) -> None:
        from audio_pipeline.stages.asr_nemo import NeMoTranscriber

        self.transcriber = NeMoTranscriber(model_name=model_name, device=device)

    def transcribe(self, audio_paths: list[Path]) -> list[str]:
        return self.transcriber.transcribe_batch(audio_paths=audio_paths, batch_size=max(1, len(audio_paths)), pretokenize=False)


class FasterWhisperBackend:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            import importlib

            module = importlib.import_module("faster_whisper")
            WhisperModel = getattr(module, "WhisperModel")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "faster-whisper backend requested but package is not installed. "
                "Install with: uv add faster-whisper"
            ) from exc

        whisper_device = "cpu" if device == "mps" else device
        self.model = WhisperModel(model_name, device=whisper_device, compute_type="int8")

    def transcribe(self, audio_paths: list[Path]) -> list[str]:
        outputs: list[str] = []
        for path in audio_paths:
            segments, _ = self.model.transcribe(str(path), vad_filter=True)
            text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            outputs.append(text)
        return outputs


def make_backend(name: str, nemo_model: str, whisper_model: str, device: str):
    if name == "nemo":
        return NemoBackend(model_name=nemo_model, device=device)
    if name == "faster-whisper":
        return FasterWhisperBackend(model_name=whisper_model, device=device)
    raise ValueError(f"Unsupported backend: {name}")


def _contains_keyword(text: str, keyword: str) -> bool:
    return keyword in text.lower()


def _keyword_metrics(results: list[dict[str, Any]], lexicon: list[dict[str, str]], category: str | None = None) -> dict[str, float | int]:
    filtered = [item for item in lexicon if category is None or item["category"] == category]
    total = 0
    hits = 0

    for item in results:
        ref = str(item["reference_text"]).lower()
        hyp = str(item["hypothesis_text"]).lower()
        for lex_item in filtered:
            keyword = lex_item["keyword"]
            if _contains_keyword(ref, keyword):
                total += 1
                if _contains_keyword(hyp, keyword):
                    hits += 1

    misses = max(0, total - hits)
    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / total) if total else 0.0,
        "substitution_rate": (misses / total) if total else 0.0,
    }


def _entity_tokens(text: str) -> list[str]:
    import re

    tokens = re.findall(r"\b\+?\d[\d\s\-()]{6,}\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\[[^\]]+\]", text)
    return [token.strip().lower() for token in tokens if token.strip()]


def compute_metrics(results: list[dict[str, Any]], lexicon: list[dict[str, str]], elapsed_sec: float) -> dict[str, Any]:
    refs = [str(item["reference_text"]).strip() for item in results]
    hyps = [str(item["hypothesis_text"]).strip() for item in results]

    joined_ref = "\n".join(refs).strip()
    joined_hyp = "\n".join(hyps).strip()
    overall_wer = wer(joined_ref, joined_hyp) if joined_ref else 0.0

    by_language: dict[str, dict[str, Any]] = {}
    for language in ["ru", "kk", "mixed"]:
        lang_rows = [item for item in results if item.get("language_profile") == language]
        lang_refs = "\n".join(str(item["reference_text"]) for item in lang_rows).strip()
        lang_hyps = "\n".join(str(item["hypothesis_text"]) for item in lang_rows).strip()
        by_language[language] = {
            "samples": len(lang_rows),
            "wer": wer(lang_refs, lang_hyps) if lang_refs else 0.0,
        }

    entity_refs = [" ".join(_entity_tokens(str(item["reference_text"]))) for item in results]
    entity_hyps = [" ".join(_entity_tokens(str(item["hypothesis_text"]))) for item in results]
    entity_ref_joined = "\n".join(item for item in entity_refs if item).strip()
    entity_hyp_joined = "\n".join(item for item in entity_hyps if item).strip()
    entity_wer = wer(entity_ref_joined, entity_hyp_joined) if entity_ref_joined else 0.0

    keyword_all = _keyword_metrics(results, lexicon)
    keyword_slang = _keyword_metrics(results, lexicon, category="slang")

    total_chars = sum(len(str(item["hypothesis_text"])) for item in results)
    chars_per_sec = total_chars / max(1e-6, elapsed_sec)

    return {
        "samples_total": len(results),
        "overall_wer": overall_wer,
        "wer_ru": float(by_language["ru"]["wer"]),
        "wer_kk": float(by_language["kk"]["wer"]),
        "wer_mixed": float(by_language["mixed"]["wer"]),
        "entity_wer": entity_wer,
        "domain_keyword_accuracy": float(keyword_all["hit_rate"]),
        "domain_keyword_substitution_rate": float(keyword_all["substitution_rate"]),
        "slang_keyword_accuracy": float(keyword_slang["hit_rate"]),
        "latency_total_sec": elapsed_sec,
        "chars_per_sec": chars_per_sec,
        "subjective_intelligibility_proxy": max(0.0, 1.0 - overall_wer),
        "by_language": by_language,
        "keyword_metrics": {
            "all": keyword_all,
            "slang": keyword_slang,
        },
    }


def evaluate_backend(
    backend_name: str,
    rows: list[EvalRow],
    nemo_model: str,
    whisper_model: str,
    device: str,
    lexicon: list[dict[str, str]],
) -> dict[str, Any]:
    backend = make_backend(backend_name, nemo_model=nemo_model, whisper_model=whisper_model, device=device)

    batch_paths = [row.audio_path for row in rows]
    start = time.perf_counter()
    hypotheses = backend.transcribe(batch_paths)
    elapsed = time.perf_counter() - start

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        hypothesis = hypotheses[index] if index < len(hypotheses) else ""
        sample = {
            "fragment_id": row.fragment_id,
            "audio_path": str(row.audio_path),
            "language_profile": row.language_profile,
            "reference_text": row.reference_text,
            "hypothesis_text": hypothesis,
            "sample_wer": wer(row.reference_text, hypothesis) if row.reference_text else 0.0,
        }
        results.append(sample)

    metrics = compute_metrics(results, lexicon=lexicon, elapsed_sec=elapsed)
    return {
        "backend": backend_name,
        "metrics": metrics,
        "samples": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR benchmark over a gold evaluation set")
    parser.add_argument("--dataset", default="evaluation/asr_eval_dataset_template.csv", help="CSV with audio_path and reference_text")
    parser.add_argument("--audio-column", default="audio_path", help="Audio file column in dataset")
    parser.add_argument("--reference-column", default="reference_text", help="Reference transcript column in dataset")
    parser.add_argument("--language-column", default="language_profile", help="Language label column in dataset")
    parser.add_argument("--backends", default="nemo", help="Comma-separated backends: nemo,faster-whisper")
    parser.add_argument("--nemo-model", default="nvidia/stt_kk_ru_fastconformer_hybrid_large")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="cpu", help="cpu/cuda/mps")
    parser.add_argument("--lexicon", default="evaluation/domain_lexicon_template.csv", help="Keyword lexicon CSV")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample limit for quick experiments")
    parser.add_argument("--output", default="evaluation/reports/asr_benchmark.json", help="Output JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_dataset(
        dataset_path=dataset_path,
        audio_column=args.audio_column,
        reference_column=args.reference_column,
        language_column=args.language_column,
    )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        raise ValueError("No evaluation rows found with non-empty audio and reference")

    lexicon = load_lexicon(Path(args.lexicon) if args.lexicon else None)

    reports: list[dict[str, Any]] = []
    for backend_name in [item.strip() for item in args.backends.split(",") if item.strip()]:
        reports.append(
            evaluate_backend(
                backend_name=backend_name,
                rows=rows,
                nemo_model=args.nemo_model,
                whisper_model=args.whisper_model,
                device=args.device,
                lexicon=lexicon,
            )
        )

    output_payload = {
        "dataset": str(dataset_path),
        "audio_column": args.audio_column,
        "reference_column": args.reference_column,
        "language_column": args.language_column,
        "rows_total": len(rows),
        "reports": reports,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
