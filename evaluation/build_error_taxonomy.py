#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def collect_substitutions(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens)

    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue

        ref_span = ref_tokens[i1:i2]
        hyp_span = hyp_tokens[j1:j2]
        length = min(len(ref_span), len(hyp_span))
        for index in range(length):
            pairs.append((ref_span[index], hyp_span[index]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build substitution taxonomy from ASR benchmark report")
    parser.add_argument("--benchmark", default="evaluation/reports/asr_benchmark.json", help="Path to run_asr_benchmark output")
    parser.add_argument("--backend", default="nemo", help="Backend section to analyze")
    parser.add_argument("--top", type=int, default=50, help="Top-N substitutions")
    parser.add_argument("--output", default="evaluation/reports/error_taxonomy.json", help="Output taxonomy JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark report not found: {benchmark_path}")

    with benchmark_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    reports = payload.get("reports", [])
    target_report: dict[str, Any] | None = None
    for report in reports:
        if str(report.get("backend", "")).strip() == args.backend:
            target_report = report
            break

    if target_report is None:
        raise ValueError(f"Backend '{args.backend}' not found in benchmark report")

    counter: Counter[tuple[str, str]] = Counter()
    for sample in target_report.get("samples", []):
        reference = str(sample.get("reference_text", ""))
        hypothesis = str(sample.get("hypothesis_text", ""))
        for pair in collect_substitutions(reference, hypothesis):
            counter[pair] += 1

    top_items = counter.most_common(max(1, int(args.top)))
    rows = [
        {
            "reference_token": source,
            "hypothesis_token": target,
            "count": count,
        }
        for (source, target), count in top_items
    ]

    output_payload = {
        "backend": args.backend,
        "source_report": str(benchmark_path),
        "substitutions_total": sum(counter.values()),
        "top_substitutions": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved substitution taxonomy: {output_path}")


if __name__ == "__main__":
    main()
