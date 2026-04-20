from __future__ import annotations

from typing import Any

from app.models import ChunkRecord, PipelineSegment, TranscribedSegment
from app.domain.contracts import ChunkQualityAnalyzer
from app.stages.text_metrics import has_word_sequence_drift, token_language


def _duration(start: float, end: float) -> float:
    return max(0.0, float(end - start))


def _code_switch_presence(text: str) -> bool:
    labels = [token_language(token) for token in text.split()]
    labels = [label for label in labels if label in {"ru", "kk", "lat", "mixed"}]
    if len(labels) < 2:
        return False
    return any(labels[index] != labels[index - 1] for index in range(1, len(labels)))


def _dominant_language(text: str) -> str:
    counts: dict[str, int] = {"ru": 0, "kk": 0, "lat": 0, "mixed": 0}
    for token in text.split():
        label = token_language(token)
        if label in counts:
            counts[label] += 1
    if sum(counts.values()) == 0:
        return "unknown"
    return max(counts, key=counts.get)  # type: ignore[arg-type, return-value]


def _build_chunk_quality_analytics(
    chunks: list[ChunkRecord],
    transcripts: list[TranscribedSegment],
    final_segments: list[PipelineSegment],
    low_confidence_min_cps: float,
    low_confidence_max_cps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcript_by_index = {int(item.index): item for item in transcripts}
    final_by_chunk_path = {item.chunk_path: item for item in final_segments if item.chunk_path}

    sorted_chunks = sorted(chunks, key=lambda value: (value.start, value.end, value.index))
    overlap_indices: set[int] = set()
    for position in range(len(sorted_chunks) - 1):
        left = sorted_chunks[position]
        right = sorted_chunks[position + 1]
        if left.end > right.start:
            overlap_indices.add(int(left.index))
            overlap_indices.add(int(right.index))

    rows: list[dict[str, Any]] = []
    low_confidence_chunks = 0
    code_switch_chunks = 0
    overlap_chunks = 0
    cleaned_out_chunks = 0
    enhancement_changed_chunks = 0
    suspicious_substitution_chunks = 0
    chars_per_sec_values: list[float] = []

    for chunk in chunks:
        transcript = transcript_by_index.get(int(chunk.index))
        text = (transcript.raw_text if transcript else "").strip()
        status = (transcript.status if transcript else "missing")

        duration = _duration(chunk.start, chunk.end)
        chars_count = len(text)
        chars_per_sec = chars_count / max(1e-6, duration)
        chars_per_sec_values.append(chars_per_sec)

        code_switch = _code_switch_presence(text)
        overlap_marker = int(chunk.index) in overlap_indices
        low_confidence_marker = (
            status != "ok"
            or not text
            or chars_per_sec < float(low_confidence_min_cps)
            or chars_per_sec > float(low_confidence_max_cps)
        )

        final_segment = final_by_chunk_path.get(chunk.chunk_path)
        cleaned_out = final_segment is None
        enhancement_changed = False
        suspicious_substitution = False

        if final_segment is not None:
            source_text = (final_segment.anonymized_text or final_segment.raw_text).strip()
            enhanced_text = final_segment.enhanced_text.strip()
            enhancement_changed = bool(source_text and enhanced_text and source_text != enhanced_text)
            suspicious_substitution = bool(
                source_text
                and enhanced_text
                and has_word_sequence_drift(source_text, enhanced_text)
            )

        if low_confidence_marker:
            low_confidence_chunks += 1
        if code_switch:
            code_switch_chunks += 1
        if overlap_marker:
            overlap_chunks += 1
        if cleaned_out:
            cleaned_out_chunks += 1
        if enhancement_changed:
            enhancement_changed_chunks += 1
        if suspicious_substitution:
            suspicious_substitution_chunks += 1

        rows.append(
            {
                "chunk_index": int(chunk.index),
                "speaker": str(chunk.speaker),
                "start": float(chunk.start),
                "end": float(chunk.end),
                "duration_sec": duration,
                "chunk_path": str(chunk.chunk_path),
                "asr_status": status,
                "raw_text": text,
                "raw_text_length": chars_count,
                "chars_per_sec": chars_per_sec,
                "code_switch_presence": code_switch,
                "dominant_language": _dominant_language(text),
                "low_confidence_marker": low_confidence_marker,
                "overlap_marker": overlap_marker,
                "cleaned_out": cleaned_out,
                "enhancement_changed_text": enhancement_changed,
                "suspicious_substitution_flag": suspicious_substitution,
            }
        )

    summary: dict[str, Any] = {
        "chunks_total": len(chunks),
        "low_confidence_chunks": low_confidence_chunks,
        "low_confidence_chunk_ratio": (low_confidence_chunks / max(1, len(chunks))),
        "code_switch_chunks": code_switch_chunks,
        "code_switch_chunk_ratio": (code_switch_chunks / max(1, len(chunks))),
        "overlap_chunks": overlap_chunks,
        "cleaned_out_chunks": cleaned_out_chunks,
        "enhancement_changed_chunks": enhancement_changed_chunks,
        "suspicious_substitution_chunks": suspicious_substitution_chunks,
        "avg_chars_per_sec": (sum(chars_per_sec_values) / len(chars_per_sec_values)) if chars_per_sec_values else 0.0,
        "low_confidence_min_cps": float(low_confidence_min_cps),
        "low_confidence_max_cps": float(low_confidence_max_cps),
    }
    return rows, summary


class QualityAnalyzer(ChunkQualityAnalyzer):
    """Analyzes quality of chunks and transcriptions."""

    def __init__(self, low_confidence_min_cps: float, low_confidence_max_cps: float) -> None:
        self._low_confidence_min_cps = low_confidence_min_cps
        self._low_confidence_max_cps = low_confidence_max_cps

    def analyze(
        self,
        chunks: list[ChunkRecord],
        transcripts: list[TranscribedSegment],
        final_segments: list[PipelineSegment],
        low_confidence_min_cps: float | None = None,
        low_confidence_max_cps: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build chunk quality analytics."""
        effective_min = low_confidence_min_cps if low_confidence_min_cps is not None else self._low_confidence_min_cps
        effective_max = low_confidence_max_cps if low_confidence_max_cps is not None else self._low_confidence_max_cps

        return _build_chunk_quality_analytics(
            chunks, transcripts, final_segments, effective_min, effective_max
        )
