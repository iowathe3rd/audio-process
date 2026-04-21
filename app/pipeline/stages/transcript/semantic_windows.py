from __future__ import annotations

from statistics import median
from typing import Any

from app.models import PipelineSegment
from app.domain.contracts import SemanticWindowBuilder


def _window_duration(items: list[tuple[int, PipelineSegment]]) -> float:
    if not items:
        return 0.0
    return max(0.0, float(items[-1][1].end - items[0][1].start))


def _build_window(
    index: int,
    items: list[tuple[int, PipelineSegment]],
) -> dict[str, Any]:
    segment_positions = [position for position, _ in items]
    speakers = [segment.speaker for _, segment in items]
    unique_speakers: list[str] = []
    for speaker in speakers:
        if not unique_speakers or unique_speakers[-1] != speaker:
            unique_speakers.append(speaker)

    texts = [segment.raw_text.strip() for _, segment in items]
    return {
        "index": index,
        "segment_positions": segment_positions,
        "start": float(items[0][1].start),
        "end": float(items[-1][1].end),
        "duration_sec": max(0.0, float(items[-1][1].end - items[0][1].start)),
        "speakers": unique_speakers,
        "segments_count": len(items),
        "chars_count": sum(len(text) for text in texts),
        "texts": texts,
    }


def _build_semantic_windows(
    segments: list[PipelineSegment],
    max_chars: int,
    max_duration_sec: float,
    max_gap_sec: float,
    max_speaker_switches: int,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    safe_max_chars = max(80, int(max_chars))
    safe_max_duration = max(5.0, float(max_duration_sec))
    safe_max_gap = max(0.0, float(max_gap_sec))
    safe_max_switches = max(0, int(max_speaker_switches))

    indexed_segments = sorted(
        [
            (position, item)
            for position, item in enumerate(segments)
            if item.raw_text.strip()
        ],
        key=lambda value: (value[1].start, value[1].end),
    )

    if not indexed_segments:
        return [], {
            "windows_total": 0,
            "segments_total": len(segments),
            "segments_covered": 0,
            "avg_segments_per_window": 0.0,
            "avg_chars_per_window": 0.0,
            "avg_window_duration_sec": 0.0,
            "median_window_duration_sec": 0.0,
            "max_segments_per_window": 0,
            "max_chars": safe_max_chars,
            "max_duration_sec": safe_max_duration,
            "max_gap_sec": safe_max_gap,
            "max_speaker_switches": safe_max_switches,
        }

    windows_raw: list[list[tuple[int, PipelineSegment]]] = []
    current: list[tuple[int, PipelineSegment]] = []
    current_chars = 0
    current_speaker_switches = 0

    for item in indexed_segments:
        position, segment = item
        text = segment.raw_text.strip()

        if not current:
            current = [item]
            current_chars = len(text)
            current_speaker_switches = 0
            continue

        previous_segment = current[-1][1]
        gap = max(0.0, float(segment.start - previous_segment.end))
        speaker_switch = 1 if previous_segment.speaker != segment.speaker else 0

        prospective_chars = current_chars + len(text)
        prospective_duration = max(0.0, float(segment.end - current[0][1].start))
        prospective_switches = current_speaker_switches + speaker_switch

        should_split = (
            gap > safe_max_gap
            or prospective_chars > safe_max_chars
            or prospective_duration > safe_max_duration
            or prospective_switches > safe_max_switches
        )

        if should_split:
            windows_raw.append(current)
            current = [item]
            current_chars = len(text)
            current_speaker_switches = 0
            continue

        current.append((position, segment))
        current_chars = prospective_chars
        current_speaker_switches = prospective_switches

    if current:
        windows_raw.append(current)

    windows = [_build_window(index, items) for index, items in enumerate(windows_raw)]
    durations = [float(window["duration_sec"]) for window in windows]
    segment_counts = [int(window["segments_count"]) for window in windows]
    chars_counts = [int(window["chars_count"]) for window in windows]

    report: dict[str, int | float] = {
        "windows_total": len(windows),
        "segments_total": len(segments),
        "segments_covered": sum(segment_counts),
        "avg_segments_per_window": (sum(segment_counts) / len(segment_counts)) if segment_counts else 0.0,
        "avg_chars_per_window": (sum(chars_counts) / len(chars_counts)) if chars_counts else 0.0,
        "avg_window_duration_sec": (sum(durations) / len(durations)) if durations else 0.0,
        "median_window_duration_sec": median(durations) if durations else 0.0,
        "max_segments_per_window": max(segment_counts) if segment_counts else 0,
        "max_chars": safe_max_chars,
        "max_duration_sec": safe_max_duration,
        "max_gap_sec": safe_max_gap,
        "max_speaker_switches": safe_max_switches,
    }
    return windows, report


class SemanticWindowGrouper(SemanticWindowBuilder):
    """Groups segments into semantic windows for LLM processing."""

    def __init__(
        self,
        max_chars: int,
        max_duration_sec: float,
        max_gap_sec: float,
        max_speaker_switches: int,
    ) -> None:
        self._max_chars = max_chars
        self._max_duration_sec = max_duration_sec
        self._max_gap_sec = max_gap_sec
        self._max_speaker_switches = max_speaker_switches

    def build_windows(
        self,
        segments: list[PipelineSegment],
        max_chars: int | None = None,
        max_duration_sec: float | None = None,
        max_gap_sec: float | None = None,
        max_speaker_switches: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
        """Group segments into semantic windows."""
        effective_chars = max_chars if max_chars is not None else self._max_chars
        effective_duration = max_duration_sec if max_duration_sec is not None else self._max_duration_sec
        effective_gap = max_gap_sec if max_gap_sec is not None else self._max_gap_sec
        effective_switches = max_speaker_switches if max_speaker_switches is not None else self._max_speaker_switches

        return _build_semantic_windows(
            segments, effective_chars, effective_duration,
            effective_gap, effective_switches
        )