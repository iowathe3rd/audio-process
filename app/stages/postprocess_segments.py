from app.models import DiarizationSegment
from app.domain.contracts import SegmentPostprocessor
from typing import Any


def _duration(segment: DiarizationSegment) -> float:
    return max(0.0, float(segment.end - segment.start))


def _clone(segment: DiarizationSegment) -> DiarizationSegment:
    return DiarizationSegment(
        index=int(segment.index),
        speaker=str(segment.speaker),
        start=float(segment.start),
        end=float(segment.end),
    )


def _entry(segment: DiarizationSegment, source_indices: list[int] | None = None) -> dict[str, Any]:
    return {
        "segment": _clone(segment),
        "source_indices": list(source_indices or [int(segment.index)]),
        "merge_reasons": [],
    }


def _normalize_speaker_segments(
    segments: list[DiarizationSegment],
    audio_duration_sec: float,
    min_segment_duration_ms: int,
    merge_gap_ms: int,
    padding_ms: int,
    absorb_short_gap_ms: int = 220,
) -> tuple[list[DiarizationSegment], dict[str, Any], list[dict[str, Any]]]:
    min_duration_sec = max(0.0, float(min_segment_duration_ms) / 1000.0)
    merge_gap_sec = max(0.0, float(merge_gap_ms) / 1000.0)
    padding_sec = max(0.0, float(padding_ms) / 1000.0)
    absorb_short_gap_sec = max(0.0, float(absorb_short_gap_ms) / 1000.0)
    audio_duration = max(0.0, float(audio_duration_sec))

    valid_entries: list[dict[str, Any]] = []
    for item in sorted(segments, key=lambda value: (value.start, value.end)):
        start = max(0.0, float(item.start))
        end = min(audio_duration, float(item.end))
        if end <= start:
            continue
        valid_entries.append(
            _entry(
                DiarizationSegment(index=item.index, speaker=item.speaker, start=start, end=end),
                source_indices=[int(item.index)],
            )
        )

    if not valid_entries:
        return (
            [],
            {
                "input_segments": len(segments),
                "output_segments": 0,
                "merged_segments_count": 0,
                "overlap_conflicts_count": 0,
                "short_segments_before": 0,
                "short_segments_after": 0,
                "short_segment_ratio_before": 0.0,
                "short_segment_ratio_after": 0.0,
                "absorbed_short_segments_count": 0,
                "reassigned_short_overlap_count": 0,
            },
            [],
        )

    short_before = sum(1 for item in valid_entries if _duration(item["segment"]) < min_duration_sec)

    merged: list[dict[str, Any]] = []
    merged_segments_count = 0
    for current in valid_entries:
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]["segment"]
        current_segment = current["segment"]
        gap = float(current_segment.start - previous.end)
        combined_duration = float(current_segment.end - previous.start)
        previous_duration = _duration(previous)
        current_duration = _duration(current_segment)

        should_merge = (
            current_segment.speaker == previous.speaker
            and gap <= merge_gap_sec
            and (
                combined_duration >= min_duration_sec
                or previous_duration < min_duration_sec
                or current_duration < min_duration_sec
            )
        )

        if should_merge:
            previous.end = max(previous.end, current_segment.end)
            merged[-1]["source_indices"].extend(current["source_indices"])
            merged[-1]["merge_reasons"].append("same_speaker_gap")
            merged_segments_count += 1
        else:
            merged.append(current)

    absorbed_short_segments_count = 0
    index = 0
    while index < len(merged):
        item = merged[index]
        segment = item["segment"]
        if _duration(segment) >= min_duration_sec or len(merged) <= 1:
            index += 1
            continue

        left = merged[index - 1] if index > 0 else None
        right = merged[index + 1] if (index + 1) < len(merged) else None

        left_gap = float("inf")
        right_gap = float("inf")
        if left is not None:
            left_gap = max(0.0, float(segment.start - left["segment"].end))
        if right is not None:
            right_gap = max(0.0, float(right["segment"].start - segment.end))

        target = ""
        if left is not None and left_gap <= absorb_short_gap_sec:
            target = "left"
        if right is not None and right_gap <= absorb_short_gap_sec and right_gap < left_gap:
            target = "right"

        if target == "left" and left is not None:
            left["segment"].end = max(left["segment"].end, segment.end)
            left["source_indices"].extend(item["source_indices"])
            left["merge_reasons"].append("absorbed_short")
            merged.pop(index)
            absorbed_short_segments_count += 1
            merged_segments_count += 1
            continue

        if target == "right" and right is not None:
            right["segment"].start = min(right["segment"].start, segment.start)
            right["source_indices"] = item["source_indices"] + right["source_indices"]
            right["merge_reasons"].append("absorbed_short")
            merged.pop(index)
            absorbed_short_segments_count += 1
            merged_segments_count += 1
            continue

        index += 1

    # Optional context padding for better ASR stability on boundaries.
    padded: list[dict[str, Any]] = []
    for item in merged:
        segment = item["segment"]
        start = max(0.0, segment.start - padding_sec)
        end = min(audio_duration, segment.end + padding_sec)
        if end <= start:
            continue
        padded.append(
            {
                "segment": DiarizationSegment(
                    index=segment.index,
                    speaker=segment.speaker,
                    start=start,
                    end=end,
                ),
                "source_indices": list(item["source_indices"]),
                "merge_reasons": list(item["merge_reasons"]),
            }
        )

    # Resolve overlap conflicts introduced by padding.
    overlap_conflicts_count = 0
    reassigned_short_overlap_count = 0
    padded.sort(key=lambda value: (value["segment"].start, value["segment"].end))
    for index in range(len(padded) - 1):
        left = padded[index]["segment"]
        right = padded[index + 1]["segment"]
        if left.end <= right.start:
            continue

        overlap_conflicts_count += 1
        if left.speaker == right.speaker:
            left.end = max(left.end, right.end)
            padded[index]["source_indices"].extend(padded[index + 1]["source_indices"])
            padded[index]["merge_reasons"].append("same_speaker_overlap")
            merged_segments_count += 1
            right.start = right.end
            continue

        left_duration = _duration(left)
        right_duration = _duration(right)
        overlap_reassign_threshold = max(0.08, min_duration_sec * 0.5)
        if min(left_duration, right_duration) <= overlap_reassign_threshold:
            if left_duration <= right_duration:
                padded[index + 1]["segment"].start = min(
                    padded[index + 1]["segment"].start,
                    padded[index]["segment"].start,
                )
                padded[index + 1]["source_indices"] = (
                    padded[index]["source_indices"] + padded[index + 1]["source_indices"]
                )
                padded[index + 1]["merge_reasons"].append("reassigned_short_overlap")
                padded[index]["segment"].end = padded[index]["segment"].start
            else:
                padded[index]["segment"].end = max(
                    padded[index]["segment"].end,
                    padded[index + 1]["segment"].end,
                )
                padded[index]["source_indices"].extend(padded[index + 1]["source_indices"])
                padded[index]["merge_reasons"].append("reassigned_short_overlap")
                padded[index + 1]["segment"].start = padded[index + 1]["segment"].end

            merged_segments_count += 1
            reassigned_short_overlap_count += 1
            continue

        midpoint = (left.end + right.start) / 2.0
        left.end = max(left.start, midpoint)
        right.start = min(right.end, midpoint)

    output_entries: list[dict[str, Any]] = []
    for item in padded:
        segment = item["segment"]
        if segment.end <= segment.start:
            continue
        output_entries.append(item)

    output_entries.sort(key=lambda value: (value["segment"].start, value["segment"].end))
    output: list[DiarizationSegment] = []
    merged_groups: list[dict[str, Any]] = []
    for index, item in enumerate(output_entries):
        segment = item["segment"]
        segment.index = index
        output.append(segment)
        merged_groups.append(
            {
                "output_index": index,
                "speaker": segment.speaker,
                "start": float(segment.start),
                "end": float(segment.end),
                "duration_sec": _duration(segment),
                "source_indices": sorted({int(value) for value in item["source_indices"]}),
                "merge_reasons": sorted({str(value) for value in item["merge_reasons"]}),
            }
        )

    short_after = sum(1 for item in output if _duration(item) < min_duration_sec)

    report: dict[str, Any] = {
        "input_segments": len(segments),
        "output_segments": len(output),
        "merged_segments_count": merged_segments_count,
        "overlap_conflicts_count": overlap_conflicts_count,
        "short_segments_before": short_before,
        "short_segments_after": short_after,
        "short_segment_ratio_before": short_before / max(1, len(valid_entries)),
        "short_segment_ratio_after": short_after / max(1, len(output)),
        "absorbed_short_segments_count": absorbed_short_segments_count,
        "reassigned_short_overlap_count": reassigned_short_overlap_count,
        "min_segment_duration_ms": int(min_segment_duration_ms),
        "merge_gap_ms": int(merge_gap_ms),
        "padding_ms": int(padding_ms),
        "absorb_short_gap_ms": int(absorb_short_gap_ms),
    }
    return output, report, merged_groups


class SegmentNormalizer(SegmentPostprocessor):
    """Post-processes diarization segments for ASR readiness.

    Normalizes, merges, and pads segments for optimal transcription.
    """

    def postprocess(
        self,
        segments: list[DiarizationSegment],
        audio_duration_sec: float,
        min_duration_ms: int,
        merge_gap_ms: int,
        padding_ms: int,
        absorb_short_gap_ms: int,
    ) -> tuple[list[DiarizationSegment], dict[str, Any], list[Any]]:
        """Normalize, merge, and pad diarization segments."""
        return _normalize_speaker_segments(
            segments, audio_duration_sec, min_duration_ms,
            merge_gap_ms, padding_ms, absorb_short_gap_ms
        )
