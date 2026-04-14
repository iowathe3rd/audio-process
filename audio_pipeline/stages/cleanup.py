from audio_pipeline.models import PipelineSegment


def _duration(segment: PipelineSegment) -> float:
    return max(0.0, float(segment.end - segment.start))


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _clone(segment: PipelineSegment) -> PipelineSegment:
    return PipelineSegment(
        speaker=str(segment.speaker),
        start=float(segment.start),
        end=float(segment.end),
        raw_text=str(segment.raw_text),
        anonymized_text=str(segment.anonymized_text),
        enhanced_text=str(segment.enhanced_text),
        chunk_path=str(segment.chunk_path),
        asr_status=str(segment.asr_status),
    )


def cleanup_transcript_segments(
    segments: list[PipelineSegment],
    min_duration_ms: int,
    duplicate_window_ms: int,
) -> tuple[list[PipelineSegment], dict[str, int | float]]:
    min_duration_sec = max(0.0, float(min_duration_ms) / 1000.0)
    duplicate_window_sec = max(0.0, float(duplicate_window_ms) / 1000.0)

    sorted_segments = sorted(segments, key=lambda value: (value.start, value.end))

    skipped_short_removed = 0
    empty_segments_removed = 0
    short_empty_segments_removed = 0

    filtered: list[PipelineSegment] = []
    for item in sorted_segments:
        text = item.raw_text.strip()
        duration = _duration(item)

        if item.asr_status == "skipped_short":
            skipped_short_removed += 1
            continue

        if not text:
            empty_segments_removed += 1
            if duration < min_duration_sec:
                short_empty_segments_removed += 1
            continue

        filtered.append(_clone(item))

    duplicate_segments_removed = 0
    deduplicated: list[PipelineSegment] = []
    for item in filtered:
        if not deduplicated:
            deduplicated.append(item)
            continue

        previous = deduplicated[-1]
        same_speaker = previous.speaker == item.speaker
        same_text = _normalize_text(previous.raw_text) == _normalize_text(item.raw_text)
        close_in_time = (
            abs(previous.start - item.start) <= duplicate_window_sec
            and abs(previous.end - item.end) <= duplicate_window_sec
        )

        overlap = max(0.0, min(previous.end, item.end) - max(previous.start, item.start))
        min_duration = max(1e-6, min(_duration(previous), _duration(item)))
        heavy_overlap = overlap / min_duration >= 0.8

        if same_speaker and same_text and (close_in_time or heavy_overlap):
            duplicate_segments_removed += 1
            if _duration(item) > _duration(previous):
                deduplicated[-1] = item
            continue

        deduplicated.append(item)

    overlap_conflicts_count = 0
    for index in range(len(deduplicated) - 1):
        left = deduplicated[index]
        right = deduplicated[index + 1]
        if left.end <= right.start:
            continue

        overlap_conflicts_count += 1
        midpoint = (left.end + right.start) / 2.0
        left.end = max(left.start, midpoint)
        right.start = min(right.end, midpoint)

    collapsed_segments_removed = 0
    output: list[PipelineSegment] = []
    for item in deduplicated:
        if item.end <= item.start:
            collapsed_segments_removed += 1
            continue

        if not item.raw_text.strip():
            collapsed_segments_removed += 1
            continue

        output.append(item)

    report: dict[str, int | float] = {
        "input_segments": len(segments),
        "output_segments": len(output),
        "skipped_short_removed": skipped_short_removed,
        "empty_segments_removed": empty_segments_removed,
        "short_empty_segments_removed": short_empty_segments_removed,
        "duplicate_segments_removed": duplicate_segments_removed,
        "overlap_conflicts_count": overlap_conflicts_count,
        "collapsed_segments_removed": collapsed_segments_removed,
        "cleanup_min_duration_ms": int(min_duration_ms),
        "cleanup_duplicate_window_ms": int(duplicate_window_ms),
    }
    return output, report
