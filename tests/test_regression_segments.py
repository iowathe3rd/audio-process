import unittest

from audio_pipeline.models import DiarizationSegment, PipelineSegment
from audio_pipeline.stages.cleanup import cleanup_transcript_segments
from audio_pipeline.stages.postprocess_segments import normalize_speaker_segments


class RegressionSegmentPipelineTests(unittest.TestCase):
    def test_postprocess_merges_short_adjacent_segments(self) -> None:
        diarization_segments = [
            DiarizationSegment(index=0, speaker="SPEAKER_00", start=0.00, end=0.12),
            DiarizationSegment(index=1, speaker="SPEAKER_00", start=0.15, end=0.33),
            DiarizationSegment(index=2, speaker="SPEAKER_00", start=0.37, end=0.58),
            DiarizationSegment(index=3, speaker="SPEAKER_01", start=0.65, end=0.85),
            DiarizationSegment(index=4, speaker="SPEAKER_01", start=0.90, end=1.10),
        ]

        normalized, report, merged_groups = normalize_speaker_segments(
            segments=diarization_segments,
            audio_duration_sec=2.0,
            min_segment_duration_ms=400,
            merge_gap_ms=300,
            padding_ms=50,
            absorb_short_gap_ms=220,
        )

        self.assertLess(len(normalized), len(diarization_segments))
        self.assertGreaterEqual(int(report["merged_segments_count"]), 2)
        self.assertLessEqual(int(report["short_segments_after"]), int(report["short_segments_before"]))
        self.assertTrue(all(segment.end > segment.start for segment in normalized))
        self.assertEqual(len(merged_groups), len(normalized))

    def test_cleanup_removes_skipped_empty_and_duplicates(self) -> None:
        merged_segments = [
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.00,
                end=0.15,
                raw_text="",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c0.wav",
                asr_status="skipped_short",
            ),
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.16,
                end=0.34,
                raw_text="",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c1.wav",
                asr_status="ok",
            ),
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.35,
                end=1.10,
                raw_text="hello world",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c2.wav",
                asr_status="ok",
            ),
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.36,
                end=1.08,
                raw_text="hello   world",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c3.wav",
                asr_status="ok",
            ),
            PipelineSegment(
                speaker="SPEAKER_01",
                start=1.05,
                end=1.50,
                raw_text="next reply",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c4.wav",
                asr_status="ok",
            ),
        ]

        cleaned, report = cleanup_transcript_segments(
            segments=merged_segments,
            min_duration_ms=300,
            duplicate_window_ms=250,
        )

        self.assertEqual(int(report["skipped_short_removed"]), 1)
        self.assertEqual(int(report["empty_segments_removed"]), 1)
        self.assertGreaterEqual(int(report["duplicate_segments_removed"]), 1)
        self.assertTrue(all(segment.raw_text.strip() for segment in cleaned))
        self.assertTrue(all(segment.asr_status != "skipped_short" for segment in cleaned))
        self.assertEqual(len(cleaned), 2)


if __name__ == "__main__":
    unittest.main()
