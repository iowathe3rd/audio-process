from __future__ import annotations
import unittest
from app.models import DiarizationSegment, PipelineSegment
from app.stages.cleanup import SegmentCleaner
from app.stages.postprocess_segments import SegmentNormalizer


class RegressionTests(unittest.TestCase):
    def test_postprocess_and_cleanup_chain(self) -> None:
        # Original: 0.0-1.0s Speaker A
        segments = [DiarizationSegment(index=0, speaker="A", start=0.0, end=1.0)]
        normalizer = SegmentNormalizer()
        
        # padding 0ms, merge gap 0ms etc to keep simple
        processed, report, groups = normalizer.postprocess(
            segments=segments,
            audio_duration_sec=10.0,
            min_duration_ms=0,
            merge_gap_ms=0,
            padding_ms=0,
            absorb_short_gap_ms=0
        )
        
        self.assertEqual(len(processed), 1)
        
        # Fake transcript
        pipeline_segments = [
            PipelineSegment(
                speaker="A",
                start=0.0,
                end=1.0,
                raw_text="Hello world",
                anonymized_text="",
                enhanced_text="",
                chunk_path="",
            )
        ]
        
        cleaner = SegmentCleaner(min_duration_ms=0, duplicate_window_ms=0)
        cleaned, cleanup_report = cleaner.cleanup(pipeline_segments, 0, 0)
        
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].raw_text, "Hello world")


if __name__ == "__main__":
    unittest.main()
