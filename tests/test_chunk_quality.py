from __future__ import annotations
import unittest
from app.models import ChunkRecord, PipelineSegment, TranscribedSegment
from app.stages.chunk_quality import QualityAnalyzer


class ChunkQualityTests(unittest.TestCase):
    def test_analyzer_flags_low_cps(self) -> None:
        # 1 char in 10 seconds = 0.1 CPS (very low)
        chunks = [ChunkRecord(index=0, speaker="A", start=0.0, end=10.0, chunk_path="")]
        transcripts = [TranscribedSegment(index=0, raw_text="H")]
        final = [
            PipelineSegment(
                speaker="A",
                start=0.0,
                end=10.0,
                raw_text="H",
                anonymized_text="H",
                enhanced_text="H",
                chunk_path="",
            )
        ]

        analyzer = QualityAnalyzer(low_confidence_min_cps=1.0, low_confidence_max_cps=20.0)
        rows, report = analyzer.analyze(chunks, transcripts, final, 1.0, 20.0)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["low_confidence_marker"])
        self.assertGreater(report["low_confidence_chunk_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
