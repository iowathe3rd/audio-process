import unittest

from audio_pipeline.models import ChunkRecord, PipelineSegment, TranscribedSegment
from audio_pipeline.stages.chunk_quality import build_chunk_quality_analytics


class ChunkQualityTests(unittest.TestCase):
    def test_chunk_quality_marks_suspicious_substitution(self) -> None:
        chunks = [
            ChunkRecord(index=0, speaker="SPEAKER_00", start=0.0, end=1.0, chunk_path="c0.wav"),
            ChunkRecord(index=1, speaker="SPEAKER_01", start=0.9, end=2.0, chunk_path="c1.wav"),
        ]
        transcripts = [
            TranscribedSegment(index=0, raw_text="ну нашу курят это же противозаконно", status="ok"),
            TranscribedSegment(index=1, raw_text="123456789", status="ok"),
        ]
        final_segments = [
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.0,
                end=1.0,
                raw_text="ну нашу курят это же противозаконно",
                anonymized_text="ну нашу курят это же противозаконно",
                enhanced_text="Ну нашу крадут это же противозаконно",
                chunk_path="c0.wav",
                asr_status="ok",
            )
        ]

        rows, summary = build_chunk_quality_analytics(
            chunks=chunks,
            transcripts=transcripts,
            final_segments=final_segments,
            low_confidence_min_cps=1.0,
            low_confidence_max_cps=20.0,
        )

        row0 = rows[0]
        row1 = rows[1]

        self.assertTrue(row0["suspicious_substitution_flag"])
        self.assertTrue(row0["enhancement_changed_text"])
        self.assertTrue(row0["overlap_marker"])
        self.assertTrue(row1["cleaned_out"])
        self.assertGreaterEqual(float(summary["low_confidence_chunk_ratio"]), 0.0)


if __name__ == "__main__":
    unittest.main()
