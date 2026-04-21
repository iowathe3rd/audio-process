from __future__ import annotations
import unittest
from typing import Any

from app.models import PipelineSegment
from app.pipeline.stages.transcript.semantic_windows import SemanticWindowGrouper
from app.pipeline.stages.text.vertex_text import VertexTextProcessor


class _DriftProcessor:
    def enhance_batch(self, texts: list[str]) -> tuple[list[str], dict[str, Any]]:
        # Intentional meaning shift to verify drift guard fallback.
        return ["ну нашу крадут это же противозаконно" for _ in texts], {
            "llm_calls_total": 1,
            "llm_latency_total_ms": 10.0,
            "chars_in_call": sum(len(item) for item in texts),
            "segments_in_call": len(texts),
            "windows_failed": 0,
            "fallback_segments": 0,
        }

    def anonymize_batch(self, texts: list[str]) -> tuple[list[str], dict[str, Any]]:
        return texts, {}


class SemanticWindowsAndContractsTests(unittest.TestCase):
    def test_semantic_windows_reduce_call_granularity(self) -> None:
        segments = [
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.0,
                end=1.2,
                raw_text="алло здравствуйте",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c0.wav",
                asr_status="ok",
            ),
            PipelineSegment(
                speaker="SPEAKER_00",
                start=1.3,
                end=2.0,
                raw_text="у меня вопрос",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c1.wav",
                asr_status="ok",
            ),
            PipelineSegment(
                speaker="SPEAKER_01",
                start=2.1,
                end=3.1,
                raw_text="слушаю вас",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c2.wav",
                asr_status="ok",
            ),
            PipelineSegment(
                speaker="SPEAKER_00",
                start=7.8,
                end=8.6,
                raw_text="я по поводу жалобы",
                anonymized_text="",
                enhanced_text="",
                chunk_path="c3.wav",
                asr_status="ok",
            ),
        ]

        grouper = SemanticWindowGrouper(
            max_chars=140,
            max_duration_sec=12.0,
            max_gap_sec=1.0,
            max_speaker_switches=3,
        )
        windows, report = grouper.build_windows(
            segments=segments,
            max_chars=140,
            max_duration_sec=12.0,
            max_gap_sec=1.0,
            max_speaker_switches=3,
        )

        self.assertEqual(int(report["windows_total"]), 2)
        self.assertLess(int(report["windows_total"]), len(segments))
        self.assertEqual(int(report["segments_covered"]), len(segments))

    def test_enhancement_contract_blocks_semantic_drift(self) -> None:
        segments = [
            PipelineSegment(
                speaker="SPEAKER_00",
                start=0.0,
                end=2.5,
                raw_text="ну нашу курят это же противозаконно",
                anonymized_text="ну нашу курят это же противозаконно",
                enhanced_text="",
                chunk_path="c0.wav",
                asr_status="ok",
            )
        ]
        windows = [
            {
                "index": 0,
                "segment_positions": [0],
                "start": 0.0,
                "end": 2.5,
                "segments_count": 1,
                "chars_count": 36,
            }
        ]

        # Use a mock-like processor
        class MockVertexProcessor(VertexTextProcessor):
            def __init__(self):
                self.enabled = True
                self.client = True
                self.strict = True
            
            def enhance_batch(self, texts):
                return _DriftProcessor().enhance_batch(texts)

        processor = MockVertexProcessor()

        enhanced, report = processor.enhance(
            segments=segments,
            semantic_windows=windows,
            mode="llm",
            low_confidence_positions=None,
            skip_low_confidence=False
        )

        self.assertEqual(enhanced[0].enhanced_text, "ну нашу курят это же противозаконно")
        self.assertEqual(int(report["semantic_drift_flags_total"]), 1)


if __name__ == "__main__":
    unittest.main()
