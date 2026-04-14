import unittest
from pathlib import Path

from audio_pipeline.models import ChunkRecord
from audio_pipeline.stages.asr_transcribe import transcribe_chunks


class _StubAdapter:
    def __init__(self) -> None:
        self.requested_provider_name = "chirp"
        self.provider_name = "nemo"
        self.model_name = "nvidia/model"
        self.fallback_used = True
        self.fallback_reason = "primary_init_failed"

    def transcribe_batch(self, audio_paths: list[Path], batch_size: int, pretokenize: bool) -> list[str]:
        del batch_size
        del pretokenize
        return ["hello" for _ in audio_paths]


class AsrTranscribeTests(unittest.TestCase):
    def test_transcribe_chunks_reports_provider_metadata(self) -> None:
        chunks = [
            ChunkRecord(index=0, speaker="SPEAKER_00", start=0.0, end=0.10, chunk_path="a.wav"),
            ChunkRecord(index=1, speaker="SPEAKER_00", start=0.10, end=1.20, chunk_path="b.wav"),
        ]

        transcripts, report = transcribe_chunks(
            chunks=chunks,
            adapter=_StubAdapter(),
            batch_size=4,
            orchestration_batch_size=8,
            min_chunk_duration_sec=0.25,
            pretokenize=False,
        )

        self.assertEqual(len(transcripts), 2)
        self.assertEqual(transcripts[0].status, "skipped_short")
        self.assertEqual(transcripts[1].status, "ok")
        self.assertEqual(report["asr_provider_requested"], "chirp")
        self.assertEqual(report["asr_provider_effective"], "nemo")
        self.assertTrue(report["asr_fallback_used"])


if __name__ == "__main__":
    unittest.main()
