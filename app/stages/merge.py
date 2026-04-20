from app.models import ChunkRecord, PipelineSegment, TranscribedSegment
from app.domain.contracts import SegmentMerger


class ChunkTranscriptMerger(SegmentMerger):
    """Merges chunk records with transcribed segments."""

    def merge(
        self,
        chunks: list[ChunkRecord],
        transcripts: list[TranscribedSegment],
    ) -> list[PipelineSegment]:
        text_by_index = {item.index: item.raw_text for item in transcripts}
        status_by_index = {item.index: item.status for item in transcripts}

        merged: list[PipelineSegment] = []
        for chunk in chunks:
            merged.append(
                PipelineSegment(
                    speaker=chunk.speaker,
                    start=chunk.start,
                    end=chunk.end,
                    raw_text=text_by_index.get(chunk.index, ""),
                    anonymized_text="",
                    enhanced_text="",
                    chunk_path=chunk.chunk_path,
                    asr_status=status_by_index.get(chunk.index, "error"),
                )
            )

        return merged
