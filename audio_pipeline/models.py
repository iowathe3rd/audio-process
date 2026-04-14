from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DiarizationSegment:
    index: int
    speaker: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DiarizationSegment":
        return cls(
            index=int(value["index"]),
            speaker=str(value["speaker"]),
            start=float(value["start"]),
            end=float(value["end"]),
        )


@dataclass
class ChunkRecord:
    index: int
    speaker: str
    start: float
    end: float
    chunk_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChunkRecord":
        return cls(
            index=int(value["index"]),
            speaker=str(value["speaker"]),
            start=float(value["start"]),
            end=float(value["end"]),
            chunk_path=str(value["chunk_path"]),
        )


@dataclass
class TranscribedSegment:
    index: int
    raw_text: str
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscribedSegment":
        return cls(
            index=int(value["index"]),
            raw_text=str(value.get("raw_text", "")),
            status=str(value.get("status", "ok")),
        )


@dataclass
class PipelineSegment:
    speaker: str
    start: float
    end: float
    raw_text: str
    anonymized_text: str
    enhanced_text: str
    chunk_path: str
    asr_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PipelineSegment":
        return cls(
            speaker=str(value["speaker"]),
            start=float(value["start"]),
            end=float(value["end"]),
            raw_text=str(value.get("raw_text", "")),
            anonymized_text=str(value.get("anonymized_text", "")),
            enhanced_text=str(value.get("enhanced_text", "")),
            chunk_path=str(value.get("chunk_path", "")),
            asr_status=str(value.get("asr_status", "ok")),
        )
