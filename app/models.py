from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Any


class DiarizationSegment(BaseModel):
    model_config = ConfigDict(strict=True)

    index: int
    speaker: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DiarizationSegment:
        return cls(**value)


class ChunkRecord(BaseModel):
    model_config = ConfigDict(strict=True)

    index: int
    speaker: str
    start: float
    end: float
    chunk_path: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChunkRecord:
        return cls(**value)


class TranscribedSegment(BaseModel):
    model_config = ConfigDict(strict=True)

    index: int
    raw_text: str
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TranscribedSegment:
        return cls(**value)


class PipelineSegment(BaseModel):
    model_config = ConfigDict(strict=True)

    speaker: str
    start: float
    end: float
    raw_text: str
    anonymized_text: str
    enhanced_text: str
    chunk_path: str
    asr_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PipelineSegment:
        return cls(**value)
