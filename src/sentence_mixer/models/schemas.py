"""Pydantic data models for Sentence Mixer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class VideoStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    id: int | None = None
    path: Path
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    audio_sample_rate: int
    created_at: datetime = Field(default_factory=datetime.now)
    status: VideoStatus = VideoStatus.PENDING

    @field_validator("duration")
    @classmethod
    def duration_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duration must be positive")
        return v


class Segment(BaseModel):
    id: int | None = None
    video_id: int
    start_time: float
    end_time: float
    text: str
    speaker: str | None = None
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in range [0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def start_before_end(self) -> "Segment":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")
        return self


class Word(BaseModel):
    id: int | None = None
    segment_id: int
    video_id: int
    word: str
    normalized_word: str
    start_time: float
    end_time: float
    confidence: float
    speaker: str | None = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in range [0.0, 1.0]")
        return v

    @field_validator("normalized_word")
    @classmethod
    def normalized_word_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("normalized_word must be non-empty")
        return v

    @model_validator(mode="after")
    def start_before_end(self) -> "Word":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")
        return self


class WordCandidate(BaseModel):
    word: Word
    video: VideoMetadata
    duration: float = Field(description="end_time - start_time")
    score: float = 0.0

    @field_validator("duration")
    @classmethod
    def duration_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duration must be positive")
        return v


class TokenInfo(BaseModel):
    """Extended token with punctuation context from the original sentence."""

    normalized: str
    original: str
    trailing_punctuation: str | None = None


class GapEntry(BaseModel):
    """A silence gap between clips in the EDL."""

    duration_ms: float
    reason: str  # "default", "comma", "sentence_end"


class RankingConfig(BaseModel):
    confidence_weight: float = 0.35
    duration_weight: float = 0.25
    boundary_quality_weight: float = 0.20
    diversity_weight: float = 0.20
    min_confidence: float = 0.3
    min_duration: float = 0.03
    max_duration: float = 3.0
    ideal_duration_min: float = 0.1
    ideal_duration_max: float = 1.5
    prefer_same_speaker: bool = True


class ClipEntry(BaseModel):
    source_video: Path
    source_filename: str
    word: str
    start_time: float
    end_time: float
    padded_start: float
    padded_end: float
    confidence: float
    speaker: str | None = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in range [0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def start_before_end(self) -> "ClipEntry":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")
        return self


class EDLManifest(BaseModel):
    sentence: str
    variation_index: int
    clips: list[ClipEntry]
    gaps: list[GapEntry] = []
    skipped_words: list[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    total_duration: float = 0.0

    @field_validator("clips")
    @classmethod
    def clips_non_empty(cls, v: list[ClipEntry]) -> list[ClipEntry]:
        if not v:
            raise ValueError("clips list must be non-empty")
        return v


class TranscriptionResult(BaseModel):
    """Container for transcription output."""

    segments: list[Segment]
    words: list[Word]


class RenderConfig(BaseModel):
    output_resolution: tuple[int, int] = (1920, 1080)
    output_fps: float = 30.0
    pixel_format: str = "yuv420p"
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    clip_padding: float = 0.10
    # V1.5 additions
    default_gap_ms: float = 80.0
    punctuation_pause_enabled: bool = True
    comma_gap_ms: float = 200.0
    sentence_end_gap_ms: float = 400.0
    subtitles_enabled: bool = True
    # V2 additions
    playback_speed: float = 0.9


class PhraseCandidate(BaseModel):
    """A multi-word match from a single segment."""

    words: list[Word]
    segment: Segment
    video: VideoMetadata
    start_time: float
    end_time: float
    duration: float
    score: float = 0.0

    @model_validator(mode="after")
    def validate_words_consecutive(self) -> PhraseCandidate:
        """Ensure all words belong to the same segment."""
        for w in self.words:
            if w.segment_id != self.segment.id:
                raise ValueError("All words must belong to the same segment")
        return self
