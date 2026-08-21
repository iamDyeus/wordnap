"""Unit tests for Pydantic schemas validation."""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from wordnap.models.schemas import (
    ClipEntry,
    EDLManifest,
    RankingConfig,
    RenderConfig,
    Segment,
    VideoMetadata,
    VideoStatus,
    Word,
    WordCandidate,
)


# --- VideoMetadata tests ---


class TestVideoMetadata:
    def test_valid_video_metadata(self):
        vm = VideoMetadata(
            path=Path("/videos/test.mp4"),
            filename="test.mp4",
            duration=120.5,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        assert vm.duration == 120.5
        assert vm.status == VideoStatus.PENDING
        assert vm.id is None

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError, match="duration must be positive"):
            VideoMetadata(
                path=Path("/videos/test.mp4"),
                filename="test.mp4",
                duration=0.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError, match="duration must be positive"):
            VideoMetadata(
                path=Path("/videos/test.mp4"),
                filename="test.mp4",
                duration=-1.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )


# --- Segment tests ---


class TestSegment:
    def test_valid_segment(self):
        seg = Segment(
            video_id=1,
            start_time=0.5,
            end_time=3.0,
            text="hello world",
            confidence=0.95,
        )
        assert seg.start_time == 0.5
        assert seg.end_time == 3.0

    def test_start_must_be_less_than_end(self):
        with pytest.raises(ValidationError, match="start_time must be less than end_time"):
            Segment(
                video_id=1,
                start_time=3.0,
                end_time=2.0,
                text="hello",
                confidence=0.9,
            )

    def test_equal_start_and_end_rejected(self):
        with pytest.raises(ValidationError, match="start_time must be less than end_time"):
            Segment(
                video_id=1,
                start_time=2.0,
                end_time=2.0,
                text="hello",
                confidence=0.9,
            )

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="confidence must be in range"):
            Segment(
                video_id=1,
                start_time=0.0,
                end_time=1.0,
                text="hello",
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError, match="confidence must be in range"):
            Segment(
                video_id=1,
                start_time=0.0,
                end_time=1.0,
                text="hello",
                confidence=1.1,
            )

    def test_confidence_at_boundaries(self):
        seg_zero = Segment(
            video_id=1, start_time=0.0, end_time=1.0, text="a", confidence=0.0
        )
        seg_one = Segment(
            video_id=1, start_time=0.0, end_time=1.0, text="b", confidence=1.0
        )
        assert seg_zero.confidence == 0.0
        assert seg_one.confidence == 1.0


# --- Word tests ---


class TestWord:
    def test_valid_word(self):
        w = Word(
            segment_id=1,
            video_id=1,
            word="Hello",
            normalized_word="hello",
            start_time=0.5,
            end_time=0.8,
            confidence=0.92,
        )
        assert w.normalized_word == "hello"

    def test_start_must_be_less_than_end(self):
        with pytest.raises(ValidationError, match="start_time must be less than end_time"):
            Word(
                segment_id=1,
                video_id=1,
                word="Hello",
                normalized_word="hello",
                start_time=1.0,
                end_time=0.5,
                confidence=0.9,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError, match="confidence must be in range"):
            Word(
                segment_id=1,
                video_id=1,
                word="Hello",
                normalized_word="hello",
                start_time=0.0,
                end_time=0.5,
                confidence=1.5,
            )

    def test_empty_normalized_word_rejected(self):
        with pytest.raises(ValidationError, match="normalized_word must be non-empty"):
            Word(
                segment_id=1,
                video_id=1,
                word="!!!",
                normalized_word="",
                start_time=0.0,
                end_time=0.5,
                confidence=0.9,
            )

    def test_whitespace_only_normalized_word_rejected(self):
        with pytest.raises(ValidationError, match="normalized_word must be non-empty"):
            Word(
                segment_id=1,
                video_id=1,
                word="  ",
                normalized_word="   ",
                start_time=0.0,
                end_time=0.5,
                confidence=0.9,
            )


# --- WordCandidate tests ---


class TestWordCandidate:
    def _make_word(self):
        return Word(
            segment_id=1,
            video_id=1,
            word="test",
            normalized_word="test",
            start_time=0.5,
            end_time=1.0,
            confidence=0.9,
        )

    def _make_video(self):
        return VideoMetadata(
            path=Path("/videos/test.mp4"),
            filename="test.mp4",
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )

    def test_valid_word_candidate(self):
        wc = WordCandidate(
            word=self._make_word(),
            video=self._make_video(),
            duration=0.5,
        )
        assert wc.duration == 0.5
        assert wc.score == 0.0

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError, match="duration must be positive"):
            WordCandidate(
                word=self._make_word(),
                video=self._make_video(),
                duration=0.0,
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError, match="duration must be positive"):
            WordCandidate(
                word=self._make_word(),
                video=self._make_video(),
                duration=-0.1,
            )


# --- ClipEntry tests ---


class TestClipEntry:
    def test_valid_clip_entry(self):
        clip = ClipEntry(
            source_video=Path("/videos/test.mp4"),
            source_filename="test.mp4",
            word="hello",
            start_time=1.0,
            end_time=1.5,
            padded_start=0.9,
            padded_end=1.6,
            confidence=0.95,
        )
        assert clip.word == "hello"

    def test_start_must_be_less_than_end(self):
        with pytest.raises(ValidationError, match="start_time must be less than end_time"):
            ClipEntry(
                source_video=Path("/videos/test.mp4"),
                source_filename="test.mp4",
                word="hello",
                start_time=2.0,
                end_time=1.5,
                padded_start=1.9,
                padded_end=1.6,
                confidence=0.95,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError, match="confidence must be in range"):
            ClipEntry(
                source_video=Path("/videos/test.mp4"),
                source_filename="test.mp4",
                word="hello",
                start_time=1.0,
                end_time=1.5,
                padded_start=0.9,
                padded_end=1.6,
                confidence=2.0,
            )


# --- EDLManifest tests ---


class TestEDLManifest:
    def _make_clip(self):
        return ClipEntry(
            source_video=Path("/videos/test.mp4"),
            source_filename="test.mp4",
            word="hello",
            start_time=1.0,
            end_time=1.5,
            padded_start=0.9,
            padded_end=1.6,
            confidence=0.95,
        )

    def test_valid_edl_manifest(self):
        edl = EDLManifest(
            sentence="hello world",
            variation_index=0,
            clips=[self._make_clip()],
            total_duration=0.7,
        )
        assert edl.sentence == "hello world"
        assert len(edl.clips) == 1

    def test_empty_clips_rejected(self):
        with pytest.raises(ValidationError, match="clips list must be non-empty"):
            EDLManifest(
                sentence="hello world",
                variation_index=0,
                clips=[],
            )


# --- RankingConfig tests ---


class TestRankingConfig:
    def test_default_values(self):
        config = RankingConfig()
        assert config.confidence_weight == 0.35
        assert config.duration_weight == 0.25
        assert config.boundary_quality_weight == 0.20
        assert config.diversity_weight == 0.20
        assert config.min_confidence == 0.3
        assert config.min_duration == 0.03
        assert config.max_duration == 3.0
        assert config.ideal_duration_min == 0.1
        assert config.ideal_duration_max == 1.5
        assert config.prefer_same_speaker is True


# --- RenderConfig tests ---


class TestRenderConfig:
    def test_default_values(self):
        config = RenderConfig()
        assert config.output_resolution == (1920, 1080)
        assert config.output_fps == 30.0
        assert config.pixel_format == "yuv420p"
        assert config.audio_sample_rate == 44100
        assert config.audio_channels == 2
        assert config.video_codec == "libx264"
        assert config.audio_codec == "aac"
        assert config.clip_padding == 0.10
