"""Unit tests for SearchEngine in search/candidate.py."""

from pathlib import Path

import pytest

from sentence_mixer.database.database import Database
from sentence_mixer.models.schemas import Segment, VideoMetadata, VideoStatus, Word
from sentence_mixer.search.candidate import SearchEngine, WordNotFoundError


def _create_test_db() -> Database:
    """Create an in-memory database with test data."""
    db = Database(Path(":memory:"))
    db.initialize()
    return db


def _insert_video(db: Database, path: str = "/videos/test.mp4") -> int:
    """Insert a test video and return its ID."""
    video = VideoMetadata(
        path=Path(path),
        filename=path.split("/")[-1],
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=44100,
        status=VideoStatus.INDEXED,
    )
    return db.upsert_video(video)


def _store_words(
    db: Database, video_id: int, words_data: list[tuple[str, str, float, float, float]]
) -> None:
    """Store test words. Each tuple: (raw_word, normalized_word, start, end, confidence)."""
    segments = [
        Segment(
            video_id=video_id,
            start_time=0.0,
            end_time=60.0,
            text="test segment",
            confidence=0.9,
        )
    ]
    words = [
        Word(
            segment_id=0,
            video_id=video_id,
            word=raw,
            normalized_word=normalized,
            start_time=start,
            end_time=end,
            confidence=conf,
        )
        for raw, normalized, start, end, conf in words_data
    ]
    db.store_transcription(video_id, segments, words)


class TestWordNotFoundError:
    def test_single_missing_word(self):
        err = WordNotFoundError(["hello"])
        assert err.missing_words == ["hello"]
        assert "hello" in str(err)

    def test_multiple_missing_words(self):
        err = WordNotFoundError(["hello", "world"])
        assert err.missing_words == ["hello", "world"]
        assert "hello" in str(err)
        assert "world" in str(err)

    def test_is_exception(self):
        err = WordNotFoundError(["test"])
        assert isinstance(err, Exception)


class TestSearchEngineFindCandidates:
    def test_returns_empty_for_missing_word(self):
        db = _create_test_db()
        _insert_video(db)
        engine = SearchEngine(db)

        result = engine.find_candidates("nonexistent")
        assert result == []

    def test_returns_candidates_for_matching_word(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("Hello", "hello", 1.0, 1.5, 0.95)])
        engine = SearchEngine(db)

        result = engine.find_candidates("hello")

        assert len(result) == 1
        assert result[0].word.normalized_word == "hello"
        assert result[0].word.word == "Hello"
        assert result[0].video.id == video_id
        assert result[0].duration == 0.5

    def test_returns_multiple_candidates(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(
            db,
            video_id,
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("Hello", "hello", 5.0, 5.4, 0.85),
                ("HELLO", "hello", 10.0, 10.6, 0.7),
            ],
        )
        engine = SearchEngine(db)

        result = engine.find_candidates("hello")
        assert len(result) == 3

    def test_candidates_from_multiple_videos(self):
        db = _create_test_db()
        vid1 = _insert_video(db, "/videos/v1.mp4")
        vid2 = _insert_video(db, "/videos/v2.mp4")
        _store_words(db, vid1, [("world", "world", 2.0, 2.5, 0.9)])
        _store_words(db, vid2, [("World", "world", 3.0, 3.6, 0.8)])
        engine = SearchEngine(db)

        result = engine.find_candidates("world")

        assert len(result) == 2
        video_ids = {c.video.id for c in result}
        assert video_ids == {vid1, vid2}


class TestSearchEngineFindCandidatesBatch:
    def test_empty_token_list(self):
        db = _create_test_db()
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch([])
        assert result == {}
        assert missing == []

    def test_single_token_found(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch(["hello"])
        assert "hello" in result
        assert len(result["hello"]) == 1
        assert missing == []

    def test_multiple_tokens_found(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(
            db,
            video_id,
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 2.0, 2.5, 0.85),
            ],
        )
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch(["hello", "world"])
        assert len(result) == 2
        assert len(result["hello"]) == 1
        assert len(result["world"]) == 1
        assert missing == []

    def test_raises_word_not_found_for_missing_token(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])
        engine = SearchEngine(db)

        with pytest.raises(WordNotFoundError) as exc_info:
            engine.find_candidates_batch(["hello", "missing"])

        assert exc_info.value.missing_words == ["missing"]

    def test_raises_word_not_found_for_multiple_missing(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])
        engine = SearchEngine(db)

        with pytest.raises(WordNotFoundError) as exc_info:
            engine.find_candidates_batch(["hello", "foo", "bar"])

        assert set(exc_info.value.missing_words) == {"foo", "bar"}

    def test_preserves_token_order(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(
            db,
            video_id,
            [
                ("the", "the", 0.0, 0.3, 0.9),
                ("quick", "quick", 0.5, 1.0, 0.9),
                ("fox", "fox", 1.2, 1.6, 0.9),
            ],
        )
        engine = SearchEngine(db)

        tokens = ["the", "quick", "fox"]
        result, missing = engine.find_candidates_batch(tokens)
        assert list(result.keys()) == tokens
        assert missing == []

    def test_all_missing_raises_error(self):
        db = _create_test_db()
        engine = SearchEngine(db)

        with pytest.raises(WordNotFoundError) as exc_info:
            engine.find_candidates_batch(["nonexistent", "missing"])

        assert set(exc_info.value.missing_words) == {"nonexistent", "missing"}


class TestSearchEngineFindCandidatesBatchBestEffort:
    """Tests for strict=False (best-effort) mode."""

    def test_returns_partial_results_with_missing(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch(
            ["hello", "nonexistent"], strict=False
        )
        assert "hello" in result
        assert len(result["hello"]) == 1
        assert "nonexistent" not in result
        assert missing == ["nonexistent"]

    def test_all_missing_returns_empty_dict(self):
        db = _create_test_db()
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch(
            ["foo", "bar"], strict=False
        )
        assert result == {}
        assert set(missing) == {"foo", "bar"}

    def test_no_missing_returns_empty_missing_list(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(
            db,
            video_id,
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 2.0, 2.5, 0.85),
            ],
        )
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch(
            ["hello", "world"], strict=False
        )
        assert len(result) == 2
        assert missing == []

    def test_multiple_missing_tokens(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch(
            ["hello", "foo", "bar", "baz"], strict=False
        )
        assert "hello" in result
        assert set(missing) == {"foo", "bar", "baz"}

    def test_empty_tokens_best_effort(self):
        db = _create_test_db()
        engine = SearchEngine(db)

        result, missing = engine.find_candidates_batch([], strict=False)
        assert result == {}
        assert missing == []
