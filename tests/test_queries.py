"""Unit tests for database query helpers."""

from pathlib import Path

from sentence_mixer.database.database import Database
from sentence_mixer.database.queries import (
    find_consecutive_words_in_segment,
    find_words_batch,
    find_words_by_normalized,
    find_words_with_segment_context,
    is_indexed,
)
from sentence_mixer.models.schemas import Segment, VideoMetadata, VideoStatus, Word


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


class TestFindWordsByNormalized:
    def test_returns_empty_for_missing_word(self):
        db = _create_test_db()
        _insert_video(db)
        result = find_words_by_normalized(db, "nonexistent")
        assert result == []

    def test_returns_candidates_for_matching_word(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("Hello", "hello", 1.0, 1.5, 0.95)])

        result = find_words_by_normalized(db, "hello")

        assert len(result) == 1
        candidate = result[0]
        assert candidate.word.normalized_word == "hello"
        assert candidate.word.word == "Hello"
        assert candidate.video.id == video_id
        assert candidate.duration == 0.5

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

        result = find_words_by_normalized(db, "hello")
        assert len(result) == 3

    def test_candidates_from_multiple_videos(self):
        db = _create_test_db()
        vid1 = _insert_video(db, "/videos/v1.mp4")
        vid2 = _insert_video(db, "/videos/v2.mp4")
        _store_words(db, vid1, [("world", "world", 2.0, 2.5, 0.9)])
        _store_words(db, vid2, [("World", "world", 3.0, 3.6, 0.8)])

        result = find_words_by_normalized(db, "world")

        assert len(result) == 2
        video_ids = {c.video.id for c in result}
        assert video_ids == {vid1, vid2}

    def test_candidate_has_correct_duration(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("test", "test", 2.0, 3.7, 0.88)])

        result = find_words_by_normalized(db, "test")
        assert len(result) == 1
        assert abs(result[0].duration - 1.7) < 1e-9

    def test_candidate_video_has_correct_metadata(self):
        db = _create_test_db()
        video_id = _insert_video(db, "/videos/special.mp4")
        _store_words(db, video_id, [("cat", "cat", 0.5, 1.0, 0.92)])

        result = find_words_by_normalized(db, "cat")
        assert len(result) == 1
        assert result[0].video.filename == "special.mp4"
        assert result[0].video.duration == 60.0

    def test_score_defaults_to_zero(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hi", "hi", 0.1, 0.3, 0.95)])

        result = find_words_by_normalized(db, "hi")
        assert result[0].score == 0.0


class TestFindWordsBatch:
    def test_empty_token_list(self):
        db = _create_test_db()
        result = find_words_batch(db, [])
        assert result == {}

    def test_single_token(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])

        result = find_words_batch(db, ["hello"])
        assert "hello" in result
        assert len(result["hello"]) == 1

    def test_multiple_tokens(self):
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

        result = find_words_batch(db, ["hello", "world"])
        assert len(result) == 2
        assert len(result["hello"]) == 1
        assert len(result["world"]) == 1

    def test_missing_token_returns_empty_list(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("hello", "hello", 1.0, 1.5, 0.9)])

        result = find_words_batch(db, ["hello", "missing"])
        assert len(result["hello"]) == 1
        assert result["missing"] == []

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

        tokens = ["the", "quick", "fox"]
        result = find_words_batch(db, tokens)
        assert list(result.keys()) == tokens

    def test_duplicate_tokens_handled(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words(db, video_id, [("the", "the", 0.0, 0.3, 0.9)])

        result = find_words_batch(db, ["the", "the"])
        # Last entry wins in dict, both map to same candidates
        assert "the" in result
        assert len(result["the"]) == 1


class TestIsIndexed:
    def test_returns_false_for_unknown_path(self):
        db = _create_test_db()
        assert is_indexed(db, Path("/videos/unknown.mp4")) is False

    def test_returns_true_for_indexed_video(self):
        db = _create_test_db()
        _insert_video(db, "/videos/indexed.mp4")
        assert is_indexed(db, Path("/videos/indexed.mp4")) is True

    def test_returns_false_for_pending_video(self):
        db = _create_test_db()
        video = VideoMetadata(
            path=Path("/videos/pending.mp4"),
            filename="pending.mp4",
            duration=30.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
            status=VideoStatus.PENDING,
        )
        db.upsert_video(video)
        assert is_indexed(db, Path("/videos/pending.mp4")) is False

    def test_returns_false_for_failed_video(self):
        db = _create_test_db()
        video = VideoMetadata(
            path=Path("/videos/failed.mp4"),
            filename="failed.mp4",
            duration=30.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
            status=VideoStatus.FAILED,
        )
        db.upsert_video(video)
        assert is_indexed(db, Path("/videos/failed.mp4")) is False


def _store_words_in_segment(
    db: Database,
    video_id: int,
    segment_text: str,
    words_data: list[tuple[str, str, float, float, float]],
    segment_start: float = 0.0,
    segment_end: float = 60.0,
) -> int:
    """Store a segment with words and return the segment_id.

    Each tuple in words_data: (raw_word, normalized_word, start, end, confidence).
    Returns the DB-assigned segment ID.
    """
    segments = [
        Segment(
            video_id=video_id,
            start_time=segment_start,
            end_time=segment_end,
            text=segment_text,
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

    # Get the segment ID that was just inserted
    conn = db.connection
    row = conn.execute(
        "SELECT id FROM segments WHERE video_id = ? ORDER BY id DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    return row["id"]


class TestFindConsecutiveWordsInSegment:
    def test_returns_words_when_consecutive_match(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "hello world today",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
                ("today", "today", 2.1, 2.5, 0.8),
            ],
        )

        # Get the first word's ID
        conn = db.connection
        first_word = conn.execute(
            "SELECT id FROM words WHERE segment_id = ? ORDER BY start_time ASC LIMIT 1",
            (seg_id,),
        ).fetchone()

        result = find_consecutive_words_in_segment(
            db, seg_id, first_word["id"], ["hello", "world"]
        )

        assert result is not None
        assert len(result) == 2
        assert result[0].normalized_word == "hello"
        assert result[1].normalized_word == "world"

    def test_returns_none_when_tokens_dont_match(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )

        conn = db.connection
        first_word = conn.execute(
            "SELECT id FROM words WHERE segment_id = ? ORDER BY start_time ASC LIMIT 1",
            (seg_id,),
        ).fetchone()

        result = find_consecutive_words_in_segment(
            db, seg_id, first_word["id"], ["hello", "foo"]
        )

        assert result is None

    def test_returns_none_when_not_enough_words(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "hello",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
            ],
        )

        conn = db.connection
        first_word = conn.execute(
            "SELECT id FROM words WHERE segment_id = ? ORDER BY start_time ASC LIMIT 1",
            (seg_id,),
        ).fetchone()

        result = find_consecutive_words_in_segment(
            db, seg_id, first_word["id"], ["hello", "world"]
        )

        assert result is None

    def test_matches_starting_from_middle_of_segment(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "the quick brown fox",
            [
                ("the", "the", 1.0, 1.3, 0.9),
                ("quick", "quick", 1.4, 1.8, 0.9),
                ("brown", "brown", 1.9, 2.3, 0.9),
                ("fox", "fox", 2.4, 2.7, 0.9),
            ],
        )

        # Get the "quick" word's ID (second word)
        conn = db.connection
        second_word = conn.execute(
            "SELECT id FROM words WHERE segment_id = ? ORDER BY start_time ASC LIMIT 1 OFFSET 1",
            (seg_id,),
        ).fetchone()

        result = find_consecutive_words_in_segment(
            db, seg_id, second_word["id"], ["quick", "brown", "fox"]
        )

        assert result is not None
        assert len(result) == 3
        assert result[0].normalized_word == "quick"
        assert result[1].normalized_word == "brown"
        assert result[2].normalized_word == "fox"

    def test_single_token_match(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )

        conn = db.connection
        first_word = conn.execute(
            "SELECT id FROM words WHERE segment_id = ? ORDER BY start_time ASC LIMIT 1",
            (seg_id,),
        ).fetchone()

        result = find_consecutive_words_in_segment(
            db, seg_id, first_word["id"], ["hello"]
        )

        assert result is not None
        assert len(result) == 1
        assert result[0].normalized_word == "hello"

    def test_returns_word_objects_with_correct_fields(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("Hello", "hello", 1.0, 1.5, 0.95),
                ("World", "world", 1.6, 2.0, 0.85),
            ],
        )

        conn = db.connection
        first_word = conn.execute(
            "SELECT id FROM words WHERE segment_id = ? ORDER BY start_time ASC LIMIT 1",
            (seg_id,),
        ).fetchone()

        result = find_consecutive_words_in_segment(
            db, seg_id, first_word["id"], ["hello", "world"]
        )

        assert result is not None
        assert result[0].word == "Hello"
        assert result[0].normalized_word == "hello"
        assert result[0].start_time == 1.0
        assert result[0].end_time == 1.5
        assert result[0].confidence == 0.95
        assert result[0].video_id == video_id
        assert result[0].segment_id == seg_id


class TestFindWordsWithSegmentContext:
    def test_returns_empty_for_missing_word(self):
        db = _create_test_db()
        _insert_video(db)
        result = find_words_with_segment_context(db, "nonexistent")
        assert result == []

    def test_returns_word_and_segment_id(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )

        result = find_words_with_segment_context(db, "hello")

        assert len(result) == 1
        word, segment_id = result[0]
        assert word.normalized_word == "hello"
        assert word.word == "hello"
        assert segment_id == seg_id

    def test_returns_multiple_occurrences(self):
        db = _create_test_db()
        vid1 = _insert_video(db, "/videos/v1.mp4")
        vid2 = _insert_video(db, "/videos/v2.mp4")
        _store_words_in_segment(
            db,
            vid1,
            "hello there",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("there", "there", 1.6, 2.0, 0.8),
            ],
        )
        _store_words_in_segment(
            db,
            vid2,
            "say hello",
            [
                ("say", "say", 0.5, 0.9, 0.9),
                ("hello", "hello", 1.0, 1.4, 0.85),
            ],
        )

        result = find_words_with_segment_context(db, "hello")

        assert len(result) == 2
        normalized_words = [w.normalized_word for w, _ in result]
        assert all(nw == "hello" for nw in normalized_words)

    def test_segment_id_matches_word_segment(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        seg_id = _store_words_in_segment(
            db,
            video_id,
            "the quick brown fox",
            [
                ("the", "the", 1.0, 1.3, 0.9),
                ("quick", "quick", 1.4, 1.8, 0.9),
                ("brown", "brown", 1.9, 2.3, 0.9),
                ("fox", "fox", 2.4, 2.7, 0.9),
            ],
        )

        result = find_words_with_segment_context(db, "fox")

        assert len(result) == 1
        word, segment_id = result[0]
        assert word.segment_id == segment_id
        assert segment_id == seg_id
