"""Unit tests for the Database class using in-memory SQLite."""

from datetime import datetime
from pathlib import Path

import pytest

from sentence_mixer.database.database import Database
from sentence_mixer.models.schemas import Segment, VideoMetadata, VideoStatus, Word


@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    database = Database(db_path=Path(":memory:"))
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata."""
    return VideoMetadata(
        path=Path("/videos/test_video.mp4"),
        filename="test_video.mp4",
        duration=120.5,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=44100,
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        status=VideoStatus.PENDING,
    )


@pytest.fixture
def sample_segments():
    """Create sample segments."""
    return [
        Segment(
            video_id=1,
            start_time=0.0,
            end_time=2.5,
            text="hello world",
            speaker="speaker_1",
            confidence=0.95,
        ),
        Segment(
            video_id=1,
            start_time=2.5,
            end_time=5.0,
            text="this is a test",
            speaker="speaker_1",
            confidence=0.88,
        ),
    ]


@pytest.fixture
def sample_words():
    """Create sample words. segment_id uses 0-based index into segments list."""
    return [
        Word(
            segment_id=0,
            video_id=1,
            word="Hello",
            normalized_word="hello",
            start_time=0.0,
            end_time=0.5,
            confidence=0.95,
            speaker="speaker_1",
        ),
        Word(
            segment_id=0,
            video_id=1,
            word="world",
            normalized_word="world",
            start_time=0.6,
            end_time=1.2,
            confidence=0.92,
            speaker="speaker_1",
        ),
        Word(
            segment_id=1,
            video_id=1,
            word="test",
            normalized_word="test",
            start_time=3.0,
            end_time=3.5,
            confidence=0.88,
            speaker="speaker_1",
        ),
    ]


class TestDatabaseInitialize:
    def test_creates_tables(self, db):
        """Tables should be created on initialize."""
        conn = db.connection
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [row["name"] for row in tables]
        assert "videos" in table_names
        assert "segments" in table_names
        assert "words" in table_names

    def test_creates_indexes(self, db):
        """Indexes on normalized_word and video_id should exist."""
        conn = db.connection
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        ).fetchall()
        index_names = [row["name"] for row in indexes]
        assert "idx_words_normalized" in index_names
        assert "idx_words_video_id" in index_names

    def test_wal_mode_configured(self, db):
        """WAL mode pragma should be set (in-memory DBs report 'memory' which is expected)."""
        conn = db.connection
        result = conn.execute("PRAGMA journal_mode").fetchone()
        # In-memory databases cannot use WAL, they report 'memory'.
        # The implementation sets WAL mode which works for file-based DBs.
        assert result[0] in ("wal", "memory")

    def test_initialize_idempotent(self, db):
        """Calling initialize multiple times should not fail."""
        db.initialize()
        db.initialize()
        conn = db.connection
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len([r for r in tables if r["name"] in ("videos", "segments", "words")]) == 3


class TestUpsertVideo:
    def test_insert_new_video(self, db, sample_video_metadata):
        """Inserting a new video should return a valid id."""
        video_id = db.upsert_video(sample_video_metadata)
        assert video_id is not None
        assert video_id > 0

    def test_upsert_updates_existing(self, db, sample_video_metadata):
        """Upserting the same path should update, not create a duplicate."""
        video_id_1 = db.upsert_video(sample_video_metadata)

        # Update metadata
        updated = sample_video_metadata.model_copy(
            update={"duration": 200.0, "status": VideoStatus.INDEXED}
        )
        video_id_2 = db.upsert_video(updated)

        assert video_id_1 == video_id_2

        # Verify the update took effect
        video = db.get_video(video_id_1)
        assert video.duration == 200.0
        assert video.status == VideoStatus.INDEXED

    def test_multiple_videos_get_unique_ids(self, db):
        """Different videos should get different IDs."""
        v1 = VideoMetadata(
            path=Path("/videos/a.mp4"),
            filename="a.mp4",
            duration=60.0,
            width=1280,
            height=720,
            fps=24.0,
            audio_sample_rate=44100,
        )
        v2 = VideoMetadata(
            path=Path("/videos/b.mp4"),
            filename="b.mp4",
            duration=90.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=48000,
        )
        id1 = db.upsert_video(v1)
        id2 = db.upsert_video(v2)
        assert id1 != id2


class TestStoreTranscription:
    def test_stores_segments_and_words(self, db, sample_video_metadata, sample_segments, sample_words):
        """Segments and words should be stored correctly."""
        video_id = db.upsert_video(sample_video_metadata)
        db.store_transcription(video_id, sample_segments, sample_words)

        # Verify segments stored
        conn = db.connection
        segments = conn.execute(
            "SELECT * FROM segments WHERE video_id = ?", (video_id,)
        ).fetchall()
        assert len(segments) == 2

        # Verify words stored
        words = conn.execute(
            "SELECT * FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
        assert len(words) == 3

    def test_atomic_transaction_on_success(self, db, sample_video_metadata, sample_segments, sample_words):
        """All data should be committed atomically on success."""
        video_id = db.upsert_video(sample_video_metadata)
        db.store_transcription(video_id, sample_segments, sample_words)

        found = db.find_words("hello")
        assert len(found) == 1
        assert found[0].word == "Hello"

    def test_rejects_word_with_invalid_temporal(self, db, sample_video_metadata, sample_segments):
        """Words with start_time >= end_time should be rejected by database validation."""
        video_id = db.upsert_video(sample_video_metadata)

        # Use model_construct to bypass Pydantic validation and test DB layer validation
        invalid_word = Word.model_construct(
            id=None,
            segment_id=1,
            video_id=video_id,
            word="bad",
            normalized_word="bad",
            start_time=2.0,
            end_time=1.0,
            confidence=0.9,
            speaker=None,
        )

        with pytest.raises(ValueError, match="start_time.*end_time"):
            db.store_transcription(video_id, sample_segments, [invalid_word])

    def test_rejects_word_with_equal_start_end(self, db, sample_video_metadata, sample_segments):
        """Words with start_time == end_time should be rejected."""
        video_id = db.upsert_video(sample_video_metadata)

        invalid_word = Word.model_construct(
            id=None,
            segment_id=1,
            video_id=video_id,
            word="bad",
            normalized_word="bad",
            start_time=1.0,
            end_time=1.0,
            confidence=0.9,
            speaker=None,
        )

        with pytest.raises(ValueError, match="start_time.*end_time"):
            db.store_transcription(video_id, sample_segments, [invalid_word])

    def test_rejects_word_with_confidence_above_one(self, db, sample_video_metadata, sample_segments):
        """Words with confidence > 1.0 should be rejected."""
        video_id = db.upsert_video(sample_video_metadata)

        invalid_word = Word.model_construct(
            id=None,
            segment_id=1,
            video_id=video_id,
            word="bad",
            normalized_word="bad",
            start_time=0.0,
            end_time=0.5,
            confidence=1.5,
            speaker=None,
        )

        with pytest.raises(ValueError, match="confidence.*outside"):
            db.store_transcription(video_id, sample_segments, [invalid_word])

    def test_rejects_word_with_confidence_below_zero(self, db, sample_video_metadata, sample_segments):
        """Words with confidence < 0.0 should be rejected."""
        video_id = db.upsert_video(sample_video_metadata)

        invalid_word = Word.model_construct(
            id=None,
            segment_id=1,
            video_id=video_id,
            word="bad",
            normalized_word="bad",
            start_time=0.0,
            end_time=0.5,
            confidence=-0.1,
            speaker=None,
        )

        with pytest.raises(ValueError, match="confidence.*outside"):
            db.store_transcription(video_id, sample_segments, [invalid_word])

    def test_replaces_existing_transcription(self, db, sample_video_metadata, sample_segments, sample_words):
        """Re-storing transcription should replace old data."""
        video_id = db.upsert_video(sample_video_metadata)
        db.store_transcription(video_id, sample_segments, sample_words)

        # Store new data - use segment index 0 which maps to the new segment's DB id
        new_segments = [
            Segment(
                video_id=video_id,
                start_time=0.0,
                end_time=3.0,
                text="new text",
                confidence=0.99,
            )
        ]
        new_words = [
            Word(
                segment_id=0,  # Index 0 in new_segments list, will be mapped to DB id
                video_id=video_id,
                word="New",
                normalized_word="new",
                start_time=0.0,
                end_time=0.5,
                confidence=0.99,
            )
        ]
        db.store_transcription(video_id, new_segments, new_words)

        # Old words should be gone
        assert db.find_words("hello") == []
        assert db.find_words("world") == []

        # New word should be present
        found = db.find_words("new")
        assert len(found) == 1

    def test_rollback_on_failure(self, db, sample_video_metadata):
        """If storage fails partway through, nothing should be committed."""
        video_id = db.upsert_video(sample_video_metadata)

        segments = [
            Segment(
                video_id=video_id,
                start_time=0.0,
                end_time=2.0,
                text="hello",
                confidence=0.9,
            )
        ]

        # Use model_construct to bypass Pydantic and test DB-layer validation
        words_with_bad_entry = [
            Word(
                segment_id=0,
                video_id=video_id,
                word="good",
                normalized_word="good",
                start_time=0.0,
                end_time=0.5,
                confidence=0.9,
            ),
            Word.model_construct(
                id=None,
                segment_id=0,
                video_id=video_id,
                word="bad",
                normalized_word="bad",
                start_time=2.0,
                end_time=1.0,  # invalid - bypasses Pydantic via model_construct
                confidence=0.9,
                speaker=None,
            ),
        ]

        with pytest.raises(ValueError):
            db.store_transcription(video_id, segments, words_with_bad_entry)

        # Nothing should have been stored (validation is pre-insert)
        conn = db.connection
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM words WHERE video_id = ?", (video_id,)
        ).fetchone()
        assert count["cnt"] == 0


class TestFindWords:
    def test_finds_matching_words(self, db, sample_video_metadata, sample_segments, sample_words):
        """Should return all words matching the normalized form."""
        video_id = db.upsert_video(sample_video_metadata)
        db.store_transcription(video_id, sample_segments, sample_words)

        results = db.find_words("hello")
        assert len(results) == 1
        assert results[0].word == "Hello"
        assert results[0].normalized_word == "hello"

    def test_returns_empty_for_no_match(self, db, sample_video_metadata, sample_segments, sample_words):
        """Should return empty list when no matches exist."""
        video_id = db.upsert_video(sample_video_metadata)
        db.store_transcription(video_id, sample_segments, sample_words)

        results = db.find_words("nonexistent")
        assert results == []

    def test_finds_multiple_occurrences(self, db, sample_video_metadata):
        """Should return all occurrences across segments."""
        video_id = db.upsert_video(sample_video_metadata)

        segments = [
            Segment(video_id=video_id, start_time=0.0, end_time=2.0, text="hello", confidence=0.9),
            Segment(video_id=video_id, start_time=3.0, end_time=5.0, text="hello again", confidence=0.85),
        ]
        words = [
            Word(segment_id=0, video_id=video_id, word="hello", normalized_word="hello",
                 start_time=0.0, end_time=0.5, confidence=0.9),
            Word(segment_id=1, video_id=video_id, word="Hello", normalized_word="hello",
                 start_time=3.0, end_time=3.5, confidence=0.85),
        ]
        db.store_transcription(video_id, segments, words)

        results = db.find_words("hello")
        assert len(results) == 2

    def test_finds_words_across_videos(self, db):
        """Should find words from multiple videos."""
        v1 = VideoMetadata(
            path=Path("/videos/a.mp4"), filename="a.mp4",
            duration=60.0, width=1920, height=1080, fps=30.0, audio_sample_rate=44100,
        )
        v2 = VideoMetadata(
            path=Path("/videos/b.mp4"), filename="b.mp4",
            duration=90.0, width=1920, height=1080, fps=30.0, audio_sample_rate=44100,
        )
        id1 = db.upsert_video(v1)
        id2 = db.upsert_video(v2)

        seg1 = [Segment(video_id=id1, start_time=0.0, end_time=2.0, text="hello", confidence=0.9)]
        words1 = [Word(segment_id=0, video_id=id1, word="hello", normalized_word="hello",
                       start_time=0.0, end_time=0.5, confidence=0.9)]
        db.store_transcription(id1, seg1, words1)

        seg2 = [Segment(video_id=id2, start_time=0.0, end_time=2.0, text="hello", confidence=0.85)]
        words2 = [Word(segment_id=0, video_id=id2, word="Hello", normalized_word="hello",
                       start_time=0.0, end_time=0.6, confidence=0.85)]
        db.store_transcription(id2, seg2, words2)

        results = db.find_words("hello")
        assert len(results) == 2
        video_ids = {w.video_id for w in results}
        assert video_ids == {id1, id2}


class TestGetVideo:
    def test_retrieves_existing_video(self, db, sample_video_metadata):
        """Should return video metadata for a valid id."""
        video_id = db.upsert_video(sample_video_metadata)
        video = db.get_video(video_id)

        assert video is not None
        assert video.id == video_id
        assert video.path == sample_video_metadata.path
        assert video.filename == sample_video_metadata.filename
        assert video.duration == sample_video_metadata.duration
        assert video.width == sample_video_metadata.width
        assert video.height == sample_video_metadata.height
        assert video.fps == sample_video_metadata.fps
        assert video.audio_sample_rate == sample_video_metadata.audio_sample_rate
        assert video.status == VideoStatus.PENDING

    def test_returns_none_for_invalid_id(self, db):
        """Should return None if video_id doesn't exist."""
        result = db.get_video(999)
        assert result is None


class TestIsIndexed:
    def test_returns_false_for_unknown_path(self, db):
        """Should return False for paths not in the database."""
        assert db.is_indexed(Path("/unknown/video.mp4")) is False

    def test_returns_false_for_pending_video(self, db, sample_video_metadata):
        """Should return False for videos with pending status."""
        db.upsert_video(sample_video_metadata)
        assert db.is_indexed(sample_video_metadata.path) is False

    def test_returns_true_for_indexed_video(self, db, sample_video_metadata):
        """Should return True for videos with indexed status."""
        video_id = db.upsert_video(sample_video_metadata)
        db.update_video_status(video_id, VideoStatus.INDEXED.value)
        assert db.is_indexed(sample_video_metadata.path) is True

    def test_returns_false_for_failed_video(self, db, sample_video_metadata):
        """Should return False for videos with failed status."""
        video_id = db.upsert_video(sample_video_metadata)
        db.update_video_status(video_id, VideoStatus.FAILED.value)
        assert db.is_indexed(sample_video_metadata.path) is False


class TestUpdateVideoStatus:
    def test_updates_status(self, db, sample_video_metadata):
        """Should update the video status."""
        video_id = db.upsert_video(sample_video_metadata)
        db.update_video_status(video_id, VideoStatus.INDEXED.value)

        video = db.get_video(video_id)
        assert video.status == VideoStatus.INDEXED

    def test_updates_to_failed(self, db, sample_video_metadata):
        """Should be able to set status to failed."""
        video_id = db.upsert_video(sample_video_metadata)
        db.update_video_status(video_id, VideoStatus.FAILED.value)

        video = db.get_video(video_id)
        assert video.status == VideoStatus.FAILED


class TestDatabaseEdgeCases:
    def test_memory_database(self):
        """Should support :memory: databases for testing."""
        db = Database(db_path=Path(":memory:"))
        db.initialize()
        assert db.connection is not None
        db.close()

    def test_close_and_reopen(self):
        """Closing and creating a new instance should work."""
        db = Database(db_path=Path(":memory:"))
        db.initialize()
        db.close()
        # After close, a new database instance works
        db2 = Database(db_path=Path(":memory:"))
        db2.initialize()
        db2.close()

    def test_store_word_with_speaker_none(self, db, sample_video_metadata):
        """Words with no speaker should be stored fine."""
        video_id = db.upsert_video(sample_video_metadata)
        segments = [
            Segment(video_id=video_id, start_time=0.0, end_time=2.0, text="hi", confidence=0.9)
        ]
        words = [
            Word(
                segment_id=1, video_id=video_id, word="hi", normalized_word="hi",
                start_time=0.0, end_time=0.3, confidence=0.9, speaker=None,
            )
        ]
        db.store_transcription(video_id, segments, words)

        found = db.find_words("hi")
        assert len(found) == 1
        assert found[0].speaker is None

    def test_confidence_boundary_zero(self, db, sample_video_metadata):
        """Words with confidence=0.0 should be accepted."""
        video_id = db.upsert_video(sample_video_metadata)
        segments = [
            Segment(video_id=video_id, start_time=0.0, end_time=2.0, text="low", confidence=0.0)
        ]
        words = [
            Word(
                segment_id=1, video_id=video_id, word="low", normalized_word="low",
                start_time=0.0, end_time=0.5, confidence=0.0,
            )
        ]
        db.store_transcription(video_id, segments, words)
        found = db.find_words("low")
        assert len(found) == 1
        assert found[0].confidence == 0.0

    def test_confidence_boundary_one(self, db, sample_video_metadata):
        """Words with confidence=1.0 should be accepted."""
        video_id = db.upsert_video(sample_video_metadata)
        segments = [
            Segment(video_id=video_id, start_time=0.0, end_time=2.0, text="high", confidence=1.0)
        ]
        words = [
            Word(
                segment_id=1, video_id=video_id, word="high", normalized_word="high",
                start_time=0.0, end_time=0.5, confidence=1.0,
            )
        ]
        db.store_transcription(video_id, segments, words)
        found = db.find_words("high")
        assert len(found) == 1
        assert found[0].confidence == 1.0
