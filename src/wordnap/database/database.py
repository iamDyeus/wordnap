"""Database connection and operations using raw sqlite3."""

import sqlite3
from pathlib import Path

from wordnap.models.schemas import Segment, VideoMetadata, VideoStatus, Word


class Database:
    """SQLite database layer for video metadata, segments, and words."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            if str(self._db_path) == ":memory:":
                self._conn = sqlite3.connect(":memory:")
            else:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def initialize(self) -> None:
        """Create tables and indexes if not exist."""
        conn = self.connection
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                fps REAL NOT NULL,
                audio_sample_rate INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL REFERENCES videos(id),
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT,
                confidence REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL REFERENCES segments(id),
                video_id INTEGER NOT NULL REFERENCES videos(id),
                word TEXT NOT NULL,
                normalized_word TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                confidence REAL NOT NULL,
                speaker TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_word);
            CREATE INDEX IF NOT EXISTS idx_words_video_id ON words(video_id);
            """
        )

    def upsert_video(self, metadata: VideoMetadata) -> int:
        """Insert or update video record, return video_id."""
        conn = self.connection
        cursor = conn.execute(
            """
            INSERT INTO videos (path, filename, duration, width, height, fps, audio_sample_rate, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                filename = excluded.filename,
                duration = excluded.duration,
                width = excluded.width,
                height = excluded.height,
                fps = excluded.fps,
                audio_sample_rate = excluded.audio_sample_rate,
                created_at = excluded.created_at,
                status = excluded.status
            """,
            (
                str(metadata.path),
                metadata.filename,
                metadata.duration,
                metadata.width,
                metadata.height,
                metadata.fps,
                metadata.audio_sample_rate,
                metadata.created_at.isoformat(),
                metadata.status.value,
            ),
        )
        conn.commit()

        # Get the id (lastrowid is 0 on UPDATE, so query it)
        row = conn.execute(
            "SELECT id FROM videos WHERE path = ?", (str(metadata.path),)
        ).fetchone()
        return row["id"]

    def store_transcription(
        self, video_id: int, segments: list[Segment], words: list[Word]
    ) -> None:
        """Store segments and words for a video atomically.

        Validates temporal constraints and confidence ranges before storage.
        Raises ValueError if any word has start_time >= end_time or
        confidence outside [0.0, 1.0].
        """
        # Validate all words before storing
        for word in words:
            if word.start_time >= word.end_time:
                raise ValueError(
                    f"Word '{word.word}' has start_time ({word.start_time}) "
                    f">= end_time ({word.end_time})"
                )
            if not (0.0 <= word.confidence <= 1.0):
                raise ValueError(
                    f"Word '{word.word}' has confidence ({word.confidence}) "
                    f"outside [0.0, 1.0]"
                )

        # Validate all segments before storing
        for segment in segments:
            if segment.start_time >= segment.end_time:
                raise ValueError(
                    f"Segment has start_time ({segment.start_time}) "
                    f">= end_time ({segment.end_time})"
                )
            if not (0.0 <= segment.confidence <= 1.0):
                raise ValueError(
                    f"Segment has confidence ({segment.confidence}) "
                    f"outside [0.0, 1.0]"
                )

        conn = self.connection

        # Delete existing transcription data for this video (re-indexing)
        # Delete words first due to FK constraint on segment_id
        conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))

        # Use a transaction for atomic insert
        try:
            # Insert segments and build a mapping from original segment to db id
            segment_id_map: dict[int, int] = {}
            for i, segment in enumerate(segments):
                cursor = conn.execute(
                    """
                    INSERT INTO segments (video_id, start_time, end_time, text, speaker, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        video_id,
                        segment.start_time,
                        segment.end_time,
                        segment.text,
                        segment.speaker,
                        segment.confidence,
                    ),
                )
                segment_id_map[i] = cursor.lastrowid

            # Insert words
            for word in words:
                # Map segment_id: use the DB-assigned id from the segment_id_map.
                # Words reference segments by their position index (0-based) in the
                # segments list when the segment_id matches a key in the map.
                # Otherwise use the raw segment_id value.
                db_segment_id = segment_id_map.get(
                    word.segment_id, word.segment_id
                )
                conn.execute(
                    """
                    INSERT INTO words (segment_id, video_id, word, normalized_word, start_time, end_time, confidence, speaker)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_segment_id,
                        video_id,
                        word.word,
                        word.normalized_word,
                        word.start_time,
                        word.end_time,
                        word.confidence,
                        word.speaker,
                    ),
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def find_words(self, normalized_word: str) -> list[Word]:
        """Find all word occurrences matching normalized form."""
        conn = self.connection
        rows = conn.execute(
            "SELECT * FROM words WHERE normalized_word = ?", (normalized_word,)
        ).fetchall()

        return [
            Word(
                id=row["id"],
                segment_id=row["segment_id"],
                video_id=row["video_id"],
                word=row["word"],
                normalized_word=row["normalized_word"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                confidence=row["confidence"],
                speaker=row["speaker"],
            )
            for row in rows
        ]

    def get_video(self, video_id: int) -> VideoMetadata | None:
        """Retrieve video metadata by ID."""
        conn = self.connection
        row = conn.execute(
            "SELECT * FROM videos WHERE id = ?", (video_id,)
        ).fetchone()

        if row is None:
            return None

        return VideoMetadata(
            id=row["id"],
            path=Path(row["path"]),
            filename=row["filename"],
            duration=row["duration"],
            width=row["width"],
            height=row["height"],
            fps=row["fps"],
            audio_sample_rate=row["audio_sample_rate"],
            created_at=row["created_at"],
            status=VideoStatus(row["status"]),
        )

    def is_indexed(self, path: Path) -> bool:
        """Check if a video has already been indexed."""
        conn = self.connection
        row = conn.execute(
            "SELECT status FROM videos WHERE path = ?", (str(path),)
        ).fetchone()

        if row is None:
            return False
        return row["status"] == VideoStatus.INDEXED.value

    def update_video_status(self, video_id: int, status: str) -> None:
        """Update a video's status field."""
        conn = self.connection
        conn.execute(
            "UPDATE videos SET status = ? WHERE id = ?",
            (status, video_id),
        )
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
