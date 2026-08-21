"""Unit tests for PhraseSearchEngine."""

from pathlib import Path

from wordnap.database.database import Database
from wordnap.models.schemas import Segment, VideoMetadata, VideoStatus, Word
from wordnap.search.phrase_search import PhraseSearchEngine


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

    conn = db.connection
    row = conn.execute(
        "SELECT id FROM segments WHERE video_id = ? ORDER BY id DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    return row["id"]


class TestPhraseSearchEmptyAndSingle:
    """Test edge cases: empty tokens and single token."""

    def test_empty_tokens_returns_empty(self):
        db = _create_test_db()
        _insert_video(db)
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases([])

        assert matches == {}
        assert covered == set()

    def test_single_token_returns_empty(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(["hello"])

        assert matches == {}
        assert covered == set()


class TestPhraseSearchBigram:
    """Test bigram (two-word) phrase matching."""

    def test_bigram_matching(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(["hello", "world"])

        assert (0, 2) in matches
        assert len(matches[(0, 2)]) == 1
        candidate = matches[(0, 2)][0]
        assert len(candidate.words) == 2
        assert candidate.words[0].normalized_word == "hello"
        assert candidate.words[1].normalized_word == "world"
        assert covered == {0, 1}


class TestPhraseSearchTrigram:
    """Test trigram (three-word) phrase matching."""

    def test_trigram_matching(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words_in_segment(
            db,
            video_id,
            "the quick fox",
            [
                ("the", "the", 1.0, 1.3, 0.9),
                ("quick", "quick", 1.4, 1.8, 0.9),
                ("fox", "fox", 1.9, 2.3, 0.9),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(["the", "quick", "fox"])

        assert (0, 3) in matches
        assert len(matches[(0, 3)]) == 1
        candidate = matches[(0, 3)][0]
        assert len(candidate.words) == 3
        assert candidate.words[0].normalized_word == "the"
        assert candidate.words[1].normalized_word == "quick"
        assert candidate.words[2].normalized_word == "fox"
        assert covered == {0, 1, 2}


class TestPhraseSearchNoMatch:
    """Test that non-consecutive words don't match as phrases."""

    def test_no_match_when_words_not_consecutive(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        # "hello" and "world" exist but with "gap" between them
        _store_words_in_segment(
            db,
            video_id,
            "hello gap world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("gap", "gap", 1.6, 1.9, 0.9),
                ("world", "world", 2.0, 2.5, 0.85),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(["hello", "world"])

        # Should NOT find "hello world" as a phrase since they're not consecutive
        assert (0, 2) not in matches
        assert covered == set()


class TestPhraseSearchLongestFirst:
    """Test longest-first greedy strategy."""

    def test_longest_phrase_takes_priority(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        # Segment contains "hello world today" as consecutive words
        _store_words_in_segment(
            db,
            video_id,
            "hello world today",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
                ("today", "today", 2.1, 2.5, 0.8),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(["hello", "world", "today"])

        # Should match the full trigram (3 words = max_phrase_length default)
        assert (0, 3) in matches
        assert (0, 2) not in matches  # Bigram should NOT be present
        assert covered == {0, 1, 2}

    def test_covered_positions_prevent_shorter_overlapping_matches(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        # Segment has "hello world today" as consecutive
        _store_words_in_segment(
            db,
            video_id,
            "hello world today",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
                ("today", "today", 2.1, 2.5, 0.8),
            ],
        )
        engine = PhraseSearchEngine(db)

        # Search for tokens that include the trigram plus an extra token
        matches, covered = engine.find_phrases(
            ["hello", "world", "today", "extra"]
        )

        # "hello world today" should match as trigram
        assert (0, 3) in matches
        # "world today" should NOT match because positions 1,2 are already covered
        assert (1, 3) not in matches
        # Positions 0, 1, 2 should be covered
        assert {0, 1, 2}.issubset(covered)

    def test_max_phrase_length_caps_window_size(self):
        """Phrases longer than max_phrase_length are not matched."""
        db = _create_test_db()
        video_id = _insert_video(db)
        # Segment contains 5 consecutive words
        _store_words_in_segment(
            db,
            video_id,
            "one two three four five",
            [
                ("one", "one", 1.0, 1.3, 0.9),
                ("two", "two", 1.4, 1.7, 0.9),
                ("three", "three", 1.8, 2.1, 0.9),
                ("four", "four", 2.2, 2.5, 0.9),
                ("five", "five", 2.6, 2.9, 0.9),
            ],
        )
        # Default max_phrase_length=3 should prevent matching all 5 as one phrase
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(
            ["one", "two", "three", "four", "five"]
        )

        # Should NOT have a 5-word or 4-word match
        assert (0, 5) not in matches
        assert (0, 4) not in matches
        # Should have trigram matches instead
        assert (0, 3) in matches

    def test_custom_max_phrase_length(self):
        """Custom max_phrase_length allows longer phrases."""
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words_in_segment(
            db,
            video_id,
            "one two three four five",
            [
                ("one", "one", 1.0, 1.3, 0.9),
                ("two", "two", 1.4, 1.7, 0.9),
                ("three", "three", 1.8, 2.1, 0.9),
                ("four", "four", 2.2, 2.5, 0.9),
                ("five", "five", 2.6, 2.9, 0.9),
            ],
        )
        # Allow up to 5-word phrases
        engine = PhraseSearchEngine(db, max_phrase_length=5)

        matches, covered = engine.find_phrases(
            ["one", "two", "three", "four", "five"]
        )

        # Should match all 5 as one phrase
        assert (0, 5) in matches
        assert covered == {0, 1, 2, 3, 4}


class TestPhraseSearchCandidateDetails:
    """Test that PhraseCandidate objects have correct fields."""

    def test_phrase_candidate_timing(self):
        db = _create_test_db()
        video_id = _insert_video(db)
        _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, _ = engine.find_phrases(["hello", "world"])

        candidate = matches[(0, 2)][0]
        assert candidate.start_time == 1.0
        assert candidate.end_time == 2.0
        assert candidate.duration == 1.0

    def test_phrase_candidate_has_video_and_segment(self):
        db = _create_test_db()
        video_id = _insert_video(db, "/videos/special.mp4")
        _store_words_in_segment(
            db,
            video_id,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, _ = engine.find_phrases(["hello", "world"])

        candidate = matches[(0, 2)][0]
        assert candidate.video.id == video_id
        assert candidate.video.filename == "special.mp4"
        assert candidate.segment is not None
        assert candidate.segment.video_id == video_id

    def test_multiple_candidates_for_same_phrase(self):
        db = _create_test_db()
        vid1 = _insert_video(db, "/videos/v1.mp4")
        vid2 = _insert_video(db, "/videos/v2.mp4")
        _store_words_in_segment(
            db,
            vid1,
            "hello world",
            [
                ("hello", "hello", 1.0, 1.5, 0.9),
                ("world", "world", 1.6, 2.0, 0.85),
            ],
        )
        _store_words_in_segment(
            db,
            vid2,
            "hello world",
            [
                ("hello", "hello", 0.5, 1.0, 0.88),
                ("world", "world", 1.1, 1.6, 0.82),
            ],
        )
        engine = PhraseSearchEngine(db)

        matches, covered = engine.find_phrases(["hello", "world"])

        assert (0, 2) in matches
        assert len(matches[(0, 2)]) == 2
        assert covered == {0, 1}
