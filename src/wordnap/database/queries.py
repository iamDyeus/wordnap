"""Database query helpers for word search and video lookup.

Higher-level query functions that return WordCandidate objects (Word + VideoMetadata)
for use by the search engine and generation pipeline.
"""

from pathlib import Path

from wordnap.database.database import Database
from wordnap.models.schemas import VideoMetadata, Word, WordCandidate


def find_words_by_normalized(
    db: Database, normalized_word: str
) -> list[WordCandidate]:
    """Find all Word records matching a normalized form, enriched with video metadata.

    Returns WordCandidate objects combining the Word record with its source
    VideoMetadata and computed duration.

    Args:
        db: Database instance to query.
        normalized_word: The normalized word to search for.

    Returns:
        List of WordCandidate objects for all matching words.
        Words whose video metadata cannot be found are skipped.
    """
    words = db.find_words(normalized_word)
    candidates: list[WordCandidate] = []

    for word in words:
        video = db.get_video(word.video_id)
        if video is None:
            continue
        candidate = WordCandidate(
            word=word,
            video=video,
            duration=word.end_time - word.start_time,
        )
        candidates.append(candidate)

    return candidates


def find_words_batch(
    db: Database, tokens: list[str]
) -> dict[str, list[WordCandidate]]:
    """Batch search for multiple normalized tokens.

    Returns a dict mapping each token to its list of matching WordCandidate
    objects, preserving the original token order in the dict.

    Args:
        db: Database instance to query.
        tokens: Ordered list of normalized tokens to search for.

    Returns:
        Dict mapping each token to its list of WordCandidate matches.
        Tokens with no matches map to an empty list.
    """
    results: dict[str, list[WordCandidate]] = {}

    for token in tokens:
        results[token] = find_words_by_normalized(db, token)

    return results


def find_consecutive_words_in_segment(
    db: Database, segment_id: int, start_word_id: int, expected_tokens: list[str]
) -> list[Word] | None:
    """Verify that consecutive words in a segment match expected tokens.

    Queries words in the segment ordered by start_time, starting from the
    word with start_word_id, and checks the next N words match expected_tokens.

    Args:
        db: Database instance.
        segment_id: The segment to search within.
        start_word_id: The word ID to start from.
        expected_tokens: Ordered list of normalized tokens to match.

    Returns:
        List of matching Word objects if consecutive match found, None otherwise.
    """
    conn = db.connection
    rows = conn.execute(
        """
        SELECT * FROM words
        WHERE segment_id = ? AND start_time >= (
            SELECT start_time FROM words WHERE id = ?
        )
        ORDER BY start_time ASC
        LIMIT ?
        """,
        (segment_id, start_word_id, len(expected_tokens)),
    ).fetchall()

    if len(rows) != len(expected_tokens):
        return None

    words = []
    for row, expected in zip(rows, expected_tokens):
        if row["normalized_word"] != expected:
            return None
        words.append(
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
        )

    return words


def find_words_with_segment_context(
    db: Database, normalized_word: str
) -> list[tuple[Word, int]]:
    """Find all words matching normalized form, returning (Word, segment_id) pairs.

    Used by phrase search to find starting points for phrase matching.

    Args:
        db: Database instance to query.
        normalized_word: The normalized word to search for.

    Returns:
        List of (Word, segment_id) tuples for all matching words.
    """
    conn = db.connection
    rows = conn.execute(
        "SELECT * FROM words WHERE normalized_word = ?", (normalized_word,)
    ).fetchall()

    return [
        (
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
            ),
            row["segment_id"],
        )
        for row in rows
    ]


def is_indexed(db: Database, path: Path) -> bool:
    """Check if a video file has already been indexed.

    Delegates to the Database.is_indexed() method. Provided here for
    consistent access through the queries module.

    Args:
        db: Database instance to query.
        path: Path to the video file.

    Returns:
        True if the video has status 'indexed', False otherwise.
    """
    return db.is_indexed(path)
