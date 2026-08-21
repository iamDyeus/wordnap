"""Phrase-level search engine for multi-word matching."""

import logging

from wordnap.database.database import Database
from wordnap.database.queries import (
    find_consecutive_words_in_segment,
    find_words_with_segment_context,
)
from wordnap.models.schemas import PhraseCandidate, Segment, Word

logger = logging.getLogger(__name__)


class PhraseSearchEngine:
    """Finds multi-word phrase matches from indexed segments.

    Uses a longest-first greedy strategy: tries to match the longest
    possible subsequence of tokens as a phrase, then works down to bigrams.
    """

    def __init__(self, db: Database, max_phrase_length: int = 3):
        self._db = db
        self._max_phrase_length = max_phrase_length

    def find_phrases(
        self, tokens: list[str]
    ) -> tuple[dict[tuple[int, int], list[PhraseCandidate]], set[int]]:
        """Find phrase matches for a token sequence.

        Args:
            tokens: Ordered list of normalized tokens.

        Returns:
            Tuple of:
            - Dict mapping (start_pos, end_pos) to list of PhraseCandidates
            - Set of token positions covered by phrase matches
        """
        if len(tokens) < 2:
            return {}, set()

        covered: set[int] = set()
        phrase_matches: dict[tuple[int, int], list[PhraseCandidate]] = {}

        # Try longest windows first, down to bigrams (capped by max_phrase_length)
        max_window = min(len(tokens), self._max_phrase_length)
        for window_size in range(max_window, 1, -1):
            for start_pos in range(len(tokens) - window_size + 1):
                end_pos = start_pos + window_size  # exclusive

                # Skip if any position in this window is already covered
                positions = set(range(start_pos, end_pos))
                if positions & covered:
                    continue

                token_window = tokens[start_pos:end_pos]
                candidates = self._find_phrase_in_segments(token_window)

                if candidates:
                    phrase_matches[(start_pos, end_pos)] = candidates
                    covered.update(positions)

        return phrase_matches, covered

    def _find_phrase_in_segments(
        self, token_window: list[str]
    ) -> list[PhraseCandidate]:
        """Search for segments containing consecutive tokens matching the window.

        Algorithm:
        1. Find all words matching the first token
        2. For each match, verify subsequent words in that segment match
        3. Construct PhraseCandidate from matched word sequence
        """
        if not token_window:
            return []

        first_token = token_window[0]
        candidates: list[PhraseCandidate] = []

        # Find all occurrences of the first token
        word_contexts = find_words_with_segment_context(self._db, first_token)

        for word, segment_id in word_contexts:
            if word.id is None:
                continue

            # If it's just one token, we don't need phrase matching
            if len(token_window) == 1:
                continue

            # Verify consecutive words match
            matched_words = find_consecutive_words_in_segment(
                self._db, segment_id, word.id, token_window
            )

            if matched_words is None:
                continue

            # Build PhraseCandidate
            video = self._db.get_video(word.video_id)
            if video is None:
                continue

            # Get segment
            conn = self._db.connection
            seg_row = conn.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            ).fetchone()
            if seg_row is None:
                continue

            segment = Segment(
                id=seg_row["id"],
                video_id=seg_row["video_id"],
                start_time=seg_row["start_time"],
                end_time=seg_row["end_time"],
                text=seg_row["text"],
                speaker=seg_row["speaker"],
                confidence=seg_row["confidence"],
            )

            phrase_candidate = PhraseCandidate(
                words=matched_words,
                segment=segment,
                video=video,
                start_time=matched_words[0].start_time,
                end_time=matched_words[-1].end_time,
                duration=matched_words[-1].end_time - matched_words[0].start_time,
            )
            candidates.append(phrase_candidate)

        return candidates
