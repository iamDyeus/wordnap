"""Candidate search engine for finding word clips in the database."""

from wordnap.database.database import Database
from wordnap.database.queries import find_words_batch, find_words_by_normalized
from wordnap.models.schemas import WordCandidate


class WordNotFoundError(Exception):
    """Raised when one or more words cannot be found in the library."""

    def __init__(self, missing_words: list[str]):
        self.missing_words = missing_words
        super().__init__(f"Words not found in library: {', '.join(missing_words)}")


class SearchEngine:
    """Finds candidate word clips from the database.

    A thin wrapper around the database query helpers, providing a clean interface
    for the generation pipeline with error reporting for missing words.
    """

    def __init__(self, db: Database):
        self._db = db

    def find_candidates(self, normalized_word: str) -> list[WordCandidate]:
        """Find all candidate clips for a given normalized word.

        Args:
            normalized_word: The normalized word to search for.

        Returns:
            List of WordCandidate objects matching the normalized word.
        """
        return find_words_by_normalized(self._db, normalized_word)

    def find_candidates_batch(
        self, tokens: list[str], strict: bool = True
    ) -> tuple[dict[str, list[WordCandidate]], list[str]]:
        """Find candidates for multiple tokens.

        Args:
            tokens: Ordered list of normalized tokens.
            strict: If True, raises WordNotFoundError on any missing token.
                    If False, returns partial results with missing word list.

        Returns:
            Tuple of:
            - Dict mapping each found token to its candidates
            - List of tokens that had no candidates (empty when strict=True and all found)

        Raises:
            WordNotFoundError: If strict=True and any token has no candidates.
        """
        results = find_words_batch(self._db, tokens)

        missing = [token for token in tokens if not results.get(token)]

        if strict and missing:
            raise WordNotFoundError(missing)

        # In best-effort mode, remove missing tokens from results
        available = {k: v for k, v in results.items() if v}
        return available, missing
