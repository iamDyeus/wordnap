"""Word normalization and sentence tokenization."""

import unicodedata

from sentence_mixer.models.schemas import TokenInfo


class Tokenizer:
    """Normalizes user input sentences into searchable tokens."""

    def normalize_word(self, word: str) -> str:
        """Normalize a word for indexing and lookup.

        Applies:
        - Unicode NFKC normalization
        - Lowercase transformation
        - Punctuation stripping (preserving intra-word hyphens)

        Args:
            word: A non-empty string to normalize.

        Returns:
            The normalized word. May be empty if input contains no
            alphanumeric characters.
        """
        # Apply Unicode NFKC normalization
        normalized = unicodedata.normalize("NFKC", word)

        # Lowercase
        normalized = normalized.lower()

        # Strip punctuation while preserving intra-word hyphens.
        # A hyphen is "intra-word" if it has alphanumeric characters on both sides.
        result = []
        for i, char in enumerate(normalized):
            if char.isalnum():
                result.append(char)
            elif char == "-":
                # Preserve hyphen only if between alphanumeric characters
                has_alnum_before = i > 0 and normalized[i - 1].isalnum()
                has_alnum_after = (
                    i < len(normalized) - 1 and normalized[i + 1].isalnum()
                )
                if has_alnum_before and has_alnum_after:
                    result.append(char)
            # All other punctuation/characters are stripped

        return "".join(result)

    def tokenize(self, sentence: str) -> list[str]:
        """Split sentence into normalized word tokens.

        Splits on whitespace, normalizes each token, and filters out
        tokens that normalize to an empty string.

        Args:
            sentence: The input sentence to tokenize.

        Returns:
            An ordered list of normalized tokens. Empty list if the input
            is empty or whitespace-only.
        """
        if not sentence or not sentence.strip():
            return []

        tokens = []
        for word in sentence.split():
            normalized = self.normalize_word(word)
            if normalized:  # Filter out tokens that normalize to empty
                tokens.append(normalized)

        return tokens

    def tokenize_with_context(self, sentence: str) -> list[TokenInfo]:
        """Split sentence into tokens preserving trailing punctuation context.

        Splits on whitespace, normalizes each token, and records trailing
        punctuation from the original word for silence gap calculation.

        Args:
            sentence: The input sentence.

        Returns:
            Ordered list of TokenInfo with normalized form and punctuation.
        """
        if not sentence or not sentence.strip():
            return []

        tokens = []
        for word in sentence.split():
            normalized = self.normalize_word(word)
            if not normalized:
                continue

            # Detect trailing punctuation
            trailing = None
            if word and not word[-1].isalnum():
                trailing = word[-1]

            tokens.append(TokenInfo(
                normalized=normalized,
                original=word,
                trailing_punctuation=trailing,
            ))

        return tokens
