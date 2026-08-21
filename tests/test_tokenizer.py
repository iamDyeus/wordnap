"""Unit tests for the Tokenizer class."""

import unicodedata

import pytest

from sentence_mixer.search.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


# --- normalize_word tests ---


class TestNormalizeWord:
    def test_lowercase_transformation(self, tokenizer):
        assert tokenizer.normalize_word("Hello") == "hello"
        assert tokenizer.normalize_word("WORLD") == "world"
        assert tokenizer.normalize_word("MiXeD") == "mixed"

    def test_strip_leading_punctuation(self, tokenizer):
        assert tokenizer.normalize_word('"hello') == "hello"
        assert tokenizer.normalize_word("(world") == "world"

    def test_strip_trailing_punctuation(self, tokenizer):
        assert tokenizer.normalize_word("hello!") == "hello"
        assert tokenizer.normalize_word("world,") == "world"
        assert tokenizer.normalize_word("end.") == "end"

    def test_strip_surrounding_punctuation(self, tokenizer):
        assert tokenizer.normalize_word('"hello"') == "hello"
        assert tokenizer.normalize_word("(world)") == "world"
        assert tokenizer.normalize_word("'quoted'") == "quoted"

    def test_preserve_intra_word_hyphens(self, tokenizer):
        assert tokenizer.normalize_word("well-known") == "well-known"
        assert tokenizer.normalize_word("self-aware") == "self-aware"
        assert tokenizer.normalize_word("state-of-the-art") == "state-of-the-art"

    def test_strip_leading_hyphen(self, tokenizer):
        # Hyphen at the start is not intra-word
        assert tokenizer.normalize_word("-hello") == "hello"

    def test_strip_trailing_hyphen(self, tokenizer):
        # Hyphen at the end is not intra-word
        assert tokenizer.normalize_word("hello-") == "hello"

    def test_unicode_nfkc_normalization(self, tokenizer):
        # NFKC converts compatibility characters to their canonical forms
        # e.g., fullwidth 'Ａ' (U+FF21) becomes 'A' then lowercased to 'a'
        fullwidth_a = "\uff21"  # Ａ
        assert tokenizer.normalize_word(fullwidth_a) == "a"

    def test_unicode_nfkc_ligature(self, tokenizer):
        # 'ﬁ' (U+FB01) normalizes to 'fi' under NFKC
        assert tokenizer.normalize_word("\ufb01sh") == "fish"

    def test_unicode_nfkc_superscript(self, tokenizer):
        # '²' (U+00B2) normalizes to '2' under NFKC
        assert tokenizer.normalize_word("x\u00b2") == "x2"

    def test_pure_punctuation_returns_empty(self, tokenizer):
        assert tokenizer.normalize_word("!!!") == ""
        assert tokenizer.normalize_word("...") == ""
        assert tokenizer.normalize_word("@#$%") == ""

    def test_single_character(self, tokenizer):
        assert tokenizer.normalize_word("a") == "a"
        assert tokenizer.normalize_word("A") == "a"
        assert tokenizer.normalize_word("5") == "5"

    def test_idempotence(self, tokenizer):
        words = ["Hello", "well-known", "WORLD!", '"quoted"', "café"]
        for word in words:
            once = tokenizer.normalize_word(word)
            twice = tokenizer.normalize_word(once)
            assert once == twice, f"Not idempotent for '{word}': '{once}' != '{twice}'"

    def test_numeric_content_preserved(self, tokenizer):
        assert tokenizer.normalize_word("123") == "123"
        assert tokenizer.normalize_word("abc123") == "abc123"

    def test_accented_characters_preserved(self, tokenizer):
        # Accented characters that are already in NFC/NFKC remain
        assert tokenizer.normalize_word("café") == "café"
        assert tokenizer.normalize_word("naïve") == "naïve"

    def test_mixed_punctuation_and_hyphens(self, tokenizer):
        # Punctuation stripped, intra-word hyphen preserved
        assert tokenizer.normalize_word('"well-known"') == "well-known"
        assert tokenizer.normalize_word("(self-aware)") == "self-aware"


# --- tokenize tests ---


class TestTokenize:
    def test_simple_sentence(self, tokenizer):
        result = tokenizer.tokenize("hello world")
        assert result == ["hello", "world"]

    def test_preserves_order(self, tokenizer):
        result = tokenizer.tokenize("the quick brown fox")
        assert result == ["the", "quick", "brown", "fox"]

    def test_normalizes_tokens(self, tokenizer):
        result = tokenizer.tokenize("Hello World!")
        assert result == ["hello", "world"]

    def test_empty_string_returns_empty_list(self, tokenizer):
        assert tokenizer.tokenize("") == []

    def test_whitespace_only_returns_empty_list(self, tokenizer):
        assert tokenizer.tokenize("   ") == []
        assert tokenizer.tokenize("\t\n") == []

    def test_multiple_spaces_between_words(self, tokenizer):
        result = tokenizer.tokenize("hello    world")
        assert result == ["hello", "world"]

    def test_leading_and_trailing_whitespace(self, tokenizer):
        result = tokenizer.tokenize("  hello world  ")
        assert result == ["hello", "world"]

    def test_tabs_and_newlines_as_separators(self, tokenizer):
        result = tokenizer.tokenize("hello\tworld\nfoo")
        assert result == ["hello", "world", "foo"]

    def test_filters_out_empty_normalized_tokens(self, tokenizer):
        # A token that is pure punctuation normalizes to empty and is filtered
        result = tokenizer.tokenize("hello ... world")
        assert result == ["hello", "world"]

    def test_hyphenated_words_preserved(self, tokenizer):
        result = tokenizer.tokenize("this is well-known")
        assert result == ["this", "is", "well-known"]

    def test_sentence_with_punctuation(self, tokenizer):
        result = tokenizer.tokenize("Hello, World! How are you?")
        assert result == ["hello", "world", "how", "are", "you"]

    def test_unicode_sentence(self, tokenizer):
        result = tokenizer.tokenize("Café Naïve")
        assert result == ["café", "naïve"]

    def test_single_word(self, tokenizer):
        result = tokenizer.tokenize("Hello")
        assert result == ["hello"]

    def test_all_punctuation_tokens_filtered(self, tokenizer):
        result = tokenizer.tokenize("!!! ... ---")
        assert result == []


# --- tokenize_with_context tests ---


class TestTokenizeWithContext:
    def test_trailing_comma_and_period(self, tokenizer):
        """Test 'Hello, world.' produces correct trailing punctuation."""
        result = tokenizer.tokenize_with_context("Hello, world.")
        assert len(result) == 2
        assert result[0].normalized == "hello"
        assert result[0].original == "Hello,"
        assert result[0].trailing_punctuation == ","
        assert result[1].normalized == "world"
        assert result[1].original == "world."
        assert result[1].trailing_punctuation == "."

    def test_no_trailing_punctuation(self, tokenizer):
        """Test 'hello world' produces None trailing_punctuation."""
        result = tokenizer.tokenize_with_context("hello world")
        assert len(result) == 2
        assert result[0].normalized == "hello"
        assert result[0].trailing_punctuation is None
        assert result[1].normalized == "world"
        assert result[1].trailing_punctuation is None

    def test_exclamation_and_question(self, tokenizer):
        """Test 'Wait! Really?' produces '!' and '?'."""
        result = tokenizer.tokenize_with_context("Wait! Really?")
        assert len(result) == 2
        assert result[0].normalized == "wait"
        assert result[0].trailing_punctuation == "!"
        assert result[1].normalized == "really"
        assert result[1].trailing_punctuation == "?"

    def test_empty_input_returns_empty(self, tokenizer):
        """Test empty input returns []."""
        assert tokenizer.tokenize_with_context("") == []
        assert tokenizer.tokenize_with_context("   ") == []

    def test_normalize_consistency_with_tokenize(self, tokenizer):
        """tokenize_with_context produces same normalized values as tokenize()."""
        sentences = [
            "Hello, World! How are you?",
            "well-known self-aware state-of-the-art",
            "Café Naïve",
            "this is a test...",
            '"quoted" words (here)',
        ]
        for sentence in sentences:
            context_tokens = tokenizer.tokenize_with_context(sentence)
            plain_tokens = tokenizer.tokenize(sentence)
            context_normalized = [t.normalized for t in context_tokens]
            assert context_normalized == plain_tokens, (
                f"Mismatch for '{sentence}': {context_normalized} != {plain_tokens}"
            )
