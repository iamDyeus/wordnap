"""Unit tests for EDLGenerator in editing/edl.py."""

from pathlib import Path

import pytest

from wordnap.editing.edl import EDLGenerator
from wordnap.models.schemas import (
    GapEntry,
    PhraseCandidate,
    Segment,
    TokenInfo,
    VideoMetadata,
    VideoStatus,
    Word,
    WordCandidate,
)


def _make_video(
    path: str = "/videos/test.mp4", duration: float = 60.0
) -> VideoMetadata:
    """Create a test VideoMetadata instance."""
    return VideoMetadata(
        id=1,
        path=Path(path),
        filename=path.split("/")[-1],
        duration=duration,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=44100,
        status=VideoStatus.INDEXED,
    )


def _make_word(
    word: str = "hello",
    normalized: str = "hello",
    start: float = 1.0,
    end: float = 1.5,
    confidence: float = 0.9,
    speaker: str | None = None,
    video_id: int = 1,
    segment_id: int = 1,
) -> Word:
    """Create a test Word instance."""
    return Word(
        id=1,
        segment_id=segment_id,
        video_id=video_id,
        word=word,
        normalized_word=normalized,
        start_time=start,
        end_time=end,
        confidence=confidence,
        speaker=speaker,
    )


def _make_candidate(
    word: str = "hello",
    start: float = 1.0,
    end: float = 1.5,
    confidence: float = 0.9,
    video_duration: float = 60.0,
    video_path: str = "/videos/test.mp4",
    speaker: str | None = None,
) -> WordCandidate:
    """Create a test WordCandidate instance."""
    video = _make_video(path=video_path, duration=video_duration)
    w = _make_word(
        word=word, normalized=word.lower(), start=start, end=end,
        confidence=confidence, speaker=speaker, video_id=video.id or 1,
    )
    return WordCandidate(word=w, video=video, duration=end - start)


class TestEDLGeneratorInit:
    def test_default_padding(self):
        gen = EDLGenerator()
        assert gen.clip_padding == 0.10

    def test_custom_padding(self):
        gen = EDLGenerator(clip_padding=0.25)
        assert gen.clip_padding == 0.25

    def test_zero_padding(self):
        gen = EDLGenerator(clip_padding=0.0)
        assert gen.clip_padding == 0.0

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            EDLGenerator(clip_padding=-0.1)


class TestEDLGeneratorGenerate:
    def test_empty_selections_raises(self):
        gen = EDLGenerator()
        with pytest.raises(ValueError, match="non-empty"):
            gen.generate([], sentence="hello", variation_index=0)

    def test_single_selection(self):
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="hello", start=1.0, end=1.5)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        assert manifest.sentence == "hello"
        assert manifest.variation_index == 0
        assert len(manifest.clips) == 1

        clip = manifest.clips[0]
        assert clip.word == "hello"
        assert clip.start_time == 1.0
        assert clip.end_time == 1.5
        assert clip.padded_start == 0.9
        assert clip.padded_end == 1.6
        assert clip.confidence == 0.9
        assert clip.source_video == Path("/videos/test.mp4")
        assert clip.source_filename == "test.mp4"

    def test_multiple_selections_preserves_order(self):
        gen = EDLGenerator(clip_padding=0.10)
        candidates = [
            _make_candidate(word="the", start=2.0, end=2.3),
            _make_candidate(word="quick", start=5.0, end=5.5),
            _make_candidate(word="fox", start=10.0, end=10.4),
        ]
        manifest = gen.generate(candidates, sentence="the quick fox", variation_index=1)

        assert len(manifest.clips) == 3
        assert manifest.clips[0].word == "the"
        assert manifest.clips[1].word == "quick"
        assert manifest.clips[2].word == "fox"
        assert manifest.variation_index == 1

    def test_total_duration_computed_correctly(self):
        gen = EDLGenerator(clip_padding=0.10)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5),
            _make_candidate(word="world", start=3.0, end=3.4),
        ]
        manifest = gen.generate(candidates, sentence="hello world", variation_index=0)

        # hello: padded 0.9 to 1.6 = 0.7
        # world: padded 2.9 to 3.5 = 0.6
        # default gap: 80ms = 0.08s
        expected = 0.7 + 0.6 + 0.08
        assert abs(manifest.total_duration - expected) < 1e-9

    def test_source_attribution(self):
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(
            word="test", start=1.0, end=1.5, video_path="/library/my_video.mp4"
        )
        manifest = gen.generate([candidate], sentence="test", variation_index=0)

        clip = manifest.clips[0]
        assert clip.source_video == Path("/library/my_video.mp4")
        assert clip.source_filename == "my_video.mp4"

    def test_speaker_included(self):
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(word="hello", start=1.0, end=1.5, speaker="SPEAKER_01")
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        assert manifest.clips[0].speaker == "SPEAKER_01"

    def test_speaker_none_when_not_set(self):
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(word="hello", start=1.0, end=1.5, speaker=None)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        assert manifest.clips[0].speaker is None


class TestEDLGeneratorPaddingClamping:
    def test_padding_clamped_at_video_start(self):
        """When word is near the start of video, padded_start clamped to 0."""
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="hello", start=0.05, end=0.5, video_duration=60.0)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 0.0  # max(0, 0.05 - 0.10) = 0.0
        assert clip.padded_end == 0.6    # min(60.0, 0.5 + 0.10)

    def test_padding_clamped_at_video_end(self):
        """When word is near the end of video, padded_end clamped to duration."""
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="bye", start=59.5, end=59.95, video_duration=60.0)
        manifest = gen.generate([candidate], sentence="bye", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 59.4   # max(0, 59.5 - 0.10)
        assert clip.padded_end == 60.0     # min(60.0, 59.95 + 0.10) = 60.0

    def test_padding_clamped_both_sides(self):
        """Very short video where padding exceeds both boundaries."""
        gen = EDLGenerator(clip_padding=0.5)
        candidate = _make_candidate(word="hi", start=0.1, end=0.9, video_duration=1.0)
        manifest = gen.generate([candidate], sentence="hi", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 0.0  # max(0, 0.1 - 0.5)
        assert clip.padded_end == 1.0    # min(1.0, 0.9 + 0.5)

    def test_zero_padding_no_change(self):
        """With zero padding, padded times equal original times."""
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(word="hello", start=5.0, end=5.5, video_duration=60.0)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 5.0
        assert clip.padded_end == 5.5

    def test_large_padding_clamped(self):
        """Very large padding still respects boundaries."""
        gen = EDLGenerator(clip_padding=100.0)
        candidate = _make_candidate(word="hello", start=5.0, end=5.5, video_duration=10.0)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 0.0   # max(0, 5.0 - 100)
        assert clip.padded_end == 10.0    # min(10.0, 5.5 + 100)

    def test_padded_start_less_than_or_equal_to_start_time(self):
        """padded_start should always be <= start_time."""
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="hello", start=5.0, end=5.5)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start <= clip.start_time

    def test_padded_end_greater_than_or_equal_to_end_time(self):
        """padded_end should always be >= end_time."""
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="hello", start=5.0, end=5.5)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_end >= clip.end_time

    def test_word_at_exact_start(self):
        """Word starting at time 0."""
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="hello", start=0.0, end=0.5, video_duration=60.0)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 0.0  # max(0, 0.0 - 0.10)
        assert clip.padded_end == 0.6

    def test_word_at_exact_end(self):
        """Word ending exactly at video duration."""
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="end", start=59.5, end=60.0, video_duration=60.0)
        manifest = gen.generate([candidate], sentence="end", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 59.4
        assert clip.padded_end == 60.0  # min(60.0, 60.0 + 0.10)


class TestEDLGeneratorMultipleVideos:
    def test_clips_from_different_videos(self):
        """Selections can come from different source videos."""
        gen = EDLGenerator(clip_padding=0.05)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5, video_path="/videos/v1.mp4", video_duration=30.0),
            _make_candidate(word="world", start=5.0, end=5.5, video_path="/videos/v2.mp4", video_duration=120.0),
        ]
        manifest = gen.generate(candidates, sentence="hello world", variation_index=0)

        assert manifest.clips[0].source_video == Path("/videos/v1.mp4")
        assert manifest.clips[0].source_filename == "v1.mp4"
        assert manifest.clips[1].source_video == Path("/videos/v2.mp4")
        assert manifest.clips[1].source_filename == "v2.mp4"

    def test_padding_respects_each_videos_duration(self):
        """Each clip's padding is clamped to its own video's duration."""
        gen = EDLGenerator(clip_padding=1.0)
        candidates = [
            _make_candidate(word="a", start=0.1, end=0.5, video_path="/v1.mp4", video_duration=2.0),
            _make_candidate(word="b", start=0.1, end=0.5, video_path="/v2.mp4", video_duration=0.8),
        ]
        manifest = gen.generate(candidates, sentence="a b", variation_index=0)

        # v1: padded_start=max(0, 0.1-1.0)=0, padded_end=min(2.0, 0.5+1.0)=1.5
        assert manifest.clips[0].padded_start == 0.0
        assert manifest.clips[0].padded_end == 1.5

        # v2: padded_start=max(0, 0.1-1.0)=0, padded_end=min(0.8, 0.5+1.0)=0.8
        assert manifest.clips[1].padded_start == 0.0
        assert manifest.clips[1].padded_end == 0.8


class TestEDLGeneratorComputeGap:
    """Tests for EDLGenerator.compute_gap() with various punctuation types."""

    def test_default_gap_when_token_info_is_none(self):
        gen = EDLGenerator(default_gap_ms=80.0)
        gap = gen.compute_gap(None)
        assert gap.duration_ms == 80.0
        assert gap.reason == "default"

    def test_default_gap_when_no_trailing_punctuation(self):
        gen = EDLGenerator(default_gap_ms=80.0)
        token = TokenInfo(normalized="hello", original="hello", trailing_punctuation=None)
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 80.0
        assert gap.reason == "default"

    def test_comma_gap(self):
        gen = EDLGenerator(comma_gap_ms=200.0)
        token = TokenInfo(normalized="hello", original="hello,", trailing_punctuation=",")
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 200.0
        assert gap.reason == "comma"

    def test_period_gap(self):
        gen = EDLGenerator(sentence_end_gap_ms=400.0)
        token = TokenInfo(normalized="world", original="world.", trailing_punctuation=".")
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 400.0
        assert gap.reason == "sentence_end"

    def test_exclamation_gap(self):
        gen = EDLGenerator(sentence_end_gap_ms=400.0)
        token = TokenInfo(normalized="wow", original="wow!", trailing_punctuation="!")
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 400.0
        assert gap.reason == "sentence_end"

    def test_question_mark_gap(self):
        gen = EDLGenerator(sentence_end_gap_ms=400.0)
        token = TokenInfo(normalized="what", original="what?", trailing_punctuation="?")
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 400.0
        assert gap.reason == "sentence_end"

    def test_other_punctuation_uses_default(self):
        gen = EDLGenerator(default_gap_ms=80.0)
        token = TokenInfo(normalized="hello", original="hello;", trailing_punctuation=";")
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 80.0
        assert gap.reason == "default"

    def test_punctuation_pause_disabled_uses_default(self):
        gen = EDLGenerator(
            punctuation_pause_enabled=False,
            default_gap_ms=80.0,
            comma_gap_ms=200.0,
        )
        token = TokenInfo(normalized="hello", original="hello,", trailing_punctuation=",")
        gap = gen.compute_gap(token)
        assert gap.duration_ms == 80.0
        assert gap.reason == "default"

    def test_custom_gap_values(self):
        gen = EDLGenerator(
            default_gap_ms=50.0,
            comma_gap_ms=150.0,
            sentence_end_gap_ms=300.0,
        )
        token_comma = TokenInfo(normalized="a", original="a,", trailing_punctuation=",")
        token_period = TokenInfo(normalized="b", original="b.", trailing_punctuation=".")
        token_none = TokenInfo(normalized="c", original="c", trailing_punctuation=None)

        assert gen.compute_gap(token_comma).duration_ms == 150.0
        assert gen.compute_gap(token_period).duration_ms == 300.0
        assert gen.compute_gap(token_none).duration_ms == 50.0


class TestEDLGeneratorGapsInGenerate:
    """Tests for generate() producing correct gap entries from token_infos."""

    def test_gaps_generated_between_clips(self):
        gen = EDLGenerator(clip_padding=0.10, default_gap_ms=80.0)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5),
            _make_candidate(word="world", start=3.0, end=3.4),
        ]
        token_infos = [
            TokenInfo(normalized="hello", original="hello", trailing_punctuation=None),
            TokenInfo(normalized="world", original="world", trailing_punctuation=None),
        ]
        manifest = gen.generate(
            candidates, sentence="hello world", variation_index=0,
            token_infos=token_infos,
        )

        assert len(manifest.gaps) == 1
        assert manifest.gaps[0].duration_ms == 80.0
        assert manifest.gaps[0].reason == "default"

    def test_comma_gap_in_generate(self):
        gen = EDLGenerator(clip_padding=0.10, comma_gap_ms=200.0)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5),
            _make_candidate(word="world", start=3.0, end=3.4),
        ]
        token_infos = [
            TokenInfo(normalized="hello", original="hello,", trailing_punctuation=","),
            TokenInfo(normalized="world", original="world", trailing_punctuation=None),
        ]
        manifest = gen.generate(
            candidates, sentence="hello, world", variation_index=0,
            token_infos=token_infos,
        )

        assert len(manifest.gaps) == 1
        assert manifest.gaps[0].duration_ms == 200.0
        assert manifest.gaps[0].reason == "comma"

    def test_multiple_gaps_with_mixed_punctuation(self):
        gen = EDLGenerator(
            clip_padding=0.0,
            default_gap_ms=80.0,
            comma_gap_ms=200.0,
            sentence_end_gap_ms=400.0,
        )
        candidates = [
            _make_candidate(word="hi", start=1.0, end=1.3),
            _make_candidate(word="there", start=2.0, end=2.5),
            _make_candidate(word="friend", start=4.0, end=4.5),
        ]
        token_infos = [
            TokenInfo(normalized="hi", original="hi,", trailing_punctuation=","),
            TokenInfo(normalized="there", original="there.", trailing_punctuation="."),
            TokenInfo(normalized="friend", original="friend", trailing_punctuation=None),
        ]
        manifest = gen.generate(
            candidates, sentence="hi, there. friend", variation_index=0,
            token_infos=token_infos,
        )

        assert len(manifest.gaps) == 2
        assert manifest.gaps[0].duration_ms == 200.0
        assert manifest.gaps[0].reason == "comma"
        assert manifest.gaps[1].duration_ms == 400.0
        assert manifest.gaps[1].reason == "sentence_end"

    def test_no_token_infos_uses_default_gaps(self):
        gen = EDLGenerator(clip_padding=0.10, default_gap_ms=80.0)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5),
            _make_candidate(word="world", start=3.0, end=3.4),
        ]
        manifest = gen.generate(
            candidates, sentence="hello world", variation_index=0,
            token_infos=None,
        )

        assert len(manifest.gaps) == 1
        assert manifest.gaps[0].duration_ms == 80.0
        assert manifest.gaps[0].reason == "default"

    def test_total_duration_includes_gaps(self):
        gen = EDLGenerator(clip_padding=0.0, default_gap_ms=100.0)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5),
            _make_candidate(word="world", start=3.0, end=3.4),
        ]
        manifest = gen.generate(
            candidates, sentence="hello world", variation_index=0,
        )

        # hello: 0.5s, world: 0.4s, gap: 100ms = 0.1s
        expected = 0.5 + 0.4 + 0.1
        assert abs(manifest.total_duration - expected) < 1e-9

    def test_single_selection_no_gaps(self):
        gen = EDLGenerator(clip_padding=0.10)
        candidate = _make_candidate(word="hello", start=1.0, end=1.5)
        manifest = gen.generate(
            [candidate], sentence="hello", variation_index=0,
            token_infos=[TokenInfo(normalized="hello", original="hello.", trailing_punctuation=".")],
        )

        assert len(manifest.gaps) == 0


class TestEDLGeneratorPhraseCandidate:
    """Tests for generate() with PhraseCandidate producing correct clip boundaries."""

    def _make_phrase_candidate(
        self,
        words_data: list[tuple[str, float, float, float]],
        video_path: str = "/videos/test.mp4",
        video_duration: float = 60.0,
    ) -> PhraseCandidate:
        """Create a PhraseCandidate from word data tuples (word, start, end, confidence)."""
        video = _make_video(path=video_path, duration=video_duration)
        segment = Segment(
            id=1,
            video_id=video.id or 1,
            start_time=words_data[0][1],
            end_time=words_data[-1][2],
            text=" ".join(w[0] for w in words_data),
            speaker="SPEAKER_01",
            confidence=0.9,
        )
        words = [
            Word(
                id=i + 1,
                segment_id=1,
                video_id=video.id or 1,
                word=wd[0],
                normalized_word=wd[0].lower(),
                start_time=wd[1],
                end_time=wd[2],
                confidence=wd[3],
                speaker="SPEAKER_01",
            )
            for i, wd in enumerate(words_data)
        ]
        return PhraseCandidate(
            words=words,
            segment=segment,
            video=video,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            duration=words[-1].end_time - words[0].start_time,
        )

    def test_phrase_candidate_clip_boundaries(self):
        """PhraseCandidate clip uses first word start and last word end."""
        gen = EDLGenerator(clip_padding=0.0)
        phrase = self._make_phrase_candidate([
            ("hello", 1.0, 1.5, 0.9),
            ("world", 1.6, 2.0, 0.85),
        ])
        manifest = gen.generate([phrase], sentence="hello world", variation_index=0)

        clip = manifest.clips[0]
        assert clip.start_time == 1.0  # first word start
        assert clip.end_time == 2.0    # last word end
        assert clip.word == "hello world"

    def test_phrase_candidate_with_padding(self):
        """PhraseCandidate respects clip padding."""
        gen = EDLGenerator(clip_padding=0.10)
        phrase = self._make_phrase_candidate([
            ("the", 2.0, 2.3, 0.95),
            ("quick", 2.4, 2.8, 0.88),
            ("fox", 2.9, 3.2, 0.92),
        ])
        manifest = gen.generate([phrase], sentence="the quick fox", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == pytest.approx(1.9)   # max(0, 2.0 - 0.10)
        assert clip.padded_end == pytest.approx(3.3)     # min(60.0, 3.2 + 0.10)

    def test_phrase_candidate_confidence_averaged(self):
        """PhraseCandidate confidence is average of word confidences."""
        gen = EDLGenerator(clip_padding=0.0)
        phrase = self._make_phrase_candidate([
            ("hello", 1.0, 1.5, 0.9),
            ("world", 1.6, 2.0, 0.8),
        ])
        manifest = gen.generate([phrase], sentence="hello world", variation_index=0)

        clip = manifest.clips[0]
        assert abs(clip.confidence - 0.85) < 1e-9  # (0.9 + 0.8) / 2

    def test_phrase_candidate_speaker_from_first_word(self):
        """PhraseCandidate speaker comes from first word."""
        gen = EDLGenerator(clip_padding=0.0)
        phrase = self._make_phrase_candidate([
            ("hello", 1.0, 1.5, 0.9),
            ("world", 1.6, 2.0, 0.85),
        ])
        manifest = gen.generate([phrase], sentence="hello world", variation_index=0)

        assert manifest.clips[0].speaker == "SPEAKER_01"

    def test_mixed_word_and_phrase_candidates(self):
        """Generate works with a mix of WordCandidate and PhraseCandidate."""
        gen = EDLGenerator(clip_padding=0.0, default_gap_ms=80.0)
        word_candidate = _make_candidate(word="we", start=1.0, end=1.3)
        phrase = self._make_phrase_candidate([
            ("need", 3.0, 3.4, 0.9),
            ("to", 3.5, 3.7, 0.88),
        ])
        manifest = gen.generate(
            [word_candidate, phrase],
            sentence="we need to",
            variation_index=0,
        )

        assert len(manifest.clips) == 2
        assert manifest.clips[0].word == "we"
        assert manifest.clips[0].start_time == 1.0
        assert manifest.clips[1].word == "need to"
        assert manifest.clips[1].start_time == 3.0
        assert manifest.clips[1].end_time == 3.7
        assert len(manifest.gaps) == 1

    def test_phrase_candidate_padding_clamped_at_video_start(self):
        """PhraseCandidate padding clamped at video start."""
        gen = EDLGenerator(clip_padding=0.5)
        phrase = self._make_phrase_candidate(
            words_data=[("hi", 0.1, 0.4, 0.9), ("there", 0.5, 0.8, 0.85)],
            video_duration=60.0,
        )
        manifest = gen.generate([phrase], sentence="hi there", variation_index=0)

        clip = manifest.clips[0]
        assert clip.padded_start == 0.0  # max(0, 0.1 - 0.5)


class TestEDLGeneratorSkippedWords:
    """Tests for generate() with skipped_words."""

    def test_skipped_words_empty_by_default(self):
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(word="hello", start=1.0, end=1.5)
        manifest = gen.generate([candidate], sentence="hello", variation_index=0)

        assert manifest.skipped_words == []

    def test_skipped_words_populated(self):
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(word="hello", start=1.0, end=1.5)
        manifest = gen.generate(
            [candidate],
            sentence="hello beautiful world",
            variation_index=0,
            skipped_words=["beautiful", "world"],
        )

        assert manifest.skipped_words == ["beautiful", "world"]

    def test_skipped_words_none_treated_as_empty(self):
        gen = EDLGenerator(clip_padding=0.0)
        candidate = _make_candidate(word="test", start=1.0, end=1.5)
        manifest = gen.generate(
            [candidate],
            sentence="test word",
            variation_index=0,
            skipped_words=None,
        )

        assert manifest.skipped_words == []

    def test_skipped_words_does_not_affect_clips(self):
        """Skipped words are just metadata; clips remain based on selections."""
        gen = EDLGenerator(clip_padding=0.0)
        candidates = [
            _make_candidate(word="hello", start=1.0, end=1.5),
            _make_candidate(word="world", start=3.0, end=3.4),
        ]
        manifest = gen.generate(
            candidates,
            sentence="hello beautiful world",
            variation_index=0,
            skipped_words=["beautiful"],
        )

        assert len(manifest.clips) == 2
        assert manifest.clips[0].word == "hello"
        assert manifest.clips[1].word == "world"
        assert manifest.skipped_words == ["beautiful"]


class TestEDLGeneratorNewInit:
    """Tests for new constructor parameters."""

    def test_default_new_params(self):
        gen = EDLGenerator()
        assert gen.default_gap_ms == 80.0
        assert gen.punctuation_pause_enabled is True
        assert gen.comma_gap_ms == 200.0
        assert gen.sentence_end_gap_ms == 400.0

    def test_custom_new_params(self):
        gen = EDLGenerator(
            default_gap_ms=50.0,
            punctuation_pause_enabled=False,
            comma_gap_ms=150.0,
            sentence_end_gap_ms=350.0,
        )
        assert gen.default_gap_ms == 50.0
        assert gen.punctuation_pause_enabled is False
        assert gen.comma_gap_ms == 150.0
        assert gen.sentence_end_gap_ms == 350.0

    def test_backward_compat_clip_padding_only(self):
        """Existing callers that pass only clip_padding still work."""
        gen = EDLGenerator(clip_padding=0.25)
        assert gen.clip_padding == 0.25
        assert gen.default_gap_ms == 80.0
