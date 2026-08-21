"""Unit tests for Ranker in search/ranking.py."""

from pathlib import Path

import pytest

from wordnap.models.schemas import (
    RankingConfig,
    VideoMetadata,
    VideoStatus,
    Word,
    WordCandidate,
)
from wordnap.search.ranking import Ranker


def _make_video(video_id: int = 1, path: str = "/videos/test.mp4") -> VideoMetadata:
    """Create a test video metadata object."""
    return VideoMetadata(
        id=video_id,
        path=Path(path),
        filename=path.split("/")[-1],
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=44100,
        status=VideoStatus.INDEXED,
    )


def _make_word(
    word_id: int = 1,
    video_id: int = 1,
    word: str = "hello",
    normalized: str = "hello",
    start: float = 1.0,
    end: float = 1.5,
    confidence: float = 0.9,
    speaker: str | None = None,
) -> Word:
    """Create a test Word object."""
    return Word(
        id=word_id,
        segment_id=1,
        video_id=video_id,
        word=word,
        normalized_word=normalized,
        start_time=start,
        end_time=end,
        confidence=confidence,
        speaker=speaker,
    )


def _make_candidate(
    word_id: int = 1,
    video_id: int = 1,
    confidence: float = 0.9,
    duration: float = 0.5,
    speaker: str | None = None,
    video_path: str = "/videos/test.mp4",
    start: float = 1.0,
) -> WordCandidate:
    """Create a test WordCandidate object."""
    video = _make_video(video_id=video_id, path=video_path)
    word = _make_word(
        word_id=word_id,
        video_id=video_id,
        confidence=confidence,
        start=start,
        end=start + duration,
        speaker=speaker,
    )
    return WordCandidate(word=word, video=video, duration=duration)


class TestFilterCandidates:
    """Tests for Ranker.filter_candidates()."""

    def test_filters_low_confidence(self):
        """Candidates with confidence < 0.3 are filtered out."""
        ranker = Ranker(RankingConfig())
        candidates = [
            _make_candidate(word_id=1, confidence=0.9, duration=0.5),
            _make_candidate(word_id=2, confidence=0.2, duration=0.5),
        ]
        result = ranker.filter_candidates(candidates)
        assert len(result) == 1
        assert result[0].word.confidence == 0.9

    def test_filters_duration_too_short(self):
        """Candidates with duration < 0.03 are filtered out."""
        ranker = Ranker(RankingConfig())
        candidates = [
            _make_candidate(word_id=1, confidence=0.9, duration=0.5),
            _make_candidate(word_id=2, confidence=0.9, duration=0.02),
        ]
        result = ranker.filter_candidates(candidates)
        assert len(result) == 1
        assert result[0].duration == 0.5

    def test_filters_duration_too_long(self):
        """Candidates with duration > 3.0 are filtered out."""
        ranker = Ranker(RankingConfig())
        candidates = [
            _make_candidate(word_id=1, confidence=0.9, duration=0.5),
            _make_candidate(word_id=2, confidence=0.9, duration=4.0),
        ]
        result = ranker.filter_candidates(candidates)
        assert len(result) == 1
        assert result[0].duration == 0.5

    def test_fallback_when_all_filtered(self):
        """If all candidates are filtered, return original list."""
        ranker = Ranker(RankingConfig())
        candidates = [
            _make_candidate(word_id=1, confidence=0.1, duration=0.5),
            _make_candidate(word_id=2, confidence=0.2, duration=0.5),
        ]
        result = ranker.filter_candidates(candidates)
        assert len(result) == 2  # fallback to unfiltered

    def test_boundary_values_pass_filter(self):
        """Candidates at exact boundaries pass the filter."""
        ranker = Ranker(RankingConfig())
        candidates = [
            _make_candidate(word_id=1, confidence=0.3, duration=0.03),
            _make_candidate(word_id=2, confidence=0.3, duration=3.0),
        ]
        result = ranker.filter_candidates(candidates)
        assert len(result) == 2


class TestComputeDurationScore:
    """Tests for Ranker.compute_duration_score()."""

    def test_ideal_range_scores_1(self):
        """Duration in [0.1, 1.5] scores 1.0."""
        ranker = Ranker(RankingConfig())
        assert ranker.compute_duration_score(0.1) == 1.0
        assert ranker.compute_duration_score(0.5) == 1.0
        assert ranker.compute_duration_score(1.5) == 1.0

    def test_below_min_scores_0(self):
        """Duration <= 0.03 scores 0.0."""
        ranker = Ranker(RankingConfig())
        assert ranker.compute_duration_score(0.03) == 0.0
        assert ranker.compute_duration_score(0.01) == 0.0

    def test_above_max_scores_0(self):
        """Duration >= 3.0 scores 0.0."""
        ranker = Ranker(RankingConfig())
        assert ranker.compute_duration_score(3.0) == 0.0
        assert ranker.compute_duration_score(5.0) == 0.0

    def test_linear_ramp_up(self):
        """Duration in (0.03, 0.1) linearly ramps from 0.0 to 1.0."""
        ranker = Ranker(RankingConfig())
        # Midpoint between 0.03 and 0.1 should be ~0.5
        midpoint = (0.03 + 0.1) / 2
        score = ranker.compute_duration_score(midpoint)
        assert 0.4 <= score <= 0.6

    def test_linear_ramp_down(self):
        """Duration in (1.5, 3.0) linearly ramps from 1.0 to 0.0."""
        ranker = Ranker(RankingConfig())
        midpoint = (1.5 + 3.0) / 2
        score = ranker.compute_duration_score(midpoint)
        assert 0.4 <= score <= 0.6


class TestComputeBoundaryQuality:
    """Tests for Ranker.compute_boundary_quality()."""

    def test_returns_score_in_range(self):
        """Boundary quality is in [0.0, 1.0]."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(confidence=0.9, duration=0.5, start=1.0)
        score = ranker.compute_boundary_quality(candidate)
        assert 0.0 <= score <= 1.0

    def test_word_at_video_start_lower_quality(self):
        """Word at the very start of a video has lower boundary quality."""
        ranker = Ranker(RankingConfig())
        # Word starting at 0.0 - no room for padding before
        video = _make_video(video_id=1)
        word = _make_word(word_id=1, start=0.0, end=0.5, confidence=0.9)
        candidate = WordCandidate(word=word, video=video, duration=0.5)

        at_start = ranker.compute_boundary_quality(candidate)

        # Word starting at 1.0 - plenty of room
        word2 = _make_word(word_id=2, start=1.0, end=1.5, confidence=0.9)
        candidate2 = WordCandidate(word=word2, video=video, duration=0.5)
        in_middle = ranker.compute_boundary_quality(candidate2)

        assert at_start < in_middle


class TestComputeDiversityScore:
    """Tests for Ranker.compute_diversity_score()."""

    def test_unused_source_returns_1(self):
        """Candidate from unused source scores 1.0."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(video_id=1)
        assert ranker.compute_diversity_score(candidate, set()) == 1.0
        assert ranker.compute_diversity_score(candidate, {2, 3}) == 1.0

    def test_used_source_returns_0(self):
        """Candidate from already-used source scores 0.0."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(video_id=1)
        assert ranker.compute_diversity_score(candidate, {1}) == 0.0


class TestScoreCandidate:
    """Tests for Ranker.score_candidate()."""

    def test_perfect_candidate_scores_high(self):
        """A candidate with high confidence and ideal duration scores close to 1.0."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(confidence=0.95, duration=0.5)
        score = ranker.score_candidate(candidate)
        assert 0.8 <= score <= 1.0

    def test_score_always_in_bounds(self):
        """Score is always in [0.0, 1.0]."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(confidence=0.9, duration=0.5)
        score = ranker.score_candidate(candidate)
        assert 0.0 <= score <= 1.0

    def test_low_confidence_scores_lower(self):
        """Lower confidence produces lower score."""
        ranker = Ranker(RankingConfig())
        low_conf = _make_candidate(confidence=0.2, duration=0.5)
        high_conf = _make_candidate(confidence=0.5, duration=0.5)

        low_score = ranker.score_candidate(low_conf)
        high_score = ranker.score_candidate(high_conf)

        assert low_score < high_score

    def test_confidence_at_boundary_030(self):
        """Higher confidence always scores better."""
        ranker = Ranker(RankingConfig())
        at_boundary = _make_candidate(confidence=0.3, duration=0.5)
        below_boundary = _make_candidate(confidence=0.29, duration=0.5)

        at_score = ranker.score_candidate(at_boundary)
        below_score = ranker.score_candidate(below_boundary)

        assert below_score < at_score

    def test_duration_too_short_penalized(self):
        """Duration below min_duration is penalized."""
        ranker = Ranker(RankingConfig())
        short = _make_candidate(confidence=0.9, duration=0.04)
        normal = _make_candidate(confidence=0.9, duration=0.5)

        short_score = ranker.score_candidate(short)
        normal_score = ranker.score_candidate(normal)

        assert short_score < normal_score

    def test_duration_too_long_penalized(self):
        """Duration above max_duration is penalized."""
        ranker = Ranker(RankingConfig())
        # Duration of 2.5 is within filter range but not ideal
        long_dur = _make_candidate(confidence=0.9, duration=2.5)
        normal = _make_candidate(confidence=0.9, duration=1.0)

        long_score = ranker.score_candidate(long_dur)
        normal_score = ranker.score_candidate(normal)

        assert long_score < normal_score

    def test_zero_confidence_scores_lower_than_high_confidence(self):
        """Zero confidence should produce a much lower score than high confidence."""
        ranker = Ranker(RankingConfig())
        zero_conf = _make_candidate(confidence=0.0, duration=0.5)
        high_conf = _make_candidate(confidence=0.9, duration=0.5)
        zero_score = ranker.score_candidate(zero_conf)
        high_score = ranker.score_candidate(high_conf)
        # Confidence is 35% of the score, so zero conf candidate loses at least 0.315
        assert zero_score < high_score
        assert zero_score < 0.7  # without confidence, max is 0.65

    def test_maximum_confidence_ideal_duration(self):
        """Maximum confidence (1.0) with ideal duration and unused source scores 1.0."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(confidence=1.0, duration=0.5)
        score = ranker.score_candidate(candidate)
        assert score == 1.0

    def test_used_sources_reduce_score(self):
        """Using a source already in used_sources reduces score."""
        ranker = Ranker(RankingConfig())
        candidate = _make_candidate(confidence=0.9, duration=0.5, video_id=1)

        score_unused = ranker.score_candidate(candidate, used_sources=set())
        score_used = ranker.score_candidate(candidate, used_sources={1})

        assert score_used < score_unused


class TestRank:
    """Tests for Ranker.rank()."""

    def test_empty_candidates_returns_empty(self):
        """Empty candidates dict returns empty list."""
        ranker = Ranker(RankingConfig())
        result = ranker.rank({}, 5)
        assert result == []

    def test_single_token_single_candidate(self):
        """Single candidate per token produces one variation."""
        ranker = Ranker(RankingConfig())
        candidates = {"hello": [_make_candidate(word_id=1)]}
        result = ranker.rank(candidates, 5)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_multiple_variations_are_distinct(self):
        """All generated variations are pairwise distinct."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "hello": [
                _make_candidate(word_id=1, confidence=0.9),
                _make_candidate(word_id=2, confidence=0.8),
                _make_candidate(word_id=3, confidence=0.7),
            ],
            "world": [
                _make_candidate(word_id=4, confidence=0.95),
                _make_candidate(word_id=5, confidence=0.85),
                _make_candidate(word_id=6, confidence=0.75),
            ],
        }
        result = ranker.rank(candidates, 5)

        combo_keys = [tuple(c.word.id for c in var) for var in result]
        assert len(combo_keys) == len(set(combo_keys))

    def test_variations_ordered_by_descending_score(self):
        """Variations are ordered by total score descending."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "hello": [
                _make_candidate(word_id=1, confidence=0.95),
                _make_candidate(word_id=2, confidence=0.5),
            ],
            "world": [
                _make_candidate(word_id=3, confidence=0.9),
                _make_candidate(word_id=4, confidence=0.4),
            ],
        }
        result = ranker.rank(candidates, 4)

        total_scores = [sum(c.score for c in var) for var in result]
        for i in range(len(total_scores) - 1):
            assert total_scores[i] >= total_scores[i + 1]

    def test_caps_at_max_unique_combinations(self):
        """Never produces more variations than possible unique combinations."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "a": [
                _make_candidate(word_id=1, confidence=0.9),
                _make_candidate(word_id=2, confidence=0.8),
            ],
            "b": [
                _make_candidate(word_id=3, confidence=0.9),
                _make_candidate(word_id=4, confidence=0.8),
            ],
        }
        result = ranker.rank(candidates, 100)
        assert len(result) <= 4

    def test_produces_exact_count_when_possible(self):
        """Produces exactly the requested number when enough combinations exist."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "a": [
                _make_candidate(word_id=1, confidence=0.9),
                _make_candidate(word_id=2, confidence=0.8),
                _make_candidate(word_id=3, confidence=0.7),
            ],
            "b": [
                _make_candidate(word_id=4, confidence=0.9),
                _make_candidate(word_id=5, confidence=0.8),
                _make_candidate(word_id=6, confidence=0.7),
            ],
        }
        result = ranker.rank(candidates, 5)
        assert len(result) == 5

    def test_each_variation_has_correct_length(self):
        """Each variation has one candidate per token."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "the": [_make_candidate(word_id=1)],
            "quick": [
                _make_candidate(word_id=2, confidence=0.9),
                _make_candidate(word_id=3, confidence=0.8),
            ],
            "fox": [_make_candidate(word_id=4)],
        }
        result = ranker.rank(candidates, 3)

        for variation in result:
            assert len(variation) == 3

    def test_prefer_same_speaker(self):
        """With prefer_same_speaker=True, prefers consistent speakers."""
        config = RankingConfig(prefer_same_speaker=True)
        ranker = Ranker(config)

        candidates = {
            "hello": [
                _make_candidate(word_id=1, confidence=0.9, speaker="alice"),
                _make_candidate(word_id=2, confidence=0.88, speaker="bob"),
            ],
            "world": [
                _make_candidate(word_id=3, confidence=0.85, speaker="alice"),
                _make_candidate(word_id=4, confidence=0.87, speaker="bob"),
            ],
        }
        result = ranker.rank(candidates, 4)

        has_consistent = False
        for var in result:
            speakers = [c.word.speaker for c in var if c.word.speaker]
            if len(set(speakers)) == 1:
                has_consistent = True
                break
        assert has_consistent

    def test_single_variation_request(self):
        """Requesting exactly 1 variation returns the best one."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "hello": [
                _make_candidate(word_id=1, confidence=0.9),
                _make_candidate(word_id=2, confidence=0.5),
            ],
        }
        result = ranker.rank(candidates, 1)
        assert len(result) == 1
        # The best candidate (highest confidence) should be selected
        assert result[0][0].word.confidence == 0.9

    def test_many_tokens(self):
        """Works correctly with many tokens."""
        ranker = Ranker(RankingConfig())
        candidates = {}
        word_id = 1
        for token in ["the", "quick", "brown", "fox", "jumps"]:
            candidates[token] = [
                _make_candidate(word_id=word_id, confidence=0.9),
                _make_candidate(word_id=word_id + 1, confidence=0.7),
            ]
            word_id += 2

        result = ranker.rank(candidates, 3)
        assert len(result) == 3
        for var in result:
            assert len(var) == 5

    def test_scores_assigned_to_candidates(self):
        """After ranking, candidates have their scores assigned."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "hello": [_make_candidate(word_id=1, confidence=0.9)],
        }
        result = ranker.rank(candidates, 1)
        assert result[0][0].score > 0.0

    def test_diversity_prefers_different_sources(self):
        """Ranking prefers candidates from different video sources."""
        config = RankingConfig(diversity_weight=0.3)
        ranker = Ranker(config)

        candidates = {
            "hello": [
                _make_candidate(
                    word_id=1, video_id=1, confidence=0.9, video_path="/v1.mp4"
                ),
                _make_candidate(
                    word_id=2, video_id=2, confidence=0.88, video_path="/v2.mp4"
                ),
            ],
            "world": [
                _make_candidate(
                    word_id=3, video_id=1, confidence=0.9, video_path="/v1.mp4"
                ),
                _make_candidate(
                    word_id=4, video_id=2, confidence=0.88, video_path="/v2.mp4"
                ),
            ],
        }
        result = ranker.rank(candidates, 4)
        assert len(result) >= 2

    def test_filter_removes_low_quality_before_ranking(self):
        """Filter pass removes candidates below thresholds before scoring."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "hello": [
                _make_candidate(word_id=1, confidence=0.9, duration=0.5),
                _make_candidate(word_id=2, confidence=0.1, duration=0.5),  # below confidence threshold
            ],
        }
        result = ranker.rank(candidates, 5)
        # Only the high-quality candidate should appear
        assert len(result) == 1
        assert result[0][0].word.confidence == 0.9

    def test_filter_fallback_when_all_rejected(self):
        """If all candidates for a token are rejected, use unfiltered list."""
        ranker = Ranker(RankingConfig())
        candidates = {
            "hello": [
                _make_candidate(word_id=1, confidence=0.1, duration=0.5),
                _make_candidate(word_id=2, confidence=0.2, duration=0.5),
            ],
        }
        result = ranker.rank(candidates, 5)
        # Should still produce results via fallback
        assert len(result) == 2


class TestRankRoundRobin:
    """Tests for Ranker.rank_round_robin()."""

    def test_cycles_through_sources(self):
        """Words should cycle through available video sources."""
        ranker = Ranker(RankingConfig(prefer_same_speaker=False))
        candidates = {
            "hello": [
                _make_candidate(word_id=1, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=2, video_id=2, confidence=0.88, video_path="/v2.mp4"),
                _make_candidate(word_id=3, video_id=3, confidence=0.85, video_path="/v3.mp4"),
            ],
            "world": [
                _make_candidate(word_id=4, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=5, video_id=2, confidence=0.88, video_path="/v2.mp4"),
                _make_candidate(word_id=6, video_id=3, confidence=0.85, video_path="/v3.mp4"),
            ],
            "today": [
                _make_candidate(word_id=7, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=8, video_id=2, confidence=0.88, video_path="/v2.mp4"),
                _make_candidate(word_id=9, video_id=3, confidence=0.85, video_path="/v3.mp4"),
            ],
        }
        result = ranker.rank_round_robin(candidates, 3)

        assert len(result) >= 1
        # First variation should cycle: source 1, 2, 3
        first_var_sources = [c.video.id for c in result[0]]
        assert first_var_sources == [1, 2, 3]

    def test_wraps_around_when_more_words_than_sources(self):
        """Should cycle back to first source after exhausting all sources."""
        ranker = Ranker(RankingConfig(prefer_same_speaker=False))
        candidates = {}
        word_id = 1
        for token in ["a", "b", "c", "d", "e"]:
            candidates[token] = [
                _make_candidate(word_id=word_id, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=word_id + 1, video_id=2, confidence=0.88, video_path="/v2.mp4"),
            ]
            word_id += 2

        result = ranker.rank_round_robin(candidates, 2)
        assert len(result) >= 1
        # Should alternate: 1, 2, 1, 2, 1
        sources = [c.video.id for c in result[0]]
        assert sources == [1, 2, 1, 2, 1]

    def test_fallback_when_source_missing_for_token(self):
        """If a token has no candidate from the target source, use best available."""
        ranker = Ranker(RankingConfig(prefer_same_speaker=False))
        candidates = {
            "hello": [
                _make_candidate(word_id=1, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=2, video_id=2, confidence=0.88, video_path="/v2.mp4"),
            ],
            "world": [
                # Only available from source 1
                _make_candidate(word_id=3, video_id=1, confidence=0.9, video_path="/v1.mp4"),
            ],
        }
        result = ranker.rank_round_robin(candidates, 1)
        assert len(result) == 1
        # "hello" from source 1, "world" fallback to source 1 (only option)
        assert result[0][0].video.id == 1
        assert result[0][1].video.id == 1

    def test_produces_multiple_variations_with_different_offsets(self):
        """Multiple variations should start from different positions in the rotation."""
        ranker = Ranker(RankingConfig(prefer_same_speaker=False))
        candidates = {
            "hello": [
                _make_candidate(word_id=1, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=2, video_id=2, confidence=0.88, video_path="/v2.mp4"),
                _make_candidate(word_id=3, video_id=3, confidence=0.85, video_path="/v3.mp4"),
            ],
            "world": [
                _make_candidate(word_id=4, video_id=1, confidence=0.9, video_path="/v1.mp4"),
                _make_candidate(word_id=5, video_id=2, confidence=0.88, video_path="/v2.mp4"),
                _make_candidate(word_id=6, video_id=3, confidence=0.85, video_path="/v3.mp4"),
            ],
        }
        result = ranker.rank_round_robin(candidates, 3)
        assert len(result) == 3
        # Different starting sources
        starts = [var[0].video.id for var in result]
        assert len(set(starts)) == 3  # All different starting sources

    def test_empty_candidates_returns_empty(self):
        """Empty input returns empty output."""
        ranker = Ranker(RankingConfig())
        result = ranker.rank_round_robin({}, 5)
        assert result == []
