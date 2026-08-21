"""Unit tests for clip boundary calculation functions."""

import pytest

from sentence_mixer.editing.boundaries import (
    compute_clip_duration,
    compute_padded_boundaries,
    format_timestamp,
)


class TestComputePaddedBoundaries:
    """Tests for compute_padded_boundaries()."""

    def test_normal_padding(self):
        """Standard case: padding applied within video bounds."""
        start, end = compute_padded_boundaries(5.0, 6.0, 0.1, 60.0)
        assert start == pytest.approx(4.9)
        assert end == pytest.approx(6.1)

    def test_zero_padding(self):
        """Zero padding returns original times unchanged."""
        start, end = compute_padded_boundaries(3.5, 4.2, 0.0, 60.0)
        assert start == pytest.approx(3.5)
        assert end == pytest.approx(4.2)

    def test_clamp_start_to_zero(self):
        """Padded start clamped to 0 when padding exceeds start_time."""
        start, end = compute_padded_boundaries(0.05, 0.5, 0.1, 60.0)
        assert start == 0.0
        assert end == pytest.approx(0.6)

    def test_clamp_end_to_video_duration(self):
        """Padded end clamped to video_duration when padding exceeds bounds."""
        start, end = compute_padded_boundaries(59.5, 59.9, 0.2, 60.0)
        assert start == pytest.approx(59.3)
        assert end == 60.0

    def test_clamp_both_boundaries(self):
        """Both boundaries clamped when video is very short."""
        start, end = compute_padded_boundaries(0.1, 0.2, 0.5, 0.3)
        assert start == 0.0
        assert end == 0.3

    def test_large_padding(self):
        """Large padding still clamps correctly."""
        start, end = compute_padded_boundaries(10.0, 11.0, 100.0, 30.0)
        assert start == 0.0
        assert end == 30.0

    def test_word_at_video_start(self):
        """Word at the very start of the video."""
        start, end = compute_padded_boundaries(0.0, 0.5, 0.1, 60.0)
        assert start == 0.0
        assert end == pytest.approx(0.6)

    def test_word_at_video_end(self):
        """Word at the very end of the video."""
        start, end = compute_padded_boundaries(59.5, 60.0, 0.1, 60.0)
        assert start == pytest.approx(59.4)
        assert end == 60.0

    def test_small_floating_point_values(self):
        """Handles small floating point values correctly."""
        start, end = compute_padded_boundaries(0.001, 0.002, 0.0005, 10.0)
        assert start == pytest.approx(0.0005)
        assert end == pytest.approx(0.0025)


class TestFormatTimestamp:
    """Tests for format_timestamp()."""

    def test_typical_value(self):
        """Standard timestamp formatted with 3 decimal places."""
        assert format_timestamp(41.12) == "41.120"

    def test_zero(self):
        """Zero formats correctly."""
        assert format_timestamp(0.0) == "0.000"

    def test_small_value(self):
        """Small value preserves leading zeros in decimal."""
        assert format_timestamp(0.05) == "0.050"

    def test_integer_value(self):
        """Integer seconds get .000 suffix."""
        assert format_timestamp(10.0) == "10.000"

    def test_high_precision_truncated(self):
        """Values beyond 3 decimal places are rounded."""
        assert format_timestamp(1.23456) == "1.235"

    def test_large_value(self):
        """Large timestamp values format correctly."""
        assert format_timestamp(3600.5) == "3600.500"

    def test_three_decimal_exact(self):
        """Value already at 3 decimal places stays the same."""
        assert format_timestamp(5.123) == "5.123"


class TestComputeClipDuration:
    """Tests for compute_clip_duration()."""

    def test_normal_duration(self):
        """Basic duration computation."""
        assert compute_clip_duration(4.9, 6.1) == pytest.approx(1.2)

    def test_zero_duration(self):
        """Same start and end produce zero duration."""
        assert compute_clip_duration(5.0, 5.0) == 0.0

    def test_small_duration(self):
        """Very short clip duration."""
        assert compute_clip_duration(10.0, 10.05) == pytest.approx(0.05)

    def test_large_duration(self):
        """Duration for a long clip."""
        assert compute_clip_duration(0.0, 60.0) == pytest.approx(60.0)

    def test_precision(self):
        """Duration computation maintains floating point precision."""
        duration = compute_clip_duration(1.001, 1.002)
        assert duration == pytest.approx(0.001)
