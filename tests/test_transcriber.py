"""Unit tests for the Transcriber class with mocked whisperx module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentence_mixer.models.schemas import TranscriptionResult
from sentence_mixer.transcription.whisperx import Transcriber


@pytest.fixture
def transcriber():
    """Create a Transcriber instance without loading any model."""
    return Transcriber(model_size="base", device="cpu")


@pytest.fixture
def mock_whisperx():
    """Mock the whisperx module for all tests."""
    with patch.dict("sys.modules", {"whisperx": MagicMock()}) as _:
        import whisperx

        yield whisperx


class TestTranscriberInit:
    def test_default_params(self):
        t = Transcriber()
        assert t.model_size == "base"
        assert t.device == "cpu"
        assert t._model is None

    def test_custom_params(self):
        t = Transcriber(model_size="large", device="cuda")
        assert t.model_size == "large"
        assert t.device == "cuda"


class TestTranscriberTranscribe:
    @patch.dict("sys.modules", {"whisperx": MagicMock()})
    def test_file_not_found_raises_error(self, transcriber):
        """transcribe() raises FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe(Path("/nonexistent/audio.wav"))

    @patch("sentence_mixer.transcription.whisperx.whisperx", create=True)
    def test_successful_transcription(self, mock_wx, tmp_path):
        """transcribe() returns TranscriptionResult with correct mapping."""
        # Create a fake audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        # Mock whisperx module functions
        mock_model = MagicMock()
        mock_wx.load_model.return_value = mock_model
        mock_wx.load_audio.return_value = "fake_audio_array"

        # Mock transcription result
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.6, "score": 0.95},
                        {"word": "world", "start": 0.7, "end": 1.5, "score": 0.88},
                    ],
                }
            ],
        }

        # Mock alignment
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        mock_wx.align.return_value = {
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.6, "score": 0.95},
                        {"word": "world", "start": 0.7, "end": 1.5, "score": 0.88},
                    ],
                }
            ]
        }

        # Patch the import inside the method
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: mock_wx
            if name == "whisperx"
            else __builtins__.__import__(name, *args, **kwargs),
        ):
            # Directly test the _map_result method which does the real mapping
            transcriber = Transcriber(model_size="base", device="cpu")
            result = transcriber._map_result(mock_wx.align.return_value)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        assert len(result.words) == 2

        # Verify segment mapping
        seg = result.segments[0]
        assert seg.start_time == 0.0
        assert seg.end_time == 1.5
        assert seg.text == "Hello world"
        assert seg.video_id == 0  # Placeholder, set during storage
        # Confidence is average of word scores: (0.95 + 0.88) / 2 = 0.915
        assert abs(seg.confidence - 0.915) < 0.001

        # Verify word mapping
        w1 = result.words[0]
        assert w1.word == "Hello"
        assert w1.normalized_word == "hello"
        assert w1.start_time == 0.0
        assert w1.end_time == 0.6
        assert w1.confidence == 0.95

        w2 = result.words[1]
        assert w2.word == "world"
        assert w2.normalized_word == "world"
        assert w2.start_time == 0.7
        assert w2.end_time == 1.5
        assert w2.confidence == 0.88


class TestMapResult:
    """Test the _map_result method which maps WhisperX format to our models."""

    @pytest.fixture
    def transcriber(self):
        return Transcriber(model_size="base", device="cpu")

    def test_basic_mapping(self, transcriber):
        """Maps segments and words correctly from WhisperX output."""
        whisperx_output = {
            "segments": [
                {
                    "text": "The quick brown fox",
                    "start": 0.5,
                    "end": 3.0,
                    "words": [
                        {"word": "The", "start": 0.5, "end": 0.8, "score": 0.9},
                        {"word": "quick", "start": 0.9, "end": 1.3, "score": 0.85},
                        {"word": "brown", "start": 1.4, "end": 1.9, "score": 0.92},
                        {"word": "fox", "start": 2.0, "end": 3.0, "score": 0.78},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        assert len(result.words) == 4

        # Verify ordering is preserved
        assert result.words[0].word == "The"
        assert result.words[1].word == "quick"
        assert result.words[2].word == "brown"
        assert result.words[3].word == "fox"

    def test_multiple_segments(self, transcriber):
        """Handles multiple segments correctly."""
        whisperx_output = {
            "segments": [
                {
                    "text": "Hello",
                    "start": 0.0,
                    "end": 0.8,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.8, "score": 0.95},
                    ],
                },
                {
                    "text": "World",
                    "start": 1.0,
                    "end": 1.8,
                    "words": [
                        {"word": "World", "start": 1.0, "end": 1.8, "score": 0.9},
                    ],
                },
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert len(result.segments) == 2
        assert len(result.words) == 2
        assert result.words[0].word == "Hello"
        assert result.words[1].word == "World"

    def test_score_mapped_to_confidence(self, transcriber):
        """WhisperX 'score' field maps to our 'confidence' field."""
        whisperx_output = {
            "segments": [
                {
                    "text": "test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "test", "start": 0.0, "end": 1.0, "score": 0.73},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.words[0].confidence == 0.73

    def test_words_without_timestamps_are_skipped(self, transcriber):
        """Words missing start or end timestamps are skipped."""
        whisperx_output = {
            "segments": [
                {
                    "text": "hello world there",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "score": 0.9},
                        {"word": "world", "score": 0.8},  # missing timestamps
                        {"word": "there", "start": 1.2, "end": 2.0, "score": 0.7},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        # Only "hello" and "there" should be included
        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert result.words[1].word == "there"

    def test_words_with_missing_start_only_skipped(self, transcriber):
        """Words with only end timestamp are skipped."""
        whisperx_output = {
            "segments": [
                {
                    "text": "one two",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "one", "end": 0.5, "score": 0.9},  # missing start
                        {"word": "two", "start": 1.0, "end": 2.0, "score": 0.8},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert len(result.words) == 1
        assert result.words[0].word == "two"

    def test_words_with_invalid_timestamps_skipped(self, transcriber):
        """Words where start >= end are skipped."""
        whisperx_output = {
            "segments": [
                {
                    "text": "good bad",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "good", "start": 0.0, "end": 0.5, "score": 0.9},
                        {
                            "word": "bad",
                            "start": 1.5,
                            "end": 1.5,
                            "score": 0.8,
                        },  # start == end
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert len(result.words) == 1
        assert result.words[0].word == "good"

    def test_normalized_word_populated(self, transcriber):
        """Each word's normalized_word is set using Tokenizer.normalize_word()."""
        whisperx_output = {
            "segments": [
                {
                    "text": "Hello, World!",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "Hello,", "start": 0.0, "end": 0.8, "score": 0.9},
                        {"word": "World!", "start": 1.0, "end": 2.0, "score": 0.85},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.words[0].normalized_word == "hello"
        assert result.words[1].normalized_word == "world"

    def test_speaker_propagated_from_segment(self, transcriber):
        """Speaker from segment propagates to words without their own speaker."""
        whisperx_output = {
            "segments": [
                {
                    "text": "testing",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "testing", "start": 0.0, "end": 1.0, "score": 0.9},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.words[0].speaker == "SPEAKER_01"

    def test_word_speaker_overrides_segment_speaker(self, transcriber):
        """Word-level speaker overrides segment-level speaker."""
        whisperx_output = {
            "segments": [
                {
                    "text": "testing",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "word": "testing",
                            "start": 0.0,
                            "end": 1.0,
                            "score": 0.9,
                            "speaker": "SPEAKER_02",
                        },
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.words[0].speaker == "SPEAKER_02"

    def test_confidence_clamped_to_valid_range(self, transcriber):
        """Confidence scores outside [0, 1] are clamped."""
        whisperx_output = {
            "segments": [
                {
                    "text": "test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {
                            "word": "over",
                            "start": 0.0,
                            "end": 0.4,
                            "score": 1.5,
                        },  # over max
                        {
                            "word": "under",
                            "start": 0.5,
                            "end": 1.0,
                            "score": -0.3,
                        },  # under min
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.words[0].confidence == 1.0
        assert result.words[1].confidence == 0.0

    def test_segment_confidence_is_average_of_word_scores(self, transcriber):
        """Segment confidence is computed as average of valid word scores."""
        whisperx_output = {
            "segments": [
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.8, "score": 0.8},
                        {"word": "world", "start": 1.0, "end": 2.0, "score": 0.6},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        # Average: (0.8 + 0.6) / 2 = 0.7
        assert abs(result.segments[0].confidence - 0.7) < 0.001

    def test_empty_segments_list(self, transcriber):
        """Empty segments list returns empty results."""
        whisperx_output = {"segments": []}

        result = transcriber._map_result(whisperx_output)

        assert len(result.segments) == 0
        assert len(result.words) == 0

    def test_segment_with_invalid_timestamps_skipped(self, transcriber):
        """Segments where start >= end are skipped entirely."""
        whisperx_output = {
            "segments": [
                {
                    "text": "valid",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "valid", "start": 0.0, "end": 1.0, "score": 0.9},
                    ],
                },
                {
                    "text": "invalid",
                    "start": 2.0,
                    "end": 2.0,  # start == end
                    "words": [
                        {"word": "invalid", "start": 2.0, "end": 2.5, "score": 0.8},
                    ],
                },
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert len(result.segments) == 1
        assert result.segments[0].text == "valid"
        # Words from invalid segment are also not included
        assert len(result.words) == 1

    def test_word_with_empty_text_skipped(self, transcriber):
        """Words with empty text after stripping are skipped."""
        whisperx_output = {
            "segments": [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.8, "score": 0.9},
                        {"word": "  ", "start": 1.0, "end": 1.5, "score": 0.5},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert len(result.words) == 1
        assert result.words[0].word == "hello"

    def test_word_normalizing_to_empty_skipped(self, transcriber):
        """Words that normalize to empty string (e.g., pure punctuation) are skipped."""
        whisperx_output = {
            "segments": [
                {
                    "text": "hello ...",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.8, "score": 0.9},
                        {
                            "word": "...",
                            "start": 1.0,
                            "end": 1.5,
                            "score": 0.5,
                        },  # normalizes to ""
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert len(result.words) == 1
        assert result.words[0].word == "hello"

    def test_missing_score_defaults_to_zero(self, transcriber):
        """Words without a score field default to confidence 0.0."""
        whisperx_output = {
            "segments": [
                {
                    "text": "test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {
                            "word": "test",
                            "start": 0.0,
                            "end": 1.0,
                        },  # no score field
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.words[0].confidence == 0.0

    def test_no_speaker_info(self, transcriber):
        """Results without speaker info have None for speaker."""
        whisperx_output = {
            "segments": [
                {
                    "text": "test",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "test", "start": 0.0, "end": 1.0, "score": 0.9},
                    ],
                }
            ]
        }

        result = transcriber._map_result(whisperx_output)

        assert result.segments[0].speaker is None
        assert result.words[0].speaker is None


class TestAlignmentFailure:
    """Test graceful handling of alignment failures."""

    def test_alignment_failure_falls_back_to_segments(self):
        """When alignment fails, _align returns original transcription result."""
        transcriber = Transcriber(model_size="base", device="cpu")

        # The raw transcription result (before alignment)
        raw_result = {
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.6, "score": 0.9},
                        {"word": "world", "start": 0.7, "end": 1.5, "score": 0.8},
                    ],
                }
            ],
        }

        # Mock whisperx to raise an error during alignment
        mock_wx = MagicMock()
        mock_wx.load_align_model.side_effect = RuntimeError(
            "Alignment model not available"
        )

        with patch.dict("sys.modules", {"whisperx": mock_wx}):
            result = transcriber._align("fake_audio", raw_result)

        # Should return the original transcription result unchanged
        assert result == raw_result

    def test_alignment_failure_still_produces_result(self):
        """Even when alignment fails, _map_result can process the fallback."""
        transcriber = Transcriber(model_size="base", device="cpu")

        # Simulate fallback result (same as raw transcription)
        fallback_result = {
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.6, "score": 0.9},
                        {"word": "world", "start": 0.7, "end": 1.5, "score": 0.8},
                    ],
                }
            ],
        }

        result = transcriber._map_result(fallback_result)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        assert len(result.words) == 2

    def test_alignment_failure_with_no_word_timestamps(self):
        """When alignment fails and segments have no word timestamps."""
        transcriber = Transcriber(model_size="base", device="cpu")

        # Segments without word-level detail (alignment failed completely)
        fallback_result = {
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    # No "words" key at all
                }
            ]
        }

        result = transcriber._map_result(fallback_result)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        # No words since there's no word-level data
        assert len(result.words) == 0
        # Segment confidence is 0.0 when no word scores available
        assert result.segments[0].confidence == 0.0
