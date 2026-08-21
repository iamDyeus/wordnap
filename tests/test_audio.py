"""Tests for AudioExtractor - audio extraction from video files."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from sentence_mixer.ingestion.audio import AudioExtractor, AudioExtractionError


class TestAudioExtractorCaching:
    """Test caching behavior with real filesystem (tmp_path)."""

    def test_skips_extraction_when_output_exists(self, tmp_path: Path) -> None:
        """If the WAV file already exists, extraction is skipped entirely."""
        extractor = AudioExtractor()
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "audio_cache"
        output_dir.mkdir()

        # Pre-create the cached output file
        cached_wav = output_dir / "video.wav"
        cached_wav.write_text("existing audio data")

        with patch("sentence_mixer.ingestion.audio.subprocess.run") as mock_run:
            result = extractor.extract_audio(video_path, output_dir)

        # subprocess.run should never be called
        mock_run.assert_not_called()
        assert result == cached_wav

    def test_creates_output_dir_if_not_exists(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        extractor = AudioExtractor()
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "nested" / "audio_cache"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("sentence_mixer.ingestion.audio.subprocess.run", return_value=mock_result):
            extractor.extract_audio(video_path, output_dir)

        assert output_dir.exists()

    def test_returns_correct_output_path(self, tmp_path: Path) -> None:
        """Output filename is derived from video filename with .wav extension."""
        extractor = AudioExtractor()
        video_path = tmp_path / "my_video_file.mp4"
        video_path.touch()
        output_dir = tmp_path / "cache"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("sentence_mixer.ingestion.audio.subprocess.run", return_value=mock_result):
            result = extractor.extract_audio(video_path, output_dir)

        assert result == output_dir / "my_video_file.wav"


class TestAudioExtractorFFmpeg:
    """Test FFmpeg command construction and error handling."""

    def test_calls_ffmpeg_with_correct_arguments(self, tmp_path: Path) -> None:
        """FFmpeg is called with list arguments for mono WAV at 16kHz."""
        extractor = AudioExtractor()
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("sentence_mixer.ingestion.audio.subprocess.run", return_value=mock_result) as mock_run:
            extractor.extract_audio(video_path, output_dir)

        expected_output = output_dir / "test.wav"
        mock_run.assert_called_once_with(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(expected_output),
            ],
            capture_output=True,
            text=True,
        )

    def test_custom_sample_rate(self, tmp_path: Path) -> None:
        """Custom sample rate is passed to FFmpeg command."""
        extractor = AudioExtractor()
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("sentence_mixer.ingestion.audio.subprocess.run", return_value=mock_result) as mock_run:
            extractor.extract_audio(video_path, output_dir, sample_rate=44100)

        call_args = mock_run.call_args[0][0]
        ar_index = call_args.index("-ar")
        assert call_args[ar_index + 1] == "44100"

    def test_raises_error_on_ffmpeg_failure(self, tmp_path: Path) -> None:
        """Raises AudioExtractionError with stderr when FFmpeg fails."""
        extractor = AudioExtractor()
        video_path = tmp_path / "bad_video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "No such file or directory: codec not found"

        with patch("sentence_mixer.ingestion.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractionError) as exc_info:
                extractor.extract_audio(video_path, output_dir)

        assert exc_info.value.video_path == video_path
        assert "codec not found" in exc_info.value.stderr
        assert str(video_path) in str(exc_info.value)

    def test_cleans_up_partial_output_on_failure(self, tmp_path: Path) -> None:
        """Partial output file is removed when FFmpeg fails."""
        extractor = AudioExtractor()
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        partial_output = output_dir / "test.wav"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Conversion failed"

        def fake_ffmpeg_run(*args, **kwargs):
            # Simulate FFmpeg creating a partial file before failing
            partial_output.write_text("partial data")
            return mock_result

        with patch("sentence_mixer.ingestion.audio.subprocess.run", side_effect=fake_ffmpeg_run):
            with pytest.raises(AudioExtractionError):
                extractor.extract_audio(video_path, output_dir)

        assert not partial_output.exists()

    def test_uses_list_args_not_shell(self, tmp_path: Path) -> None:
        """subprocess.run is called without shell=True (security)."""
        extractor = AudioExtractor()
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("sentence_mixer.ingestion.audio.subprocess.run", return_value=mock_result) as mock_run:
            extractor.extract_audio(video_path, output_dir)

        # Verify shell=True was NOT passed
        call_kwargs = mock_run.call_args[1]
        assert "shell" not in call_kwargs or call_kwargs["shell"] is False
