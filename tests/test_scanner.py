"""Unit tests for the Scanner class."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from wordnap.ingestion.scanner import Scanner
from wordnap.models.schemas import VideoMetadata, VideoStatus


# Sample ffprobe JSON output for a valid video file
SAMPLE_FFPROBE_OUTPUT = {
    "streams": [
        {
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
        },
        {
            "codec_type": "audio",
            "sample_rate": "44100",
        },
    ],
    "format": {
        "duration": "120.5",
    },
}

SAMPLE_FFPROBE_NTSC = {
    "streams": [
        {
            "codec_type": "video",
            "width": 1280,
            "height": 720,
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "30000/1001",
        },
        {
            "codec_type": "audio",
            "sample_rate": "48000",
        },
    ],
    "format": {
        "duration": "60.0",
    },
}


class TestScanDirectory:
    """Tests for scan_directory() with real filesystem using tmp_path."""

    def test_discovers_supported_extensions(self, tmp_path: Path):
        """scan_directory finds all files with supported extensions."""
        # Create files with various extensions
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mkv").touch()
        (tmp_path / "video3.avi").touch()
        (tmp_path / "video4.mov").touch()
        (tmp_path / "video5.webm").touch()
        (tmp_path / "document.pdf").touch()
        (tmp_path / "image.png").touch()
        (tmp_path / "audio.mp3").touch()

        scanner = Scanner()

        with patch.object(scanner, "probe_video") as mock_probe:
            mock_probe.return_value = VideoMetadata(
                path=Path("dummy"),
                filename="dummy",
                duration=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )
            results = scanner.scan_directory(tmp_path)

        # Should call probe_video exactly 5 times (one for each supported ext)
        assert mock_probe.call_count == 5

    def test_recursive_discovery(self, tmp_path: Path):
        """scan_directory recursively discovers files in subdirectories."""
        sub1 = tmp_path / "subdir1"
        sub1.mkdir()
        sub2 = tmp_path / "subdir1" / "subdir2"
        sub2.mkdir()

        (tmp_path / "root.mp4").touch()
        (sub1 / "level1.mkv").touch()
        (sub2 / "level2.avi").touch()

        scanner = Scanner()

        with patch.object(scanner, "probe_video") as mock_probe:
            mock_probe.return_value = VideoMetadata(
                path=Path("dummy"),
                filename="dummy",
                duration=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )
            results = scanner.scan_directory(tmp_path)

        assert mock_probe.call_count == 3

    def test_ignores_unsupported_extensions(self, tmp_path: Path):
        """scan_directory ignores files without supported extensions."""
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.json").touch()
        (tmp_path / "script.py").touch()
        (tmp_path / "image.jpg").touch()

        scanner = Scanner()

        with patch.object(scanner, "probe_video") as mock_probe:
            results = scanner.scan_directory(tmp_path)

        mock_probe.assert_not_called()
        assert results == []

    def test_case_insensitive_extensions(self, tmp_path: Path):
        """scan_directory handles uppercase extensions."""
        (tmp_path / "VIDEO.MP4").touch()
        (tmp_path / "movie.MKV").touch()

        scanner = Scanner()

        with patch.object(scanner, "probe_video") as mock_probe:
            mock_probe.return_value = VideoMetadata(
                path=Path("dummy"),
                filename="dummy",
                duration=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )
            results = scanner.scan_directory(tmp_path)

        assert mock_probe.call_count == 2

    def test_skips_already_indexed_files(self, tmp_path: Path):
        """scan_directory skips files that are already indexed."""
        video1 = tmp_path / "video1.mp4"
        video2 = tmp_path / "video2.mp4"
        video1.touch()
        video2.touch()

        scanner = Scanner(indexed_paths={video1})

        with patch.object(scanner, "probe_video") as mock_probe:
            mock_probe.return_value = VideoMetadata(
                path=Path("dummy"),
                filename="dummy",
                duration=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )
            results = scanner.scan_directory(tmp_path)

        # Should only probe video2, not video1
        assert mock_probe.call_count == 1
        mock_probe.assert_called_once_with(video2)

    def test_handles_unreadable_files(self, tmp_path: Path):
        """scan_directory logs warning and continues on unreadable files."""
        (tmp_path / "good.mp4").touch()
        (tmp_path / "bad.mp4").touch()
        (tmp_path / "also_good.mp4").touch()

        scanner = Scanner()

        call_count = 0

        def side_effect(path):
            nonlocal call_count
            call_count += 1
            if "bad" in path.name:
                raise subprocess.CalledProcessError(1, "ffprobe")
            return VideoMetadata(
                path=path,
                filename=path.name,
                duration=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                audio_sample_rate=44100,
            )

        with patch.object(scanner, "probe_video", side_effect=side_effect):
            results = scanner.scan_directory(tmp_path)

        # Should have 2 successful results (good and also_good)
        assert len(results) == 2
        assert call_count == 3

    def test_empty_directory(self, tmp_path: Path):
        """scan_directory returns empty list for empty directory."""
        scanner = Scanner()
        results = scanner.scan_directory(tmp_path)
        assert results == []

    def test_ignores_directories(self, tmp_path: Path):
        """scan_directory does not try to probe directories even if named like videos."""
        (tmp_path / "video.mp4").mkdir()  # directory named like a video

        scanner = Scanner()

        with patch.object(scanner, "probe_video") as mock_probe:
            results = scanner.scan_directory(tmp_path)

        mock_probe.assert_not_called()
        assert results == []


class TestProbeVideo:
    """Tests for probe_video() with mocked subprocess calls."""

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_extracts_metadata_from_valid_video(self, mock_run: MagicMock):
        """probe_video extracts all metadata fields correctly."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps(SAMPLE_FFPROBE_OUTPUT),
            returncode=0,
        )

        scanner = Scanner()
        result = scanner.probe_video(Path("/videos/test.mp4"))

        assert result.path == Path("/videos/test.mp4")
        assert result.filename == "test.mp4"
        assert result.duration == 120.5
        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30.0
        assert result.audio_sample_rate == 44100
        assert result.status == VideoStatus.PENDING

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_parses_ntsc_frame_rate(self, mock_run: MagicMock):
        """probe_video correctly parses NTSC frame rates like 30000/1001."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps(SAMPLE_FFPROBE_NTSC),
            returncode=0,
        )

        scanner = Scanner()
        result = scanner.probe_video(Path("/videos/ntsc.mp4"))

        assert result.fps == pytest.approx(29.97, rel=1e-2)
        assert result.audio_sample_rate == 48000
        assert result.width == 1280
        assert result.height == 720

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_uses_correct_ffprobe_command(self, mock_run: MagicMock):
        """probe_video calls ffprobe with correct arguments."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps(SAMPLE_FFPROBE_OUTPUT),
            returncode=0,
        )

        video_path = Path("/videos/test.mp4")
        scanner = Scanner()
        scanner.probe_video(video_path)

        mock_run.assert_called_once_with(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_raises_on_ffprobe_failure(self, mock_run: MagicMock):
        """probe_video raises SubprocessError when ffprobe fails."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffprobe", stderr="Error reading file"
        )

        scanner = Scanner()
        with pytest.raises(subprocess.CalledProcessError):
            scanner.probe_video(Path("/videos/corrupt.mp4"))

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_raises_on_missing_video_stream(self, mock_run: MagicMock):
        """probe_video raises ValueError when no video stream exists."""
        data = {
            "streams": [
                {"codec_type": "audio", "sample_rate": "44100"},
            ],
            "format": {"duration": "60.0"},
        }
        mock_run.return_value = MagicMock(
            stdout=json.dumps(data),
            returncode=0,
        )

        scanner = Scanner()
        with pytest.raises(ValueError, match="No video stream"):
            scanner.probe_video(Path("/videos/audio_only.mp4"))

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_raises_on_missing_duration(self, mock_run: MagicMock):
        """probe_video raises ValueError when duration is missing."""
        data = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
            ],
            "format": {},
        }
        mock_run.return_value = MagicMock(
            stdout=json.dumps(data),
            returncode=0,
        )

        scanner = Scanner()
        with pytest.raises(ValueError, match="No duration"):
            scanner.probe_video(Path("/videos/no_duration.mp4"))

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_handles_video_without_audio_stream(self, mock_run: MagicMock):
        """probe_video handles video with no audio stream (sample_rate=0)."""
        data = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "24/1",
                },
            ],
            "format": {"duration": "30.0"},
        }
        mock_run.return_value = MagicMock(
            stdout=json.dumps(data),
            returncode=0,
        )

        scanner = Scanner()
        result = scanner.probe_video(Path("/videos/silent.mp4"))

        assert result.audio_sample_rate == 0
        assert result.width == 640
        assert result.height == 480
        assert result.fps == 24.0

    @patch("wordnap.ingestion.scanner.subprocess.run")
    def test_falls_back_to_avg_frame_rate(self, mock_run: MagicMock):
        """probe_video falls back to avg_frame_rate when r_frame_rate is 0/0."""
        data = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "0/0",
                    "avg_frame_rate": "25/1",
                },
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                },
            ],
            "format": {"duration": "90.0"},
        }
        mock_run.return_value = MagicMock(
            stdout=json.dumps(data),
            returncode=0,
        )

        scanner = Scanner()
        result = scanner.probe_video(Path("/videos/test.mp4"))

        assert result.fps == 25.0


class TestParseFrameRate:
    """Tests for the _parse_frame_rate static method."""

    def test_integer_fraction(self):
        assert Scanner._parse_frame_rate("30/1") == 30.0

    def test_ntsc_fraction(self):
        assert Scanner._parse_frame_rate("30000/1001") == pytest.approx(
            29.97, rel=1e-2
        )

    def test_24_fps(self):
        assert Scanner._parse_frame_rate("24/1") == 24.0

    def test_zero_denominator(self):
        assert Scanner._parse_frame_rate("30/0") == 0.0

    def test_plain_number(self):
        assert Scanner._parse_frame_rate("30") == 30.0

    def test_invalid_string(self):
        assert Scanner._parse_frame_rate("invalid") == 0.0

    def test_empty_string(self):
        assert Scanner._parse_frame_rate("") == 0.0
