"""Video file discovery and metadata extraction via ffprobe."""

import json
import logging
import subprocess
from pathlib import Path

from wordnap.models.schemas import VideoMetadata, VideoStatus

logger = logging.getLogger(__name__)


class Scanner:
    """Discovers video files in a directory and extracts metadata via ffprobe."""

    SUPPORTED_EXTENSIONS: set[str] = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    def __init__(self, indexed_paths: set[Path] | None = None):
        """Initialize the scanner.

        Args:
            indexed_paths: Set of paths already indexed (to skip re-indexing).
        """
        self._indexed_paths = indexed_paths or set()

    def scan_directory(self, directory: Path) -> list[VideoMetadata]:
        """Recursively scan directory for video files and extract metadata.

        Discovers all files with supported extensions, skips already-indexed
        files, and handles unreadable files by logging a warning and continuing.

        Args:
            directory: Root directory to scan recursively.

        Returns:
            List of VideoMetadata objects for successfully probed files.
        """
        results: list[VideoMetadata] = []

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            if file_path in self._indexed_paths:
                logger.info("Skipping already-indexed file: %s", file_path)
                continue

            try:
                metadata = self.probe_video(file_path)
                results.append(metadata)
            except (subprocess.SubprocessError, ValueError, KeyError) as e:
                logger.warning(
                    "Could not read video file %s: %s", file_path, e
                )
                continue

        return results

    def probe_video(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from a single video file using ffprobe.

        Runs ffprobe with JSON output to extract duration, width, height,
        fps, and audio_sample_rate from the video file.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with extracted information and status=PENDING.

        Raises:
            subprocess.SubprocessError: If ffprobe fails to execute.
            ValueError: If ffprobe output cannot be parsed.
            KeyError: If expected fields are missing from ffprobe output.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)
        return self._parse_ffprobe_output(data, video_path)

    def _parse_ffprobe_output(
        self, data: dict, video_path: Path
    ) -> VideoMetadata:
        """Parse ffprobe JSON output into VideoMetadata.

        Args:
            data: Parsed JSON output from ffprobe.
            video_path: Path to the source video file.

        Returns:
            VideoMetadata populated from ffprobe data.

        Raises:
            ValueError: If required streams or fields are missing.
            KeyError: If expected keys are not in the output.
        """
        streams = data.get("streams", [])
        format_info = data.get("format", {})

        # Find video and audio streams
        video_stream = None
        audio_stream = None
        for stream in streams:
            codec_type = stream.get("codec_type", "")
            if codec_type == "video" and video_stream is None:
                video_stream = stream
            elif codec_type == "audio" and audio_stream is None:
                audio_stream = stream

        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")

        # Extract duration from format
        duration_str = format_info.get("duration")
        if duration_str is None:
            raise ValueError(f"No duration found in format for {video_path}")
        duration = float(duration_str)

        # Extract resolution
        width = int(video_stream["width"])
        height = int(video_stream["height"])

        # Extract fps from r_frame_rate (format: "30/1" or "30000/1001")
        fps = self._parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))
        if fps == 0.0:
            # Fallback to avg_frame_rate
            fps = self._parse_frame_rate(
                video_stream.get("avg_frame_rate", "0/1")
            )

        # Extract audio sample rate
        audio_sample_rate = 0
        if audio_stream is not None:
            audio_sample_rate = int(audio_stream.get("sample_rate", 0))

        return VideoMetadata(
            path=video_path,
            filename=video_path.name,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            audio_sample_rate=audio_sample_rate,
            status=VideoStatus.PENDING,
        )

    @staticmethod
    def _parse_frame_rate(rate_str: str) -> float:
        """Parse a frame rate string like '30/1' or '30000/1001' into a float.

        Args:
            rate_str: Frame rate as a fraction string (numerator/denominator).

        Returns:
            Frame rate as a float, or 0.0 if parsing fails.
        """
        try:
            if "/" in rate_str:
                num, den = rate_str.split("/")
                numerator = int(num)
                denominator = int(den)
                if denominator == 0:
                    return 0.0
                return numerator / denominator
            return float(rate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0
