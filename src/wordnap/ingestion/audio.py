"""Audio extraction from video files using FFmpeg."""

import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when FFmpeg fails to extract audio from a video file."""

    def __init__(self, video_path: Path, stderr: str) -> None:
        self.video_path = video_path
        self.stderr = stderr
        super().__init__(
            f"FFmpeg failed to extract audio from '{video_path}': {stderr}"
        )


class AudioExtractor:
    """Extracts audio tracks from video files as mono WAV for transcription."""

    def extract_audio(
        self, video_path: Path, output_dir: Path, sample_rate: int = 16000
    ) -> Path:
        """Extract audio from video as mono WAV at specified sample rate.

        Converts the video's audio track to a mono WAV file at the given
        sample rate (default 16kHz for WhisperX compatibility). Results are
        cached: if the output file already exists, extraction is skipped.

        Args:
            video_path: Path to the source video file.
            output_dir: Directory to store the extracted WAV file.
            sample_rate: Audio sample rate in Hz (default 16000).

        Returns:
            Path to the extracted (or cached) WAV file.

        Raises:
            AudioExtractionError: If FFmpeg exits with a non-zero code.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{video_path.stem}.wav"

        # Cache: skip extraction if output already exists
        if output_path.exists():
            return output_path

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Clean up partial output on failure
            if output_path.exists():
                output_path.unlink()
            raise AudioExtractionError(
                video_path=video_path,
                stderr=result.stderr,
            )

        return output_path
