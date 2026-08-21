"""Video renderer - clip extraction, normalization, and concatenation via FFmpeg."""

import os
import platform
import subprocess
import tempfile
from pathlib import Path

from wordnap.models.schemas import ClipEntry, EDLManifest, RenderConfig

# Bundled font directory (project-relative, no system font dependency)
_FONT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "fonts"


def _get_font_path() -> str:
    """Get a reliable font file path for FFmpeg drawtext filter.

    Prefers a bundled font from assets/fonts/ (project-relative) so that
    rendering works without fontconfig or system-level font access.

    Returns the path with FFmpeg drawtext escaping applied (colons escaped
    as ``\\:``, backslashes converted to forward slashes).
    """
    # First, try bundled font in assets/fonts/
    if _FONT_DIR.exists():
        for f in _FONT_DIR.iterdir():
            if f.suffix.lower() in (".ttf", ".otf"):
                return str(f).replace("\\", "/").replace(":", "\\:")

    # Fallback to system fonts
    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        arial = os.path.join(windir, "Fonts", "arial.ttf")
        if os.path.exists(arial):
            # FFmpeg drawtext requires forward slashes and escaped colons
            return arial.replace("\\", "/").replace(":", "\\:")
        # Fallback with standard path
        return "C\\:/Windows/Fonts/arial.ttf"
    else:
        # Linux/Mac font candidates
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # Best guess fallback
        return candidates[0]


class RenderError(Exception):
    """Raised when FFmpeg fails during clip extraction or concatenation."""

    def __init__(
        self, message: str, stderr: str = "", clip_entry: ClipEntry | None = None
    ) -> None:
        self.stderr = stderr
        self.clip_entry = clip_entry
        super().__init__(message)


class Renderer:
    """Extracts clips and concatenates them into final MP4 via FFmpeg."""

    def __init__(self, config: RenderConfig) -> None:
        """Initialize renderer with output configuration.

        Args:
            config: Render configuration specifying output parameters.
        """
        self.config = config

    def generate_silence(self, duration_ms: float, output_path: Path) -> Path:
        """Generate a silence audio+video segment using FFmpeg.

        Creates a short clip with black video and silent audio.
        Note: This method is retained for backward compatibility but is no
        longer called by render(). Gaps are no longer rendered as separate
        segments to avoid black frames between clips.

        Args:
            duration_ms: Duration of silence in milliseconds.
            output_path: Path for the output file.

        Returns:
            The output_path on success.

        Raises:
            RenderError: If FFmpeg fails.
        """
        duration_s = duration_ms / 1000.0
        width, height = self.config.output_resolution
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={self.config.output_fps}:d={duration_s}",
            "-f", "lavfi",
            "-i", f"anullsrc=r={self.config.audio_sample_rate}:cl={'stereo' if self.config.audio_channels == 2 else 'mono'}",
            "-t", str(duration_s),
            "-c:v", self.config.video_codec,
            "-c:a", self.config.audio_codec,
            "-pix_fmt", self.config.pixel_format,
            "-y",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RenderError(
                message=f"FFmpeg failed to generate silence ({duration_ms}ms)",
                stderr=result.stderr,
            )
        return output_path

    def extract_clip(
        self,
        video_path: Path,
        start: float,
        end: float,
        output_path: Path,
        subtitle_text: str | None = None,
    ) -> Path:
        """Extract a single clip with normalized parameters and optional subtitle overlay.

        Uses FFmpeg to trim the source video and normalize resolution, fps,
        pixel format, sample rate, channels, and codecs. When subtitle_text is
        provided and subtitles are enabled, applies a drawtext filter.

        Speed adjustment is NOT applied here — it is applied as a single
        post-concatenation pass via apply_speed() to avoid audio/video desync.

        Args:
            video_path: Path to the source video file.
            start: Start time in seconds.
            end: End time in seconds.
            output_path: Path for the extracted clip output.
            subtitle_text: Optional text to overlay as subtitle.

        Returns:
            The output_path on success.

        Raises:
            RenderError: If FFmpeg exits with a non-zero code.
        """
        width, height = self.config.output_resolution

        # Build video filter chain
        vf_filters = [f"scale={width}:{height}"]

        if subtitle_text and self.config.subtitles_enabled:
            # Escape special characters for FFmpeg drawtext
            # For subprocess list args: colons need \: and single quotes
            # are replaced with unicode right quote to avoid escaping issues
            escaped = subtitle_text.replace(":", "\\:").replace("'", "\u2019")
            font_path = _get_font_path()
            drawtext = (
                f"drawtext=fontfile='{font_path}'"
                f":text='{escaped}'"
                f":fontsize=48"
                f":fontcolor=white"
                f":borderw=3"
                f":bordercolor=black"
                f":x=(w-tw)/2"
                f":y=h-th-40"
            )
            vf_filters.append(drawtext)

        vf_string = ",".join(vf_filters)

        duration = end - start

        cmd = [
            "ffmpeg",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-vf",
            vf_string,
            "-r",
            str(self.config.output_fps),
            "-pix_fmt",
            self.config.pixel_format,
            "-ar",
            str(self.config.audio_sample_rate),
            "-ac",
            str(self.config.audio_channels),
            "-c:v",
            self.config.video_codec,
            "-c:a",
            self.config.audio_codec,
            "-bf", "0",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RenderError(
                message=f"FFmpeg failed to extract clip from '{video_path}' "
                f"({start:.3f}s - {end:.3f}s)",
                stderr=result.stderr,
            )

        return output_path

    def apply_speed(self, input_path: Path, output_path: Path) -> Path:
        """Apply playback speed adjustment to a video file.

        Uses setpts for video and atempo for audio in a single clean pass.
        This is applied post-concatenation to avoid audio/video desync that
        occurs when speed filters are applied per-clip before concat.

        Args:
            input_path: Path to the input video.
            output_path: Path for the speed-adjusted output.

        Returns:
            output_path on success.

        Raises:
            RenderError: If FFmpeg fails.
        """
        speed = self.config.playback_speed

        vf = f"setpts={1.0/speed}*PTS"
        af = f"atempo={speed}"

        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-af",
            af,
            "-c:v",
            self.config.video_codec,
            "-c:a",
            self.config.audio_codec,
            "-pix_fmt",
            self.config.pixel_format,
            "-r",
            str(self.config.output_fps),
            "-ar",
            str(self.config.audio_sample_rate),
            "-ac",
            str(self.config.audio_channels),
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RenderError(
                message=f"FFmpeg failed to apply speed ({speed}x)",
                stderr=result.stderr,
            )
        return output_path

    def concatenate(self, clips: list[Path], output_path: Path) -> Path:
        """Concatenate clips into a single MP4 using FFmpeg concat demuxer with re-encoding.

        Creates a temporary concat file listing all clips, then uses FFmpeg's
        concat demuxer to join them with full re-encoding. Re-encoding strips
        AAC encoder priming silence and produces clean timestamps, eliminating
        accumulated micro-gaps across many clips.

        Args:
            clips: Ordered list of clip file paths to concatenate.
            output_path: Path for the final concatenated output.

        Returns:
            The output_path on success.

        Raises:
            RenderError: If FFmpeg exits with a non-zero code.
        """
        concat_file = output_path.parent / f"{output_path.stem}_concat.txt"

        try:
            # Write concat file listing all clips
            with open(concat_file, "w", encoding="utf-8") as f:
                for clip_path in clips:
                    f.write(f"file '{clip_path}'\n")

            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", self.config.video_codec,
                "-c:a", self.config.audio_codec,
                "-bf", "0",
                "-pix_fmt", self.config.pixel_format,
                "-r", str(self.config.output_fps),
                "-ar", str(self.config.audio_sample_rate),
                "-ac", str(self.config.audio_channels),
                "-y", str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RenderError(
                    message=f"FFmpeg failed to concatenate {len(clips)} clips",
                    stderr=result.stderr,
                )
        finally:
            # Always clean up the concat file
            if concat_file.exists():
                concat_file.unlink()

        return output_path

    def render(self, manifest: EDLManifest, output_path: Path) -> Path:
        """Render EDL manifest into final MP4 by direct-cutting clips.

        Orchestrates the full render pipeline: extract each clip from its
        source video, concatenate all segments directly (no silence gaps
        to avoid black frames), and apply speed adjustment as a final pass
        if playback_speed != 1.0.

        Speed is applied post-concatenation to guarantee perfect audio/video
        sync — per-clip speed filters cause desync when concatenated.

        Args:
            manifest: EDL manifest describing clips to extract.
            output_path: Path for the final rendered output file.

        Returns:
            The output_path on success.

        Raises:
            RenderError: If any FFmpeg operation fails.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="wordnap_render_"))

        try:
            segments: list[Path] = []

            for i, clip_entry in enumerate(manifest.clips):
                clip_output = temp_dir / f"clip_{i:04d}.mp4"

                # Pass subtitle text if subtitles are enabled
                subtitle_text = clip_entry.word if self.config.subtitles_enabled else None

                try:
                    self.extract_clip(
                        video_path=clip_entry.source_video,
                        start=clip_entry.padded_start,
                        end=clip_entry.padded_end,
                        output_path=clip_output,
                        subtitle_text=subtitle_text,
                    )
                except RenderError as e:
                    # Attach clip_entry context to the error
                    raise RenderError(
                        message=e.args[0],
                        stderr=e.stderr,
                        clip_entry=clip_entry,
                    ) from e

                segments.append(clip_output)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Concatenate all clips directly (no silence gaps)
            if self.config.playback_speed != 1.0:
                # Concat to temp file, then apply speed as a single clean pass
                concat_output = temp_dir / "concat_raw.mp4"
                self.concatenate(segments, concat_output)
                self.apply_speed(concat_output, output_path)
            else:
                # Direct concat to output
                self.concatenate(segments, output_path)

        finally:
            # Clean up temp directory and all intermediate files
            for f in temp_dir.iterdir():
                f.unlink()
            temp_dir.rmdir()

        return output_path
