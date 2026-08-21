"""Unit tests for the video renderer with mocked subprocess calls."""

from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from wordnap.editing.renderer import Renderer, RenderError
from wordnap.models.schemas import (
    ClipEntry,
    EDLManifest,
    GapEntry,
    RenderConfig,
)


@pytest.fixture
def config():
    """Default render configuration."""
    return RenderConfig(playback_speed=1.0)


@pytest.fixture
def renderer(config):
    """Renderer instance with default config."""
    return Renderer(config)


@pytest.fixture
def sample_clip_entry():
    """A sample ClipEntry for testing."""
    return ClipEntry(
        source_video=Path("/videos/test.mp4"),
        source_filename="test.mp4",
        word="hello",
        start_time=1.0,
        end_time=1.5,
        padded_start=0.9,
        padded_end=1.6,
        confidence=0.95,
        speaker="speaker1",
    )


@pytest.fixture
def sample_manifest(sample_clip_entry):
    """A sample EDLManifest with two clips."""
    clip2 = ClipEntry(
        source_video=Path("/videos/test2.mp4"),
        source_filename="test2.mp4",
        word="world",
        start_time=2.0,
        end_time=2.5,
        padded_start=1.9,
        padded_end=2.6,
        confidence=0.88,
        speaker="speaker1",
    )
    return EDLManifest(
        sentence="hello world",
        variation_index=0,
        clips=[sample_clip_entry, clip2],
        total_duration=1.4,
    )


class TestExtractClip:
    """Tests for Renderer.extract_clip()."""

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_success(self, mock_run, renderer):
        """extract_clip returns output_path on successful FFmpeg execution."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        assert result == Path("/tmp/clip.mp4")
        mock_run.assert_called_once()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_builds_correct_command(self, mock_run, renderer):
        """extract_clip constructs FFmpeg command with all normalized parameters."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.5,
            end=3.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(Path("/videos/test.mp4")) in cmd
        assert "-ss" in cmd
        assert "1.5" in cmd
        # -ss should come before -i (input seeking)
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx
        # -t (duration) instead of -to
        assert "-t" in cmd
        assert "-to" not in cmd
        # duration = end - start = 3.0 - 1.5 = 1.5
        t_idx = cmd.index("-t")
        assert cmd[t_idx + 1] == "1.5"
        # -t should be an input option (after -i, before -vf)
        vf_idx = cmd.index("-vf")
        assert t_idx > i_idx, "-t should be after -i"
        assert t_idx < vf_idx, "-t should be before -vf (input option)"
        # With no subtitle and speed=1.0, vf should just be scale
        assert cmd[vf_idx + 1] == "scale=1920:1080"
        assert "-r" in cmd
        assert "30.0" in cmd
        assert "-pix_fmt" in cmd
        assert "yuv420p" in cmd
        assert "-ar" in cmd
        assert "44100" in cmd
        assert "-ac" in cmd
        assert "2" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-bf" in cmd
        bf_idx = cmd.index("-bf")
        assert cmd[bf_idx + 1] == "0"
        assert "-y" in cmd
        # No speed filters should be present
        assert "-af" not in cmd
        assert "setpts=" not in cmd[vf_idx + 1]

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_uses_list_args(self, mock_run, renderer):
        """extract_clip passes command as list (no shell=True)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=0.0,
            end=1.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        # First positional arg should be a list
        cmd = mock_run.call_args[0][0]
        assert isinstance(cmd, list)
        # shell should not be passed or should be False
        kwargs = mock_run.call_args[1]
        assert kwargs.get("shell", False) is False

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_raises_render_error_on_failure(self, mock_run, renderer):
        """extract_clip raises RenderError with stderr on FFmpeg failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="Error: codec not found"
        )

        with pytest.raises(RenderError) as exc_info:
            renderer.extract_clip(
                video_path=Path("/videos/test.mp4"),
                start=1.0,
                end=2.0,
                output_path=Path("/tmp/clip.mp4"),
            )

        assert exc_info.value.stderr == "Error: codec not found"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_with_custom_config(self, mock_run):
        """extract_clip uses custom config values in FFmpeg command."""
        config = RenderConfig(
            output_resolution=(1280, 720),
            output_fps=24.0,
            pixel_format="yuv444p",
            audio_sample_rate=48000,
            audio_channels=1,
            video_codec="libx265",
            audio_codec="opus",
            playback_speed=1.0,
        )
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=0.0,
            end=1.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert "scale=1280:720" in cmd[vf_idx + 1]
        assert "24.0" in cmd
        assert "yuv444p" in cmd
        assert "48000" in cmd
        assert "1" in cmd
        assert "libx265" in cmd
        assert "opus" in cmd

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_with_subtitle_adds_drawtext(self, mock_run, renderer):
        """extract_clip with subtitle_text adds drawtext to -vf filter."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
            subtitle_text="hello",
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "scale=1920:1080" in vf_value
        assert "drawtext=" in vf_value
        assert "fontfile=" in vf_value
        assert "text='hello'" in vf_value
        assert "fontsize=48" in vf_value
        assert "fontcolor=white" in vf_value
        assert "borderw=3" in vf_value
        assert "bordercolor=black" in vf_value
        assert "x=(w-tw)/2" in vf_value
        assert "y=h-th-40" in vf_value

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_without_subtitle_no_drawtext(self, mock_run, renderer):
        """extract_clip without subtitle_text has no drawtext in -vf filter."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
            subtitle_text=None,
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert vf_value == "scale=1920:1080"
        assert "drawtext" not in vf_value

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_subtitle_disabled_in_config(self, mock_run):
        """extract_clip with subtitles_enabled=False does not add drawtext even with subtitle_text."""
        config = RenderConfig(subtitles_enabled=False, playback_speed=1.0)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
            subtitle_text="hello",
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "drawtext" not in vf_value

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_subtitle_special_char_escaping(self, mock_run, renderer):
        """extract_clip escapes special characters in subtitle text for FFmpeg drawtext."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
            subtitle_text="it's a:test",
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        # Single quotes should be replaced with unicode right quote
        assert "\u2019" in vf_value
        # Colons should be escaped as \:
        assert "\\:" in vf_value
        # Original unescaped colon should not appear in the text value
        assert "a\\:test" in vf_value

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_with_speed_config_no_speed_filters(self, mock_run):
        """extract_clip does NOT apply speed filters even when playback_speed != 1.0.
        
        Speed is applied post-concatenation via apply_speed() to avoid desync."""
        config = RenderConfig(playback_speed=0.9)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        # No setpts filter — speed is applied post-concat
        assert "setpts=" not in vf_value
        # No audio filter
        assert "-af" not in cmd
        # -t should use raw duration (1.0), not adjusted by speed
        t_idx = cmd.index("-t")
        assert cmd[t_idx + 1] == "1.0"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_speed_uses_raw_duration(self, mock_run):
        """extract_clip uses raw duration for -t even when speed != 1.0 (speed applied post-concat)."""
        config = RenderConfig(playback_speed=0.88)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=10.0,
            end=10.5,
            output_path=Path("/tmp/clip.mp4"),
        )

        cmd = mock_run.call_args[0][0]
        # duration = 0.5, should NOT be divided by speed
        t_idx = cmd.index("-t")
        t_value = float(cmd[t_idx + 1])
        assert t_value == 0.5

        # -t should be an input option (between -i and -vf)
        i_idx = cmd.index("-i")
        vf_idx = cmd.index("-vf")
        assert i_idx < t_idx < vf_idx, "-t should be an input option (after -i, before -vf)"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_speed_1_uses_original_duration(self, mock_run):
        """extract_clip with speed=1.0 uses original duration for -t."""
        config = RenderConfig(playback_speed=1.0)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=5.0,
            end=7.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        cmd = mock_run.call_args[0][0]
        t_idx = cmd.index("-t")
        t_value = float(cmd[t_idx + 1])
        # With speed=1.0, duration = 2.0
        assert t_value == 2.0
        # -t should be an input option (between -i and -vf)
        i_idx = cmd.index("-i")
        vf_idx = cmd.index("-vf")
        assert i_idx < t_idx < vf_idx

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_extract_clip_speed_1_no_filters(self, mock_run, renderer):
        """extract_clip with speed=1.0 does not add speed filters."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.extract_clip(
            video_path=Path("/videos/test.mp4"),
            start=1.0,
            end=2.0,
            output_path=Path("/tmp/clip.mp4"),
        )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "setpts=" not in vf_value
        assert "-af" not in cmd


class TestApplySpeed:
    """Tests for Renderer.apply_speed()."""

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_apply_speed_builds_correct_command(self, mock_run):
        """apply_speed constructs FFmpeg command with setpts and atempo filters."""
        config = RenderConfig(playback_speed=0.88)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.apply_speed(Path("/tmp/concat.mp4"), Path("/tmp/output.mp4"))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(Path("/tmp/concat.mp4")) in cmd
        # Video filter: setpts for speed adjustment
        assert "-vf" in cmd
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        # 1.0/0.88 ≈ 1.1363636
        assert "setpts=" in vf_value
        assert "PTS" in vf_value
        # Audio filter: atempo
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        af_value = cmd[af_idx + 1]
        assert "atempo=0.88" in af_value
        # Codecs and format
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-pix_fmt" in cmd
        assert "yuv420p" in cmd
        assert "-r" in cmd
        assert "30.0" in cmd
        assert "-ar" in cmd
        assert "44100" in cmd
        assert "-ac" in cmd
        assert "2" in cmd
        assert "-y" in cmd
        assert str(Path("/tmp/output.mp4")) in cmd

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_apply_speed_returns_output_path(self, mock_run):
        """apply_speed returns the output path on success."""
        config = RenderConfig(playback_speed=0.9)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = renderer.apply_speed(Path("/tmp/in.mp4"), Path("/tmp/out.mp4"))

        assert result == Path("/tmp/out.mp4")

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_apply_speed_raises_on_failure(self, mock_run):
        """apply_speed raises RenderError on FFmpeg failure."""
        config = RenderConfig(playback_speed=1.2)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=1, stderr="speed filter error")

        with pytest.raises(RenderError) as exc_info:
            renderer.apply_speed(Path("/tmp/in.mp4"), Path("/tmp/out.mp4"))

        assert "speed" in str(exc_info.value).lower()
        assert exc_info.value.stderr == "speed filter error"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_apply_speed_fast_forward(self, mock_run):
        """apply_speed with speed > 1.0 uses correct filter values."""
        config = RenderConfig(playback_speed=1.5)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.apply_speed(Path("/tmp/in.mp4"), Path("/tmp/out.mp4"))

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        # 1.0/1.5 = 0.6666...
        assert "setpts=" in vf_value
        af_idx = cmd.index("-af")
        af_value = cmd[af_idx + 1]
        assert "atempo=1.5" in af_value


class TestGenerateSilence:
    """Tests for Renderer.generate_silence()."""

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_generate_silence_builds_correct_command(self, mock_run, renderer):
        """generate_silence builds FFmpeg command with black video and silent audio."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.generate_silence(80.0, Path("/tmp/silence.mp4"))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        # Check for lavfi color input with black video
        assert "-f" in cmd
        assert "lavfi" in cmd
        # Duration in seconds
        assert "-t" in cmd
        assert "0.08" in cmd
        # Verify video parameters
        cmd_str = " ".join(cmd)
        assert "color=c=black:s=1920x1080:r=30.0:d=0.08" in cmd_str
        # Verify audio null source
        assert "anullsrc=r=44100:cl=stereo" in cmd_str
        # Codec options
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-pix_fmt" in cmd
        assert "yuv420p" in cmd
        assert "-y" in cmd

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_generate_silence_returns_output_path(self, mock_run, renderer):
        """generate_silence returns the output path on success."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = renderer.generate_silence(100.0, Path("/tmp/silence.mp4"))

        assert result == Path("/tmp/silence.mp4")

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_generate_silence_raises_on_failure(self, mock_run, renderer):
        """generate_silence raises RenderError on FFmpeg failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="lavfi error")

        with pytest.raises(RenderError) as exc_info:
            renderer.generate_silence(80.0, Path("/tmp/silence.mp4"))

        assert "silence" in str(exc_info.value)
        assert exc_info.value.stderr == "lavfi error"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_generate_silence_mono_config(self, mock_run):
        """generate_silence uses mono when audio_channels=1."""
        config = RenderConfig(audio_channels=1, playback_speed=1.0)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        renderer.generate_silence(50.0, Path("/tmp/silence.mp4"))

        cmd_str = " ".join(mock_run.call_args[0][0])
        assert "cl=mono" in cmd_str


class TestConcatenate:
    """Tests for Renderer.concatenate()."""

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_success(self, mock_run, renderer, tmp_path):
        """concatenate returns output_path on successful FFmpeg execution."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"
        clip1.touch()
        clip2.touch()
        output = tmp_path / "output.mp4"

        result = renderer.concatenate([clip1, clip2], output)

        assert result == output
        mock_run.assert_called_once()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_builds_correct_command(self, mock_run, renderer, tmp_path):
        """concatenate uses FFmpeg concat demuxer with full re-encoding arguments."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clip1 = tmp_path / "clip1.mp4"
        clip1.touch()
        output = tmp_path / "output.mp4"

        renderer.concatenate([clip1], output)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-f" in cmd
        assert "concat" in cmd
        assert "-safe" in cmd
        assert "0" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-bf" in cmd
        bf_idx = cmd.index("-bf")
        assert cmd[bf_idx + 1] == "0"
        assert "-pix_fmt" in cmd
        assert "yuv420p" in cmd
        assert "-r" in cmd
        assert "30.0" in cmd
        assert "-ar" in cmd
        assert "44100" in cmd
        assert "-ac" in cmd
        assert "2" in cmd
        assert "-y" in cmd

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_creates_concat_file(self, mock_run, renderer, tmp_path):
        """concatenate writes a concat file listing all clip paths."""
        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"
        clip1.touch()
        clip2.touch()
        output = tmp_path / "output.mp4"

        concat_content = []

        def capture_concat(*args, **kwargs):
            # Read the concat file referenced in the command
            cmd = args[0]
            if "-f" in cmd and "concat" in cmd:
                i_idx = cmd.index("-i")
                concat_path = Path(cmd[i_idx + 1])
                if concat_path.exists():
                    concat_content.append(concat_path.read_text())
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = capture_concat

        renderer.concatenate([clip1, clip2], output)

        assert len(concat_content) == 1
        assert f"file '{clip1}'" in concat_content[0]
        assert f"file '{clip2}'" in concat_content[0]

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_cleans_up_concat_file(self, mock_run, renderer, tmp_path):
        """concatenate removes the temporary concat file after completion."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clip1 = tmp_path / "clip1.mp4"
        clip1.touch()
        output = tmp_path / "output.mp4"

        renderer.concatenate([clip1], output)

        concat_file = tmp_path / "output_concat.txt"
        assert not concat_file.exists()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_cleans_up_on_failure(self, mock_run, renderer, tmp_path):
        """concatenate removes the concat file even when FFmpeg fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="concat error")

        clip1 = tmp_path / "clip1.mp4"
        clip1.touch()
        output = tmp_path / "output.mp4"

        with pytest.raises(RenderError):
            renderer.concatenate([clip1], output)

        concat_file = tmp_path / "output_concat.txt"
        assert not concat_file.exists()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_raises_render_error_on_failure(
        self, mock_run, renderer, tmp_path
    ):
        """concatenate raises RenderError with stderr on FFmpeg failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="segfault in demuxer")

        clip1 = tmp_path / "clip1.mp4"
        clip1.touch()
        output = tmp_path / "output.mp4"

        with pytest.raises(RenderError) as exc_info:
            renderer.concatenate([clip1], output)

        assert exc_info.value.stderr == "segfault in demuxer"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_concatenate_uses_list_args(self, mock_run, renderer, tmp_path):
        """concatenate passes command as list (no shell injection)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clip1 = tmp_path / "clip1.mp4"
        clip1.touch()
        output = tmp_path / "output.mp4"

        renderer.concatenate([clip1], output)

        cmd = mock_run.call_args[0][0]
        assert isinstance(cmd, list)


class TestRender:
    """Tests for Renderer.render()."""

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_success(self, mock_run, renderer, sample_manifest, tmp_path):
        """render returns output_path after successful extraction and concatenation."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "final.mp4"

        result = renderer.render(sample_manifest, output)

        assert result == output

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_calls_extract_for_each_clip(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render calls FFmpeg once per clip for extraction plus once for concatenation (no gaps)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "final.mp4"

        # sample_manifest has no gaps, so 2 clips + 1 concat = 3 calls
        renderer.render(sample_manifest, output)

        assert mock_run.call_count == 3

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_with_gaps_no_silence_inserted(self, mock_run, tmp_path):
        """render with gaps does NOT insert silence segments (direct cuts only)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clip1 = ClipEntry(
            source_video=Path("/videos/a.mp4"),
            source_filename="a.mp4",
            word="one",
            start_time=0.0,
            end_time=0.5,
            padded_start=0.0,
            padded_end=0.6,
            confidence=0.9,
        )
        clip2 = ClipEntry(
            source_video=Path("/videos/b.mp4"),
            source_filename="b.mp4",
            word="two",
            start_time=1.0,
            end_time=1.5,
            padded_start=0.9,
            padded_end=1.6,
            confidence=0.9,
        )
        clip3 = ClipEntry(
            source_video=Path("/videos/c.mp4"),
            source_filename="c.mp4",
            word="three",
            start_time=2.0,
            end_time=2.5,
            padded_start=1.9,
            padded_end=2.6,
            confidence=0.9,
        )
        manifest = EDLManifest(
            sentence="one two three",
            variation_index=0,
            clips=[clip1, clip2, clip3],
            gaps=[
                GapEntry(duration_ms=80.0, reason="default"),
                GapEntry(duration_ms=200.0, reason="comma"),
            ],
            total_duration=2.6,
        )

        config = RenderConfig(subtitles_enabled=False, playback_speed=1.0)
        renderer = Renderer(config)
        output = tmp_path / "final.mp4"

        renderer.render(manifest, output)

        # 3 clip extractions + 1 concatenation = 4 calls (NO silence)
        assert mock_run.call_count == 4

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_with_empty_gaps_backward_compat(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render with empty gaps list works (backward compat, no silence inserted)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "final.mp4"

        # sample_manifest has gaps=[] by default
        assert sample_manifest.gaps == []
        renderer.render(sample_manifest, output)

        # 2 clips + 1 concat = 3 calls (no silence)
        assert mock_run.call_count == 3

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_with_speed_applies_post_concat(self, mock_run, sample_manifest, tmp_path):
        """render with playback_speed != 1.0 applies speed as a post-concat pass."""
        config = RenderConfig(playback_speed=0.88, subtitles_enabled=False)
        renderer = Renderer(config)
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "final.mp4"

        renderer.render(sample_manifest, output)

        # 2 clip extractions + 1 concat + 1 speed pass = 4 calls
        assert mock_run.call_count == 4

        # Last call should be the speed pass (apply_speed)
        speed_cmd = mock_run.call_args_list[3][0][0]
        assert "-vf" in speed_cmd
        vf_idx = speed_cmd.index("-vf")
        assert "setpts=" in speed_cmd[vf_idx + 1]
        assert "-af" in speed_cmd
        af_idx = speed_cmd.index("-af")
        assert "atempo=0.88" in speed_cmd[af_idx + 1]

        # Extract clips should NOT have speed filters
        for i in range(2):
            extract_cmd = mock_run.call_args_list[i][0][0]
            vf_idx = extract_cmd.index("-vf")
            assert "setpts=" not in extract_cmd[vf_idx + 1]
            assert "-af" not in extract_cmd

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_cleans_up_temp_directory(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render removes the temporary directory after completion."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "final.mp4"

        # Track temp dirs that get created
        import tempfile

        original_mkdtemp = tempfile.mkdtemp
        created_temps = []

        def track_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_temps.append(d)
            return d

        with patch("wordnap.editing.renderer.tempfile.mkdtemp", side_effect=track_mkdtemp):
            renderer.render(sample_manifest, output)

        # Temp dir should be removed
        for temp_dir in created_temps:
            assert not Path(temp_dir).exists()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_cleans_up_on_extract_failure(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render removes temp directory even when extraction fails."""
        # First extract call fails
        mock_run.return_value = MagicMock(returncode=1, stderr="extraction error")
        output = tmp_path / "final.mp4"

        import tempfile

        original_mkdtemp = tempfile.mkdtemp
        created_temps = []

        def track_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_temps.append(d)
            return d

        with patch("wordnap.editing.renderer.tempfile.mkdtemp", side_effect=track_mkdtemp):
            with pytest.raises(RenderError):
                renderer.render(sample_manifest, output)

        # Temp dir should still be cleaned up
        for temp_dir in created_temps:
            assert not Path(temp_dir).exists()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_raises_render_error_with_clip_entry(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render attaches clip_entry to RenderError on extraction failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="bad clip")
        output = tmp_path / "final.mp4"

        with pytest.raises(RenderError) as exc_info:
            renderer.render(sample_manifest, output)

        # Should have the clip entry from the first clip that failed
        assert exc_info.value.clip_entry is not None
        assert exc_info.value.clip_entry.word == "hello"
        assert exc_info.value.stderr == "bad clip"

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_uses_padded_times(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render passes padded_start and padded_end to extract_clip."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "final.mp4"

        renderer.render(sample_manifest, output)

        # First call should use padded times from first clip
        first_call_cmd = mock_run.call_args_list[0][0][0]
        # -ss comes before -i with padded_start value
        assert "0.9" in first_call_cmd
        # -t with duration (padded_end - padded_start = 1.6 - 0.9 = 0.7)
        ss_idx = first_call_cmd.index("-ss")
        i_idx = first_call_cmd.index("-i")
        assert ss_idx < i_idx
        assert "-t" in first_call_cmd
        assert "-to" not in first_call_cmd

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_creates_output_directory(
        self, mock_run, renderer, sample_manifest, tmp_path
    ):
        """render creates output directory if it doesn't exist."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = tmp_path / "subdir" / "final.mp4"

        renderer.render(sample_manifest, output)

        assert output.parent.exists()

    @patch("wordnap.editing.renderer.subprocess.run")
    def test_render_passes_subtitle_text_when_enabled(self, mock_run, tmp_path):
        """render passes word as subtitle_text when subtitles are enabled."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clip = ClipEntry(
            source_video=Path("/videos/a.mp4"),
            source_filename="a.mp4",
            word="hello",
            start_time=0.0,
            end_time=0.5,
            padded_start=0.0,
            padded_end=0.6,
            confidence=0.9,
        )
        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[clip],
            total_duration=0.6,
        )

        config = RenderConfig(subtitles_enabled=True, playback_speed=1.0)
        renderer = Renderer(config)
        output = tmp_path / "final.mp4"

        renderer.render(manifest, output)

        # First call is clip extraction - should have drawtext
        extract_cmd = mock_run.call_args_list[0][0][0]
        vf_idx = extract_cmd.index("-vf")
        vf_value = extract_cmd[vf_idx + 1]
        assert "drawtext=" in vf_value
        assert "hello" in vf_value


class TestRenderError:
    """Tests for the RenderError exception."""

    def test_render_error_stores_stderr(self):
        """RenderError stores stderr for debugging."""
        err = RenderError("failed", stderr="some error output")
        assert err.stderr == "some error output"

    def test_render_error_stores_clip_entry(self, sample_clip_entry):
        """RenderError stores the failing clip entry."""
        err = RenderError("failed", clip_entry=sample_clip_entry)
        assert err.clip_entry == sample_clip_entry

    def test_render_error_message(self):
        """RenderError has a descriptive message."""
        err = RenderError("extraction failed for clip 3")
        assert str(err) == "extraction failed for clip 3"

    def test_render_error_defaults(self):
        """RenderError has empty defaults for optional fields."""
        err = RenderError("failed")
        assert err.stderr == ""
        assert err.clip_entry is None
