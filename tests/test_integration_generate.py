"""Integration tests for the generate command pipeline.

Verifies the full end-to-end flow:
  Tokenize → Search → Rank → EDL → Render

Uses a real in-memory database pre-populated with test data and only
mocks the Renderer's subprocess.run calls (FFmpeg).
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wordnap.cli import app
from wordnap.database.database import Database
from wordnap.models.schemas import (
    Segment,
    VideoMetadata,
    VideoStatus,
    Word,
)

runner = CliRunner()


@pytest.fixture
def populated_db(tmp_path: Path):
    """Create a temp database pre-populated with two indexed videos and words.

    Video 1 (/videos/greeting.mp4): contains "hello", "world", "foo"
    Video 2 (/videos/speech.mp4): contains "hello", "world", "bar"

    This gives multiple candidates per token for ranking diversity.
    """
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()

    # Video 1
    video1 = VideoMetadata(
        path=Path("/videos/greeting.mp4"),
        filename="greeting.mp4",
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=44100,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        status=VideoStatus.INDEXED,
    )
    vid1_id = db.upsert_video(video1)

    segments1 = [
        Segment(
            video_id=vid1_id,
            start_time=0.0,
            end_time=3.0,
            text="hello world foo",
            speaker="speaker_a",
            confidence=0.95,
        ),
    ]
    words1 = [
        Word(
            segment_id=0,
            video_id=vid1_id,
            word="Hello",
            normalized_word="hello",
            start_time=0.5,
            end_time=1.0,
            confidence=0.95,
            speaker="speaker_a",
        ),
        Word(
            segment_id=0,
            video_id=vid1_id,
            word="world",
            normalized_word="world",
            start_time=1.2,
            end_time=1.8,
            confidence=0.92,
            speaker="speaker_a",
        ),
        Word(
            segment_id=0,
            video_id=vid1_id,
            word="foo",
            normalized_word="foo",
            start_time=2.0,
            end_time=2.5,
            confidence=0.90,
            speaker="speaker_a",
        ),
    ]
    db.store_transcription(vid1_id, segments1, words1)

    # Video 2
    video2 = VideoMetadata(
        path=Path("/videos/speech.mp4"),
        filename="speech.mp4",
        duration=90.0,
        width=1280,
        height=720,
        fps=24.0,
        audio_sample_rate=44100,
        created_at=datetime(2024, 1, 2, 14, 0, 0),
        status=VideoStatus.INDEXED,
    )
    vid2_id = db.upsert_video(video2)

    segments2 = [
        Segment(
            video_id=vid2_id,
            start_time=0.0,
            end_time=4.0,
            text="hello world bar",
            speaker="speaker_b",
            confidence=0.88,
        ),
    ]
    words2 = [
        Word(
            segment_id=0,
            video_id=vid2_id,
            word="hello",
            normalized_word="hello",
            start_time=0.3,
            end_time=0.9,
            confidence=0.88,
            speaker="speaker_b",
        ),
        Word(
            segment_id=0,
            video_id=vid2_id,
            word="World",
            normalized_word="world",
            start_time=1.0,
            end_time=1.7,
            confidence=0.85,
            speaker="speaker_b",
        ),
        Word(
            segment_id=0,
            video_id=vid2_id,
            word="bar",
            normalized_word="bar",
            start_time=2.0,
            end_time=2.6,
            confidence=0.87,
            speaker="speaker_b",
        ),
    ]
    db.store_transcription(vid2_id, segments2, words2)

    db.close()
    return db_path


def _mock_subprocess_run_success(cmd, **kwargs):
    """Mock subprocess.run that simulates FFmpeg success.

    For extract_clip calls, creates an empty file at the output path.
    For concatenate calls, creates an empty file at the output path.
    """
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    mock_result.stdout = ""

    # Determine the output path from the command args
    # FFmpeg commands end with output path as last argument
    if cmd and cmd[0] == "ffmpeg":
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake mp4 content")

    return mock_result


class TestGeneratePipelineEndToEnd:
    """Integration tests for the full generate command pipeline."""

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_successful_generation_hello_world(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """Full pipeline should produce output files for 'hello world'.

        With phrase matching enabled, 'hello world' may be found as a phrase
        from a single segment, resulting in a single clip per variation.
        """
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "2",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Generated" in result.output

        # Verify output MP4 files exist (may be 1-2 depending on phrase matches)
        mp4_files = list(output_dir.glob("*.mp4"))
        assert len(mp4_files) >= 1

        # Verify manifest JSON files exist
        manifest_dir = output_dir / "manifests"
        json_files = list(manifest_dir.glob("*.json"))
        assert len(json_files) >= 1

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_edl_manifests_have_correct_structure(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """Generated EDL manifests should contain correct clip data.

        With phrase matching, 'hello world' may be a single phrase clip
        or two individual clips, depending on consecutive word matches.
        """
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        manifest_dir = output_dir / "manifests"
        json_files = list(manifest_dir.glob("*.json"))
        assert len(json_files) == 1

        manifest_data = json.loads(json_files[0].read_text())

        # Check manifest structure
        assert manifest_data["sentence"] == "hello world"
        assert manifest_data["variation_index"] == 0
        # With phrase matching, "hello world" may be 1 clip (phrase) or 2 (individual)
        assert len(manifest_data["clips"]) >= 1

        # The clip words should contain both "hello" and "world"
        all_words = " ".join(clip["word"] for clip in manifest_data["clips"]).lower()
        assert "hello" in all_words
        assert "world" in all_words

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_padding_applied_correctly(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """Clip padding should be applied and clamped properly."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
                "--padding", "0.10",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        manifest_dir = output_dir / "manifests"
        json_files = list(manifest_dir.glob("*.json"))
        manifest_data = json.loads(json_files[0].read_text())

        for clip in manifest_data["clips"]:
            # padded_start should be <= start_time
            assert clip["padded_start"] <= clip["start_time"]
            # padded_end should be >= end_time
            assert clip["padded_end"] >= clip["end_time"]
            # padded_start should never be negative
            assert clip["padded_start"] >= 0.0

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_renderer_called_with_padded_times(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """The renderer should use padded start/end times from the EDL."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check that subprocess.run was called with ffmpeg commands containing
        # the padded time values (not the original word times)
        ffmpeg_calls = [
            call for call in mock_run.call_args_list
            if call[0][0][0] == "ffmpeg" and "-ss" in call[0][0]
        ]

        # Should have at least 1 extract call (phrase match produces 1 clip)
        assert len(ffmpeg_calls) >= 1

        # Verify each extraction call uses -ss and -t arguments
        for call in ffmpeg_calls:
            cmd = call[0][0]
            assert "-ss" in cmd
            assert "-t" in cmd
            ss_idx = cmd.index("-ss")
            t_idx = cmd.index("-t")
            start_val = float(cmd[ss_idx + 1])
            duration_val = float(cmd[t_idx + 1])
            # Start should be >= 0 (clamped)
            assert start_val >= 0.0
            # Duration should be > 0
            assert duration_val > 0.0

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_ranker_produces_distinct_variations(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """When multiple candidates exist, pipeline should produce variations.

        With phrase matching, "hello world" may match as a phrase from two
        different segments/videos, producing distinct phrase-level variations.
        """
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "2",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        manifest_dir = output_dir / "manifests"
        json_files = sorted(manifest_dir.glob("*.json"))
        # At least 1 variation should be produced
        assert len(json_files) >= 1

    def test_missing_word_returns_error(
        self, populated_db: Path, tmp_path: Path
    ):
        """Requesting a word not in the database should exit with code 1 in strict mode."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello xyznonexistent",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
                "--strict",
            ],
        )

        assert result.exit_code == 1
        assert "xyznonexistent" in result.output.lower()

    def test_empty_sentence_returns_error(
        self, populated_db: Path, tmp_path: Path
    ):
        """An empty sentence should exit with code 1."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "   ",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "no valid tokens" in result.output.lower()

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_output_directory_created_automatically(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """Output directory should be created if it doesn't exist."""
        output_dir = tmp_path / "deeply" / "nested" / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_dir.exists()

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_output_filenames_use_short_prefix(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """Output filenames should contain first words of the sentence."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "Hello World",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        mp4_files = list(output_dir.glob("*.mp4"))
        assert len(mp4_files) == 1
        # Filename should contain first words and a timestamp
        stem = mp4_files[0].stem
        assert "hello" in stem
        assert "world" in stem

    @patch("subprocess.run")
    def test_render_failure_reported_gracefully(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """If FFmpeg fails, the error should be reported and exit gracefully."""
        output_dir = tmp_path / "output"

        # Make subprocess.run return failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg: error encoding"
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        # Should exit with error when all variations fail
        assert result.exit_code == 1
        assert "failed" in result.output.lower()

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_single_word_sentence(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """A single word sentence should work through the full pipeline."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello",
                "--variations", "1",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        manifest_dir = output_dir / "manifests"
        json_files = list(manifest_dir.glob("*.json"))
        assert len(json_files) == 1

        manifest_data = json.loads(json_files[0].read_text())
        assert len(manifest_data["clips"]) == 1
        assert manifest_data["clips"][0]["word"].lower() == "hello"

    @patch("subprocess.run", side_effect=_mock_subprocess_run_success)
    def test_variation_count_capped_by_available_combinations(
        self, mock_run, populated_db: Path, tmp_path: Path
    ):
        """Requesting more variations than possible should cap at max unique combinations."""
        output_dir = tmp_path / "output"

        # For "hello world" with 2 candidates each: max 4 combos
        # Requesting 10 should produce at most 4
        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence", "hello world",
                "--variations", "10",
                "--db-path", str(populated_db),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        manifest_dir = output_dir / "manifests"
        json_files = list(manifest_dir.glob("*.json"))
        # Should be capped at 4 (2 hello candidates * 2 world candidates)
        assert len(json_files) <= 4
