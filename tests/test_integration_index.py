"""Integration tests for the indexing pipeline end-to-end.

Tests the full flow: Scanner → AudioExtractor → Transcriber → Database
via the CLI `index` command with mocked FFmpeg and WhisperX.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wordnap.cli import app
from wordnap.database.database import Database
from wordnap.models.schemas import (
    Segment,
    TranscriptionResult,
    VideoMetadata,
    VideoStatus,
    Word,
)

runner = CliRunner()


def _create_fake_videos(tmp_path: Path, names: list[str]) -> Path:
    """Create fake video files in a temp directory.

    Args:
        tmp_path: Pytest tmp_path fixture directory.
        names: List of filenames to create (with extensions).

    Returns:
        The directory containing the fake video files.
    """
    video_dir = tmp_path / "library"
    video_dir.mkdir()
    for name in names:
        (video_dir / name).write_bytes(b"\x00" * 100)
    return video_dir


def _make_video_metadata(path: Path) -> VideoMetadata:
    """Create a VideoMetadata for a given video path."""
    return VideoMetadata(
        path=path,
        filename=path.name,
        duration=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=44100,
        status=VideoStatus.PENDING,
    )


def _make_transcription_result() -> TranscriptionResult:
    """Create a mock transcription result with 2 segments and 3 words."""
    segments = [
        Segment(
            video_id=0,
            start_time=0.0,
            end_time=2.0,
            text="hello world",
            speaker="speaker_1",
            confidence=0.95,
        ),
        Segment(
            video_id=0,
            start_time=2.5,
            end_time=4.0,
            text="testing now",
            speaker="speaker_1",
            confidence=0.88,
        ),
    ]
    words = [
        Word(
            segment_id=0,
            video_id=0,
            word="Hello",
            normalized_word="hello",
            start_time=0.0,
            end_time=0.5,
            confidence=0.95,
            speaker="speaker_1",
        ),
        Word(
            segment_id=0,
            video_id=0,
            word="world",
            normalized_word="world",
            start_time=0.6,
            end_time=1.2,
            confidence=0.92,
            speaker="speaker_1",
        ),
        Word(
            segment_id=1,
            video_id=0,
            word="testing",
            normalized_word="testing",
            start_time=2.5,
            end_time=3.2,
            confidence=0.88,
            speaker="speaker_1",
        ),
    ]
    return TranscriptionResult(segments=segments, words=words)


class TestIndexPipelineEndToEnd:
    """Test the full indexing pipeline via CLI with mocked externals."""

    def test_indexes_videos_successfully(self, tmp_path):
        """The index command should scan, extract, transcribe, and store."""
        video_dir = _create_fake_videos(tmp_path, ["video1.mp4", "video2.mkv"])
        db_path = tmp_path / "test.db"

        video1_path = video_dir / "video1.mp4"
        video2_path = video_dir / "video2.mkv"

        # Mock scanner to return metadata for both videos
        mock_scan_results = [
            _make_video_metadata(video1_path),
            _make_video_metadata(video2_path),
        ]

        # Mock transcriber to return words for each video
        transcription = _make_transcription_result()

        with (
            patch(
                "wordnap.cli.Scanner"
            ) as MockScanner,
            patch(
                "wordnap.cli.AudioExtractor"
            ) as MockAudioExtractor,
            patch(
                "wordnap.cli.Transcriber"
            ) as MockTranscriber,
        ):
            # Set up mocks
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.return_value = transcription

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0, f"CLI error: {result.output}"
        assert "Indexed 2 videos" in result.output
        assert "stored 6 words" in result.output

        # Verify database state
        db = Database(db_path)
        db.initialize()

        # Both videos should be INDEXED
        video1 = db.get_video(1)
        video2 = db.get_video(2)
        assert video1 is not None
        assert video1.status == VideoStatus.INDEXED
        assert video2 is not None
        assert video2.status == VideoStatus.INDEXED

        # Words should be searchable
        hello_words = db.find_words("hello")
        assert len(hello_words) == 2  # One from each video

        world_words = db.find_words("world")
        assert len(world_words) == 2

        testing_words = db.find_words("testing")
        assert len(testing_words) == 2

        db.close()

    def test_segments_and_words_stored_correctly(self, tmp_path):
        """Segments and words should be stored with correct references."""
        video_dir = _create_fake_videos(tmp_path, ["video1.mp4"])
        db_path = tmp_path / "test.db"

        video_path = video_dir / "video1.mp4"
        mock_scan_results = [_make_video_metadata(video_path)]
        transcription = _make_transcription_result()

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.return_value = transcription

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0, f"CLI error: {result.output}"

        # Verify segment-word relationship
        db = Database(db_path)
        db.initialize()
        conn = db.connection

        # Get segments for this video
        segments = conn.execute(
            "SELECT * FROM segments WHERE video_id = 1 ORDER BY start_time"
        ).fetchall()
        assert len(segments) == 2
        assert segments[0]["text"] == "hello world"
        assert segments[1]["text"] == "testing now"

        seg1_id = segments[0]["id"]
        seg2_id = segments[1]["id"]

        # Get words and verify segment_id mapping
        words = conn.execute(
            "SELECT * FROM words WHERE video_id = 1 ORDER BY start_time"
        ).fetchall()
        assert len(words) == 3

        # "Hello" and "world" should reference the first segment
        assert words[0]["word"] == "Hello"
        assert words[0]["segment_id"] == seg1_id
        assert words[1]["word"] == "world"
        assert words[1]["segment_id"] == seg1_id

        # "testing" should reference the second segment
        assert words[2]["word"] == "testing"
        assert words[2]["segment_id"] == seg2_id

        db.close()

    def test_skips_already_indexed_on_rerun(self, tmp_path):
        """Already-indexed videos should be skipped on re-run."""
        video_dir = _create_fake_videos(tmp_path, ["video1.mp4"])
        db_path = tmp_path / "test.db"

        video_path = video_dir / "video1.mp4"
        mock_scan_results = [_make_video_metadata(video_path)]
        transcription = _make_transcription_result()

        # First run - indexes the video
        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.return_value = transcription

            result1 = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result1.exit_code == 0
        assert "Indexed 1 videos" in result1.output

        # Second run - scanner should receive the indexed path in its init
        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            # Scanner returns empty (simulating already-indexed filtering)
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = []

            result2 = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result2.exit_code == 0
        assert "No new video files found" in result2.output

    def test_failed_transcription_marks_video_as_failed(self, tmp_path):
        """Videos that fail transcription should be marked as FAILED."""
        video_dir = _create_fake_videos(tmp_path, ["good.mp4", "bad.mp4"])
        db_path = tmp_path / "test.db"

        good_path = video_dir / "good.mp4"
        bad_path = video_dir / "bad.mp4"

        mock_scan_results = [
            _make_video_metadata(good_path),
            _make_video_metadata(bad_path),
        ]
        transcription = _make_transcription_result()

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            # Transcriber succeeds for first, fails for second
            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.side_effect = [
                transcription,
                RuntimeError("WhisperX alignment failed"),
            ]

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0
        assert "Indexed 1 videos" in result.output
        assert "Failed" in result.output

        # Verify database state
        db = Database(db_path)
        db.initialize()

        good_video = db.get_video(1)
        bad_video = db.get_video(2)

        assert good_video is not None
        assert good_video.status == VideoStatus.INDEXED

        assert bad_video is not None
        assert bad_video.status == VideoStatus.FAILED

        # Good video's words should be stored
        hello_words = db.find_words("hello")
        assert len(hello_words) == 1

        db.close()

    def test_failed_audio_extraction_marks_video_as_failed(self, tmp_path):
        """Videos that fail audio extraction should be marked as FAILED."""
        video_dir = _create_fake_videos(tmp_path, ["broken.mp4"])
        db_path = tmp_path / "test.db"

        broken_path = video_dir / "broken.mp4"
        mock_scan_results = [_make_video_metadata(broken_path)]

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            # Audio extraction fails
            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.side_effect = RuntimeError(
                "FFmpeg failed"
            )

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0
        assert "Failed" in result.output
        assert "Indexed 0 videos" in result.output

        # Video should be marked as FAILED
        db = Database(db_path)
        db.initialize()
        video = db.get_video(1)
        assert video is not None
        assert video.status == VideoStatus.FAILED
        db.close()

    def test_no_videos_found(self, tmp_path):
        """When no videos are found, the command should report that."""
        video_dir = _create_fake_videos(tmp_path, [])
        db_path = tmp_path / "test.db"

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor"),
            patch("wordnap.cli.Transcriber"),
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = []

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0
        assert "No new video files found" in result.output

    def test_video_ids_set_on_segments_and_words(self, tmp_path):
        """The CLI should set video_id on segments and words before storage."""
        video_dir = _create_fake_videos(tmp_path, ["video1.mp4"])
        db_path = tmp_path / "test.db"

        video_path = video_dir / "video1.mp4"
        mock_scan_results = [_make_video_metadata(video_path)]
        transcription = _make_transcription_result()

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.return_value = transcription

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0

        # Verify all words have correct video_id
        db = Database(db_path)
        db.initialize()
        conn = db.connection

        words = conn.execute("SELECT * FROM words").fetchall()
        for word_row in words:
            assert word_row["video_id"] == 1

        segments = conn.execute("SELECT * FROM segments").fetchall()
        for seg_row in segments:
            assert seg_row["video_id"] == 1

        db.close()

    def test_word_normalization_during_indexing(self, tmp_path):
        """Words should be normalized during the indexing pipeline."""
        video_dir = _create_fake_videos(tmp_path, ["video1.mp4"])
        db_path = tmp_path / "test.db"

        video_path = video_dir / "video1.mp4"
        mock_scan_results = [_make_video_metadata(video_path)]

        # Create transcription with words that need normalization
        segments = [
            Segment(
                video_id=0,
                start_time=0.0,
                end_time=3.0,
                text="Hello, World!",
                speaker="speaker_1",
                confidence=0.95,
            ),
        ]
        words = [
            Word(
                segment_id=0,
                video_id=0,
                word="Hello,",
                normalized_word="hello",  # Will be overwritten by CLI
                start_time=0.0,
                end_time=0.5,
                confidence=0.95,
                speaker="speaker_1",
            ),
            Word(
                segment_id=0,
                video_id=0,
                word="World!",
                normalized_word="world",  # Will be overwritten by CLI
                start_time=0.6,
                end_time=1.2,
                confidence=0.92,
                speaker="speaker_1",
            ),
        ]
        transcription = TranscriptionResult(segments=segments, words=words)

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.return_value = transcription

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0

        # Words should be findable by normalized form
        db = Database(db_path)
        db.initialize()

        hello_words = db.find_words("hello")
        assert len(hello_words) == 1
        assert hello_words[0].word == "Hello,"  # Raw form preserved
        assert hello_words[0].normalized_word == "hello"

        world_words = db.find_words("world")
        assert len(world_words) == 1
        assert world_words[0].word == "World!"  # Raw form preserved
        assert world_words[0].normalized_word == "world"

        db.close()

    def test_summary_statistics_reported(self, tmp_path):
        """The CLI should report summary statistics after indexing."""
        video_dir = _create_fake_videos(tmp_path, ["v1.mp4", "v2.mp4", "v3.mp4"])
        db_path = tmp_path / "test.db"

        mock_scan_results = [
            _make_video_metadata(video_dir / "v1.mp4"),
            _make_video_metadata(video_dir / "v2.mp4"),
            _make_video_metadata(video_dir / "v3.mp4"),
        ]
        transcription = _make_transcription_result()

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.return_value = transcription

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0
        assert "Indexed 3 videos" in result.output
        assert "stored 9 words" in result.output

    def test_mixed_success_and_failure_continues(self, tmp_path):
        """Pipeline should continue after per-video failures and report summary."""
        video_dir = _create_fake_videos(
            tmp_path, ["v1.mp4", "v2.mp4", "v3.mp4"]
        )
        db_path = tmp_path / "test.db"

        mock_scan_results = [
            _make_video_metadata(video_dir / "v1.mp4"),
            _make_video_metadata(video_dir / "v2.mp4"),
            _make_video_metadata(video_dir / "v3.mp4"),
        ]
        transcription = _make_transcription_result()

        with (
            patch("wordnap.cli.Scanner") as MockScanner,
            patch("wordnap.cli.AudioExtractor") as MockAudioExtractor,
            patch("wordnap.cli.Transcriber") as MockTranscriber,
        ):
            scanner_instance = MockScanner.return_value
            scanner_instance.scan_directory.return_value = mock_scan_results

            extractor_instance = MockAudioExtractor.return_value
            extractor_instance.extract_audio.return_value = tmp_path / "audio.wav"

            # First succeeds, second fails, third succeeds
            transcriber_instance = MockTranscriber.return_value
            transcriber_instance.transcribe.side_effect = [
                transcription,
                RuntimeError("Transcription failed"),
                transcription,
            ]

            result = runner.invoke(
                app, ["index", str(video_dir), "--db-path", str(db_path)]
            )

        assert result.exit_code == 0
        assert "Indexed 2 videos" in result.output
        assert "stored 6 words" in result.output

        # Verify correct status on each video
        db = Database(db_path)
        db.initialize()

        v1 = db.get_video(1)
        v2 = db.get_video(2)
        v3 = db.get_video(3)

        assert v1.status == VideoStatus.INDEXED
        assert v2.status == VideoStatus.FAILED
        assert v3.status == VideoStatus.INDEXED

        db.close()
