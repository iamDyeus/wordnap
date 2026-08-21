"""Unit tests for CLI interface using Typer's CliRunner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wordnap.cli import _slugify, app
from wordnap.editing.renderer import RenderError
from wordnap.models.schemas import (
    ClipEntry,
    EDLManifest,
    RankingConfig,
    VideoMetadata,
    VideoStatus,
    Word,
    WordCandidate,
)
from wordnap.search.candidate import WordNotFoundError

runner = CliRunner()


class TestSlugify:
    """Tests for the _slugify helper function."""

    def test_basic_sentence(self):
        assert _slugify("Hello World") == "hello-world"

    def test_punctuation_stripped(self):
        assert _slugify("We need to leave this place.") == "we-need-to-leave-this-place"

    def test_multiple_spaces(self):
        assert _slugify("hello   world") == "hello-world"

    def test_empty_string(self):
        assert _slugify("") == ""

    def test_special_characters(self):
        assert _slugify("It's a test!") == "its-a-test"

    def test_long_text_truncated(self):
        result = _slugify("a " * 100)
        assert len(result) <= 80


class TestIndexCommand:
    """Tests for the index CLI command."""

    @patch("wordnap.cli.Transcriber")
    @patch("wordnap.cli.AudioExtractor")
    @patch("wordnap.cli.Scanner")
    @patch("wordnap.cli.Database")
    def test_index_no_new_videos(
        self, mock_db_cls, mock_scanner_cls, mock_audio_cls, mock_transcriber_cls
    ):
        """When no new videos are found, display appropriate message."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.connection.execute.return_value.fetchall.return_value = []

        mock_scanner = MagicMock()
        mock_scanner_cls.return_value = mock_scanner
        mock_scanner.scan_directory.return_value = []

        result = runner.invoke(app, ["index", "./library"])

        assert result.exit_code == 0
        assert "No new video files found" in result.output

    @patch("wordnap.cli.Transcriber")
    @patch("wordnap.cli.AudioExtractor")
    @patch("wordnap.cli.Scanner")
    @patch("wordnap.cli.Database")
    def test_index_success_summary(
        self, mock_db_cls, mock_scanner_cls, mock_audio_cls, mock_transcriber_cls
    ):
        """Displays correct summary after successful indexing."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.connection.execute.return_value.fetchall.return_value = []
        mock_db.upsert_video.return_value = 1

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
            status=VideoStatus.PENDING,
        )

        mock_scanner = MagicMock()
        mock_scanner_cls.return_value = mock_scanner
        mock_scanner.scan_directory.return_value = [video]

        mock_audio = MagicMock()
        mock_audio_cls.return_value = mock_audio
        mock_audio.extract_audio.return_value = Path("/data/audio/test.wav")

        # Transcriber returns a result with words
        from wordnap.models.schemas import Segment, TranscriptionResult

        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            segments=[
                Segment(
                    video_id=0,
                    start_time=0.0,
                    end_time=1.0,
                    text="hello world",
                    confidence=0.95,
                )
            ],
            words=[
                Word(
                    segment_id=0,
                    video_id=0,
                    word="hello",
                    normalized_word="hello",
                    start_time=0.0,
                    end_time=0.5,
                    confidence=0.95,
                ),
                Word(
                    segment_id=0,
                    video_id=0,
                    word="world",
                    normalized_word="world",
                    start_time=0.5,
                    end_time=1.0,
                    confidence=0.90,
                ),
            ],
        )

        result = runner.invoke(app, ["index", "./library"])

        assert result.exit_code == 0
        assert "Indexed 1 videos, stored 2 words" in result.output

    @patch("wordnap.cli.Transcriber")
    @patch("wordnap.cli.AudioExtractor")
    @patch("wordnap.cli.Scanner")
    @patch("wordnap.cli.Database")
    def test_index_handles_transcription_failure(
        self, mock_db_cls, mock_scanner_cls, mock_audio_cls, mock_transcriber_cls
    ):
        """Failed transcription marks video as FAILED and continues."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.connection.execute.return_value.fetchall.return_value = []
        mock_db.upsert_video.return_value = 1

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
            status=VideoStatus.PENDING,
        )

        mock_scanner = MagicMock()
        mock_scanner_cls.return_value = mock_scanner
        mock_scanner.scan_directory.return_value = [video]

        mock_audio = MagicMock()
        mock_audio_cls.return_value = mock_audio
        mock_audio.extract_audio.return_value = Path("/data/audio/test.wav")

        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber
        mock_transcriber.transcribe.side_effect = RuntimeError("WhisperX failed")

        result = runner.invoke(app, ["index", "./library"])

        assert result.exit_code == 0
        assert "Indexed 0 videos, stored 0 words" in result.output
        mock_db.update_video_status.assert_called_with(1, VideoStatus.FAILED.value)


class TestGenerateCommand:
    """Tests for the generate CLI command."""

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_missing_words_strict_error(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
    ):
        """When words are not found in strict mode, display error and exit 1."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.side_effect = WordNotFoundError(
            ["xyzzy", "quux"]
        )

        result = runner.invoke(
            app, ["generate", "--sentence", "xyzzy quux hello", "--strict"]
        )

        assert result.exit_code == 1
        assert "Words not found: xyzzy, quux" in result.output

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_empty_sentence_error(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
    ):
        """Empty sentence produces error and exits with code 1."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        result = runner.invoke(app, ["generate", "--sentence", "   "])

        assert result.exit_code == 1
        assert "no valid tokens" in result.output

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_success_displays_paths(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """Successful generation displays output file paths."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        # Create mock word candidate
        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.return_value = ({"hello": [candidate]}, [])

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        # EDL generator returns a manifest
        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        mock_renderer.render.return_value = tmp_path / "hello_v000.mp4"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello",
                "--variations",
                "1",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "Generated 1 variation(s)" in result.output

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_partial_render_failure(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """Partial render failures are reported while successful ones proceed."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.return_value = ({"hello": [candidate]}, [])

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        # Render succeeds
        mock_renderer.render.return_value = tmp_path / "hello_v000.mp4"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello",
                "--variations",
                "2",
                "--output-dir",
                str(tmp_path),
            ],
        )

        # With only 1 candidate, dedup produces 1 unique variation
        assert result.exit_code == 0
        assert "Generated 1 variation(s)" in result.output

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_all_renders_fail_exits_1(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """If all variations fail to render, exit with code 1."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.return_value = ({"hello": [candidate]}, [])

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        mock_renderer.render.side_effect = RenderError("FFmpeg failed")

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello",
                "--variations",
                "1",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 1
        assert "All variations failed to render" in result.output

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_best_effort_skips_missing_words(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """Best-effort mode (default) warns about missing words but continues."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        # "hello" found, "xyzzy" missing
        mock_search.find_candidates_batch.return_value = (
            {"hello": [candidate]},
            ["xyzzy"],
        )

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        manifest = EDLManifest(
            sentence="hello xyzzy",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        mock_renderer.render.return_value = tmp_path / "hello-xyzzy_v000.mp4"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello xyzzy",
                "--variations",
                "1",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "Skipping words not found in library: xyzzy" in result.output
        assert "Generated 1 variation(s)" in result.output

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_gap_flag_passed_to_edl(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """The --gap flag value is passed to EDLGenerator."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.return_value = ({"hello": [candidate]}, [])

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        mock_renderer.render.return_value = tmp_path / "hello_v000.mp4"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello",
                "--variations",
                "1",
                "--output-dir",
                str(tmp_path),
                "--gap",
                "120.0",
            ],
        )

        assert result.exit_code == 0
        # Verify EDLGenerator was constructed with the custom gap
        mock_edl_cls.assert_called_once_with(
            clip_padding=0.05,
            default_gap_ms=120.0,
            punctuation_pause_enabled=True,
        )

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_no_subtitles_flag(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """The --no-subtitles flag disables subtitles in RenderConfig."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.return_value = ({"hello": [candidate]}, [])

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        mock_renderer.render.return_value = tmp_path / "hello_v000.mp4"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello",
                "--variations",
                "1",
                "--output-dir",
                str(tmp_path),
                "--no-subtitles",
            ],
        )

        assert result.exit_code == 0
        # Verify Renderer was constructed with subtitles disabled
        render_config = mock_renderer_cls.call_args[0][0]
        assert render_config.subtitles_enabled is False

    @patch("wordnap.cli.PhraseSearchEngine")
    @patch("wordnap.cli.Renderer")
    @patch("wordnap.cli.EDLGenerator")
    @patch("wordnap.cli.Ranker")
    @patch("wordnap.cli.SearchEngine")
    @patch("wordnap.cli.Database")
    def test_generate_no_punctuation_pause_flag(
        self,
        mock_db_cls,
        mock_search_cls,
        mock_ranker_cls,
        mock_edl_cls,
        mock_renderer_cls,
        mock_phrase_cls,
        tmp_path,
    ):
        """The --no-punctuation-pause flag disables punctuation timing."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_phrase = MagicMock()
        mock_phrase_cls.return_value = mock_phrase
        mock_phrase.find_phrases.return_value = ({}, set())

        video = VideoMetadata(
            path=Path("/video/test.mp4"),
            filename="test.mp4",
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            audio_sample_rate=44100,
        )
        word = Word(
            segment_id=1,
            video_id=1,
            word="hello",
            normalized_word="hello",
            start_time=1.0,
            end_time=1.5,
            confidence=0.95,
        )
        candidate = WordCandidate(word=word, video=video, duration=0.5, score=0.9)

        mock_search = MagicMock()
        mock_search_cls.return_value = mock_search
        mock_search.find_candidates_batch.return_value = ({"hello": [candidate]}, [])

        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker
        mock_ranker.filter_candidates.side_effect = lambda candidates: candidates
        mock_ranker.score_candidate.return_value = 0.9

        manifest = EDLManifest(
            sentence="hello",
            variation_index=0,
            clips=[
                ClipEntry(
                    source_video=Path("/video/test.mp4"),
                    source_filename="test.mp4",
                    word="hello",
                    start_time=1.0,
                    end_time=1.5,
                    padded_start=0.9,
                    padded_end=1.6,
                    confidence=0.95,
                )
            ],
            total_duration=0.7,
        )

        mock_edl = MagicMock()
        mock_edl_cls.return_value = mock_edl
        mock_edl.generate.return_value = manifest

        mock_renderer = MagicMock()
        mock_renderer_cls.return_value = mock_renderer
        mock_renderer.render.return_value = tmp_path / "hello_v000.mp4"

        result = runner.invoke(
            app,
            [
                "generate",
                "--sentence",
                "hello",
                "--variations",
                "1",
                "--output-dir",
                str(tmp_path),
                "--no-punctuation-pause",
            ],
        )

        assert result.exit_code == 0
        # Verify EDLGenerator was constructed with punctuation_pause disabled
        mock_edl_cls.assert_called_once_with(
            clip_padding=0.05,
            default_gap_ms=80.0,
            punctuation_pause_enabled=False,
        )


class TestWordsCommand:
    """Tests for the words CLI command."""

    @patch("wordnap.cli.Database")
    def test_words_empty_database_exits_1(self, mock_db_cls):
        """When no words are in the database, exit with code 1."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.connection.execute.return_value.fetchall.return_value = []

        result = runner.invoke(app, ["words"])

        assert result.exit_code == 1
        assert "No words found" in result.output

    @patch("wordnap.cli.Database")
    def test_words_shows_words_alphabetically(self, mock_db_cls):
        """Words are displayed alphabetically by default."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        # Simulate sqlite3.Row-like dicts
        mock_rows = [
            {"normalized_word": "apple", "cnt": 2},
            {"normalized_word": "banana", "cnt": 1},
            {"normalized_word": "cherry", "cnt": 3},
        ]
        mock_db.connection.execute.return_value.fetchall.return_value = mock_rows

        result = runner.invoke(app, ["words"])

        assert result.exit_code == 0
        assert "apple" in result.output
        assert "banana" in result.output
        assert "cherry" in result.output
        # Verify alphabetical order
        apple_pos = result.output.index("apple")
        banana_pos = result.output.index("banana")
        cherry_pos = result.output.index("cherry")
        assert apple_pos < banana_pos < cherry_pos

    @patch("wordnap.cli.Database")
    def test_words_output_flag_writes_file(self, mock_db_cls, tmp_path):
        """The --output flag writes words to a file."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_rows = [
            {"normalized_word": "hello", "cnt": 5},
            {"normalized_word": "world", "cnt": 3},
        ]
        mock_db.connection.execute.return_value.fetchall.return_value = mock_rows

        output_file = tmp_path / "dictionary.txt"
        result = runner.invoke(app, ["words", "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "hello" in content
        assert "world" in content
        assert "Exported 2 unique words" in result.output

    @patch("wordnap.cli.Database")
    def test_words_counts_flag_shows_occurrences(self, mock_db_cls):
        """The --counts flag includes occurrence counts."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_rows = [
            {"normalized_word": "hello", "cnt": 5},
            {"normalized_word": "world", "cnt": 3},
        ]
        mock_db.connection.execute.return_value.fetchall.return_value = mock_rows

        result = runner.invoke(app, ["words", "--counts"])

        assert result.exit_code == 0
        assert "hello (5)" in result.output
        assert "world (3)" in result.output

    @patch("wordnap.cli.Database")
    def test_words_sort_by_count(self, mock_db_cls):
        """The --sort-by count option sorts by frequency descending."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_rows = [
            {"normalized_word": "apple", "cnt": 2},
            {"normalized_word": "banana", "cnt": 5},
            {"normalized_word": "cherry", "cnt": 1},
        ]
        mock_db.connection.execute.return_value.fetchall.return_value = mock_rows

        result = runner.invoke(app, ["words", "--sort-by", "count", "--counts"])

        assert result.exit_code == 0
        # banana(5) should appear before apple(2) before cherry(1)
        banana_pos = result.output.index("banana")
        apple_pos = result.output.index("apple")
        cherry_pos = result.output.index("cherry")
        assert banana_pos < apple_pos < cherry_pos

    @patch("wordnap.cli.Database")
    def test_words_sort_by_length(self, mock_db_cls):
        """The --sort-by length option sorts by word length descending."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_rows = [
            {"normalized_word": "be", "cnt": 1},
            {"normalized_word": "hello", "cnt": 1},
            {"normalized_word": "wonderful", "cnt": 1},
        ]
        mock_db.connection.execute.return_value.fetchall.return_value = mock_rows

        result = runner.invoke(app, ["words", "--sort-by", "length"])

        assert result.exit_code == 0
        # wonderful(9) before hello(5) before be(2)
        wonderful_pos = result.output.index("wonderful")
        hello_pos = result.output.index("hello")
        be_pos = result.output.index("be")
        assert wonderful_pos < hello_pos < be_pos
