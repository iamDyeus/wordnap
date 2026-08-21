"""WhisperX transcription with word-level alignment."""

import logging
from pathlib import Path

from wordnap.models.schemas import Segment, TranscriptionResult, Word
from wordnap.search.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Transcriber:
    """Produces word-level timestamps from audio using WhisperX."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """Initialize the transcriber.

        Args:
            model_size: WhisperX model size (e.g. "base", "small", "medium", "large").
            device: Compute device ("cpu" or "cuda").
        """
        self.model_size = model_size
        self.device = device
        self._model = None
        self._tokenizer = Tokenizer()

    def _load_model(self):
        """Lazy-load the WhisperX model."""
        if self._model is None:
            import whisperx

            self._model = whisperx.load_model(
                self.model_size, self.device
            )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio and return word-level aligned timestamps.

        Loads the WhisperX model, transcribes the audio, then aligns
        to produce word-level timestamps with confidence scores.

        Args:
            audio_path: Path to the audio file (mono WAV at 16kHz).

        Returns:
            TranscriptionResult with segments and words.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        import whisperx

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        # Transcribe with WhisperX
        audio = whisperx.load_audio(str(audio_path))
        result = self._model.transcribe(audio)

        # Align for word-level timestamps
        aligned_result = self._align(audio, result)

        # Map WhisperX output to our schema
        return self._map_result(aligned_result)

    def _align(self, audio, transcription_result: dict) -> dict:
        """Perform word-level alignment on transcription result.

        Args:
            audio: Loaded audio array from whisperx.load_audio.
            transcription_result: Raw transcription result from WhisperX.

        Returns:
            Aligned result with word-level timestamps.
            Falls back to segment-only result if alignment fails.
        """
        import whisperx

        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=transcription_result.get("language", "en"),
                device=self.device,
            )
            aligned = whisperx.align(
                transcription_result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
            )
            return aligned
        except Exception as e:
            logger.warning(
                "Word-level alignment failed: %s. "
                "Returning segments without word-level detail.",
                e,
            )
            return transcription_result

    def _map_result(self, whisperx_result: dict) -> TranscriptionResult:
        """Map WhisperX output format to our Segment/Word models.

        Args:
            whisperx_result: The aligned result from WhisperX with structure:
                {"segments": [{"text": ..., "start": ..., "end": ...,
                  "words": [{"word": ..., "start": ..., "end": ..., "score": ...}]}]}

        Returns:
            TranscriptionResult with mapped segments and words.
        """
        segments: list[Segment] = []
        words: list[Word] = []

        raw_segments = whisperx_result.get("segments", [])

        # Track the index into the segments list (which may differ from
        # seg_idx if segments with invalid timestamps are skipped).
        valid_seg_index = 0

        for seg_idx, raw_seg in enumerate(raw_segments):
            start_time = raw_seg.get("start", 0.0)
            end_time = raw_seg.get("end", 0.0)
            text = raw_seg.get("text", "").strip()
            speaker = raw_seg.get("speaker")

            # Compute segment confidence as average of word scores
            raw_words = raw_seg.get("words", [])
            word_scores = [
                w.get("score", 0.0)
                for w in raw_words
                if "start" in w and "end" in w
            ]
            seg_confidence = (
                sum(word_scores) / len(word_scores) if word_scores else 0.0
            )

            # Skip segments with invalid timestamps
            if start_time >= end_time:
                logger.warning(
                    "Skipping segment %d with invalid timestamps: "
                    "start=%f, end=%f",
                    seg_idx,
                    start_time,
                    end_time,
                )
                continue

            segment = Segment(
                video_id=0,  # Will be set during storage
                start_time=start_time,
                end_time=end_time,
                text=text,
                speaker=speaker,
                confidence=max(0.0, min(1.0, seg_confidence)),
            )
            segments.append(segment)

            # Map words within this segment
            for raw_word in raw_words:
                word_start = raw_word.get("start")
                word_end = raw_word.get("end")

                # Skip words without valid timestamps
                if word_start is None or word_end is None:
                    logger.debug(
                        "Skipping word '%s' without timestamps in segment %d",
                        raw_word.get("word", ""),
                        seg_idx,
                    )
                    continue

                if word_start >= word_end:
                    logger.debug(
                        "Skipping word '%s' with invalid timestamps: "
                        "start=%f, end=%f",
                        raw_word.get("word", ""),
                        word_start,
                        word_end,
                    )
                    continue

                raw_word_text = raw_word.get("word", "").strip()
                if not raw_word_text:
                    continue

                normalized = self._tokenizer.normalize_word(raw_word_text)
                if not normalized:
                    continue

                confidence = raw_word.get("score", 0.0)
                word_speaker = raw_word.get("speaker") or speaker

                word = Word(
                    segment_id=valid_seg_index,  # 0-based index into segments list
                    video_id=0,  # Will be set during storage
                    word=raw_word_text,
                    normalized_word=normalized,
                    start_time=word_start,
                    end_time=word_end,
                    confidence=max(0.0, min(1.0, confidence)),
                    speaker=word_speaker,
                )
                words.append(word)

            valid_seg_index += 1

        return TranscriptionResult(segments=segments, words=words)
