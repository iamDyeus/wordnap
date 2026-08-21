"""EDL (Edit Decision List) manifest generation."""

from __future__ import annotations

from sentence_mixer.models.schemas import (
    ClipEntry,
    EDLManifest,
    GapEntry,
    PhraseCandidate,
    TokenInfo,
    WordCandidate,
)


class EDLGenerator:
    """Generates EDL manifests from ranked word candidate selections."""

    def __init__(
        self,
        clip_padding: float = 0.10,
        default_gap_ms: float = 80.0,
        punctuation_pause_enabled: bool = True,
        comma_gap_ms: float = 200.0,
        sentence_end_gap_ms: float = 400.0,
    ):
        """Initialize with configurable clip padding and gap settings.

        Args:
            clip_padding: Seconds of padding to add before/after each word clip.
                          Must be non-negative.
            default_gap_ms: Default silence gap between clips in milliseconds.
            punctuation_pause_enabled: Whether to use punctuation-aware gap timing.
            comma_gap_ms: Silence gap after commas in milliseconds.
            sentence_end_gap_ms: Silence gap after sentence-ending punctuation in ms.
        """
        if clip_padding < 0:
            raise ValueError("clip_padding must be non-negative")
        self.clip_padding = clip_padding
        self.default_gap_ms = default_gap_ms
        self.punctuation_pause_enabled = punctuation_pause_enabled
        self.comma_gap_ms = comma_gap_ms
        self.sentence_end_gap_ms = sentence_end_gap_ms

    def compute_gap(self, token_info: TokenInfo | None) -> GapEntry:
        """Compute the silence gap after a token based on its trailing punctuation.

        Args:
            token_info: Token with punctuation context. None uses default gap.

        Returns:
            GapEntry with appropriate duration and reason.
        """
        if not self.punctuation_pause_enabled or token_info is None:
            return GapEntry(duration_ms=self.default_gap_ms, reason="default")

        punct = token_info.trailing_punctuation
        if punct == ",":
            return GapEntry(duration_ms=self.comma_gap_ms, reason="comma")
        elif punct in (".", "!", "?"):
            return GapEntry(duration_ms=self.sentence_end_gap_ms, reason="sentence_end")
        else:
            return GapEntry(duration_ms=self.default_gap_ms, reason="default")

    def generate(
        self,
        selections: list[WordCandidate | PhraseCandidate],
        sentence: str,
        variation_index: int,
        token_infos: list[TokenInfo] | None = None,
        skipped_words: list[str] | None = None,
    ) -> EDLManifest:
        """Generate an EDL manifest from selected candidates.

        Args:
            selections: Ordered list of word/phrase candidates matching token order.
            sentence: The original sentence being generated.
            variation_index: Index of this variation among all generated variations.
            token_infos: Token context for punctuation-aware gaps.
            skipped_words: Words not found in best-effort mode.

        Returns:
            EDLManifest with clip entries, gap entries, and metadata.

        Raises:
            ValueError: If selections is empty.
        """
        if not selections:
            raise ValueError("selections must be non-empty")

        clips: list[ClipEntry] = []
        gaps: list[GapEntry] = []

        for i, candidate in enumerate(selections):
            # Handle both WordCandidate and PhraseCandidate
            if isinstance(candidate, PhraseCandidate):
                start_time = candidate.start_time
                end_time = candidate.end_time
                video = candidate.video
                word_text = " ".join(w.word for w in candidate.words)
                confidence = sum(w.confidence for w in candidate.words) / len(
                    candidate.words
                )
                speaker = candidate.words[0].speaker
            else:
                start_time = candidate.word.start_time
                end_time = candidate.word.end_time
                video = candidate.video
                word_text = candidate.word.word
                confidence = candidate.word.confidence
                speaker = candidate.word.speaker

            # Apply padding with clamping
            padded_start = max(0.0, start_time - self.clip_padding)
            padded_end = min(video.duration, end_time + self.clip_padding)

            clip = ClipEntry(
                source_video=video.path,
                source_filename=video.filename,
                word=word_text,
                start_time=start_time,
                end_time=end_time,
                padded_start=padded_start,
                padded_end=padded_end,
                confidence=confidence,
                speaker=speaker,
            )
            clips.append(clip)

            # Compute gap between this clip and the next (not after last clip)
            if i < len(selections) - 1:
                token_info = (
                    token_infos[i] if token_infos and i < len(token_infos) else None
                )
                gap = self.compute_gap(token_info)
                gaps.append(gap)

        total_duration = sum(c.padded_end - c.padded_start for c in clips)
        total_duration += sum(g.duration_ms / 1000.0 for g in gaps)

        return EDLManifest(
            sentence=sentence,
            variation_index=variation_index,
            clips=clips,
            gaps=gaps,
            skipped_words=skipped_words or [],
            total_duration=total_duration,
        )
