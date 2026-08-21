# Design Document: Sentence Mixer V1.5 Improvements

## Overview

This design covers five enhancements to the Sentence Mixer pipeline: inter-word silence gaps, phrase-level matching, subtitle overlays, graceful missing-word handling, and a multi-pass filter-then-rank clip selection strategy. All changes are additive modifications to existing components with one new module (`phrase_search.py`).

## Architecture

The existing pipeline flow remains:

```
CLI → Tokenizer → SearchEngine → Ranker → EDLGenerator → Renderer → Output MP4
```

V1.5 modifications insert at specific points:

```
CLI (new flags) → Tokenizer (punctuation context) → PhraseSearchEngine + SearchEngine → Ranker (filter+score) → EDLGenerator (gap entries, skip metadata) → Renderer (silence gen, subtitles)
```

### Component Interaction Diagram

```
┌──────────┐    tokens + punctuation    ┌────────────────────┐
│ Tokenizer│──────────────────────────►│ PhraseSearchEngine  │
└──────────┘                            │ (new module)        │
                                        └────────┬───────────┘
                                                 │ PhraseCandidates + uncovered positions
                                                 ▼
                                        ┌────────────────────┐
                                        │   SearchEngine      │
                                        │ (single-word        │
                                        │  fallback for       │
                                        │  uncovered tokens)  │
                                        └────────┬───────────┘
                                                 │ candidates_per_token (mixed)
                                                 ▼
                                        ┌────────────────────┐
                                        │   Ranker (v2)       │
                                        │ filter → score      │
                                        └────────┬───────────┘
                                                 │ ranked selections
                                                 ▼
                                        ┌────────────────────┐
                                        │  EDLGenerator (v2)  │
                                        │ + gap entries       │
                                        │ + skipped_words     │
                                        └────────┬───────────┘
                                                 │ EDLManifest
                                                 ▼
                                        ┌────────────────────┐
                                        │   Renderer (v2)     │
                                        │ + silence gen       │
                                        │ + drawtext overlay  │
                                        └────────────────────┘
```

## Data Models

### New Models (in `models/schemas.py`)

```python
class PhraseCandidate(BaseModel):
    """A multi-word match from a single segment."""
    words: list[Word]
    segment: Segment
    video: VideoMetadata
    start_time: float  # first word's start_time
    end_time: float    # last word's end_time
    duration: float    # end_time - start_time
    score: float = 0.0

    @model_validator(mode="after")
    def validate_words_consecutive(self) -> "PhraseCandidate":
        """Ensure words are from the same segment and consecutive."""
        for w in self.words:
            if w.segment_id != self.segment.id:
                raise ValueError("All words must belong to the same segment")
        return self


class GapEntry(BaseModel):
    """A silence gap between clips in the EDL."""
    duration_ms: float  # duration in milliseconds
    reason: str  # "default", "comma", "sentence_end"


class TokenInfo(BaseModel):
    """Extended token with punctuation context from the original sentence."""
    normalized: str
    original: str  # original word before normalization
    trailing_punctuation: str | None = None  # e.g., ",", ".", "!", "?"


class RenderConfig(BaseModel):
    """Extended render config with new V1.5 options."""
    output_resolution: tuple[int, int] = (1920, 1080)
    output_fps: float = 30.0
    pixel_format: str = "yuv420p"
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    clip_padding: float = 0.10
    # V1.5 additions
    default_gap_ms: float = 80.0
    punctuation_pause_enabled: bool = True
    comma_gap_ms: float = 200.0
    sentence_end_gap_ms: float = 400.0
    subtitles_enabled: bool = True


class RankingConfig(BaseModel):
    """Updated ranking config for multi-pass strategy."""
    confidence_weight: float = 0.35
    duration_weight: float = 0.25
    boundary_quality_weight: float = 0.20
    diversity_weight: float = 0.20
    min_confidence: float = 0.3
    min_duration: float = 0.03
    max_duration: float = 3.0
    ideal_duration_min: float = 0.1
    ideal_duration_max: float = 1.5
    prefer_same_speaker: bool = True
```

### Modified Models

```python
class EDLManifest(BaseModel):
    """Updated to include gap entries and skip metadata."""
    sentence: str
    variation_index: int
    clips: list[ClipEntry]
    gaps: list[GapEntry]  # NEW: gap between clips[i] and clips[i+1]
    skipped_words: list[str] = []  # NEW: words not found in best-effort mode
    created_at: datetime = Field(default_factory=datetime.now)
    total_duration: float = 0.0

    @field_validator("clips")
    @classmethod
    def clips_non_empty(cls, v: list[ClipEntry]) -> list[ClipEntry]:
        if not v:
            raise ValueError("clips list must be non-empty")
        return v
```

## Component Designs

### 1. Tokenizer Enhancement

**File:** `src/sentence_mixer/search/tokenizer.py`

The `Tokenizer.tokenize()` method gains a new return type that preserves punctuation context.

```python
def tokenize_with_context(self, sentence: str) -> list[TokenInfo]:
    """Split sentence into tokens preserving trailing punctuation context.

    Splits on whitespace, normalizes each token, and records trailing
    punctuation from the original word for silence gap calculation.

    Args:
        sentence: The input sentence.

    Returns:
        Ordered list of TokenInfo with normalized form and punctuation.
    """
    if not sentence or not sentence.strip():
        return []

    tokens = []
    for word in sentence.split():
        normalized = self.normalize_word(word)
        if not normalized:
            continue

        # Detect trailing punctuation
        trailing = None
        if word and not word[-1].isalnum():
            trailing = word[-1]

        tokens.append(TokenInfo(
            normalized=normalized,
            original=word,
            trailing_punctuation=trailing,
        ))

    return tokens
```

The existing `tokenize()` method remains unchanged for backward compatibility.

### 2. Phrase Search Engine

**File:** `src/sentence_mixer/search/phrase_search.py` (new)

```python
class PhraseSearchEngine:
    """Finds multi-word phrase matches from indexed segments.

    Algorithm:
    1. For each window size from len(tokens) down to 2:
       a. Slide a window across the token list
       b. For each window, query segments containing the first word
       c. For matching segments, verify the remaining words are consecutive
    2. Use a greedy longest-first coverage strategy
    3. Return covered positions and their PhraseCandidates
    """

    def __init__(self, db: Database):
        self._db = db

    def find_phrases(
        self, tokens: list[str]
    ) -> tuple[dict[tuple[int, int], list[PhraseCandidate]], set[int]]:
        """Find phrase matches for a token sequence.

        Args:
            tokens: Ordered list of normalized tokens.

        Returns:
            Tuple of:
            - Dict mapping (start_pos, end_pos) to list of PhraseCandidates
            - Set of token positions covered by phrase matches

        The method uses longest-first greedy matching. Positions covered
        by a longer phrase are not available for shorter phrase matches.
        """
        ...

    def _find_phrase_in_segments(
        self, token_window: list[str]
    ) -> list[PhraseCandidate]:
        """Search the database for segments containing the consecutive tokens.

        SQL approach:
        1. Find all words matching the first token
        2. For each match, verify subsequent words in that segment match
           the remaining tokens in consecutive positions
        3. Construct PhraseCandidate from matched word sequence

        Args:
            token_window: Ordered list of normalized tokens to match.

        Returns:
            List of PhraseCandidates where the token window was found.
        """
        ...
```

**Database Query for Phrase Matching:**

```python
def find_consecutive_words_in_segment(
    db: Database, segment_id: int, start_word_id: int, expected_tokens: list[str]
) -> list[Word] | None:
    """Verify that consecutive words in a segment match expected tokens.

    Queries words in the segment ordered by start_time, starting from the
    word with start_word_id, and checks the next N words match expected_tokens.

    Returns the matching Word list or None if not consecutive.
    """
    conn = db.connection
    rows = conn.execute(
        """
        SELECT * FROM words
        WHERE segment_id = ? AND start_time >= (
            SELECT start_time FROM words WHERE id = ?
        )
        ORDER BY start_time ASC
        LIMIT ?
        """,
        (segment_id, start_word_id, len(expected_tokens)),
    ).fetchall()

    if len(rows) != len(expected_tokens):
        return None

    words = []
    for row, expected in zip(rows, expected_tokens):
        if row["normalized_word"] != expected:
            return None
        words.append(Word(
            id=row["id"],
            segment_id=row["segment_id"],
            video_id=row["video_id"],
            word=row["word"],
            normalized_word=row["normalized_word"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            confidence=row["confidence"],
            speaker=row["speaker"],
        ))

    return words
```

### 3. Renderer Enhancement: Silence Gaps

**File:** `src/sentence_mixer/editing/renderer.py`

The `Renderer.render()` method gains a silence generation step between clip extraction and concatenation.

```python
def generate_silence(self, duration_ms: float, output_path: Path) -> Path:
    """Generate a silence audio segment using FFmpeg's anullsrc filter.

    Args:
        duration_ms: Duration of silence in milliseconds.
        output_path: Path for the output silence file.

    Returns:
        The output_path on success.

    Raises:
        RenderError: If FFmpeg fails.
    """
    duration_s = duration_ms / 1000.0
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"anullsrc=r={self.config.audio_sample_rate}:cl=stereo",
        "-t", str(duration_s),
        "-c:a", self.config.audio_codec,
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
```

**Updated `render()` flow:**

```python
def render(self, manifest: EDLManifest, output_path: Path) -> Path:
    """Render EDL manifest with interleaved silence gaps."""
    temp_dir = Path(tempfile.mkdtemp(prefix="sentence_mixer_render_"))

    try:
        segments: list[Path] = []  # clips and silence interleaved

        for i, clip_entry in enumerate(manifest.clips):
            # Extract clip (with optional subtitle)
            clip_output = temp_dir / f"clip_{i:04d}.mp4"
            self.extract_clip(
                video_path=clip_entry.source_video,
                start=clip_entry.padded_start,
                end=clip_entry.padded_end,
                output_path=clip_output,
                subtitle_text=clip_entry.word if self.config.subtitles_enabled else None,
            )
            segments.append(clip_output)

            # Insert silence gap between clips (not after last clip)
            if i < len(manifest.clips) - 1 and i < len(manifest.gaps):
                gap = manifest.gaps[i]
                silence_output = temp_dir / f"silence_{i:04d}.mp4"
                self.generate_silence(gap.duration_ms, silence_output)
                segments.append(silence_output)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.concatenate(segments, output_path)

    finally:
        for f in temp_dir.iterdir():
            f.unlink()
        temp_dir.rmdir()

    return output_path
```

### 4. Renderer Enhancement: Subtitle Overlay

The `extract_clip()` method gains a `subtitle_text` parameter that conditionally adds a `drawtext` filter.

```python
def extract_clip(
    self,
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    subtitle_text: str | None = None,
) -> Path:
    """Extract a clip with optional subtitle overlay.

    When subtitle_text is provided, applies FFmpeg's drawtext filter
    with white text, black outline, centered at bottom of frame.
    """
    width, height = self.config.output_resolution

    # Build video filter chain
    vf_filters = [f"scale={width}:{height}"]

    if subtitle_text and self.config.subtitles_enabled:
        # Escape special characters for FFmpeg drawtext
        escaped_text = subtitle_text.replace("'", "\\'").replace(":", "\\:")
        drawtext = (
            f"drawtext=text='{escaped_text}'"
            f":fontsize=48"
            f":fontcolor=white"
            f":borderw=3"
            f":bordercolor=black"
            f":x=(w-tw)/2"
            f":y=h-th-40"
        )
        vf_filters.append(drawtext)

    vf_string = ",".join(vf_filters)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-ss", str(start),
        "-to", str(end),
        "-vf", vf_string,
        "-r", str(self.config.output_fps),
        "-pix_fmt", self.config.pixel_format,
        "-ar", str(self.config.audio_sample_rate),
        "-ac", str(self.config.audio_channels),
        "-c:v", self.config.video_codec,
        "-c:a", self.config.audio_codec,
        "-y", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RenderError(
            message=f"FFmpeg failed to extract clip from '{video_path}'",
            stderr=result.stderr,
        )
    return output_path
```

### 5. Missing Word Handling

**File:** `src/sentence_mixer/search/candidate.py`

`SearchEngine.find_candidates_batch()` is modified to support best-effort mode:

```python
def find_candidates_batch(
    self, tokens: list[str], strict: bool = True
) -> tuple[dict[str, list[WordCandidate]], list[str]]:
    """Find candidates for multiple tokens.

    Args:
        tokens: Ordered list of normalized tokens.
        strict: If True, raises WordNotFoundError on any missing token.
                If False, returns partial results with missing word list.

    Returns:
        Tuple of:
        - Dict mapping each found token to its candidates
        - List of tokens that had no candidates (empty in strict mode)

    Raises:
        WordNotFoundError: If strict=True and any token has no candidates.
    """
    results = find_words_batch(self._db, tokens)

    missing = [token for token in tokens if not results.get(token)]

    if strict and missing:
        raise WordNotFoundError(missing)

    # In best-effort mode, remove missing tokens from results
    available = {k: v for k, v in results.items() if v}

    return available, missing
```

### 6. Multi-Pass Filter-Then-Rank

**File:** `src/sentence_mixer/search/ranking.py`

The `Ranker` is updated with a two-pass system:

```python
class Ranker:
    """Two-pass filter-then-rank candidate selection."""

    def __init__(self, config: RankingConfig):
        self._config = config

    def filter_candidates(
        self, candidates: list[WordCandidate]
    ) -> list[WordCandidate]:
        """Pass 1: Remove candidates below quality thresholds.

        Filters on:
        - confidence >= min_confidence (0.3)
        - min_duration <= duration <= max_duration (0.03s to 3.0s)

        If filtering removes ALL candidates, returns the original list
        as fallback to avoid total failure.

        Args:
            candidates: List of candidates for a single token.

        Returns:
            Filtered list, or original list if filtering was too aggressive.
        """
        filtered = [
            c for c in candidates
            if c.word.confidence >= self._config.min_confidence
            and self._config.min_duration <= c.duration <= self._config.max_duration
        ]
        return filtered if filtered else candidates

    def compute_duration_score(self, duration: float) -> float:
        """Compute duration reasonableness score.

        Returns 1.0 for durations in [ideal_min, ideal_max].
        Linearly decreases to 0.0 at filter boundaries.

        Args:
            duration: Clip duration in seconds.

        Returns:
            Score in [0.0, 1.0].
        """
        config = self._config
        if config.ideal_duration_min <= duration <= config.ideal_duration_max:
            return 1.0
        elif duration < config.ideal_duration_min:
            # Linear from 0.0 at min_duration to 1.0 at ideal_min
            if duration <= config.min_duration:
                return 0.0
            return (duration - config.min_duration) / (
                config.ideal_duration_min - config.min_duration
            )
        else:
            # Linear from 1.0 at ideal_max to 0.0 at max_duration
            if duration >= config.max_duration:
                return 0.0
            return (config.max_duration - duration) / (
                config.max_duration - config.ideal_duration_max
            )

    def compute_boundary_quality(self, candidate: WordCandidate) -> float:
        """Compute boundary quality score.

        Measures how much padding space exists between the padded clip
        edges and the actual word boundaries. Larger gap suggests the
        word starts/ends at a natural silence boundary.

        Approximation: boundary_quality = (start_gap + end_gap) / (2 * max_padding)
        where start_gap = word.start_time - padded_start
              end_gap = padded_end - word.end_time

        Args:
            candidate: WordCandidate with timing information.

        Returns:
            Score in [0.0, 1.0].
        """
        # Use the configured clip_padding as reference for max possible gap
        max_padding = 0.15  # reasonable max reference
        start_gap = candidate.word.start_time - (
            candidate.word.start_time - min(max_padding, candidate.word.start_time)
        )
        end_gap = min(
            max_padding,
            candidate.video.duration - candidate.word.end_time
        )

        # Normalize: larger gaps = better boundary quality
        quality = (start_gap + end_gap) / (2 * max_padding)
        return max(0.0, min(1.0, quality))

    def compute_diversity_score(
        self, candidate: WordCandidate, used_sources: set[int | None]
    ) -> float:
        """Compute source diversity score.

        Returns 1.0 if candidate is from an unused source, 0.0 otherwise.

        Args:
            candidate: The candidate to score.
            used_sources: Set of video IDs already selected in this variation.

        Returns:
            1.0 or 0.0.
        """
        if candidate.video.id not in used_sources:
            return 1.0
        return 0.0

    def score_candidate(
        self, candidate: WordCandidate, used_sources: set[int | None]
    ) -> float:
        """Pass 2: Compute composite weighted score.

        Score = confidence * 0.35 + duration_score * 0.25
              + boundary_quality * 0.20 + diversity * 0.20

        Args:
            candidate: The candidate to score.
            used_sources: Sources already used in this variation.

        Returns:
            Composite score in [0.0, 1.0].
        """
        config = self._config
        confidence_score = candidate.word.confidence
        duration_score = self.compute_duration_score(candidate.duration)
        boundary_score = self.compute_boundary_quality(candidate)
        diversity_score = self.compute_diversity_score(candidate, used_sources)

        score = (
            confidence_score * config.confidence_weight
            + duration_score * config.duration_weight
            + boundary_score * config.boundary_quality_weight
            + diversity_score * config.diversity_weight
        )
        return max(0.0, min(1.0, score))
```

### 7. EDL Generator Enhancement

**File:** `src/sentence_mixer/editing/edl.py`

The `EDLGenerator` gains gap calculation logic:

```python
class EDLGenerator:
    """Generates EDL manifests with silence gap entries."""

    def __init__(
        self,
        clip_padding: float = 0.10,
        default_gap_ms: float = 80.0,
        punctuation_pause_enabled: bool = True,
        comma_gap_ms: float = 200.0,
        sentence_end_gap_ms: float = 400.0,
    ):
        ...

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
        """Generate an EDL manifest with gap entries.

        Args:
            selections: Ordered candidates (words or phrases).
            sentence: Original sentence.
            variation_index: Variation index.
            token_infos: Token context for punctuation-aware gaps.
            skipped_words: Words not found in best-effort mode.

        Returns:
            EDLManifest with clips, gaps, and metadata.
        """
        ...
```

### 8. CLI Updates

**File:** `src/sentence_mixer/cli.py`

New options on the `generate` command:

```python
@app.command()
def generate(
    sentence: str = typer.Option(..., "--sentence", help="Sentence to generate"),
    variations: int = typer.Option(5, "--variations", help="Number of variations"),
    db_path: Path = typer.Option("data/sentence_mixer.db", help="Database path"),
    output_dir: Path = typer.Option("output", help="Output directory"),
    padding: float = typer.Option(0.10, "--padding", help="Clip padding in seconds"),
    # V1.5 new options
    gap: float = typer.Option(80.0, "--gap", help="Default inter-word silence gap in ms"),
    punctuation_pause: bool = typer.Option(True, "--punctuation-pause/--no-punctuation-pause", help="Enable punctuation-aware silence timing"),
    subtitles: bool = typer.Option(True, "--subtitles/--no-subtitles", help="Enable subtitle overlay"),
    strict: bool = typer.Option(False, "--strict/--no-strict", help="Fail on missing words"),
) -> None:
    ...
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Missing words (strict=True) | Raise `WordNotFoundError` with list of missing words, exit code 1 |
| Missing words (strict=False) | Skip missing tokens, warn user, record in manifest |
| All words missing (any mode) | Error: "No words could be found", exit code 1 |
| Filter eliminates all candidates for every token | Error: "Insufficient quality candidates", exit code 1 |
| Filter eliminates all for one token | Fallback to unfiltered list for that token |
| FFmpeg silence generation fails | Raise `RenderError` |
| Phrase match DB query fails | Log warning, fall back to single-word matching |
| Subtitle text contains special chars | Escape for FFmpeg drawtext filter |

## File Organization

```
src/sentence_mixer/
├── search/
│   ├── tokenizer.py         # + tokenize_with_context()
│   ├── candidate.py         # + strict parameter
│   ├── phrase_search.py     # NEW: PhraseSearchEngine
│   └── ranking.py           # Replaced with filter-then-rank
├── editing/
│   ├── edl.py               # + gap computation, skipped_words
│   ├── renderer.py          # + silence generation, subtitle overlay
│   └── boundaries.py        # unchanged
├── models/
│   └── schemas.py           # + PhraseCandidate, GapEntry, TokenInfo, updated configs
├── database/
│   ├── database.py          # unchanged
│   └── queries.py           # + find_consecutive_words_in_segment()
└── cli.py                   # + new CLI flags
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Silence gap count equals clips minus one

*For any* EDL manifest with N clips (where N > 1), the render pipeline SHALL produce exactly N-1 silence gap segments interleaved between the clips.

**Validates: Requirements 1.1**

### Property 2: Punctuation-aware gap duration matches punctuation type

*For any* clip preceded by a word with trailing punctuation, when punctuation-aware timing is enabled, the silence gap duration SHALL equal the configured pause for that punctuation type (comma → comma_gap_ms, sentence-end → sentence_end_gap_ms, other/none → default_gap_ms).

**Validates: Requirements 1.3, 1.4**

### Property 3: Disabled punctuation-pause produces uniform gaps

*For any* EDL manifest generated with punctuation-aware timing disabled, all gap entries SHALL have duration equal to the configured default_gap_ms regardless of the punctuation in the original sentence.

**Validates: Requirements 1.6**

### Property 4: Phrase matching uses longest-first greedy strategy

*For any* token sequence where overlapping phrase matches exist at the same positions, the phrase search engine SHALL select the longest matching phrase, and no shorter phrase SHALL cover positions already covered by a longer one.

**Validates: Requirements 2.2, 2.5**

### Property 5: Phrase match validates consecutive words within a segment

*For any* token subsequence reported as a phrase match, the matched words SHALL all belong to the same segment AND appear consecutively (ordered by start_time) within that segment with normalized forms matching the token sequence.

**Validates: Requirements 2.3, 2.4**

### Property 6: Token coverage completeness

*For any* token sequence after phrase matching, every token position SHALL be covered exactly once—either by a phrase match or by single-word lookup—with no gaps and no overlaps.

**Validates: Requirements 2.1, 2.6**

### Property 7: PhraseCandidate clip boundaries span first-to-last word

*For any* PhraseCandidate used in EDL generation, the resulting clip entry SHALL have start_time equal to the first word's start_time and end_time equal to the last word's end_time.

**Validates: Requirements 2.7**

### Property 8: Subtitle presence matches enabled flag

*For any* clip extraction command, the drawtext filter SHALL be present in the FFmpeg video filter chain if and only if subtitles are enabled in the render config.

**Validates: Requirements 3.1, 3.5, 3.6**

### Property 9: Best-effort mode produces output from available tokens

*For any* token list where at least one token has candidates and strict mode is disabled, the pipeline SHALL produce an EDLManifest containing clips only for available tokens, and the manifest's skipped_words field SHALL contain exactly the tokens with no candidates.

**Validates: Requirements 4.1, 4.2, 4.6**

### Property 10: Strict mode rejects on any missing token

*For any* token list with at least one token having no candidates when strict mode is enabled, the pipeline SHALL raise WordNotFoundError containing the missing token(s).

**Validates: Requirements 4.4**

### Property 11: Filter pass criteria

*For any* candidate passing the filter pass, the candidate SHALL have confidence >= 0.3 AND duration in [0.03, 3.0] seconds. Conversely, any candidate NOT meeting these criteria SHALL be excluded (unless the fallback applies).

**Validates: Requirements 5.1, 5.2**

### Property 12: Filter fallback preserves candidates

*For any* token where filtering removes ALL candidates, the ranker SHALL use the original unfiltered candidate list for that token, ensuring at least one candidate is available for scoring.

**Validates: Requirements 5.3**

### Property 13: Composite score uses correct weights

*For any* candidate, the composite score SHALL equal confidence * 0.35 + duration_score * 0.25 + boundary_quality * 0.20 + diversity * 0.20, where each component is in [0.0, 1.0].

**Validates: Requirements 5.4**

### Property 14: Duration reasonableness is piecewise linear

*For any* duration value, the duration score SHALL be 1.0 when duration is in [0.1, 1.5] seconds, linearly decrease from 1.0 to 0.0 between [0.03, 0.1] and between [1.5, 3.0], and equal 0.0 at or beyond filter boundaries.

**Validates: Requirements 5.5**

### Property 15: Source diversity prefers unused sources

*For any* candidate and set of already-selected source video IDs, the diversity score SHALL be 1.0 if the candidate's source is not in the used set, and 0.0 if it is.

**Validates: Requirements 5.7**

### Property 16: Tokenizer punctuation preservation round-trip

*For any* sentence, the list of normalized tokens produced by `tokenize_with_context()` SHALL equal the list produced by `tokenize()` for the same sentence, ensuring backward compatibility of the normalization logic.

**Validates: Requirements 1.3, 1.4** (ensures punctuation detection doesn't alter tokenization)
