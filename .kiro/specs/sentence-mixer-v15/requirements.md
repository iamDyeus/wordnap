# Requirements Document

## Introduction

Sentence Mixer V1.5 introduces five improvements to the video sentence generation pipeline: configurable inter-word silence gaps with punctuation awareness, phrase-level matching from the indexed segments table, burned-in subtitle overlays, graceful handling of missing words, and a multi-pass filter-then-rank clip selection strategy. These features improve output naturalness, reduce failures on longer sentences, and provide better visual context.

## Glossary

- **Pipeline**: The end-to-end process from sentence input through tokenization, search, ranking, EDL generation, and rendering to produce an output video.
- **Renderer**: The component that extracts clips from source videos via FFmpeg and concatenates them into the final output MP4.
- **SearchEngine**: The component that queries the database for word-level candidates matching normalized tokens.
- **Ranker**: The component that scores and selects the best candidate combinations from the SearchEngine results.
- **EDLGenerator**: The component that produces Edit Decision List manifests describing clip boundaries and ordering.
- **Tokenizer**: The component that normalizes and splits user input sentences into searchable tokens.
- **CLI**: The Typer-based command-line interface exposing the `index` and `generate` commands.
- **Segment**: A database record representing a transcribed phrase or sentence from a source video, containing full text, timing, and confidence.
- **Word**: A database record representing a single transcribed word with timing, confidence, and association to a Segment.
- **WordCandidate**: A data model pairing a Word record with its source VideoMetadata and computed duration.
- **PhraseCandidate**: A sequence of consecutive Words from a single Segment that matches a multi-word subsequence of the target sentence.
- **Silence Gap**: A period of audio silence inserted between clips during rendering.
- **Subtitle Overlay**: Text rendered onto the video frame using FFmpeg's drawtext filter, showing the current word.
- **Boundary Quality**: A measure of how cleanly a clip starts or ends at a natural word boundary, assessed by proximity of padding edges to silence.

## Requirements

### Requirement 1: Inter-Word Silence Gaps

**User Story:** As a user generating sentences, I want configurable silence between word clips, so that the output sounds more natural with appropriate pauses.

#### Acceptance Criteria

1. WHEN the Renderer concatenates clips, THE Renderer SHALL insert a default silence gap between each consecutive pair of clips.
2. THE CLI SHALL accept a `--gap` option specifying the default inter-word silence duration in milliseconds with a default value of 80.
3. WHILE punctuation-aware timing is enabled, WHEN the preceding word in the original sentence ends with a comma, THE Renderer SHALL insert a silence gap of approximately 200 milliseconds between that clip and the next.
4. WHILE punctuation-aware timing is enabled, WHEN the preceding word in the original sentence ends with a period, exclamation mark, or question mark, THE Renderer SHALL insert a silence gap of approximately 400 milliseconds between that clip and the next.
5. THE CLI SHALL accept a `--punctuation-pause` flag that enables or disables punctuation-aware silence timing with a default value of enabled.
6. WHEN punctuation-aware timing is disabled, THE Renderer SHALL use only the default gap value between all clips regardless of punctuation.
7. THE Renderer SHALL generate silence gaps as audio-only segments containing zero-amplitude samples at the configured audio sample rate.

### Requirement 2: Phrase Matching

**User Story:** As a user generating sentences, I want the system to match multi-word phrases from source videos, so that the output preserves natural speech cadence across word sequences.

#### Acceptance Criteria

1. WHEN a sentence is submitted for generation, THE SearchEngine SHALL attempt phrase matching before falling back to single-word matching.
2. THE SearchEngine SHALL attempt to match the full token sequence as a single phrase first, then progressively shorter n-gram subsequences (N-1, N-2, ..., down to bigrams), before resorting to single-word lookup for unmatched tokens.
3. THE SearchEngine SHALL identify a phrase match by checking whether the normalized token sequence appears as consecutive words within a single Segment record in the database.
4. WHEN a phrase match is found, THE SearchEngine SHALL return a PhraseCandidate containing the consecutive Word records spanning the matched phrase from that Segment.
5. WHEN multiple phrase matches cover the same token positions, THE SearchEngine SHALL prefer the longest matching phrase.
6. WHEN phrase matching is complete, THE SearchEngine SHALL fall back to single-word lookup only for token positions not covered by any phrase match.
7. THE Pipeline SHALL treat a PhraseCandidate as a single clip unit during EDL generation and rendering, using the start time of the first word and the end time of the last word as clip boundaries.

### Requirement 3: Subtitle Overlay (Burn-In)

**User Story:** As a user generating sentences, I want the current word displayed as a subtitle on the video, so that viewers can follow along with the assembled speech.

#### Acceptance Criteria

1. WHILE subtitles are enabled, WHEN the Renderer produces each clip, THE Renderer SHALL overlay the corresponding word as text centered horizontally at the bottom of the video frame.
2. THE Renderer SHALL style subtitle text as white with a black outline to ensure readability across varying backgrounds.
3. THE Renderer SHALL synchronize subtitle display duration to match each clip's duration, so the word appears for the entire duration of its clip.
4. THE CLI SHALL accept a `--subtitles` flag that enables or disables subtitle overlay with a default value of enabled.
5. WHEN subtitles are disabled, THE Renderer SHALL produce clips without any text overlay.
6. THE Renderer SHALL apply subtitles using FFmpeg's drawtext filter during clip extraction.

### Requirement 4: Missing Word Handling (Best-Effort Mode)

**User Story:** As a user generating longer sentences, I want the system to produce output from available words when some words are missing, so that I get partial results instead of a complete failure.

#### Acceptance Criteria

1. WHILE strict mode is disabled, WHEN the SearchEngine cannot find candidates for one or more tokens, THE Pipeline SHALL skip the missing tokens and continue generation with the remaining available tokens.
2. WHILE strict mode is disabled, WHEN tokens are skipped, THE CLI SHALL report the list of skipped words to the user via a warning message.
3. WHILE strict mode is disabled, WHEN all tokens in the sentence are missing, THE Pipeline SHALL report an error indicating no words could be found and exit with a non-zero code.
4. WHILE strict mode is enabled, WHEN any token has no matching candidates, THE Pipeline SHALL raise a WordNotFoundError and exit with a non-zero code, preserving current behavior.
5. THE CLI SHALL accept a `--strict` flag that controls missing word handling with a default value of false (best-effort mode).
6. WHILE strict mode is disabled, THE EDLManifest SHALL record which words were skipped in the output manifest metadata.

### Requirement 5: Multi-Pass Filter-Then-Rank Clip Selection

**User Story:** As a user generating sentences, I want better clip selection that filters out poor candidates and ranks the rest using multiple quality signals, so that the output uses the most natural-sounding clips.

#### Acceptance Criteria

1. WHEN candidates are retrieved for a token, THE Ranker SHALL apply a filter pass that rejects candidates with confidence less than 0.3.
2. WHEN candidates are retrieved for a token, THE Ranker SHALL apply a filter pass that rejects candidates with duration less than 0.03 seconds or greater than 3.0 seconds.
3. WHEN no candidates remain after filtering for a token, THE Ranker SHALL fall back to the unfiltered candidate list for that token to avoid total failure.
4. THE Ranker SHALL compute a composite score using a weighted combination of four factors: confidence (weight 0.35), duration reasonableness (weight 0.25), boundary quality (weight 0.20), and source diversity (weight 0.20).
5. THE Ranker SHALL compute the duration reasonableness score by assigning a score of 1.0 to clips with duration in the range [0.1 seconds, 1.5 seconds], and a linearly decreasing score for durations outside that range down to 0.0 at the filter boundaries.
6. THE Ranker SHALL compute the boundary quality score by measuring how close the padded clip edges are to silence, preferring clips that start and end at natural word boundaries rather than mid-phoneme.
7. THE Ranker SHALL compute the source diversity score by preferring candidates from source locations not already selected in the current variation.
8. IF the filter pass eliminates all candidates for every token in the sentence, THEN THE Pipeline SHALL report an error indicating insufficient quality candidates and exit with a non-zero code.
