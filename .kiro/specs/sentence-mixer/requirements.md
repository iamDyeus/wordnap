# Requirements Document

## Introduction

Sentence Mixer is a local-first video editing engine that constructs arbitrary sentences by extracting and splicing word-level audio/video clips from an indexed library of videos. The system operates as a deterministic, two-phase pipeline — indexing (scan, extract, transcribe, store) and generation (tokenize, search, rank, render) — exposed via a CLI interface. This document specifies the functional requirements for the V1 implementation.

## Glossary

- **Indexing_Pipeline**: The subsystem responsible for scanning video directories, extracting audio, transcribing with WhisperX, and storing word-level metadata in the database.
- **Generation_Pipeline**: The subsystem responsible for tokenizing input sentences, searching candidates, ranking, generating EDL manifests, and rendering final MP4 output.
- **Scanner**: The component that discovers video files and extracts metadata via ffprobe.
- **Audio_Extractor**: The component that extracts mono WAV audio from video files using FFmpeg.
- **Transcriber**: The component that produces word-level aligned timestamps using WhisperX.
- **Database**: The SQLite storage layer holding video metadata, segments, and words.
- **Tokenizer**: The component that normalizes user input into searchable tokens.
- **Search_Engine**: The component that queries the word index for matching candidates.
- **Ranker**: The component that scores candidates and produces distinct variations.
- **EDL_Generator**: The component that produces JSON manifests describing clip extraction.
- **Renderer**: The component that extracts, normalizes, and concatenates clips into MP4.
- **CLI**: The Typer-based command-line interface exposing index and generate commands.
- **EDL_Manifest**: A JSON document describing the sequence of clips to extract and concatenate.
- **Word_Candidate**: A database word record enriched with source video metadata and a computed score.
- **Normalized_Word**: A word transformed to lowercase with punctuation stripped and Unicode NFKC applied.

## Requirements

### Requirement 1: Video Discovery and Metadata Extraction

**User Story:** As a user, I want to point the tool at a directory of videos so that the system can discover and catalog all video files for indexing.

#### Acceptance Criteria

1. WHEN a user invokes the index command with a directory path, THE Scanner SHALL recursively discover all files with supported extensions (.mp4, .mkv, .avi, .mov, .webm).
2. WHEN a video file is discovered, THE Scanner SHALL extract metadata (duration, width, height, fps, audio_sample_rate) using ffprobe.
3. WHEN a video file has already been indexed and has not changed, THE Scanner SHALL skip re-indexing that file.
4. IF ffprobe cannot read a discovered file, THEN THE Scanner SHALL log a warning and continue scanning remaining files.
5. WHEN video metadata is extracted, THE Database SHALL store or update the video record with status set to "pending".

### Requirement 2: Audio Extraction

**User Story:** As a user, I want audio extracted from my videos automatically so that the transcription engine can process them.

#### Acceptance Criteria

1. WHEN a video requires transcription, THE Audio_Extractor SHALL produce a mono WAV file at 16kHz sample rate from the video's audio track.
2. WHEN extracted audio already exists in the cache directory, THE Audio_Extractor SHALL skip re-extraction and return the cached path.
3. IF FFmpeg fails during audio extraction, THEN THE Audio_Extractor SHALL raise an error with the FFmpeg stderr output and the source video path.

### Requirement 3: Transcription and Word-Level Alignment

**User Story:** As a user, I want word-level timestamps from my videos so that individual words can be precisely located and extracted.

#### Acceptance Criteria

1. WHEN audio is provided to the Transcriber, THE Transcriber SHALL produce segment-level and word-level timestamps with confidence scores.
2. WHEN transcription completes successfully, THE Database SHALL store all segments and words with their timestamps, confidence, and speaker information.
3. WHEN words are stored, THE Database SHALL store both the raw word and the normalized form (lowercase, punctuation-stripped, Unicode NFKC).
4. IF WhisperX fails to produce word-level alignment, THEN THE Indexing_Pipeline SHALL mark the video status as "failed" and continue with remaining videos.
5. THE Database SHALL maintain an index on the normalized_word column for efficient lookup.

### Requirement 4: Word Normalization and Tokenization

**User Story:** As a user, I want my input sentence processed consistently so that word matching is reliable regardless of casing or punctuation.

#### Acceptance Criteria

1. WHEN a sentence is provided, THE Tokenizer SHALL split it on whitespace and produce an ordered list of normalized tokens.
2. THE Tokenizer SHALL normalize each word by applying: lowercase transformation, punctuation stripping (preserving intra-word hyphens), and Unicode NFKC normalization.
3. THE Tokenizer normalization SHALL be idempotent: normalizing an already-normalized word produces the same result.
4. WHEN the input sentence is empty or contains only whitespace, THE Tokenizer SHALL return an empty token list.

### Requirement 5: Candidate Search

**User Story:** As a user, I want the system to find all occurrences of each word in my sentence across the indexed library so that the best clips can be selected.

#### Acceptance Criteria

1. WHEN a normalized token is searched, THE Search_Engine SHALL query the database word index and return all matching Word_Candidate records.
2. WHEN searching for multiple tokens, THE Search_Engine SHALL return candidates grouped by token preserving the original token order.
3. IF a token has no matching candidates in the database, THEN THE Generation_Pipeline SHALL report the missing words to the user.
4. WHEN a word was stored during indexing, THE Search_Engine SHALL return it as a candidate when searched by its normalized form.

### Requirement 6: Candidate Ranking and Variation Generation

**User Story:** As a user, I want multiple variations of my sentence so that I can pick the best-sounding result.

#### Acceptance Criteria

1. THE Ranker SHALL score each candidate using weighted heuristics: transcription confidence, duration reasonableness, and source diversity.
2. THE Ranker scoring function SHALL produce values in the range [0.0, 1.0] for any valid candidate.
3. WHEN candidates have duration outside the configured [min_duration, max_duration] range, THE Ranker SHALL penalize their score.
4. WHEN candidates have transcription confidence below 0.3, THE Ranker SHALL heavily penalize their score.
5. WHEN generating multiple variations, THE Ranker SHALL produce pairwise-distinct variations where each uses a different combination of word candidates.
6. WHEN the number of possible unique combinations is less than requested variations, THE Ranker SHALL produce as many unique variations as possible.
7. THE Ranker SHALL prefer candidates from the same speaker within a variation by default.
8. THE Ranker SHALL order variations by total score in descending order.

### Requirement 7: EDL Manifest Generation

**User Story:** As a user, I want an intermediate manifest format so that clip extraction can be replayed, inspected, and debugged independently of ranking.

#### Acceptance Criteria

1. WHEN candidate selections are provided, THE EDL_Generator SHALL produce a JSON manifest with exactly one clip entry per selected word candidate.
2. THE EDL_Generator SHALL apply configurable padding (default 0.10s) before and after each clip's word timestamps.
3. THE EDL_Generator SHALL clamp padded start times to be no less than zero.
4. THE EDL_Generator SHALL clamp padded end times to not exceed the source video duration.
5. THE EDL_Generator SHALL preserve clip order matching the original token order in the sentence.
6. THE EDL_Generator SHALL include source attribution (video path, filename) in each clip entry.
7. FOR ALL valid EDLManifest objects, serializing to JSON and deserializing back SHALL produce an equivalent object.

### Requirement 8: Video Rendering

**User Story:** As a user, I want playable MP4 files as output so that I can watch and share the generated sentences.

#### Acceptance Criteria

1. WHEN an EDL manifest is provided, THE Renderer SHALL extract each clip from its source video using FFmpeg with the specified time boundaries.
2. THE Renderer SHALL normalize all clips to consistent parameters (resolution, fps, pixel format, sample rate, channels, codec) before concatenation.
3. WHEN all clips are extracted and normalized, THE Renderer SHALL concatenate them into a single MP4 using FFmpeg concat demuxer.
4. WHEN rendering completes, THE Renderer SHALL clean up temporary intermediate clip files.
5. IF FFmpeg fails during clip extraction or concatenation, THEN THE Renderer SHALL raise a RenderError with the FFmpeg stderr and failing clip entry.
6. THE Renderer SHALL construct FFmpeg commands using list arguments to prevent shell injection.

### Requirement 9: CLI Interface

**User Story:** As a user, I want a simple command-line interface so that I can index videos and generate sentences without writing code.

#### Acceptance Criteria

1. THE CLI SHALL expose an `index` command accepting a directory path argument.
2. THE CLI SHALL expose a `generate` command accepting `--sentence` (required) and `--variations` (optional, default 5) parameters.
3. WHEN the index command completes, THE CLI SHALL display a summary of videos indexed and words stored.
4. WHEN the generate command completes, THE CLI SHALL display the paths of all generated output files.
5. IF the generate command encounters missing words, THEN THE CLI SHALL display the list of words not found in the library.
6. IF the generate command encounters render failures, THEN THE CLI SHALL report which variations failed while still producing successful ones.

### Requirement 10: Data Persistence and Integrity

**User Story:** As a user, I want my indexed data preserved reliably so that I only need to index once and can generate many times.

#### Acceptance Criteria

1. THE Database SHALL create all required tables (videos, segments, words) and indexes on first initialization.
2. THE Database SHALL enforce that all stored words have start_time less than end_time.
3. THE Database SHALL enforce that confidence values are in the range [0.0, 1.0].
4. WHEN storing transcription data, THE Database SHALL use transactions to ensure atomicity (all segments and words for a video stored together or not at all).
5. THE Database SHALL use WAL mode for concurrent read access during generation.
