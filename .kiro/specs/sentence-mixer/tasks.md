# Implementation Plan: Sentence Mixer

## Overview

Implement a local-first video editing engine that indexes video libraries with word-level transcription and generates sentence variations by splicing clips. The implementation follows the project's modular architecture: schemas → database → ingestion → search/ranking → editing → CLI wiring.

## Tasks

- [x] 1. Set up project structure and core schemas
  - [x] 1.1 Initialize project with pyproject.toml, create directory structure (src/sentence_mixer/ with all subpackages), and configure pytest + hypothesis
    - Create pyproject.toml with dependencies (pydantic, typer, hypothesis, pytest, sqlalchemy)
    - Create all __init__.py files for packages: ingestion, transcription, database, search, editing, models
    - Create data/, library/, output/ directories
    - _Requirements: Project structure from design_

  - [x] 1.2 Implement Pydantic schemas in src/sentence_mixer/models/schemas.py
    - Define all models: VideoStatus, VideoMetadata, Segment, Word, WordCandidate, RankingConfig, ClipEntry, EDLManifest, RenderConfig
    - Add validators for start_time < end_time, confidence in [0,1], non-empty normalized_word
    - _Requirements: 10.2, 10.3, 7.6_

  - [ ]* 1.3 Write property test for EDL Manifest serialization round-trip
    - **Property 2: EDL Manifest Round-Trip**
    - Generate random valid EDLManifest objects with Hypothesis, serialize to JSON, deserialize, verify equality
    - **Validates: Requirement 7.7**

- [x] 2. Implement tokenizer and normalization
  - [x] 2.1 Implement Tokenizer in src/sentence_mixer/search/tokenizer.py
    - Implement normalize_word(): lowercase, strip punctuation (preserve intra-word hyphens), Unicode NFKC
    - Implement tokenize(): split on whitespace, normalize each token, return ordered list
    - Handle empty/whitespace-only input returning empty list
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 2.2 Write property test for tokenizer idempotence
    - **Property 1: Tokenizer Idempotence**
    - Generate random Unicode strings with Hypothesis, verify normalize_word(normalize_word(x)) == normalize_word(x)
    - **Validates: Requirement 4.3**

- [x] 3. Implement database layer
  - [x] 3.1 Implement Database class in src/sentence_mixer/database/database.py
    - Create tables (videos, segments, words) with proper schema
    - Create indexes on normalized_word and video_id
    - Configure WAL mode
    - Implement initialize(), upsert_video(), store_transcription(), find_words(), get_video()
    - Use transactions for atomic transcription storage
    - Validate temporal constraints and confidence ranges before storage
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 3.5_

  - [x] 3.2 Implement query helpers in src/sentence_mixer/database/queries.py
    - Implement find_words_by_normalized() returning Word records with video metadata
    - Implement batch word search for multiple tokens
    - Implement is_indexed() check for skip logic
    - _Requirements: 5.1, 5.2, 1.3_

  - [ ]* 3.3 Write property test for database validation constraints
    - **Property 12: Database Rejects Invalid Temporal and Confidence Data**
    - Generate Word records with start_time >= end_time or confidence outside [0,1], verify storage raises validation error
    - **Validates: Requirements 10.2, 10.3**

- [x] 4. Implement ingestion pipeline
  - [x] 4.1 Implement Scanner in src/sentence_mixer/ingestion/scanner.py
    - Implement scan_directory(): recursive file discovery filtering by SUPPORTED_EXTENSIONS
    - Implement probe_video(): call ffprobe to extract metadata
    - Skip already-indexed files
    - Handle unreadable files (log warning, continue)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 4.2 Implement AudioExtractor in src/sentence_mixer/ingestion/audio.py
    - Implement extract_audio(): FFmpeg conversion to mono WAV 16kHz
    - Implement caching: skip extraction if output already exists
    - Use subprocess with list arguments for FFmpeg commands
    - Raise descriptive error on FFmpeg failure
    - _Requirements: 2.1, 2.2, 2.3, 8.6_

  - [x] 4.3 Implement Transcriber in src/sentence_mixer/transcription/whisperx.py
    - Implement transcribe(): load WhisperX, produce word-level aligned timestamps
    - Return TranscriptionResult with segments and words including confidence and speaker
    - Handle alignment failures gracefully
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 4.4 Write property tests for scanner extension filtering and stored word forms
    - **Property 9: Scanner Extension Filtering**
    - Generate random filenames with mixed extensions, verify only supported ones are returned
    - **Property 13: Stored Words Preserve Both Forms**
    - Generate random words, simulate storage path, verify raw and normalized forms are correct
    - **Validates: Requirements 1.1, 3.3**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement search and ranking
  - [x] 6.1 Implement SearchEngine in src/sentence_mixer/search/candidate.py
    - Implement find_candidates(): query database for word matches, return WordCandidate list
    - Implement find_candidates_batch(): batch search preserving token order
    - Report missing words (tokens with no candidates)
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.2 Implement Ranker in src/sentence_mixer/search/ranking.py
    - Implement score_candidate(): weighted scoring (confidence * weight + duration_score * weight + diversity_score * weight), bounded to [0, 1]
    - Penalize durations outside [min_duration, max_duration] range
    - Heavily penalize confidence below 0.3
    - Implement rank(): produce N distinct variations, prefer same-speaker, order by total score descending
    - Cap variations at max possible unique combinations
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 6.3 Write property tests for scoring and ranking
    - **Property 4: Scoring is Bounded**
    - Generate random valid candidates, verify score in [0.0, 1.0]
    - **Property 10: Scoring Penalizes Poor Candidates**
    - Generate candidate pairs differing only in duration/confidence, verify penalty
    - **Property 3: Ranking Produces Distinct Variations**
    - Generate candidate sets, produce variations, verify pairwise distinct
    - **Property 8: Variation Count Respects Constraints**
    - Generate limited candidate sets, verify count = min(requested, possible)
    - **Property 11: Ranking Produces Descending Score Order**
    - Generate variations, verify descending total score order
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5, 6.6, 6.8**

- [x] 7. Implement EDL generation
  - [x] 7.1 Implement EDLGenerator in src/sentence_mixer/editing/edl.py
    - Implement generate(): create EDLManifest from selections
    - Apply configurable clip padding before/after word timestamps
    - Clamp padded_start >= 0 and padded_end <= video_duration
    - Preserve clip order matching token order
    - Include source attribution in each clip entry
    - Compute total_duration as sum of clip durations
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 7.2 Write property tests for EDL generation
    - **Property 5: EDL Structural Correctness**
    - Generate random selection lists, verify clip count matches and order preserved
    - **Property 6: Padding Clamping Correctness**
    - Generate clips with edge-case timestamps and various padding values, verify bounds
    - **Validates: Requirements 7.1, 7.3, 7.4, 7.5**

- [x] 8. Implement renderer
  - [x] 8.1 Implement Renderer in src/sentence_mixer/editing/renderer.py
    - Implement extract_clip(): FFmpeg trim with normalized parameters
    - Implement concatenate(): FFmpeg concat demuxer
    - Implement render(): orchestrate extraction → normalization → concatenation → cleanup
    - Use list arguments for all subprocess calls
    - Raise RenderError with stderr on FFmpeg failure
    - Clean up temp clips after successful concatenation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 8.2 Implement clip boundary calculation in src/sentence_mixer/editing/boundaries.py
    - Helper functions for computing padded boundaries with clamping
    - Normalize time values for FFmpeg arguments
    - _Requirements: 7.2, 7.3, 7.4_

- [x] 9. Implement CLI interface
  - [x] 9.1 Implement CLI commands in src/sentence_mixer/cli.py
    - Implement `index` command: accept directory path, run indexing pipeline, display summary
    - Implement `generate` command: accept --sentence and --variations, run generation pipeline, display output paths
    - Handle and display missing word errors
    - Handle and display partial render failures
    - Wire all components together
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 10. Integration wiring and end-to-end validation
  - [x] 10.1 Wire indexing pipeline end-to-end
    - Connect Scanner → AudioExtractor → Transcriber → Database in index command
    - Handle per-video failures (mark failed, continue)
    - Report summary statistics
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 3.1, 3.2, 3.4_

  - [x] 10.2 Wire generation pipeline end-to-end
    - Connect Tokenizer → SearchEngine → Ranker → EDLGenerator → Renderer in generate command
    - Handle missing words, partial failures
    - Output files to output/ directory
    - _Requirements: 4.1, 5.1, 5.3, 6.1, 7.1, 8.1, 8.3, 9.4_

  - [ ]* 10.3 Write property test for word search consistency
    - **Property 7: Word Search Consistency**
    - Store random words via database layer, search by normalized form, verify found
    - **Validates: Requirements 5.1, 5.4**

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- FFmpeg and WhisperX should be mocked in unit/property tests; integration tests use small real files
- The `hypothesis` library is used for all property-based tests

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "3.2", "4.1"] },
    { "id": 4, "tasks": ["3.3", "4.2", "4.3"] },
    { "id": 5, "tasks": ["4.4", "6.1"] },
    { "id": 6, "tasks": ["6.2", "7.1"] },
    { "id": 7, "tasks": ["6.3", "7.2", "8.1"] },
    { "id": 8, "tasks": ["8.2", "9.1"] },
    { "id": 9, "tasks": ["10.1", "10.2"] },
    { "id": 10, "tasks": ["10.3"] }
  ]
}
```
