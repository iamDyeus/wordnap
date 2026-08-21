# Implementation Plan: Sentence Mixer V1.5 Improvements

## Overview

Incremental enhancement of the existing Sentence Mixer pipeline to add inter-word silence gaps, phrase-level matching, subtitle overlays, best-effort missing word handling, and a multi-pass filter-then-rank clip selection strategy. All changes modify existing files with one new module (`phrase_search.py`).

## Tasks

- [ ] 1. Update data models in schemas
  - [ ] 1.1 Add new models (TokenInfo, GapEntry, PhraseCandidate) and update existing configs (RankingConfig, RenderConfig, EDLManifest)
    - Add `TokenInfo` model with `normalized`, `original`, and `trailing_punctuation` fields
    - Add `GapEntry` model with `duration_ms` and `reason` fields
    - Add `PhraseCandidate` model with `words`, `segment`, `video`, `start_time`, `end_time`, `duration`, `score` fields and `validate_words_consecutive` validator
    - Update `RankingConfig` to new weights: confidence 0.35, duration 0.25, boundary_quality 0.20, diversity 0.20; add `min_confidence`, `max_duration`, `ideal_duration_min`, `ideal_duration_max` fields
    - Update `RenderConfig` with `default_gap_ms`, `punctuation_pause_enabled`, `comma_gap_ms`, `sentence_end_gap_ms`, `subtitles_enabled` fields
    - Update `EDLManifest` with `gaps: list[GapEntry]` and `skipped_words: list[str]` fields
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 2.4, 2.7, 3.4, 4.6, 5.1, 5.4, 5.5_

  - [ ]* 1.2 Write unit tests for new and updated models
    - Test `TokenInfo` construction with and without trailing punctuation
    - Test `GapEntry` creation for each reason type
    - Test `PhraseCandidate` validator rejects words from different segments
    - Test `EDLManifest` with gaps and skipped_words fields
    - Test updated `RankingConfig` and `RenderConfig` defaults
    - _Requirements: 1.1, 2.4, 5.1_

- [ ] 2. Enhance Tokenizer with punctuation context
  - [ ] 2.1 Add `tokenize_with_context()` method to Tokenizer class
    - Add new method in `src/sentence_mixer/search/tokenizer.py`
    - Splits on whitespace, normalizes each token, detects trailing punctuation from the original word
    - Returns `list[TokenInfo]` preserving punctuation context for silence gap calculation
    - Existing `tokenize()` method remains unchanged for backward compatibility
    - _Requirements: 1.3, 1.4_

  - [ ]* 2.2 Write property test for tokenizer punctuation preservation
    - **Property 16: Tokenizer punctuation preservation round-trip**
    - **Validates: Requirements 1.3, 1.4**

  - [ ]* 2.3 Write unit tests for `tokenize_with_context()`
    - Test sentence with comma produces TokenInfo with `trailing_punctuation=","`
    - Test sentence with period, exclamation, question marks
    - Test sentence with no punctuation produces all `None` trailing_punctuation
    - Test empty and whitespace-only input returns empty list
    - _Requirements: 1.3, 1.4_

- [ ] 3. Create phrase search module
  - [ ] 3.1 Add `find_consecutive_words_in_segment()` query to `database/queries.py`
    - Add function that verifies consecutive words in a segment match expected tokens
    - Queries words ordered by start_time starting from a given word, checks the next N words match
    - Returns matching Word list or None if not consecutive
    - _Requirements: 2.3, 2.4_

  - [ ] 3.2 Create `src/sentence_mixer/search/phrase_search.py` with `PhraseSearchEngine` class
    - Implement `find_phrases(tokens)` with longest-first greedy matching strategy
    - For each window size from len(tokens) down to 2, slide across token list
    - Query segments containing first word, verify remaining words are consecutive
    - Return dict mapping (start_pos, end_pos) to list of PhraseCandidates and set of covered positions
    - Implement `_find_phrase_in_segments(token_window)` helper
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.3 Write property tests for phrase search
    - **Property 4: Phrase matching uses longest-first greedy strategy**
    - **Validates: Requirements 2.2, 2.5**

  - [ ]* 3.4 Write property test for consecutive word validation
    - **Property 5: Phrase match validates consecutive words within a segment**
    - **Validates: Requirements 2.3, 2.4**

  - [ ]* 3.5 Write property test for token coverage completeness
    - **Property 6: Token coverage completeness**
    - **Validates: Requirements 2.1, 2.6**

- [ ] 4. Update SearchEngine with best-effort mode
  - [ ] 4.1 Modify `SearchEngine.find_candidates_batch()` to accept `strict` parameter
    - Add `strict: bool = True` parameter to `find_candidates_batch()` in `src/sentence_mixer/search/candidate.py`
    - When `strict=False`, return partial results with missing word list instead of raising
    - Return type changes to `tuple[dict[str, list[WordCandidate]], list[str]]`
    - When `strict=True`, preserve existing behavior (raise `WordNotFoundError`)
    - _Requirements: 4.1, 4.2, 4.4, 4.6_

  - [ ]* 4.2 Write property tests for best-effort mode
    - **Property 9: Best-effort mode produces output from available tokens**
    - **Validates: Requirements 4.1, 4.2, 4.6**

  - [ ]* 4.3 Write property test for strict mode
    - **Property 10: Strict mode rejects on any missing token**
    - **Validates: Requirements 4.4**

- [ ] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Update Ranker with filter-then-rank strategy
  - [ ] 6.1 Rewrite `Ranker` class in `src/sentence_mixer/search/ranking.py` with filter-then-rank
    - Add `filter_candidates()` method that rejects confidence < 0.3 and duration outside [0.03, 3.0]s, with fallback to unfiltered list
    - Add `compute_duration_score()` with piecewise linear scoring: 1.0 in [0.1, 1.5]s, linearly decreasing outside
    - Add `compute_boundary_quality()` measuring proximity to natural silence boundaries
    - Add `compute_diversity_score()` returning 1.0 for unused sources, 0.0 otherwise
    - Update `score_candidate()` to use composite weights: confidence 0.35 + duration 0.25 + boundary 0.20 + diversity 0.20
    - Update `rank()` method to apply filter pass before scoring, accept `used_sources` tracking
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 6.2 Write property test for filter pass criteria
    - **Property 11: Filter pass criteria**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 6.3 Write property test for filter fallback
    - **Property 12: Filter fallback preserves candidates**
    - **Validates: Requirements 5.3**

  - [ ]* 6.4 Write property test for composite scoring
    - **Property 13: Composite score uses correct weights**
    - **Validates: Requirements 5.4**

  - [ ]* 6.5 Write property test for duration reasonableness
    - **Property 14: Duration reasonableness is piecewise linear**
    - **Validates: Requirements 5.5**

  - [ ]* 6.6 Write property test for source diversity
    - **Property 15: Source diversity prefers unused sources**
    - **Validates: Requirements 5.7**

- [ ] 7. Update EDLGenerator with gap entries and skipped words
  - [ ] 7.1 Update `EDLGenerator` in `src/sentence_mixer/editing/edl.py`
    - Add constructor parameters: `default_gap_ms`, `punctuation_pause_enabled`, `comma_gap_ms`, `sentence_end_gap_ms`
    - Implement `compute_gap(token_info)` method mapping punctuation to gap durations
    - Update `generate()` to accept `token_infos` and `skipped_words` parameters
    - Generate `GapEntry` for each consecutive clip pair based on punctuation context
    - Populate `EDLManifest.gaps` and `EDLManifest.skipped_words`
    - Handle `PhraseCandidate` by using first word's start_time and last word's end_time as clip boundaries
    - _Requirements: 1.1, 1.3, 1.4, 1.6, 2.7, 4.6_

  - [ ]* 7.2 Write property test for gap count
    - **Property 1: Silence gap count equals clips minus one**
    - **Validates: Requirements 1.1**

  - [ ]* 7.3 Write property test for punctuation-aware gaps
    - **Property 2: Punctuation-aware gap duration matches punctuation type**
    - **Validates: Requirements 1.3, 1.4**

  - [ ]* 7.4 Write property test for disabled punctuation-pause
    - **Property 3: Disabled punctuation-pause produces uniform gaps**
    - **Validates: Requirements 1.6**

  - [ ]* 7.5 Write property test for phrase candidate clip boundaries
    - **Property 7: PhraseCandidate clip boundaries span first-to-last word**
    - **Validates: Requirements 2.7**

- [ ] 8. Update Renderer with silence generation and subtitle overlay
  - [ ] 8.1 Add `generate_silence()` method to `Renderer` class
    - Implement in `src/sentence_mixer/editing/renderer.py`
    - Generate silence audio segment using FFmpeg's `anullsrc` filter
    - Accept `duration_ms` and `output_path`, raise `RenderError` on failure
    - _Requirements: 1.1, 1.7_

  - [ ] 8.2 Update `extract_clip()` to support subtitle overlay via drawtext filter
    - Add `subtitle_text: str | None = None` parameter
    - When subtitle_text provided and subtitles enabled, add drawtext filter: white text, black outline, centered at bottom
    - Escape special characters for FFmpeg drawtext
    - Build video filter chain combining scale and optional drawtext
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [ ] 8.3 Update `render()` method to interleave silence gaps between clips
    - Read `manifest.gaps` to insert silence segments between extracted clips
    - Pass subtitle text to `extract_clip()` based on subtitles_enabled config
    - Concatenate interleaved clips and silence into final output
    - _Requirements: 1.1, 1.7, 3.1, 3.5_

  - [ ]* 8.4 Write property test for subtitle presence
    - **Property 8: Subtitle presence matches enabled flag**
    - **Validates: Requirements 3.1, 3.5, 3.6**

  - [ ]* 8.5 Write unit tests for silence generation and render flow
    - Test `generate_silence()` constructs correct FFmpeg command
    - Test `render()` interleaves silence gaps correctly
    - Test subtitle overlay escapes special characters
    - _Requirements: 1.1, 1.7, 3.1, 3.2_

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Update CLI and wire all components together
  - [ ] 10.1 Update `generate` command in `src/sentence_mixer/cli.py` with new flags and pipeline wiring
    - Add CLI options: `--gap`, `--punctuation-pause/--no-punctuation-pause`, `--subtitles/--no-subtitles`, `--strict/--no-strict`
    - Wire `tokenize_with_context()` to get `TokenInfo` list
    - Integrate `PhraseSearchEngine` for phrase matching before single-word fallback
    - Pass `strict` flag to `SearchEngine.find_candidates_batch()`
    - Handle best-effort mode: warn on skipped words, error if all words missing
    - Pass new config values to `RenderConfig`, `EDLGenerator`, and `Renderer`
    - Pass `token_infos` and `skipped_words` to `EDLGenerator.generate()`
    - _Requirements: 1.2, 1.5, 2.1, 2.6, 3.4, 4.1, 4.2, 4.3, 4.5_

  - [ ]* 10.2 Write unit tests for CLI generate command with new flags
    - Test `--gap` option sets default gap in ms
    - Test `--no-punctuation-pause` disables punctuation-aware timing
    - Test `--no-subtitles` disables subtitle overlay
    - Test `--strict` mode raises on missing words
    - Test best-effort mode warns on skipped words
    - _Requirements: 1.2, 1.5, 3.4, 4.2, 4.5_

- [ ] 11. Integration tests
  - [ ]* 11.1 Write integration test for full pipeline with silence gaps and subtitles
    - Test end-to-end generation with punctuation-aware gaps
    - Verify manifest contains correct gap entries matching punctuation
    - Verify subtitles flag propagates through to render
    - _Requirements: 1.1, 1.3, 1.4, 3.1_

  - [ ]* 11.2 Write integration test for phrase matching pipeline
    - Test that multi-word phrases from same segment are used as single clips
    - Verify phrase candidates produce correct clip boundaries
    - _Requirements: 2.1, 2.7_

  - [ ]* 11.3 Write integration test for best-effort mode
    - Test generation succeeds with some words missing when strict=False
    - Verify skipped_words recorded in manifest
    - Test strict=True raises WordNotFoundError
    - _Requirements: 4.1, 4.4, 4.6_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All modifications target existing files except the new `phrase_search.py` module
- The `tokenize()` method is preserved unchanged for backward compatibility
- The `SearchEngine.find_candidates_batch()` signature changes (returns tuple in non-strict mode); callers in CLI must be updated

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "6.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "4.1", "6.2", "6.3", "6.4", "6.5", "6.6"] },
    { "id": 3, "tasks": ["3.2", "4.2", "4.3", "7.1"] },
    { "id": 4, "tasks": ["3.3", "3.4", "3.5", "7.2", "7.3", "7.4", "7.5", "8.1", "8.2"] },
    { "id": 5, "tasks": ["8.3", "8.4", "8.5"] },
    { "id": 6, "tasks": ["10.1"] },
    { "id": 7, "tasks": ["10.2", "11.1", "11.2", "11.3"] }
  ]
}
```
