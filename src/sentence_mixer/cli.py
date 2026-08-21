"""CLI interface for Sentence Mixer."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import typer

from sentence_mixer.database.database import Database
from sentence_mixer.editing.edl import EDLGenerator
from sentence_mixer.editing.renderer import RenderError, Renderer
from sentence_mixer.ingestion.audio import AudioExtractor
from sentence_mixer.ingestion.scanner import Scanner
from sentence_mixer.models.schemas import RankingConfig, RenderConfig, VideoStatus
from sentence_mixer.search.candidate import SearchEngine, WordNotFoundError
from sentence_mixer.search.phrase_search import PhraseSearchEngine
from sentence_mixer.search.ranking import Ranker
from sentence_mixer.search.tokenizer import Tokenizer
from sentence_mixer.transcription.whisperx import Transcriber

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="sentence-mixer",
    help="Sentence Mixer - Construct sentences from video libraries",
)


@app.command()
def index(
    directory: Path = typer.Argument(..., help="Directory containing video files"),
    db_path: Path = typer.Option("data/sentence_mixer.db", help="Database path"),
) -> None:
    """Index a directory of video files for word-level search."""
    # Initialize database
    db = Database(db_path)
    db.initialize()

    # Create scanner with already-indexed paths
    indexed_paths: set[Path] = set()
    conn = db.connection
    rows = conn.execute(
        "SELECT path FROM videos WHERE status = ?", (VideoStatus.INDEXED.value,)
    ).fetchall()
    for row in rows:
        indexed_paths.add(Path(row["path"]))

    scanner = Scanner(indexed_paths=indexed_paths)

    # Scan directory for videos
    typer.echo(f"Scanning {directory} for video files...")
    videos = scanner.scan_directory(directory)

    if not videos:
        typer.echo("No new video files found to index.")
        return

    typer.echo(f"Found {len(videos)} new video(s) to index.")

    # Set up audio extractor and transcriber
    audio_dir = Path("data/audio")
    audio_extractor = AudioExtractor()
    transcriber = Transcriber()
    tokenizer = Tokenizer()

    indexed_count = 0
    word_count = 0

    for video_meta in videos:
        typer.echo(f"  Indexing: {video_meta.filename}")

        video_id = db.upsert_video(video_meta)

        try:
            # Extract audio
            audio_path = audio_extractor.extract_audio(video_meta.path, audio_dir)

            # Transcribe
            result = transcriber.transcribe(audio_path)

            # Normalize words and set video_id
            for seg_idx, segment in enumerate(result.segments):
                segment.video_id = video_id

            for word in result.words:
                word.video_id = video_id
                word.normalized_word = tokenizer.normalize_word(word.word)

            # Store transcription
            db.store_transcription(video_id, result.segments, result.words)
            db.update_video_status(video_id, VideoStatus.INDEXED.value)

            indexed_count += 1
            word_count += len(result.words)

        except Exception as e:
            logger.warning("Failed to index %s: %s", video_meta.filename, e)
            db.update_video_status(video_id, VideoStatus.FAILED.value)
            typer.echo(f"    Failed: {e}", err=True)
            continue

    typer.echo(f"Indexed {indexed_count} videos, stored {word_count} words")
    db.close()


@app.command()
def generate(
    sentence: str = typer.Option(..., "--sentence", help="Sentence to generate"),
    variations: int = typer.Option(5, "--variations", help="Number of variations"),
    db_path: Path = typer.Option("data/sentence_mixer.db", help="Database path"),
    output_dir: Path = typer.Option("output", help="Output directory"),
    padding: float = typer.Option(0.05, "--padding", help="Clip padding in seconds"),
    # V1.5 new options
    gap: float = typer.Option(80.0, "--gap", help="Default inter-word silence gap in ms"),
    punctuation_pause: bool = typer.Option(
        True,
        "--punctuation-pause/--no-punctuation-pause",
        help="Enable punctuation-aware silence timing",
    ),
    subtitles: bool = typer.Option(
        True,
        "--subtitles/--no-subtitles",
        help="Enable subtitle overlay",
    ),
    strict: bool = typer.Option(
        False,
        "--strict/--no-strict",
        help="Fail on missing words (default: best-effort)",
    ),
    speed: float = typer.Option(
        0.9,
        "--speed",
        help="Playback speed multiplier (0.5-2.0, default 0.9 for slight slowdown)",
    ),
    round_robin: bool = typer.Option(
        False,
        "--round-robin/--no-round-robin",
        help="Cycle words through different video sources (multi-speaker mode)",
    ),
) -> None:
    """Generate a sentence from indexed video library."""
    # Open database
    db = Database(db_path)
    db.initialize()

    # 1. Tokenize with context (for punctuation-aware gaps)
    tokenizer = Tokenizer()
    token_infos = tokenizer.tokenize_with_context(sentence)
    tokens = [t.normalized for t in token_infos]

    if not tokens:
        typer.echo("Error: Sentence contains no valid tokens.", err=True)
        raise typer.Exit(code=1)

    # 2. Phrase matching (try multi-word sequences first)
    phrase_engine = PhraseSearchEngine(db)
    phrase_matches, covered_positions = phrase_engine.find_phrases(tokens)

    # 3. Single-word search for uncovered positions
    uncovered_tokens = [tokens[i] for i in range(len(tokens)) if i not in covered_positions]
    search_engine = SearchEngine(db)

    if uncovered_tokens:
        try:
            word_candidates, missing_words = search_engine.find_candidates_batch(
                uncovered_tokens, strict=strict
            )
        except WordNotFoundError as e:
            typer.echo(f"Words not found: {', '.join(e.missing_words)}", err=True)
            raise typer.Exit(code=1)
    else:
        word_candidates = {}
        missing_words = []

    # Report missing words in best-effort mode
    if missing_words:
        typer.echo(
            f"Warning: Skipping words not found in library: {', '.join(missing_words)}",
            err=True,
        )

    # Check if we have anything to work with
    if not word_candidates and not phrase_matches:
        typer.echo("Error: Could not generate any variations.", err=True)
        raise typer.Exit(code=1)

    # 4. Set up ranker and filter candidates
    ranking_config = RankingConfig(prefer_same_speaker=not round_robin)
    ranker = Ranker(ranking_config)

    if word_candidates:
        filtered_candidates = {
            token: ranker.filter_candidates(candidates)
            for token, candidates in word_candidates.items()
        }
    else:
        filtered_candidates = {}

    # 5. Build final selections — one candidate per POSITION (not per unique token)
    final_variations: list[list] = []
    for var_idx in range(variations):
        selection = _build_merged_selection(
            tokens, token_infos, phrase_matches, covered_positions,
            filtered_candidates, missing_words, ranker, round_robin, var_idx,
        )
        if selection:
            final_variations.append(selection)

    # Deduplicate identical variations
    seen_keys: set[tuple] = set()
    unique_variations: list[list] = []
    for sel in final_variations:
        key = tuple(id(s) for s in sel)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_variations.append(sel)
    final_variations = unique_variations

    if not final_variations:
        typer.echo("Error: Could not generate any variations.", err=True)
        raise typer.Exit(code=1)

    # 6. Generate EDL manifests and render
    edl_generator = EDLGenerator(
        clip_padding=padding,
        default_gap_ms=gap,
        punctuation_pause_enabled=punctuation_pause,
    )
    render_config = RenderConfig(
        clip_padding=padding,
        default_gap_ms=gap,
        punctuation_pause_enabled=punctuation_pause,
        subtitles_enabled=subtitles,
        playback_speed=speed,
    )
    renderer = Renderer(render_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Use first 4 normalized tokens + timestamp for short, unique filenames
    words_prefix = "-".join(tokens[:4])
    timestamp = datetime.now().strftime("%H%M%S")
    slug = f"{words_prefix}_{timestamp}"

    successful_outputs: list[Path] = []
    failed_variations_idx: list[int] = []

    # Build the token_infos for EDL generation (only for non-skipped tokens)
    skipped_words = missing_words if missing_words else []

    for i, selection in enumerate(final_variations):
        # Build token_infos matching the selection (skip missing tokens)
        selection_token_infos = _build_selection_token_infos(
            tokens, token_infos, phrase_matches, covered_positions, missing_words
        )

        edl_manifest = edl_generator.generate(
            selections=selection,
            sentence=sentence,
            variation_index=i,
            token_infos=selection_token_infos,
            skipped_words=skipped_words,
        )

        # Save manifest JSON
        manifest_path = manifest_dir / f"{slug}_v{i:03d}.json"
        manifest_path.write_text(
            edl_manifest.model_dump_json(indent=2), encoding="utf-8"
        )

        # Render video
        output_path = output_dir / f"{slug}_v{i:03d}.mp4"

        try:
            renderer.render(edl_manifest, output_path)
            successful_outputs.append(output_path)
        except RenderError as e:
            logger.warning("Render failed for variation %d: %s", i, e)
            failed_variations_idx.append(i)
            continue

    # Report results
    if successful_outputs:
        typer.echo(f"Generated {len(successful_outputs)} variation(s):")
        for path in successful_outputs:
            typer.echo(f"  {path}")

    if failed_variations_idx:
        typer.echo(
            f"Warning: {len(failed_variations_idx)} variation(s) failed to render: "
            f"{', '.join(str(v) for v in failed_variations_idx)}",
            err=True,
        )

    if not successful_outputs:
        typer.echo("Error: All variations failed to render.", err=True)
        raise typer.Exit(code=1)

    db.close()


def _build_merged_selection(
    tokens: list[str],
    token_infos: list,
    phrase_matches: dict[tuple[int, int], list],
    covered_positions: set[int],
    word_candidates: dict[str, list],
    missing_words: list[str],
    ranker: "Ranker",
    round_robin: bool = False,
    var_idx: int = 0,
) -> list:
    """Build a selection list in token order, one candidate per POSITION.

    For positions covered by phrase matches, use the top PhraseCandidate.
    For uncovered positions, pick the best candidate from word_candidates[token].
    Skip positions whose token is in missing_words.

    This approach fixes the bug where duplicate tokens (e.g. "your" appearing
    at multiple positions) would be collapsed by the dict-based ranker, causing
    word skipping and ordering errors.

    Args:
        tokens: Full ordered token list.
        token_infos: Full ordered TokenInfo list.
        phrase_matches: Dict mapping (start, end) to list of PhraseCandidates.
        covered_positions: Set of positions covered by phrase matches.
        word_candidates: Dict mapping token string to its candidate list.
        missing_words: Words not found in library.
        ranker: Ranker instance for scoring candidates.
        round_robin: Whether to cycle through video sources.
        var_idx: Variation index (offsets round-robin starting position).

    Returns:
        Ordered list of WordCandidate/PhraseCandidate for EDL generation.
    """
    selection = []
    phrase_positions_used: set[int] = set()
    used_sources: set[int | None] = set()

    # For round-robin, collect all available sources
    all_sources: list[int] = []
    if round_robin:
        seen: set[int] = set()
        for candidates in word_candidates.values():
            for c in candidates:
                if c.video.id is not None and c.video.id not in seen:
                    seen.add(c.video.id)
                    all_sources.append(c.video.id)

    position_counter = var_idx  # Offset by var_idx for different variations

    for pos in range(len(tokens)):
        token = tokens[pos]

        # Skip missing words at uncovered positions
        if token in missing_words and pos not in covered_positions:
            continue

        if pos in covered_positions:
            # This position is covered by a phrase match
            if pos in phrase_positions_used:
                # Already handled as part of a phrase
                continue

            # Find which phrase covers this position
            for (start, end), candidates in phrase_matches.items():
                if start <= pos < end:
                    if candidates:
                        selection.append(candidates[0])
                    for p in range(start, end):
                        phrase_positions_used.add(p)
                    position_counter += 1
                    break
        else:
            # Uncovered position — pick best candidate for THIS position
            if token in missing_words:
                continue

            candidates = word_candidates.get(token, [])
            if not candidates:
                continue

            if round_robin and all_sources:
                # Round-robin: pick from the next source in rotation
                target_source_idx = position_counter % len(all_sources)
                target_source = all_sources[target_source_idx]
                best = _pick_candidate_from_source(candidates, target_source, ranker)
                if best is None:
                    best = _pick_best_candidate(candidates, ranker, used_sources)
            else:
                best = _pick_best_candidate(candidates, ranker, used_sources)

            if best is not None:
                selection.append(best)
                if best.video.id is not None:
                    used_sources.add(best.video.id)

            position_counter += 1

    return selection


def _pick_best_candidate(candidates: list, ranker: "Ranker", used_sources: set) -> object | None:
    """Pick the highest-scoring candidate considering used sources."""
    if not candidates:
        return None
    scored = [(ranker.score_candidate(c, used_sources), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _pick_candidate_from_source(candidates: list, target_source: int, ranker: "Ranker") -> object | None:
    """Pick the best candidate from a specific source video."""
    source_candidates = [c for c in candidates if c.video.id == target_source]
    if not source_candidates:
        return None
    scored = [(ranker.score_candidate(c), c) for c in source_candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _build_selection_token_infos(
    tokens: list[str],
    token_infos: list,
    phrase_matches: dict[tuple[int, int], list],
    covered_positions: set[int],
    missing_words: list[str],
) -> list:
    """Build token_infos list matching the selection order.

    For phrase matches, use the TokenInfo of the last token in the phrase
    (its trailing punctuation determines the gap after the phrase clip).
    For single-word selections, use the corresponding TokenInfo.
    Skip missing words.

    Returns:
        Ordered list of TokenInfo matching the selection.
    """
    result = []
    phrase_positions_used: set[int] = set()

    for pos in range(len(tokens)):
        token = tokens[pos]

        if token in missing_words and pos not in covered_positions:
            continue

        if pos in covered_positions:
            if pos in phrase_positions_used:
                continue

            # Find which phrase covers this position
            for (start, end), candidates in phrase_matches.items():
                if start <= pos < end:
                    # Use the TokenInfo for the last token in the phrase
                    result.append(token_infos[end - 1])
                    for p in range(start, end):
                        phrase_positions_used.add(p)
                    break
        else:
            if token not in missing_words:
                result.append(token_infos[pos])

    return result


def _slugify(text: str, max_length: int = 80) -> str:
    """Create a filesystem-safe slug from text.

    Args:
        text: Input text to slugify.
        max_length: Maximum slug length (default 80).

    Returns:
        A lowercase, hyphen-separated version of the text with
        only alphanumeric characters and hyphens, truncated to max_length.
    """
    slug = text.lower().strip()
    result = []
    for char in slug:
        if char.isalnum():
            result.append(char)
        elif char in (" ", "-", "_"):
            if result and result[-1] != "-":
                result.append("-")
    full_slug = "".join(result).strip("-")

    # Truncate to max_length, don't cut mid-word if possible
    if len(full_slug) <= max_length:
        return full_slug

    truncated = full_slug[:max_length]
    # Try to cut at a hyphen boundary for cleaner names
    last_hyphen = truncated.rfind("-")
    if last_hyphen > max_length // 2:
        truncated = truncated[:last_hyphen]
    return truncated.strip("-")


@app.command()
def words(
    db_path: Path = typer.Option("data/sentence_mixer.db", help="Database path"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path (prints to stdout if not set)"),
    min_count: int = typer.Option(1, "--min-count", help="Minimum occurrence count to include"),
    show_counts: bool = typer.Option(False, "--counts/--no-counts", help="Show occurrence counts"),
    sort_by: str = typer.Option("alpha", "--sort-by", help="Sort by: alpha, count, length"),
) -> None:
    """Export all available words from the indexed library.

    Useful for giving to an LLM to generate scripts using only available words.
    """
    db = Database(db_path)
    db.initialize()
    conn = db.connection

    rows = conn.execute(
        """
        SELECT normalized_word, COUNT(*) as cnt
        FROM words
        GROUP BY normalized_word
        HAVING cnt >= ?
        ORDER BY normalized_word ASC
        """,
        (min_count,),
    ).fetchall()

    db.close()

    if not rows:
        typer.echo("No words found in the database.", err=True)
        raise typer.Exit(code=1)

    # Apply sorting
    if sort_by == "count":
        rows = sorted(rows, key=lambda r: r["cnt"], reverse=True)
    elif sort_by == "length":
        rows = sorted(rows, key=lambda r: len(r["normalized_word"]), reverse=True)
    # else: already sorted alphabetically from SQL

    # Build output lines
    lines = []
    if show_counts:
        for row in rows:
            lines.append(f"{row['normalized_word']} ({row['cnt']})")
    else:
        for row in rows:
            lines.append(row["normalized_word"])

    content = "\n".join(lines)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        typer.echo(f"Exported {len(rows)} unique words to {output}")
    else:
        typer.echo(content)

    # Always print summary to stderr so it doesn't pollute the word list
    typer.echo(
        f"\nTotal: {len(rows)} unique words, {sum(r['cnt'] for r in rows)} occurrences",
        err=True,
    )


if __name__ == "__main__":
    app()
