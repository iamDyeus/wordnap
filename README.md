# Wordnap

> Kidnapping words from videos.

---

You've seen it in movies. A shadowy figure sends a video — spliced together from news clips, one word at a time, different faces delivering each syllable. It's unsettling. It's precise. And doing it manually? That's hours of scrubbing through footage, finding timestamps, cutting frames.

Wordnap does it in seconds.

Feed it a library of video files. It listens to every word spoken, indexes them with millisecond precision, and waits. When you type a sentence — any sentence — it assembles a new video, pulling each word from existing footage. One speaker per word if you want. Subtitles burned in. Speed adjusted so every syllable lands.

It's a video editing engine that treats speech as a searchable database. Index once. Generate forever.


## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Video      │     │  WhisperX   │     │   SQLite     │     │   FFmpeg    │
│  Library    │────▶│  Transcribe │────▶│   Word DB    │────▶│   Render    │
│  (.mp4s)    │     │  + Align    │     │  (indexed)   │     │   Output    │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘

1. INDEX: Scan videos → extract audio → transcribe → store word timestamps
2. SEARCH: Tokenize sentence → find each word in the DB → rank candidates
3. RENDER: Extract clips → concatenate → apply speed/subtitles → final MP4
```

## Features

- 🎯 **Word-level precision** — WhisperX alignment gives exact timestamps for every spoken word
- 🔄 **Round-robin mode** — Cycle words through different speakers for the ransom-note effect
- 📝 **Burned-in subtitles** — Current word displayed on screen as it plays
- 🐢 **Speed control** — Slow down output for clarity (0.5x - 2.0x)
- 🧠 **Phrase matching** — Prefers natural 2-3 word sequences over individual words when available
- 📖 **Dictionary export** — Export all available words for LLM script generation
- ⚡ **Index once, generate forever** — Transcription is cached in SQLite, generation is instant
- 🎭 **Best-effort mode** — Skips missing words instead of failing

## Requirements

- Python 3.11+
- FFmpeg (must be on PATH)
- GPU recommended for WhisperX (CPU works but is slow)
- ~2GB disk space for WhisperX model

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/wordnap.git
cd wordnap

# Install uv (if you don't have it)
# https://docs.astral.sh/uv/getting-started/installation/
pip install uv

# Create venv and install all dependencies
uv venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

uv synchttps://github.com/iamDyeus/wordnap.git

# Install WhisperX (requires PyTorch)
uv pip install whisperx
```

### Verify FFmpeg is installed:

```bash
ffmpeg -version
```

If not installed: [Download FFmpeg](https://ffmpeg.org/download.html) and add to PATH.

## Quick Start

### 1. Add videos to library

```bash
# Put video files in the library/ folder
# Supports: .mp4, .mkv, .avi, .mov, .webm
cp your_videos/*.mp4 library/
```

Good source material: speeches, interviews, podcasts, lectures — anything with clear speech.

### 2. Index your library

```bash
python -m wordnap.cli index ./library
```

This runs WhisperX on each video and stores every word with its timestamp. Takes ~2 min per 10-min video with GPU.

**Output:**

```
Scanning library for video files...
Found 3 new video(s) to index.
  Indexing: obama.mp4
  Indexing: trump.mp4
  Indexing: bush.mp4
Indexed 3 videos, stored 6123 words
```

### 3. Check your vocabulary

```bash
# Export all available words
python -m wordnap.cli words --output data/dictionary.txt

# See most common words with counts
python -m wordnap.cli words --counts --sort-by count
```

**Pro tip:** Feed `data/dictionary.txt` to ChatGPT/Claude with the prompt:

> "Write a funny 3-sentence script using ONLY these words. Make it sound like a serious political threat."

### 4. Generate a video

```bash
python -m wordnap.cli generate \
  --sentence "We are holding your attention hostage. Just a joke. Have a great day." \
  --round-robin \
  --subtitles \
  --speed 0.88 \
  --variations 1
```

**Output:**

```
Generated 1 variation(s):
  output/we-are-holding-your_143052_v000.mp4
```

## CLI Reference

### `index` — Transcribe and index videos

```bash
python -m wordnap.cli index ./library [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--db-path` | `data/wordnap.db` | Database file location |

Indexing is **idempotent** — re-running skips already-indexed files.

---

### `generate` — Create sentence videos

```bash
python -m wordnap.cli generate --sentence "..." [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--sentence` | *(required)* | The sentence to generate |
| `--variations` | `5` | Number of output variations |
| `--speed` | `0.9` | Playback speed (0.5–2.0). Lower = slower/clearer |
| `--padding` | `0.05` | Seconds of audio padding around each word |
| `--round-robin` | `off` | Cycle through different speakers per word |
| `--subtitles` / `--no-subtitles` | `on` | Burn subtitle text onto video |
| `--punctuation-pause` / `--no-punctuation-pause` | `on` | Longer pauses at commas/periods |
| `--gap` | `80.0` | Default silence gap between words (ms) |
| `--strict` / `--no-strict` | `off` | Fail on missing words vs skip them |
| `--output-dir` | `output` | Where to save generated videos |
| `--db-path` | `data/wordnap.db` | Database file location |

---

### `words` — Export available vocabulary

```bash
python -m wordnap.cli words [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output` / `-o` | stdout | Save to file |
| `--min-count` | `1` | Minimum word occurrences to include |
| `--counts` / `--no-counts` | `off` | Show occurrence counts |
| `--sort-by` | `alpha` | Sort: `alpha`, `count`, or `length` |

---

## Architecture

```
wordnap/
├── src/wordnap/
│   ├── cli.py                  # Typer CLI — index, generate, words commands
│   ├── ingestion/
│   │   ├── scanner.py          # Recursive video discovery + ffprobe metadata
│   │   ├── audio.py            # FFmpeg audio extraction (mono 16kHz WAV)
│   │   └── ffprobe.py          # Video metadata extraction
│   ├── transcription/
│   │   ├── whisperx.py         # WhisperX transcription + word-level alignment
│   │   └── alignment.py        # Alignment utilities
│   ├── database/
│   │   ├── database.py         # SQLite layer (videos, segments, words tables)
│   │   └── queries.py          # Query helpers + phrase search queries
│   ├── search/
│   │   ├── tokenizer.py        # Sentence → normalized tokens + punctuation context
│   │   ├── phrase_search.py    # Multi-word phrase matching (up to 3 words)
│   │   ├── candidate.py        # Search engine + WordNotFoundError
│   │   └── ranking.py          # Filter-then-rank + round-robin selection
│   ├── editing/
│   │   ├── edl.py              # EDL manifest generation (clips + gaps)
│   │   ├── renderer.py         # FFmpeg clip extraction + concat + speed
│   │   └── boundaries.py       # Clip boundary calculation utilities
│   └── models/
│       └── schemas.py          # Pydantic models (20+ data types)
├── library/                    # ← Put your videos here
├── data/
│   ├── wordnap.db       # SQLite database (auto-created)
│   ├── audio/                  # Cached extracted WAV files
│   └── dictionary.txt          # Exported word list
├── output/                     # ← Generated videos appear here
│   ├── *.mp4
│   └── manifests/*.json        # Clip-by-clip source attribution
├── assets/fonts/               # Bundled font for subtitles
└── tests/                      # 400+ unit & integration tests
```

### Pipeline Flow

```
USER INPUT: "We need to be different"
         │
         ▼
┌─── TOKENIZER ───┐
│ ["we", "need",   │
│  "to", "be",     │
│  "different"]    │
└────────┬─────────┘
         │
         ▼
┌─── PHRASE SEARCH ──┐     ┌─── WORD SEARCH ──┐
│ "need to" found as │     │ "we" → 47 clips  │
│ phrase in segment  │     │ "be" → 23 clips  │
│                    │     │ "different" → 3   │
└────────┬───────────┘     └────────┬──────────┘
         │                          │
         ▼                          ▼
┌──────── RANKER ─────────────────────┐
│ Score by: confidence (35%)           │
│           duration (25%)             │
│           boundary quality (20%)     │
│           source diversity (20%)     │
│                                      │
│ Round-robin: trump→obama→bush→trump  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────── EDL MANIFEST ───────────┐
│ { clips: [...], gaps: [...] }    │
│ JSON file for reproducibility    │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────── RENDERER ───────────────┐
│ 1. Extract each clip (FFmpeg)    │
│ 2. Add subtitles (drawtext)      │
│ 3. Concatenate (re-encode)       │
│ 4. Apply speed (post-process)    │
└────────────────┬─────────────────┘
                 │
                 ▼
            output.mp4
```

### Database Schema

```sql
videos    (id, path, filename, duration, width, height, fps, status)
segments  (id, video_id, start_time, end_time, text, speaker, confidence)
words     (id, segment_id, video_id, word, normalized_word, start_time, end_time, confidence)
```

Key index: `normalized_word` — enables instant word lookup across all videos.

## Examples

```bash
# Simple generation
python -m wordnap.cli generate --sentence "I do not like this" --variations 1

# Multi-speaker ransom note
python -m wordnap.cli generate \
  --sentence "We know what you did. You cannot hide." \
  --round-robin --speed 0.85 --variations 3

# Fast, no subtitles
python -m wordnap.cli generate \
  --sentence "The world is changing" \
  --no-subtitles --speed 1.0

# Strict mode (fail if any word missing)
python -m wordnap.cli generate --sentence "hello world" --strict

# Very slow for dramatic effect
python -m wordnap.cli generate \
  --sentence "We. Are. Watching. You." \
  --speed 0.7 --round-robin
```

## Tips & Tricks

- **More videos = bigger vocabulary.** Each 10-min clip adds ~1,500+ unique words.
- **`--speed 0.85-0.90`** makes individual words much more intelligible.
- **Export your word list** → feed to LLM → get scripts guaranteed to render.
- **`--round-robin`** is the star feature for multi-speaker content.
- **Indexing is cached.** You pay the WhisperX cost once per video.
- **Check the manifest JSONs** in `output/manifests/` to see exactly which clip was used for each word.

## Resetting the Database

```bash
# Clear all indexed data
python -c "
from pathlib import Path
from wordnap.database.database import Database
db = Database(Path('data/wordnap.db'))
db.initialize()
conn = db.connection
conn.execute('DELETE FROM words')
conn.execute('DELETE FROM segments')
conn.execute('DELETE FROM videos')
conn.commit()
db.close()
print('Database cleared.')
"

# Or just delete the file
rm data/wordnap.db
```

## Running Tests

```bash
pytest                    # Run all 400+ tests
pytest tests/ -v          # Verbose output
pytest tests/test_cli.py  # Run specific test file
```

## How the Ranking Works

When a word has multiple candidates (e.g., "the" appears 200 times across all videos), the ranker picks the best one using:

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| Confidence | 35% | WhisperX's certainty that the word is correct |
| Duration | 25% | Prefers 0.1s–1.5s clips (not too short, not too long) |
| Boundary Quality | 20% | Prefers clips with silence at start/end (clean cuts) |
| Source Diversity | 20% | Prefers clips from different videos (in round-robin mode) |

## Limitations

- No AI voice generation — this only splices existing footage
- Quality depends on source clarity — mumbled speech produces bad clips
- Short function words ("a", "the", "is") can sound clipped
- Very long sentences may have some missing words depending on vocabulary
- Speed < 0.8x starts sounding unnatural

## Origin Story

The idea came from a movie scene — hackers stitching together a video message from clips of public figures, word by word, to deliver demands nobody actually said. That stuck with me for years. Not because of the plot, but because of the craft — the precision of cutting individual words and making them flow.

Wordnap automates that craft. What would take an editor hours of scrubbing through footage, finding timestamps, and cutting frames — this does in seconds. It's a niche tool, but for that niche, nothing else exists.

## License

MIT
