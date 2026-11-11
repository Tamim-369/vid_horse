# Test Pipeline Scripts

## Overview
Three test scripts to verify the fully-local Pollinations-based video generation pipeline with subtitles.

## Prerequisites
```bash
# Install dependencies
uv sync
# or
pip install -e .

# Ensure Ollama is running with required models:
ollama pull gemma3:4b      # main writer/reviewer
ollama pull d3:2b          # decision/vision judge
ollama pull gemma3:1b      # asset chooser
ollama pull qwen3-vl:2b    # vision model
ollama pull nomic-embed-text  # embeddings

# Set Pollinations API key in .env
cp .env.example .env
# Edit .env and add: POLLINATIONS_API_KEY=your_key
```

## Test 1: Basic Pipeline (`test_pipeline_1_basic.py`)
**Purpose**: End-to-end test with a hardcoded lesson idea (no book chunk needed)

```bash
uv run python experiments/test_pipeline_1_basic.py
```

**What it tests**:
- ScriptEngineBlock: fact sheet → beats → scenes → validation → polish
- Music annotation (role-based music plan)
- Asset pack: Pollinations image generation for every sentence
- VoiceOver: Chatterbox TTS with sentence-level caching
- Render: Black-style video with word-by-word subtitles

**Expected output**: `output/test-01-illusion-of-control/test-01-illusion-of-control-short-black.mp4`

---

## Test 2: Book Chunk Pipeline (`test_pipeline_2_book_chunk.py`)
**Purpose**: Full pipeline using an actual psychology book chunk

```bash
# First, build book content (one-time)
uv run python experiments/build_psychology_book_content.py

# Then run test
uv run python experiments/test_pipeline_2_book_chunk.py
```

**What it tests**:
- LessonPicker: extracts teachable idea from real book chunk
- Full pipeline from book text to rendered video
- Ledger tracking (marks chunk as done)

**Expected output**: `output/<book>--<part>x<NNN>/<slug>-short-black.mp4`

---

## Test 3: Subtitle Timing (`test_pipeline_3_subtitles.py`)
**Purpose**: Verify word-by-word subtitle alignment accuracy

```bash
uv run python experiments/test_pipeline_3_subtitles.py
```

**What it tests**:
- Hand-crafted script with known sentence durations
- Whisper `tiny.en` word-level alignment
- 2-word subtitle groups burned at exact speech timestamps
- Per-line audio file duration verification

**Expected output**: `output/test-03-subtitle-timing/test-03-subtitle-timing-short-black.mp4`

---

## Verification Checklist

After each test, verify:

| File | Should Exist | Notes |
|------|-------------|-------|
| `short.json` | ✓ | VideoScript with hook, scenes, outro |
| `assets-short.json` | ✓ | AssetPack with Pollinations-generated images |
| `audio/voice.wav` | ✓ | Joined narration track |
| `audio/voice-XX.wav` | ✓ | Per-line cached WAVs |
| `<slug>-short-black.mp4` | ✓ | Final 1080×1920 video |

**Subtitle verification** (Test 3):
- Check `audio/voice-XX.wav` durations match expected sentence lengths
- In video: subtitles appear as 2-word groups, synchronized to speech
- No subtitle drift or missing words

---

## Architecture: Fully Local + Pollinations

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL ONLY (Ollama)                      │
├─────────────────────────────────────────────────────────────┤
│  LLM:     gemma3:4b (writer)  │  d3:2b (decision/vision)   │
│           gemma3:1b (assets)  │  qwen3-vl:2b (vision)      │
│  Embed:   nomic-embed-text (RAG)                              │
│  TTS:     Chatterbox Turbo Nano (voice clone)                │
│  RAG:     sqlite-vec (local vector DB)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              POLLINATIONS (External API)                    │
├─────────────────────────────────────────────────────────────┤
│  • Flux model (default)                                     │
│  • 1080×1920 vertical                                       │
│  • Dark cinematic noir style (hardcoded)                   │
│  • ALL images: motif library + on-demand generation        │
│  • NO Pexels, Openverse, NASA, Wikimedia                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Changes Made

### 1. `src/blocks/sentence_assets.py`
- **Removed**: Pexels, Openverse, NASA, Wikimedia download logic
- **Added**: `_generate_pollinations_image()` - primary image generator
- **Modified**: `_gather_candidates()` now generates via Pollinations only
- **Modified**: `_build_asset_from_concept()` uses "pollinations" source label
- **Renamed**: `used_stock_files` → `used_generated_files`
- **Increased**: `_ON_DEMAND_CAP` from 8 to 12

### 2. `src/render/black_style.py`
- **Fixed**: Removed duplicate function definitions (`_black_frame`, `_word_flash`, `_asset_clip`, `_join_cards`)

### 3. `experiments/test_pipeline_*.py`
- **Created**: 3 comprehensive test scripts

### 4. `.env.example`
- **Updated**: Emphasizes Pollinations as PRIMARY image source
- **Commented out**: Pexels/Openverse (no longer used)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No Pollinations API key" | Add `POLLINATIONS_API_KEY` to `.env` |
| "Model not found" | Run `ollama pull <model>` for each required model |
| "CUDA OOM" | Reduce `ollama_num_ctx` in settings, ensure thermal gate works |
| "Subtitles misaligned" | Check Whisper model (`base.en`); verify per-line WAV durations |
| "Video render fails" | Ensure ffmpeg is installed; check `render-work` dir permissions |

---

## Running All Tests

```bash
# Sequential (recommended for thermal management)
uv run python experiments/test_pipeline_1_basic.py
uv run python experiments/test_pipeline_3_subtitles.py
uv run python experiments/test_pipeline_2_book_chunk.py  # requires book content
```

Each test takes ~3-5 minutes depending on Pollinations response time and local model speed.