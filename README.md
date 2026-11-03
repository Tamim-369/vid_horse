# ash_speaks

AI-powered vertical video factory. Feed it a niche, get back polished conspiracy-theory shorts — script, assets, voice, music, render — fully automated.

```
ash_speaks discover     # populate catalog from Wikipedia + fandom
ash_speaks theories     # browse catalogued topics
ash_speaks run-all X Y Z -f short   # script → assets → voice → render
```

## Quick Start

```bash
# 1. Install (uv recommended)
pip install -e .

# 2. Configure API keys in .env
cp .env.example .env
# add your GROQ_API_KEY or OLLAMA_API_KEY

# 3. Discover theories (one-time)
ash_speaks discover -l 50

# 4. List available topics
ash_speaks theories -l 10 -o popularity

# 5. Generate 10 shorts
ash_speaks run-all lost-cosmonauts montauk-project philadelphia-experiment \
  flat-earth mkultra denver-airport haarp roswell incident-1947 titanic-switch -f short
```

## What It Does

| Stage | Description |
|-------|-------------|
| **discover** | Crawls Wikipedia category trees + fandom wikis, stores theories in local SQLite catalog with pageview metrics |
| **research** | Fetches article text, extracts claims, builds a structured brief |
| **script** | LLM writes a vertical-format script (hook → 4–6 scenes → CTA) with visual direction per scene |
| **assets** | Multi-source image/video hunt (Commons, NASA, Openverse, Wikidata, Fandom, Yandex) → CLIP scoring → vision-model judge → LLM chooser → deduplicated asset pack |
| **voice** | Clones your reference voice (Chatterbox TTS), caches line-by-line |
| **render** | Motion stills, title cards, SFX shelf, music bed, burned-in subtitles → 1080×1920 MP4 |

## Architecture

```
src/
├── blocks/              # Pipeline stages (research, script, assets, voice, render)
│   ├── asset_research/  # 12 focused modules (media, chooser, staging, routing, judging, packing…)
│   ├── script_generation/
│   └── topic_research/
├── pipeline/            # Orchestration & factory
├── providers/           # LLM (Groq, Ollama local, Ollama Cloud), vision, HTTP, media sources
├── catalog/             # Theory discovery & storage
├── render/              # Compositing (cards, audio_mix, assemble)
└── cli/                 # ash_speaks entrypoint
```

## Current Niche

**Conspiracy theories / fringe mysteries** — hard-coded but trivial to change:

```python
# config/settings.py
discovery_categories = ["Conspiracy theories"]
discovery_seed_pages = ["List of conspiracy theories"]
fandom_wikis = ["conspiracy.fandom.com"]
```

Add cryptids, cold cases, abandoned places, UFOs, lost media… by editing three lines.

## Quality Gates (Assets)

- **CLIP** visual–text relevance scoring
- **Vision judge** (qwen3-vl) verifies each image matches the scene
- **LLM chooser** picks best asset with rationale
- **Dedupe + no back-to-back repeats** + proper-noun grounding
- **License filter** — CC-BY / CC-BY-SA / Public Domain only
- **Min score threshold** (configurable)

## Known Bottlenecks

| Component | Issue | Fix Path |
|-----------|-------|----------|
| Vision judge (qwen3-vl) | 30–90s/call on CPU | GPU / faster model / batch judge |
| Groq LLMs | 4 keys × 8k TPM → rate limits | More keys / Ollama Cloud / local |
| Ollama Cloud | Monthly credit cap | Add credits / local fallback |
| Yandex / web search | Fragile scraping | Official APIs / drop |

## Roadmap (We're Just Getting Started)

- [ ] GPU-accelerated vision judging (batch CLIP + vision LLM)
- [ ] Multi-niche discovery with per-niche configs
- [ ] Scheduled cron: discover → render → upload (YouTube Shorts / TikTok / Reels API)
- [ ] A/B thumbnail/title generation
- [ ] Viewer retention analytics loop back into script/asset selection
- [ ] Local-first: fully offline mode with quantized models
- [ ] Plugin system for custom niches / render styles / voice profiles

## Config

All settings via `.env` (prefix `VIDEOAI_`):

```bash
VIDEOAI_LLM_PROVIDER=ollama-cloud   # groq | ollama | ollama-cloud
VIDEOAI_OLLAMA_CLOUD_API_KEY=xxx
VIDEOAI_GROQ_API_KEY_FIRST=xxx
VIDEOAI_OUTPUT_DIR=output
VIDEOAI_CACHE_DIR=.cache/videoai
```

See `config/settings.py` for full list.

## License

MIT — but generated content inherits source licenses (CC-BY-SA for Wikimedia assets, etc.). Respect attributions.