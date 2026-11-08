# Dark-Psychology Book → Script Pipeline: Task Plan

Source material lives in `content/psychology_books/` (one file per lesson,
`manifest.json` per book). Goal: one teaching video per lesson.

## Tasks (do in order, one at a time)

1. **Taken-parts ledger**
   - Track which lesson files already produced a script.
   - Store as `content/psychology_books/taken.json` (map: lesson file path →
     script output dir / status).
   - Agent reads ledger before choosing the next chunk; marks chunk "taken"
     after the script is accepted.

2. **Agentic RAG retrieval hook**
   - `ScriptEngineBlock` accepts an optional `RAGRetriever`.
   - Before fact-sheet generation, retrieve top chunks for the lesson topic
     from the ingested knowledge base (`assets/rag.sqlite3`).
   - Pass retrieved chunks into `generate_fact_sheet` as grounded source
     material so facts come from the book text, not model memory.

3. **Teacher persona prompts**
   - Rewrite `script_engine` SYSTEM prompts (facts / beats / writer) so the
     narrator teaches like a patient teacher to a 10-year-old-level audience:
     simple words, one idea per scene, concrete examples, no jargon,
     encourage "now you know" moments.

4. **Lesson-picker agent**
   - New block: given a lesson file, read it (plus neighbouring taken parts
     only when the current part needs prior context), extract ONE lesson/idea
     worth a 45-second script, return topic + source chunks + context notes.
   - Only chunks used as *context* stay un-taken; only the chunk whose lesson
     becomes the script gets marked.

5. **CLI / runner**
   - `experiments/generate_book_scripts.py`: walk `content/psychology_books/`,
     skip taken parts, run lesson-picker → script engine, save script per
     lesson under `output/<book>/<lesson>/`, update ledger.

6. **Ingestion + validation**
   - Ingest the psychology books into `assets/rag.sqlite3` so RAG can
     retrieve from them.
   - End-to-end run on ONE lesson with the local provider to validate the
     chain before batch processing.

## Status

### Original plan (scripts from chunks; superseded chunk format below)
- [x] 1 taken-parts ledger
- [x] 2 agentic RAG hook
- [x] 3 teacher persona
- [x] 4 lesson-picker agent
- [x] 5 runner
- [x] 6 ingestion + e2e validation
- [x] 7 output hardening

### Phase R (2026-11): re-extraction & ledger integrity (fix of broken chunking)
The original chunker split books per PDF PAGE with a fuzzy-UPPERCASE
heading regex, so running page headers ("THE CROWD") and printer's imprints
became "lesson titles" and each page became a "lesson" — including a full
video about edition/printing dates built from the imprint page. Rebuilt:

- [x] R1 ledger integrity — `src/utils/ledger.py`: failed chunks are
      recorded with `status: failed` + retry counter and stay pickable; only
      successful runs become `done`; after 3 retries a chunk is abandoned
      (marked done with failure note). Runners use per-run skip sets so a
      failing chunk doesn't loop within one run.
- [x] R2 re-extraction — `experiments/build_psychology_book_content.py`
      rebuilt: fuzzy clustering of running headers into furniture (and
      header-run chapter spans for scanned books, spread-majority across
      facing pages), printed-TOC parsing fallback for books with TOC,
      heading-lines fallback, paragraph reconstruction + dehyphenation,
      sentence-bounded idea chunks (~260 words), `teachable` flags in
      manifest, page OCR cache under `.cache/book_pages/`.
      Result: real chapter structure per book (e.g. The Crowd: 6 parts /
      188 teachable idea-chunks vs. the old 239 page-fragments; Delusions:
      16 chapters incl. Mississippi Scheme, Tulipomania, Crusades...).
- [x] R3 collision-safe slugs — `output/<book>--<part>xNNN/` (chunk
      numbering restarts per chapter; old `<book>--<NNN>` collided).
- [x] R4 front matter (title/imprint/publisher pages) excluded from
      lesson picking via manifest `teachable` flag.
- Old outputs archived under `output/_archive/pre-rechunk/`; ledger
  reset (backup: `taken.json.pre-rechunk`).

### Phase S (2026-11): grounded scripts + scored visuals + batch runner
- [x] S1 grounding (Phase 3) — prompt rewrite across `facts.py` /
      `beats.py` / `writer.py`: the script teaches THE BOOK ("never quote
      the source" directive removed; chunk text is now authoritative —
      2400 chars per chunk instead of 600), chapter title passed as
      context, lesson-picker allowed to return "no teachable idea" for
      imprint/TOC/OCR-noise pages, and a `grounding_score` metric warns
      when narration drifts off the book text.
- [x] S2 scored visuals (Phase 4) — `sentence_assets.py` no longer takes
      the first Pexels hit: per sentence it gathers keyword + concept
      candidates from Pexels (Openverse fallback, no key needed) and CLIP
      scores each against the narration; the best-scoring candidate wins.
      Weak matches are logged and reported. Render treats placeholder
      assets (fetch failed) as black cards instead of crashing.
- [x] S3 batch runner (Phase 5) — `experiments/generate_book_batch.py`:
      chunk -> lesson -> script -> assets -> voice -> black-style render in
      one command; resumable (existing MP4 skips), failures retryable, and
      a per-run markdown QA report (durations, grounding scores, weak
      visuals, full scripts).
- [x] S4 provider fixes — Groq reasoning models (`gpt-oss-*`) pinned to
      reasoning_effort=low (they were burning the whole token budget on
      hidden reasoning and returning EMPTY JSON); Pexels key now loads
      `.env` when experiments import providers without the CLI entrypoint.
- [x] S5 motif library (user complaint: "uses no assets from the asset
      motif pollination directory") — `sentence_assets.py` now CLIP-indexes
      all 100 `assets/motifs_pollinations` PNGs once per process; every
      sentence picks the best of the motif pool vs stock (Pexels+Openverse),
      ties favor the signed motif art, no immediate motif repeats. Extras
      carry `source=motif|stock`, `motif`, `motif_name`, `category`.
      Verified e2e: 4 motif + 1 stock visuals, CLIP 0.24–0.28, no
      placeholders. QA report lists every visual.
- [x] S6 narration coherence (user complaint: "sounds like a human acting
      like a robot") — new `script_engine/polish.py`: after per-scene
      writing, ONE LLM pass rewrites hook + all scenes + outro as natural
      spoken prose (contractions, bridges between lines, telegraph
      fragments fixed into complete sentences). Failure-safe: original
      lines returned on any count/length/validation problem.
- [x] S7 re-render — all first-chapter chunks re-unmarked and regenerated
      with the new visual + narration code.

## Phase 2: connect book scripts to the video pipeline

Goal: lesson script → motif images → voice → rendered MP4, reusing the
existing stage commands (`ash_speaks voice/render`) which read
`output/<slug>/short.json` + `assets-short.json`.

- [x] **P1. Motif asset bridge** — `src/blocks/motif_assets.py`:
      script → AssetPack with one local motif PNG per scene, content-scored
      keyword match, no adjacent category repeats; `render/media._fetch_media`
      now serves local `file://` paths.
- [x] **P2. Stage-compatible runner** — `experiments/generate_book_videos.py`:
      lesson → `output/<book--NNN>/{short.json,short.md,assets-short.json,
      lesson.json}` + ledger update; existing voice/render stages work
      unchanged.
- [x] **P3. E2E video** — verified: `schopenhauer_artofalwaysbeingright--012`
      → voice (Chatterbox) → rendered MP4 (45.7s, 8.1MB) with motif images,
      voiceover and music bed.
