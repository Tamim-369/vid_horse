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

- [x] 1 taken-parts ledger
- [x] 2 agentic RAG hook
- [x] 3 teacher persona
- [x] 4 lesson-picker agent
- [x] 5 runner
- [x] 6 ingestion + e2e validation
- [x] 7 output hardening (validated over 10 consecutive Schopenhauer lessons):
      hook ≤12 words w/ retry + dangling-word truncation · placeholder-echo
      rejection ("Text /", "query") · reasoning-echo stripping · narration/
      visual dedup · unlabeled keyword-tail stripping · visual keyword
      rebuild (>5 words) · POV retry loop + mechanical first→second-person
      repair · contraction-safe quote cleanup · dangling conjunction/digit
      cleanup

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
