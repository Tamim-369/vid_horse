#!/usr/bin/env python3
"""Test Script 2: Book Chunk Pipeline Test
Tests: lesson_picker (real book chunk) -> script_engine -> sentence_assets -> voice -> render_black_style
Uses an actual psychology book chunk from content/psychology_books/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings, setup_logging
from blocks.lesson_picker import pick_lesson, lesson_files, next_untaken
from blocks.script_engine.block import ScriptEngineBlock
from blocks.sentence_assets import build_asset_pack
from blocks.music import annotate_music
from providers.llm import create_llm_provider
from render.black_style import render_black_style
from voice.tts import VoiceOver
from utils.ledger import TakenLedger


def main() -> int:
    setup_logging("INFO")
    settings = get_settings()

    # Find first available book with teachable chunks
    books_root = Path("content/psychology_books")
    book_dirs = sorted([d for d in books_root.iterdir() if d.is_dir()])

    if not book_dirs:
        print("ERROR: No books found in content/psychology_books/")
        print("Run: uv run python experiments/build_psychology_book_content.py first")
        return 1

    ledger = TakenLedger()
    lesson_path = None
    book_dir = None

    for bd in book_dirs:
        lesson_path = next_untaken(bd, ledger)
        if lesson_path:
            book_dir = bd
            break

    if not lesson_path:
        print("No untaken lessons found in any book")
        return 1

    print(f"\n{'='*60}")
    print(f"TEST 2: Book Chunk Pipeline")
    print(f"Book: {book_dir.name}")
    print(f"Lesson: {lesson_path.name}")
    print(f"{'='*60}\n")

    # Create output slug
    rel = lesson_path.relative_to(books_root)
    book_slug = rel.parts[0]
    part_match = rel.parts[1] if len(rel.parts) > 2 else "part_00"
    import re
    part_num = re.match(r"part_(\d+)", part_match)
    part = part_match.group(1) if part_match else "00"
    idx_match = re.match(r"(\d+)", lesson_path.name)
    idx = idx_match.group(1) if idx_match else lesson_path.stem[:6]
    slug = f"{book_slug[:40]}--{part}x{idx}"

    out_dir = Path("output") / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pick Lesson
    print("[1/6] Picking lesson from book chunk...")
    llm = create_llm_provider(settings)
    idea = pick_lesson(llm, lesson_path, ledger=ledger)
    print(f"    Topic: {idea.topic}")
    print(f"    Lesson: {idea.lesson}")
    print(f"    Source chunks: {len(idea.source_chunks)}")

    # Save lesson
    (out_dir / "lesson.json").write_text(idea.model_dump_json(indent=2), encoding="utf-8")

    # 2. Script Generation
    print("[2/6] Generating script...")
    engine = ScriptEngineBlock(llm, target_seconds=45)
    script = engine.run(
        idea.topic,
        source_chunks=idea.source_chunks,
        context_note=idea.context_notes or ""
    )
    print(f"    Generated {len(script.scenes)} scenes")

    (out_dir / "short.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")

    # 3. Music Annotation
    print("[3/6] Annotating music...")
    try:
        script = annotate_music(llm, script)
        print(f"    Music plan: base_role={script.music.base_role if script.music else 'none'}")
    except Exception as e:
        print(f"    Music annotation failed (continuing): {e}")

    # 4. Asset Pack
    print("[4/6] Building asset pack (generating Pollinations images)...")
    pack = build_asset_pack(script, slug=slug)
    (out_dir / "assets-short.json").write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    total_assets = sum(len(s.assets) for s in pack.scenes)
    print(f"    Generated {total_assets} visual assets")

    # 5. Voice Generation
    print("[5/6] Generating voiceover...")
    vo = VoiceOver()
    voice_path = vo.render_script(script, audio_dir, stem="voice")
    print(f"    Voice saved to {voice_path}")

    # 6. Render Video
    print("[6/6] Rendering video...")
    video_path = out_dir / f"{slug}-short-black.mp4"
    render_black_style(
        script, pack, audio_dir, video_path,
        stem="voice", genre=None, music="", slug=slug
    )
    print(f"    Video saved to {video_path}")

    # Mark as done in ledger
    ledger.mark(str(lesson_path), script_dir=str(out_dir), note=idea.lesson)

    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION:")
    print(f"{'='*60}")
    for f in ["short.json", "assets-short.json", "lesson.json", "audio/voice.wav", f"{slug}-short-black.mp4"]:
        p = out_dir / f
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {f} MISSING")

    print(f"\nSCRIPT PREVIEW:")
    print(f"  Hook: {script.hook}")
    for i, scene in enumerate(script.scenes):
        print(f"  Scene {i+1}: {scene.narration[:80]}... | Visual: {scene.visual}")
    print(f"  Outro: {script.outro}")

    return 0


if __name__ == "__main__":
    sys.exit(main())