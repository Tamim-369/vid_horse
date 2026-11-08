"""Book-lesson → stage-compatible video outputs (script + motif assets).

For each untaken lesson: pick the lesson idea, write the script, assign
motif images, and save everything under output/<book--lesson>/ in exactly
the layout the existing `ash_speaks voice` and `ash_speaks render` stages
read (short.json + assets-short.json). Marks the lesson in the ledger.

Usage:
    uv run python experiments/generate_book_videos.py [book_dir] [--limit N]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocks.lesson_picker import neighbours, next_untaken, pick_lesson
from blocks.motif_assets import build_asset_pack
from blocks.script_engine.block import ScriptEngineBlock
from blocks.music import annotate_music
from blocks.script_generation import render_markdown
from config.settings import get_settings
from providers.llm import create_llm_provider
from utils.ledger import TakenLedger

BOOKS_ROOT = Path("content/psychology_books")
OUTPUT_ROOT = Path("output")


def lesson_slug(lesson_path: Path) -> str:
    """Flat, filesystem-safe slug: <book>--<part><chunk>, e.g. X--04x012.

    Chunk numbering restarts within each chapter part directory, so the
    part number is part of the slug to keep every output dir unique.
    """

    rel = lesson_path.relative_to(BOOKS_ROOT)
    book = rel.parts[0]
    part_match = re.match(r"part_(\d+)", rel.parts[1]) if len(rel.parts) > 2 else None
    part = part_match.group(1) if part_match else "00"
    match = re.match(r"(\d+)", lesson_path.name)
    index = match.group(1) if match else lesson_path.stem[:6]
    return f"{book[:40]}--{part}x{index}"


def book_dirs() -> list[Path]:
    return sorted(p for p in BOOKS_ROOT.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("book_dir", nargs="?", default=None)
    parser.add_argument("--limit", type=int, default=1, help="Max videos this run.")
    parser.add_argument("--seconds", type=int, default=45)
    args = parser.parse_args()

    settings = get_settings()
    llm = create_llm_provider(settings)
    ledger = TakenLedger()
    engine = ScriptEngineBlock(llm, target_seconds=args.seconds)

    dirs = [Path(args.book_dir)] if args.book_dir else book_dirs()
    made = 0
    skip_run: set[str] = set()  # failed this run; don't retry within the same run
    for book_dir in dirs:
        while made < args.limit:
            lesson = next_untaken(book_dir, ledger, skip=skip_run)
            if lesson is None:
                print(f"[done] {book_dir.name}: every lesson taken")
                break
            started = time.monotonic()
            context = neighbours(lesson, count=1)
            try:
                idea = pick_lesson(llm, lesson, context_paths=context, ledger=ledger)
                script = engine.run(idea.topic, source_chunks=idea.source_chunks or None)
                try:
                    script = annotate_music(llm, script)
                except Exception:
                    pass
                slug = lesson_slug(lesson)
                out_dir = OUTPUT_ROOT / slug
                out_dir.mkdir(parents=True, exist_ok=True)

                pack = build_asset_pack(script, slug=slug)
                (out_dir / "short.json").write_text(
                    script.model_dump_json(indent=2), encoding="utf-8"
                )
                (out_dir / "short.md").write_text(render_markdown(script), encoding="utf-8")
                (out_dir / "assets-short.json").write_text(
                    pack.model_dump_json(indent=2), encoding="utf-8"
                )
                (out_dir / "lesson.json").write_text(
                    idea.model_dump_json(indent=2), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                abandoned = ledger.fail(lesson, note=f"failed: {exc}")
                state = "abandoned" if abandoned else "failed (will retry next run)"
                print(f"[skip] {lesson.name}: {exc} [{state}]")
                skip_run.add(str(lesson))
                continue

            ledger.mark(str(lesson), script_dir=str(out_dir), note=idea.lesson)
            made += 1
            print(
                f"[ok] {slug} <- {lesson.name} "
                f"({time.monotonic() - started:.0f}s) -> {out_dir}"
            )
        if made >= args.limit:
            break

    print(f"generated {made} video pack(s); {len(ledger.taken_paths())} lessons taken")
    print("next: ash_speaks voice <slug> && ash_speaks render <slug>")


if __name__ == "__main__":
    main()
