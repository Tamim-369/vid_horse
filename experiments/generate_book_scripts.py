"""Generate one teaching script per book lesson, skipping already-taken parts.

Usage:
    uv run python experiments/generate_book_scripts.py [book_dir] [--limit N]

Without a book_dir it walks every book under content/psychology_books/.
Each lesson becomes output/<book>/<lesson_stem>/script.{json,md} and the
lesson is marked in content/psychology_books/taken.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocks.lesson_picker import neighbours, next_untaken, pick_lesson
from blocks.script_engine.block import ScriptEngineBlock
from blocks.script_generation import render_markdown
from config.settings import get_settings
from providers.llm import create_llm_provider
from utils.ledger import TakenLedger

BOOKS_ROOT = Path("content/psychology_books")


def book_dirs() -> list[Path]:
    return sorted(p for p in BOOKS_ROOT.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("book_dir", nargs="?", default=None)
    parser.add_argument("--limit", type=int, default=1, help="Max scripts this run.")
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
            except Exception as exc:  # noqa: BLE001
                abandoned = ledger.fail(lesson, note=f"failed: {exc}")
                state = "abandoned" if abandoned else "failed (will retry next run)"
                print(f"[skip] {lesson.name}: {exc} [{state}]")
                skip_run.add(str(lesson))
                continue

            rel = lesson.relative_to(book_dir)
            part_dir = rel.parts[1] if len(rel.parts) > 2 else "part_00"
            out_dir = Path("output") / book_dir.name / part_dir / lesson.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "script.json").write_text(
                script.model_dump_json(indent=2), encoding="utf-8"
            )
            (out_dir / "script.md").write_text(render_markdown(script), encoding="utf-8")
            ledger.mark(str(lesson), script_dir=str(out_dir), note=idea.lesson)
            made += 1
            print(
                f"[ok] {book_dir.name}/{lesson.name} -> {out_dir} "
                f"({time.monotonic() - started:.0f}s)"
            )
        if made >= args.limit:
            break

    print(f"generated {made} script(s); {len(ledger.taken_paths())} lessons taken")


if __name__ == "__main__":
    main()
