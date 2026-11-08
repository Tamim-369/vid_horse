"""Batch book-video factory: chunk -> script -> assets -> voice -> render.

For every untaken teachable chunk of a book (first book with content by
default): pick the lesson idea, write a grounded script, assign CLIP-scored
sentence visuals, synthesize the voice, render the black-style MP4, and mark
the ledger. A markdown QA report is written per run.

Usage:
    uv run python experiments/generate_book_batch.py [book_dir]
        [--limit N] [--seconds 45] [--refresh-style]

Resume: a chunk directory with its final MP4 present is skipped; failed
chunks stay retryable in the ledger (they never consume a unit).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocks.lesson_picker import chapter_of, neighbours, next_untaken, pick_lesson
from blocks.music import annotate_music
from blocks.sentence_assets import build_asset_pack
from blocks.script_engine.block import ScriptEngineBlock
from blocks.script_engine.validators import grounding_score
from blocks.script_generation import render_markdown
from config.settings import get_settings
from providers.llm import create_llm_provider
from utils.ledger import TakenLedger

BOOKS_ROOT = Path("content/psychology_books")
OUTPUT_ROOT = Path("output")


def lesson_slug(lesson_path: Path) -> str:
    """Flat, filesystem-safe slug: <book>--<part>x<NNN>."""

    rel = lesson_path.relative_to(BOOKS_ROOT)
    book = rel.parts[0]
    part_match = re.match(r"part_(\d+)", rel.parts[1]) if len(rel.parts) > 2 else None
    part = part_match.group(1) if part_match else "00"
    match = re.match(r"(\d+)", lesson_path.name)
    index = match.group(1) if match else lesson_path.stem[:6]
    return f"{book[:40]}--{part}x{index}"


def book_dirs() -> list[Path]:
    return sorted(p for p in BOOKS_ROOT.iterdir() if p.is_dir())


def _qa_lines(rows: list[dict]) -> list[str]:
    lines = ["# Batch QA report", "", f"run: {datetime.now().isoformat(timespec='seconds')}", "", ""]
    lines.append("| # | chunk | idea | chapter | dur s | grounding | weak visuals |")
    lines.append("|---|-------|------|---------|-------|-----------|--------------|")
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row.get('chunk','')} | {row.get('idea','')} | {row.get('chapter','')} "
            f"| {row.get('dur','')} | {row.get('grounding','')} | {row.get('weak','')}/{row.get('clipd','')} |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"## {row.get('chunk','')} — {row.get('idea','')}")
        lines.append(f"- chapter: `{row.get('chapter','')}`")
        lines.append(f"- narration grounding: {row.get('grounding','')}")
        lines.append("```")
        lines.append(row.get("script_text", ""))
        lines.append("```")
        if row.get("weak_detail"):
            lines.append(f"- weak visuals: {row['weak_detail']}")
        lines.append("")
    return lines


def render_one(
    lesson: Path,
    *,
    seconds: int,
    llm,
    engine: ScriptEngineBlock,
    ledger: TakenLedger,
    out_dir: Path,
) -> dict:
    """Full stage run for one chunk; returns a QA row. Raises on failure."""
    started = time.monotonic()
    chapter = chapter_of(lesson)

    idea = pick_lesson(llm, lesson, context_paths=neighbours(lesson, count=1), ledger=ledger)
    script = engine.run(
        idea.topic, source_chunks=idea.source_chunks or None, context_note=chapter
    )
    try:
        script = annotate_music(llm, script)
    except Exception:  # noqa: BLE001
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    pack = build_asset_pack(script, slug=out_dir.name)
    (out_dir / "short.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "short.md").write_text(render_markdown(script), encoding="utf-8")
    (out_dir / "assets-short.json").write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "lesson.json").write_text(idea.model_dump_json(indent=2), encoding="utf-8")

    # voice
    audio_dir = out_dir / "audio"
    voice_path = audio_dir / "voice.wav"
    if not voice_path.exists():
        from voice.tts import VoiceOver

        voice_path = VoiceOver().render_script(script, audio_dir, stem="voice")
    duration = _wav_seconds(voice_path)

    # render (black-style)
    video_path = out_dir / f"{out_dir.name}-short-black.mp4"
    if not video_path.exists() or video_path.stat().st_size == 0:
        from render.black_style import render_black_style

        render_black_style(
            script, pack, audio_dir, video_path, stem="voice", genre=None, music="", slug=out_dir.name
        )

    # QA metrics
    clipd = weak = 0
    weak_detail: list[str] = []
    for scene in pack.scenes:
        for asset in scene.assets:
            cs = (asset.extra or {}).get("clip_score")
            if cs is None:
                continue
            clipd += 1
            if cs < 0.18:
                weak += 1
                weak_detail.append(f"{cs:.2f} [{(asset.extra or {}).get('query', '')}]")
    grounding = grounding_score(
        " ".join([script.hook] + [s.narration for s in script.scenes] + [script.outro]),
        (lesson.read_text(encoding="utf-8", errors="replace")),
    )

    ledger.mark(str(lesson), script_dir=str(out_dir), note=idea.lesson)
    return {
        "chunk": lesson.name,
        "idea": idea.topic,
        "chapter": chapter[:40],
        "dur": f"{duration:.0f}",
        "grounding": f"{grounding:.2f}",
        "weak": str(weak),
        "clipd": str(clipd),
        "script_text": render_markdown(script),
        "weak_detail": "; ".join(weak_detail[:4]),
        "seconds": round(time.monotonic() - started),
    }


def _wav_seconds(path: Path) -> float:
    import contextlib

    try:
        import torchaudio as ta

        with contextlib.suppress(Exception):
            wav, sr = ta.load(str(path))
            return float(wav.shape[-1]) / sr
    except ImportError:
        pass
    return float(len(path.read_bytes()) / (16000 * 2))  # fallback: raw estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("book_dir", nargs="?", default=None)
    parser.add_argument("--limit", type=int, default=10, help="Max videos this run.")
    parser.add_argument("--seconds", type=int, default=45)
    args = parser.parse_args()

    settings = get_settings()
    llm = create_llm_provider(settings)
    ledger = TakenLedger()
    engine = ScriptEngineBlock(llm, target_seconds=args.seconds)

    dirs = [Path(args.book_dir)] if args.book_dir else book_dirs()
    rows: list[dict] = []
    failures: list[str] = []
    made = 0
    for book_dir in dirs:
        while made < args.limit:
            lesson = next_untaken(book_dir, ledger)
            if lesson is None:
                print(f"[done] {book_dir.name}: every chunk taken")
                break
            slug = lesson_slug(lesson)
            out_dir = OUTPUT_ROOT / slug
            final = out_dir / f"{slug}-short-black.mp4"
            if final.exists() and final.stat().st_size > 0:
                print(f"[skip-resume] {slug}: mp4 already present")
                continue
            try:
                row = render_one(
                    lesson,
                    seconds=args.seconds,
                    llm=llm,
                    engine=engine,
                    ledger=ledger,
                    out_dir=out_dir,
                )
            except Exception as exc:  # noqa: BLE001
                abandoned = ledger.fail(lesson, note=f"failed: {exc}")
                state = "abandoned" if abandoned else "failed (retryable next run)"
                print(f"[skip] {lesson.name}: {exc} [{state}]")
                failures.append(f"{lesson.name}: {exc}")
                break  # same lesson returns first next iteration; avoid tight loop
            rows.append(row)
            made += 1
            print(
                f"[ok] {slug} <- {lesson.name} ({row['seconds']}s) "
                f"grounding={row['grounding']} weak={row['weak']}/{row['clipd']}"
            )
        if made >= args.limit:
            break

    qa_path = OUTPUT_ROOT / ("qa-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".md")
    qa_path.write_text("\n".join(_qa_lines(rows)), encoding="utf-8")
    print(f"generated {made} video(s); QA report -> {qa_path}")
    if failures:
        print(f"failures ({len(failures)}):")
        for f in failures[:10]:
            print("  -", f)


if __name__ == "__main__":
    main()
