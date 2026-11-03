"""One-command faceless-video interface: ``videoai``.

A thin skin over the stage pipeline for the common case. ``videoai TOPIC``
runs research -> script -> assets -> voice -> render -> title -> publish
in one line; ``--batch`` scales it to a moving assembly line over many
topics. Fine-grained stage control stays available via ``ash_speaks``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

CLI_DESCRIPTION = "Faceless YouTube video pipeline"
BATCH_DEFAULT_TARGET = 5

console = Console()


def build_parser() -> argparse.ArgumentParser:
    """The exact command surface ``videoai`` exposes."""

    parser = argparse.ArgumentParser(prog="videoai", description=CLI_DESCRIPTION)
    parser.add_argument("topic", nargs="?", help="Single topic to make a video for")
    parser.add_argument("--batch", action="store_true",
                        help="Generate topics and make videos for all")
    parser.add_argument("--use-saved", dest="use_saved", action="store_true",
                        help="Use saved topic batch instead of generating")
    parser.add_argument("--upload", action="store_true",
                        help="(no-op) uploading is the default; use --no-upload to disable")
    parser.add_argument("--no-upload", dest="no_upload", action="store_true",
                        help="Render WITHOUT uploading to YouTube")
    parser.add_argument("--voice", metavar="VOICE",
                        help="Force a specific voice id (see --list-voices)")
    parser.add_argument("--list-voices", dest="list_voices", action="store_true",
                        help="List all registered voices and exit")
    parser.add_argument("--script-only", dest="script_only", action="store_true",
                        help="Only generate topics + scripts (no assets, audio, video, or upload)")
    parser.add_argument("--limit", type=int, metavar="LIMIT",
                        help="Posts per source when researching topics")
    parser.add_argument("--target", type=int, metavar="TARGET",
                        help="How many topics to research")
    parser.add_argument("--concurrency", type=int, metavar="N",
                        help="Videos to render in parallel in batch mode (moving assembly line)")
    return parser


# -- voices --------------------------------------------------------------------
def _voices() -> dict[str, Path]:
    """Registered voices: every reference wav shipped in assets/voice."""

    return {path.stem: path for path in sorted(Path("assets", "voice").glob("*.wav"))}


def _default_voice() -> str:
    from config.settings import get_settings

    return get_settings().voice_prompt.stem


def _voice_path(voice_id: str) -> Path:
    voices = _voices()
    if voice_id not in voices:
        console.print(
            f"[red]Unknown voice '{voice_id}'. Registered: "
            + ", ".join(sorted(voices)) + "[/red]"
        )
        sys.exit(1)
    return voices[voice_id]


def _list_voices() -> None:
    default = _default_voice()
    voices = _voices()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Voice")
    table.add_column("File")
    for name, path in sorted(voices.items()):
        name = f"{name}  [green](default)[/green]" if name == default else name
        table.add_row(name, str(path))
    console.print(table)


# -- topic resolution -----------------------------------------------------------
def _resolve_topic(settings, term: str):
    """Find the theory behind a free-text topic, with helpful near-misses."""

    from catalog.store import TheoryCatalog
    from utils.text import slugify

    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slugify(term))
        if theory is not None:
            return theory
        rows = list(catalog.list(limit=1000))

    lowered = term.lower()
    for row in rows:
        if row.name.lower() == lowered or row.slug == lowered:
            return row
    hits = [row for row in rows if lowered in row.name.lower() or lowered in row.slug]
    if hits:
        return hits[0]

    close = difflib.get_close_matches(term, [row.name for row in rows], n=3, cutoff=0.4)
    hint = f" Did you mean: {', '.join(close)}?" if close else (
        " Run 'ash_speaks discover' first."
    )
    console.print(f"[red]No topic matches '{term}'.{hint}[/red]")
    sys.exit(1)


# -- topic batch ------------------------------------------------------------------
def _batch_path(settings) -> Path:
    return Path(settings.output_dir) / "batch.json"


def _generate_batch(settings, target: int, limit: int | None):
    """Fresh random topic batch; tops up the catalog when it runs dry."""

    from catalog.discovery import discover, list_active_niches
    from catalog.store import TheoryCatalog

    with TheoryCatalog(settings.catalog_path) as catalog:
        pool = list(catalog.list(limit=2000))
    if len(pool) < target:
        with TheoryCatalog(settings.catalog_path) as catalog, console.status(
            "Topping up the catalog with fresh topics..."
        ):
            discover(catalog, None, limit=limit or 20, with_metrics=False,
                     settings=settings, niches=list_active_niches())
        with TheoryCatalog(settings.catalog_path) as catalog:
            pool = list(catalog.list(limit=2000))
    if not pool:
        console.print("[red]No topics available; discovery found nothing.[/red]")
        sys.exit(1)
    chosen = random.sample(pool, k=min(target, len(pool)))
    path = _batch_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "generated": datetime.now(UTC).isoformat(),
            "topics": [theory.slug for theory in chosen],
        }, indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]Batch saved -> {path}[/dim]")
    return chosen


def _saved_batch(settings, target: int | None):
    """Reload the saved batch file instead of spending discovery again."""

    path = _batch_path(settings)
    if not path.exists():
        console.print(
            f"[red]No saved batch at {path}; run 'videoai --batch' once first.[/red]"
        )
        sys.exit(1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    slugs = payload["topics"]
    if target is not None:
        slugs = slugs[:target]
    from catalog.store import TheoryCatalog

    found: list = []
    with TheoryCatalog(settings.catalog_path) as catalog:
        for slug in slugs:
            theory = catalog.get(slug)
            if theory is not None:
                found.append(theory)
    if not found:
        console.print("[red]Saved batch holds no catalogued topics.[/red]")
        sys.exit(1)
    console.print(f"[dim]Using saved batch: {', '.join(t.slug for t in found)}[/dim]")
    return found


# -- one video -----------------------------------------------------------------
def _make_one(theory, *, fmt: str, settings, upload: bool, script_only: bool, voice: Path | None):
    """Run the whole stage flow for one theory; every stage resumes from disk."""

    from blocks.script_generation import render_markdown
    from core.models import AssetPack, ResearchBrief, VideoScript
    from pipeline import build_content_pipeline
    from utils import computing

    tag = f"[bold cyan]{theory.slug}[/bold cyan] "
    slug = theory.slug
    out_dir = Path(settings.output_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    brief_path = out_dir / "brief.json"
    script_path = out_dir / f"{fmt}.json"
    assets_path = out_dir / f"assets-{fmt}.json"
    video_path = out_dir / f"{slug}-{fmt}.mp4"
    stem = "voice" if fmt == "short" else f"voice-{fmt}"

    pipeline = build_content_pipeline(settings)

    # research -------------------------------------------------------------
    if brief_path.exists():
        brief = ResearchBrief.model_validate_json(brief_path.read_text(encoding="utf-8"))
        console.print(f"{tag}[dim]research: saved brief reused[/dim]")
    else:
        started = time.monotonic()
        with console.status("Researching..."):
            brief = pipeline.research(theory)
        console.print(f"{tag}[green]research[/green] done in {time.monotonic() - started:.0f}s")
    _save(settings.output_dir, slug, "brief.json", brief.model_dump_json(indent=2))

    # script ---------------------------------------------------------------
    if script_path.exists():
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        console.print(f"{tag}[dim]script: saved script reused[/dim]")
    else:
        started = time.monotonic()
        with console.status("Writing script..."):
            script = pipeline.script(
                brief, fmt,
                target_seconds=(
                    settings.script_short_seconds
                    if fmt == "short"
                    else settings.script_long_seconds
                ),
            )
        console.print(f"{tag}[green]script[/green] done in {time.monotonic() - started:.0f}s")
        script = _annotate_music(settings, script)
    _save(settings.output_dir, slug, f"{fmt}.json", script.model_dump_json(indent=2))
    _save(settings.output_dir, slug, f"{fmt}.md", render_markdown(script))
    if script_only:
        console.print(f"{tag}[yellow]--script-only: stopping after the script.[/yellow]")
        return

    # assets ---------------------------------------------------------------
    if assets_path.exists():
        pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))
        console.print(f"{tag}[dim]assets: saved pack reused[/dim]")
    else:
        started = time.monotonic()
        with console.status("Hunting assets..."):
            pack = pipeline.assets(theory, brief, script)
        console.print(f"{tag}[green]assets[/green] done in {time.monotonic() - started:.0f}s")
    _save(settings.output_dir, slug, f"assets-{fmt}.json", pack.model_dump_json(indent=2))

    # voice ------------------------------------------------------------------
    from voice.tts import VoiceOver

    started = time.monotonic()
    label = voice.stem if voice else _default_voice()
    console.print(f"{tag}narrating with '{label}'...")
    vo = VoiceOver(prompt_path=voice) if voice else VoiceOver()
    vo.render_script(script, out_dir / "audio", stem=stem)
    _design_sfx(settings, slug, script)
    console.print(f"{tag}[green]voice[/green] done in {time.monotonic() - started:.0f}s")

    # render -----------------------------------------------------------------
    from render.assemble import render

    started = time.monotonic()
    console.print(f"{tag}rendering with {computing.render_workers()} card workers...")
    render(script, pack, out_dir / "audio", video_path, stem=stem, genre=brief.genre)
    console.print(f"{tag}[green]render[/green] {video_path} in {time.monotonic() - started:.0f}s")

    # title -------------------------------------------------------------------
    from blocks.title_description import TitleDescriptionBlock
    from providers.llm import create_llm_provider

    with console.status("Crafting title..."):
        title_pack = TitleDescriptionBlock(create_llm_provider(settings)).run(
            script, topic=theory.name
        )
    _save(settings.output_dir, slug, f"{fmt}.title.json", title_pack.model_dump_json(indent=2))
    console.print(f"{tag}[bold]{title_pack.title}[/bold]")

    # publish -------------------------------------------------------------------
    if not upload:
        console.print(f"{tag}[yellow]--no-upload: skipping YouTube.[/yellow]")
    elif not settings.auto_publish:
        console.print(
            f"{tag}[yellow]AUTO_PUBLISH off; upload skipped "
            "(set VIDEOAI_AUTO_PUBLISH=true, or run 'ash_speaks publish "
            f"{slug} --push').[/yellow]"
        )
    else:
        from publish import YouTubePublisher

        with console.status("Uploading to YouTube..."):
            video_id = YouTubePublisher().upload(video_path, title_pack)
        console.print(f"{tag}[green]published[/green] https://youtu.be/{video_id}")


def _annotate_music(settings, script):
    from blocks.music import annotate_music
    from providers.llm import create_llm_provider

    return annotate_music(create_llm_provider(settings), script)


def _design_sfx(settings, slug: str, script):
    from cli.commands import _design_sfx as design

    return design(settings, slug, script)


def _save(output_dir, slug: str, name: str, content: str) -> Path:
    directory = Path(output_dir) / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


# -- batch assembly line ---------------------------------------------------------
def _parallel_cap() -> int:
    """How many full pipelines the RAM budget can hold at once."""

    from utils import computing

    weight = computing.stage_weight("render").ram_gb  # heaviest resident stage
    return max(1, int(computing.machine().available_gb() // weight))


def _run_batch(args, settings) -> None:
    target = args.target or BATCH_DEFAULT_TARGET
    if args.use_saved:
        theories = _saved_batch(settings, target)
    else:
        theories = _generate_batch(settings, target, args.limit)
    console.print(f"[bold]Assembly line: {len(theories)} video(s) started[/bold]")

    concurrency = args.concurrency or 1
    cap = _parallel_cap()
    if concurrency > cap:
        console.print(
            f"[dim]--concurrency {concurrency} exceeds the RAM budget; capping to {cap}.[/dim]"
        )
        concurrency = cap
    concurrency = max(1, min(concurrency, len(theories)))

    # the faceless pipeline ships vertical shorts
    single_voice = _voice_path(args.voice) if args.voice else None

    def run(theory):
        try:
            _make_one(
                theory,
                fmt="short",
                settings=settings,
                upload=not args.no_upload,
                script_only=args.script_only,
                voice=single_voice,
            )
            return theory.slug, None
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the line
            console.print(f"[bold red]{theory.slug}: FAILED[/bold red] {exc}")
            return theory.slug, exc

    failures = []
    if concurrency == 1 or len(theories) == 1:
        for theory in theories:
            _slug, error = run(theory)
            if error is not None:
                failures.append(_slug)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _slug, error in pool.map(run, theories):
                if error is not None:
                    failures.append(_slug)
    made = len(theories) - len(failures)
    console.print(f"[green]Done: {made}/{len(theories)} video(s).[/green]")
    if failures:
        console.print(f"[yellow]Failed: {', '.join(failures)}[/yellow]")


# -- entry ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """Run the ``videoai`` command."""

    from config import get_settings, setup_logging

    setup_logging(get_settings().log_level)
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.list_voices:
        _list_voices()
        return

    if args.topic and args.batch:
        parser.error("--batch generates topics itself; drop TOPIC or drop --batch")

    if args.batch:
        _run_batch(args, settings)
        return

    if not args.topic:
        parser.error("provide a TOPIC, or use --batch")

    if args.voice:
        _voice_path(args.voice)  # fail fast on an unknown id

    theory = _resolve_topic(settings, args.topic)
    _make_one(
        theory,
        fmt="short",
        settings=settings,
        upload=not args.no_upload,
        script_only=args.script_only,
        voice=_voice_path(args.voice) if args.voice else None,
    )


if __name__ == "__main__":
    main()
