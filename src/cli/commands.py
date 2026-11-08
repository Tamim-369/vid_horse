"""CLI command implementations."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from blocks.script_generation import render_markdown
from catalog.discovery import discover
from catalog.store import TheoryCatalog
from config import get_settings
from core.models import AssetPack, ResearchBrief, ScriptFormat, Theory, VideoScript
from pipeline import build_content_pipeline
from utils import computing

console = Console()

_FORMATS: tuple[ScriptFormat, ...] = ("short", "long")


def register(app: typer.Typer) -> None:
    """Attach all commands to ``app``.

    Everyday commands stay visible in ``--help``; the five specialist
    commands (music re-label, horror corpus/shelf maintenance) stay
    fully functional but hidden to keep the help screen scannable.
    """

    app.command(name="discover")(discover_cmd)
    app.command(name="theories")(theories_cmd)
    app.command(name="research")(research_cmd)
    app.command(name="script")(script_cmd)
    app.command(name="music", hidden=True)(music_cmd)
    app.command(name="assets")(assets_cmd)
    app.command(name="title")(title_cmd)
    app.command(name="publish")(publish_cmd)
    app.command(name="voice")(voice_cmd)
    app.command(name="render")(render_cmd)
    app.command(name="run-all")(run_all_cmd)
    app.command(name="make")(make_cmd)
    app.command(name="horror-pull", hidden=True)(horror_pull_cmd)
    app.command(name="sfx-pull", hidden=True)(sfx_pull_cmd)
    app.command(name="shelf-seed", hidden=True)(shelf_seed_cmd)
    app.command(name="story", hidden=True)(story_cmd)


# -- discovery --------------------------------------------------------------
def discover_cmd(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max theories per source."),
    metrics: bool = typer.Option(
        True, "--metrics/--no-metrics", help="Fetch pageviews (slower)."
    ),
    niche: str | None = typer.Option(
        None,
        "--niche",
        "-n",
        help=(
            "Niche(s) to discover (comma-separated). "
            "Use 'all' for all active niches, or 'list' to see available."
        ),
    ),
) -> None:
    """Discover theories across configured niches and merge into local catalog."""

    from catalog.discovery import list_active_niches, list_niches

    settings = get_settings()

    if niche is not None:
        if niche.strip().lower() == "list":
            all_niches = list_niches()
            console.print("[bold]Available niches:[/bold]")
            for key, cfg in all_niches.items():
                active = " ✓" if key in list_active_niches() else ""
                console.print(
                    f"  [cyan]{key}[/cyan]{active}: {cfg['label']} — {cfg['description']}"
                )
            raise typer.Exit(0)

        wanted = [n.strip().lower() for n in niche.split(",") if n.strip()]
        if "all" in wanted:
            wanted = list_active_niches()
        unknown = [n for n in wanted if n not in list_niches()]
        if unknown:
            console.print(
                f"[red]Unknown niche(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(list_niches()))}[/red]"
            )
            raise typer.Exit(1)
    else:
        # Default: all active niches
        wanted = list_active_niches()

    console.print(f"[dim]Discovering niches: {', '.join(wanted)}[/dim]")

    with TheoryCatalog(settings.catalog_path) as catalog:
        with console.status("Discovering theories..."):
            report = discover(
                catalog, None, limit=limit, with_metrics=metrics, settings=settings, niches=wanted
            )
        total = catalog.count()

    console.print(
        f"[green]Discovered {report.discovered} hits. "
        f"Catalog holds {total} theories.[/green]"
    )
    if report.skipped:
        console.print(f"[dim]Skipped {report.skipped} list/meta pages.[/dim]")
    for name, count in sorted(report.per_source.items()):
        console.print(f"  {name}: {count}")
    if report.per_niche:
        console.print("[dim]Per niche:[/dim]")
        for n, c in sorted(report.per_niche.items()):
            console.print(f"  {n}: {c}")


def theories_cmd(
    limit: int = typer.Option(25, "--limit", "-l"),
    tier: str | None = typer.Option(None, "--tier", help="mainstream|popular|moderate|underground"),
    order: str = typer.Option("popularity", "--order", "-o", help="popularity|size|name"),
    search: str | None = typer.Option(
        None, "--search", "-s", help="Only show theories matching this text."
    ),
) -> None:
    """List catalogued theories."""

    settings = get_settings()
    with TheoryCatalog(settings.catalog_path) as catalog:
        rows = catalog.list(limit=limit * 10 if search else limit, order=order, tier=tier)
    if search:
        needle = search.lower()
        rows = [
            theory
            for theory in rows
            if needle in theory.name.lower() or needle in theory.slug.lower()
        ][:limit]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tier")
    table.add_column("Slug", overflow="fold")
    table.add_column("Theory", overflow="fold")
    table.add_column("Views/mo", justify="right")
    table.add_column("KB", justify="right")
    table.add_column("Langs", justify="right")

    for theory in rows:
        metrics = theory.metrics
        views = f"{metrics.avg_monthly_views:,.0f}" if metrics.avg_monthly_views else "-"
        kb = f"{metrics.size_bytes / 1024:.0f}" if metrics.size_bytes else "-"
        langs = str(metrics.language_count) if metrics.language_count is not None else "-"
        table.add_row(
            theory.popularity_tier, theory.slug, theory.name, views, kb, langs
        )

    console.print(table)


# -- research + scripting ---------------------------------------------------
def research_cmd(
    slug: str = typer.Argument(..., help="Catalog slug, e.g. 'roswell-incident'."),
    chars: int | None = typer.Option(None, "--chars", help="Article budget in characters."),
) -> None:
    """Research one catalogued theory and print its brief."""

    settings = get_settings()
    theory = _load_theory(settings, slug)

    with console.status(f"Researching '{theory.name}'..."):
        pipeline = build_content_pipeline(settings)
        brief = pipeline.research(theory, max_chars=chars)

    _print_brief(brief)
    path = _save(settings.output_dir, slug, "brief.json", brief.model_dump_json(indent=2))
    console.print(f"[dim]Saved {path}[/dim]")


def script_cmd(
    slug: str = typer.Argument(..., help="Catalog slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long|both"),
    chars: int | None = typer.Option(None, "--chars", help="Article budget in characters."),
) -> None:
    """Research a theory and write its video script(s)."""

    settings = get_settings()
    theory = _load_theory(settings, slug)

    if format == "both":
        formats = _FORMATS
    elif format in _FORMATS:
        formats = (format,)  # type: ignore[assignment]
    else:
        _fail(f"Unknown format '{format}'. Choose short, long or both.")

    pipeline = build_content_pipeline(settings)
    started = time.monotonic()
    with console.status(f"Researching '{theory.name}'..."):
        brief = pipeline.research(theory, max_chars=chars)
    console.print(f"[dim]research: {time.monotonic() - started:.1f}s[/dim]")
    _save(settings.output_dir, slug, "brief.json", brief.model_dump_json(indent=2))

    for fmt in formats:
        seconds = (
            settings.script_short_seconds
            if fmt == "short"
            else settings.script_long_seconds
        )
        started = time.monotonic()
        with console.status(f"Writing {fmt} script..."):
            script = pipeline.script(brief, fmt, target_seconds=seconds)
        console.print(f"[dim]script({fmt}): {time.monotonic() - started:.1f}s[/dim]")
        from blocks.music import annotate_music
        from providers.llm import create_llm_provider

        started = time.monotonic()
        script = annotate_music(create_llm_provider(settings), script)
        console.print(f"[dim]music: {time.monotonic() - started:.1f}s[/dim]")
        path = _save(settings.output_dir, slug, f"{fmt}.json", script.model_dump_json(indent=2))
        _save(settings.output_dir, slug, f"{fmt}.md", render_markdown(script))
        music_note = (
            f", music={script.music.base_role}"
            + (f" +{len(script.music.scene_roles)} override" if script.music.scene_roles else "")
            if script.music
            else ", music=default (annotate failed)"
        )
        console.print(
            f"[green]{script.title}[/green] "
            f"({fmt}, {len(script.scenes)} scenes{music_note}) -> {path}"
        )


# -- helpers ----------------------------------------------------------------
def music_cmd(
    slug: str = typer.Argument(..., help="Catalog slug."),
    format: str = typer.Option("both", "--format", "-f", help="short|long|both"),
) -> None:
    """Re-label the saved script's music roles without regenerating it."""

    from blocks.music import annotate_music
    from config.settings import get_settings
    from core.models import VideoScript
    from providers.llm import create_llm_provider

    settings = get_settings()
    if format == "both":
        formats = _FORMATS
    elif format in _FORMATS:
        formats = (format,)  # type: ignore[assignment]
    else:
        _fail(f"Unknown format '{format}'. Choose short, long or both.")
    for fmt in formats:
        script_path = Path(settings.output_dir) / slug / f"{fmt}.json"
        if not script_path.exists():
            _fail(f"No {fmt} script for '{slug}'. Run 'ash_speaks script {slug}' first.")
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        with console.status(f"Labelling {fmt} music..."):
            script = annotate_music(create_llm_provider(settings), script)
        script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        if script.music is None:
            console.print(f"{script.title} ({fmt}): music=default (annotate failed)")
            continue
        overrides = ", ".join(
            f"scene {key} -> {role}" for key, role in sorted(script.music.scene_roles.items())
        )
        note = f"; overrides: {overrides}" if overrides else "; single bed"
        console.print(
            f"{script.title} ({fmt}): base={script.music.base_role}{note} -> {script_path}"
        )


# -- helpers ----------------------------------------------------------------
def title_cmd(
    slug: str = typer.Argument(..., help="Catalog slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Generate a viral title + description + tags for a finished script."""

    settings = get_settings()
    theory = _load_theory(settings, slug)
    script_path = settings.output_dir / slug / f"{format}.json"
    if not script_path.exists():
        _fail(f"No {format} script for '{slug}'. Run 'ash_speaks script {slug}' first.")
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    from blocks.title_description import TitleDescriptionBlock
    from providers.llm import create_llm_provider

    llm = create_llm_provider(settings)
    block = TitleDescriptionBlock(llm)
    started = time.monotonic()
    with console.status(f"Crafting title for '{theory.name}'..."):
        pack = block.run(script, topic=theory.name)
    console.print(f"[dim]title: {time.monotonic() - started:.1f}s[/dim]")
    path = _save(settings.output_dir, slug, f"{format}.title.json", pack.model_dump_json(indent=2))
    console.print(Panel.fit(
        f"[bold]{pack.title}[/bold]\n\n{pack.description}\n\ntags: {', '.join(pack.tags)}\n"
        f"hashtags: {' '.join(pack.hashtags)}",
        title="Title pack",
    ))
    console.print(f"[green]saved[/green] {path}")


def assets_cmd(
    slug: str = typer.Argument(..., help="Catalog slug, e.g. 'baron-52'."),
    format: str = typer.Option("short", "--format", "-f", help="short|long|both"),
) -> None:
    """Match topical wiki media to the saved script's scenes."""

    settings = get_settings()
    theory = _load_theory(settings, slug)

    brief_path = Path(settings.output_dir) / slug / "brief.json"
    if not brief_path.exists():
        _fail(f"No brief for '{slug}'. Run 'ash_speaks research {slug}' first.")
    brief = ResearchBrief.model_validate_json(brief_path.read_text(encoding="utf-8"))

    if format == "both":
        formats = _FORMATS
    elif format in _FORMATS:
        formats = (format,)
    else:
        _fail(f"Unknown format '{format}'. Choose short, long or both.")

    pipeline = build_content_pipeline(settings)
    for fmt in formats:
        script_path = Path(settings.output_dir) / slug / f"{fmt}.json"
        if not script_path.exists():
            _fail(f"No {fmt} script for '{slug}'. Run 'ash_speaks script {slug}' first.")
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        with console.status(f"Matching media for the {fmt} script..."):
            trace: dict[str, str] = {}
            pack = pipeline.assets(theory, brief, script, trace=trace)
        for name, content in trace.items():
            _save(settings.output_dir, slug, name, content)
        path = _save(
            settings.output_dir,
            slug,
            f"assets-{fmt}.json",
            pack.model_dump_json(indent=2),
        )
        gallery = _save(settings.output_dir, slug, f"assets-{fmt}.md", _render_assets(pack))
        served = sum(1 for scene in pack.scenes if scene.assets)
        console.print(
            f"[green]{pack.topic}[/green] ({fmt}: {served}/{len(pack.scenes)} "
            f"scenes matched) -> {path} / {gallery}"
        )


def voice_cmd(
    slugs: list[str] = typer.Argument(
        ..., help="One or more catalog slugs; the voice model loads once."
    ),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Narrate saved script(s) with the cloned voice; cached line by line.

    Multiple slugs in one invocation share a single process, so the
    Chatterbox model loads exactly once for the whole batch — that load
    used to dominate per-slug voice time in back-to-back sessions.
    """

    from voice.tts import VoiceOver

    settings = get_settings()
    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")
    stem = "voice" if format == "short" else f"voice-{format}"
    vo = VoiceOver()
    for slug in slugs:
        script_path = Path(settings.output_dir) / slug / f"{format}.json"
        if not script_path.exists():
            console.print(f"[red]{slug}: no {format} script; run 'ash_speaks script {slug}'[/red]")
            continue
        script = VideoScript.model_validate_json(
            script_path.read_text(encoding="utf-8")
        )
        with console.status(f"Narrating {slug} ({format})..."):
            path = vo.render_script(
                script, Path(settings.output_dir) / slug / "audio", stem=stem
            )
        sfx_path = _design_sfx(settings, slug, script)
        console.print(f"[green]{slug}:[/green] voiceover ready -> {path}")
        if sfx_path:
            console.print(f"[green]{slug}:[/green] sound design -> {sfx_path}")


def shelf_seed_cmd(
    slugs: list[str] = typer.Argument(..., help="Catalog slugs whose saved packs seed the shelf."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Populate the asset shelf from already-approved asset packs.

    Every pack asset re-embeds through the real CLIP scorer and lands
    in the shelf with its recorded score — exactly what new research
    passes would store. Cheap way to boot the shelf: no network search
    beyond the image fetch, no LLM calls.
    """

    import httpx
    import numpy as np

    from utils.clip import ClipScorer
    from utils.shelf import AssetShelf

    settings = get_settings()
    shelf = AssetShelf(settings.cache_dir / "shelf.sqlite3")
    scorer = ClipScorer(settings.http_user_agent, threads=computing.torch_threads())
    client = httpx.Client(
        headers={"User-Agent": settings.http_user_agent},
        timeout=settings.http_timeout,
        follow_redirects=True,
    )
    total = stored = 0
    for slug in slugs:
        pack_path = Path(settings.output_dir) / slug / f"assets-{format}.json"
        if not pack_path.exists():
            console.print(f"[red]{slug}: no {format} pack[/red]")
            continue
        pack = AssetPack.model_validate_json(pack_path.read_text(encoding="utf-8"))
        for scene in pack.scenes:
            for item in scene.assets:
                total += 1
                embed = scorer.image_embed(item.thumb_url or item.url, client)
                if embed is None:
                    console.print(f"[yellow]{slug}: no embed for {item.file[:50]}[/yellow]")
                    continue
                ok = shelf.store(
                    {
                        "file": item.file,
                        "url": item.url,
                        "thumburl": item.thumb_url,
                        "width": item.width,
                        "height": item.height,
                        "license": item.license,
                        "artist": item.artist,
                        "kind": item.kind,
                    },
                    np.asarray(embed, dtype=np.float32),
                    float(item.clip_score or 0.0),
                )
                stored += 1 if ok else 0
        n_scenes = sum(len(scene.assets) for scene in pack.scenes)
        console.print(f"[green]{slug}: {n_scenes} asset(s) in pack[/green]")
    shelf.close()
    console.print(f"[bold green]done: {stored}/{total} stored[/bold green]")


def _design_sfx(settings, slug: str, script: VideoScript) -> Path | None:
    """Horror scripts get a tiny sound design; cached once per save."""

    if not any("original horror story" in note.lower() for note in script.source_notes):
        return None
    sfx_path = Path(settings.output_dir) / slug / "sfx.json"
    if sfx_path.exists():
        return sfx_path
    from blocks.sfx_design import SfxDesignerBlock
    from providers.llm import create_llm_provider

    designer = SfxDesignerBlock(create_llm_provider(settings), Path("assets/sfx/sfx.json"))
    plan = designer.run(script, scene_count=len(script.scenes))
    sfx_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return sfx_path


def publish_cmd(
    slug: str = typer.Argument(..., help="Catalog slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    video: Path | None = typer.Option(None, "--video", help="Override video path."),
    push: bool = typer.Option(
        False, "--push", help="Upload for real even with AUTO_PUBLISH off."
    ),
) -> None:
    """Upload the rendered video to YouTube (respects AUTO_PUBLISH)."""

    settings = get_settings()
    theory = _load_theory(settings, slug)
    out_dir = settings.output_dir / slug
    video_path = video or (out_dir / f"{format}.mp4")
    if not video_path.exists():
        _fail(f"No video at {video_path}. Run 'ash_speaks render {slug}' first.")
    title_path = out_dir / f"{format}.title.json"
    if title_path.exists():
        from blocks.title_description import TitlePack

        pack = TitlePack.model_validate_json(title_path.read_text(encoding="utf-8"))
    else:
        from blocks.title_description import TitleDescriptionBlock
        from core.models import VideoScript
        from providers.llm import create_llm_provider

        script_path = out_dir / f"{format}.json"
        if not script_path.exists():
            _fail(f"No {format} script for '{slug}'. Run 'ash_speaks script {slug}' first.")
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        pack = TitleDescriptionBlock(create_llm_provider(settings)).run(script, topic=theory.name)

    from publish import YouTubePublisher

    publisher = YouTubePublisher()
    if not settings.auto_publish and not push:
        console.print("[yellow]AUTO_PUBLISH is off; dry-running metadata only.[/yellow]")
        console.print("[dim]Pass --push to upload anyway.[/dim]")
        console.print(pack.model_dump_json(indent=2))
        return
    with console.status("Uploading to YouTube..."):
        video_id = publisher.upload(video_path, pack)
    console.print(f"[green]published[/green] https://youtu.be/{video_id}")


def render_cmd(
    slug: str = typer.Argument(..., help="Catalog slug, e.g. 'baron-52'."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    music: str = typer.Option(
        "", "--music", help="Bed override, e.g. 'swell-tension.mp3'."
    ),
    subtitle_mode: str = typer.Option(
        "all", "--subtitle-mode", help="all (subs+highlights), highlights (only highlighted words), none"
    ),
) -> None:
    """Assemble the final vertical mp4 from script, assets and voice tracks."""

    from render.assemble import render

    settings = get_settings()
    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")
    if subtitle_mode not in {"all", "highlights", "none"}:
        _fail("subtitle_mode must be all, highlights, or none.")
    directory = Path(settings.output_dir) / slug
    assets_path = directory / f"assets-{format}.json"
    script_path = directory / f"{format}.json"
    if not script_path.exists():
        _fail(f"No {format} script for '{slug}'. Run 'ash_speaks script {slug}' first.")
    if not assets_path.exists():
        _fail(f"No {format} assets for '{slug}'. Run 'ash_speaks assets {slug}' first.")
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))
    genre = None
    brief_path = directory / "brief.json"
    if brief_path.exists():
        genre = ResearchBrief.model_validate_json(
            brief_path.read_text(encoding="utf-8")
        ).genre

    stem = "voice" if format == "short" else f"voice-{format}"
    suffix = f"-{subtitle_mode}" if subtitle_mode != "all" else ""
    out_path = directory / f"{slug}-{format}{suffix}.mp4"
    with console.status(f"Assembling the {format} video ({subtitle_mode})..."):
        try:
            render(
                script, pack, directory / "audio", out_path,
                stem=stem, genre=genre, music=music, subtitle_mode=subtitle_mode,
                slug=slug,
            )
        except FileNotFoundError as error:
            _fail(str(error).split(";")[0])
    console.print("[green]Video ready[/green] -> " + str(out_path))


_ALL_STAGES = ("research", "script", "assets", "voice", "render", "title", "publish")


# -- slug scheduler ----------------------------------------------------------
def make_cmd(
    count: int = typer.Option(1, "--count", "-n", help="How many videos to make."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    discover_limit: int = typer.Option(20, "--discover", help="Discover if catalog is empty."),
    publish: bool = typer.Option(
        False, "--publish/--no-publish", help="Upload to YouTube (needs AUTO_PUBLISH on)."
    ),
    only: str | None = typer.Option(
        None, "--only",
        help="Restrict stages, comma-separated: research,script,assets,voice,render,title,publish.",
    ),
    skip: str | None = typer.Option(
        None, "--skip", help="Stages to skip, comma-separated (implies --no-publish).",
    ),
) -> None:
    """Make one or more complete videos end-to-end on random catalog topics.

    Picks random theories from the local catalog (discovering first if it is
    empty), then runs research -> script -> assets -> voice -> render ->
    title -> publish for each, in a single command.
    """

    import random

    settings = get_settings()
    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")

    wanted = _stage_set(only, skip)
    if not publish:
        wanted.discard("publish")

    with TheoryCatalog(settings.catalog_path) as catalog:
        theories = catalog.list(limit=500)
    if not theories:
        with console.status("Catalog empty; discovering theories..."):
            discover(limit=discover_limit, metrics=False, niche="all")
        theories = catalog.list(limit=500)
    if not theories:
        _fail("No theories available; run 'ash_speaks discover' first.")

    chosen = random.sample(theories, k=min(count, len(theories)))
    pipeline = build_content_pipeline(settings)
    from blocks.title_description import TitleDescriptionBlock
    from providers.llm import create_llm_provider
    from publish import YouTubePublisher
    from render.assemble import render
    from voice.tts import VoiceOver

    vo = VoiceOver()
    stem = "voice" if format == "short" else f"voice-{format}"
    publisher = YouTubePublisher()
    llm = create_llm_provider(settings)
    title_block = TitleDescriptionBlock(llm)

    for theory in chosen:
        slug = theory.slug
        console.rule(f"[bold]{theory.name}[/bold] ({slug})")
        out_dir = Path(settings.output_dir) / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # research + script
        brief_path = out_dir / "brief.json"
        script_path = out_dir / f"{format}.json"
        assets_path = out_dir / f"assets-{format}.json"
        out_path = out_dir / f"{slug}-{format}.mp4"
        title_path = out_dir / f"{format}.title.json"

        if "research" in wanted and not brief_path.exists():
            with console.status("Researching..."):
                brief = pipeline.research(theory)
            _save(settings.output_dir, slug, "brief.json", brief.model_dump_json(indent=2))
        elif brief_path.exists():
            brief = ResearchBrief.model_validate_json(brief_path.read_text(encoding="utf-8"))
        else:
            brief = None  # only research + render need it

        if "script" in wanted:
            if brief is None:
                _fail(f"No brief at {brief_path}; run without skipping research.")
            seconds = (
                settings.script_short_seconds if format == "short" else settings.script_long_seconds
            )
            with console.status("Writing script..."):
                script = pipeline.script(brief, format, target_seconds=seconds)
            from blocks.music import annotate_music
            script = annotate_music(create_llm_provider(settings), script)
            _save(settings.output_dir, slug, f"{format}.json", script.model_dump_json(indent=2))
            _save(settings.output_dir, slug, f"{format}.md", render_markdown(script))
        elif script_path.exists():
            script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        else:
            _fail(f"No {format} script; run without skipping script.")

        # assets
        if "assets" in wanted:
            with console.status("Matching assets..."):
                trace: dict[str, str] = {}
                pack = pipeline.assets(theory, brief, script, trace=trace)
            for name, content in trace.items():
                _save(settings.output_dir, slug, name, content)
            _save(
                settings.output_dir, slug, f"assets-{format}.json", pack.model_dump_json(indent=2)
            )
        elif assets_path.exists():
            pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))
        else:
            pack = None

        # voice
        if "voice" in wanted:
            with console.status("Narrating..."):
                vo.render_script(script, out_dir / "audio", stem=stem)
            _design_sfx(settings, slug, script)

        # render
        if "render" in wanted:
            if pack is None:
                _fail(f"No {format} asset pack; run without skipping assets.")
            with console.status("Rendering..."):
                try:
                    render(
                        script, pack, out_dir / "audio", out_path,
                        stem=stem, genre=brief.genre if brief else None,
                    )
                except FileNotFoundError as error:
                    _fail(str(error).split(";")[0])
            console.print(f"[green]video[/green] -> {out_path}")

        # title
        if "title" in wanted:
            with console.status("Crafting title..."):
                title_pack = title_block.run(script, topic=theory.name)
            _save(
                settings.output_dir, slug, f"{format}.title.json",
                title_pack.model_dump_json(indent=2)
            )
            console.print(f"[bold]{title_pack.title}[/bold]")
        elif title_path.exists():
            from blocks.title_description import TitlePack
            title_pack = TitlePack.model_validate_json(title_path.read_text(encoding="utf-8"))
        else:
            title_pack = None

        # publish
        if "publish" in wanted and title_pack is not None:
            if settings.auto_publish:
                with console.status("Publishing to YouTube..."):
                    video_id = publisher.upload(out_path, title_pack)
                console.print(f"[green]published[/green] https://youtu.be/{video_id}")
            else:
                console.print("[yellow]AUTO_PUBLISH off; skipping upload.[/yellow]")
        elif "publish" in wanted:
            console.print("[dim]no title; skipping upload.[/dim]")

    console.print(f"[green]Done: {len(chosen)} video(s).[/green]")


def _stage_set(only: str | None, skip: str | None) -> set[str]:
    """Parse ``--only``/``--skip`` into one stage set with validation."""

    wanted: set[str] = set(_ALL_STAGES)
    for name, value in (("only", only), ("skip", skip)):
        if not value:
            continue
        stages = [s.strip() for s in value.split(",") if s.strip()]
        unknown = [s for s in stages if s not in _ALL_STAGES]
        if unknown:
            _fail(
                f"Unknown stage(s) in --{name}: {', '.join(unknown)}. "
                f"Stages: {', '.join(_ALL_STAGES)}."
            )
        wanted = set(stages) if name == "only" else wanted - set(stages)
    if not wanted:
        _fail("Nothing left to run: every stage was skipped.")
    return wanted


def run_all_cmd(
    slugs: list[str] = typer.Argument(..., help="Catalog slugs in queue order."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    ram: float | None = typer.Option(
        None, "--ram", help="RAM budget override in GiB (profiles default it)."
    ),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if present."),
) -> None:
    """Run every slug through script→assets→voice→render, resource-budgeted."""

    from cli.runall import run_queue

    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")
    run_queue(
        slugs=list(slugs),
        fmt=format,
        ram_budget=ram,
        force=force,
        output_dir=Path(get_settings().output_dir),
    )


# -- horror corpus ------------------------------------------------------------
def horror_pull_cmd(
    count: int = typer.Option(100, "--count", "-n", help="Stories to have in total."),
) -> None:
    """Grow the creepypasta study corpus (CC BY-SA, kept with attribution)."""

    from pipeline.horror_corpus import pull_stories

    settings = get_settings()
    out_path = Path(settings.output_dir) / "corpus" / "horror" / "creepypasta.jsonl"
    with console.status("Pulling creepypastas..."):
        path = pull_stories(settings, count, out_path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    console.print(f"[green]Corpus[/green] {len(lines)} stories -> {path}")


# -- sound effects shelf ------------------------------------------------------
def sfx_pull_cmd() -> None:
    """Grow the horror SFX shelf (PD/CC audio from Wikimedia Commons)."""

    from pipeline.sfx import pull_sfx
    from providers.sources import build_http_client
    from providers.sources.mediawiki import MediaWikiClient

    settings = get_settings()
    client = build_http_client(settings)
    commons = MediaWikiClient(
        "https://commons.wikimedia.org/w/api.php",
        client=client,
        delay=settings.request_delay,
    )
    out_dir = Path("assets") / "sfx"
    with console.status("Pulling sound effects..."):
        manifest = pull_sfx(settings, commons, out_dir)
    files = json.loads(manifest.read_text(encoding="utf-8"))
    console.print(f"[green]SFX shelf[/green] {len(files)} files -> {out_dir}")


# -- horror story -------------------------------------------------------------
def story_cmd(
    theme: str | None = typer.Argument(None, help="Optional flavour, e.g. 'ghost'."),
    scenes: int = typer.Option(6, "--scenes", "-s", help="Story beats."),
    fresh_style: bool = typer.Option(False, "--fresh-style", help="Re-mine the style pack."),
) -> None:
    """Write an original horror story as a script the rest of the pipeline eats."""

    from blocks.horror_story import HorrorStoryBlock
    from blocks.horror_style import HorrorStyleBlock
    from core.models import HorrorStylePack, ResearchBrief
    from providers.llm import create_llm_provider
    from utils.text import slugify

    settings = get_settings()
    corpus = Path(settings.output_dir) / "corpus" / "horror" / "creepypasta.jsonl"
    if not corpus.exists():
        _fail("No horror corpus. Run 'ash_speaks horror-pull' first.")
    style_path = corpus.parent / "style.json"

    if style_path.exists() and not fresh_style:
        style_pack = HorrorStylePack.model_validate_json(style_path.read_text("utf-8"))
    else:
        llm = create_llm_provider(settings)
        with console.status("Mining horror style patterns from the corpus..."):
            style_pack = HorrorStyleBlock(llm).run(corpus)
        style_path.write_text(style_pack.model_dump_json(indent=2), encoding="utf-8")

    llm = create_llm_provider(settings)
    with console.status("Writing an original horror story..."):
        script = HorrorStoryBlock(llm, scenes=scenes).run(style_pack, theme=theme)

    slug = slugify(script.title)
    directory = Path(settings.output_dir) / slug
    directory.mkdir(parents=True, exist_ok=True)
    script_path = _save(
        settings.output_dir, slug, "short.json", script.model_dump_json(indent=2)
    )
    _save(settings.output_dir, slug, "short.md", render_markdown(script))
    brief = ResearchBrief(
        title=script.title,
        one_line=script.hook,
        origin="An original horror story written for this channel.",
        background=script.outro,
        claims=[],
        why_compelling=[],
    )
    _save(settings.output_dir, slug, "brief.json", brief.model_dump_json(indent=2))
    console.print(
        f"[green]{script.title}[/green] -> {script_path}\n"
        f"Next: ash_speaks assets {slug} -f short && ash_speaks voice {slug} && "
        f"ash_speaks render {slug}"
    )


def _render_assets(pack: AssetPack) -> str:
    """Human-readable scene-by-scene gallery with attribution."""

    header = pack.article_title
    if pack.article_url:
        header = f"Source article: [{pack.article_title}]({pack.article_url})"
    lines = [f"# Media — {pack.topic}", "", header, ""]
    for scene in pack.scenes:
        heading = "## Hook line" if scene.scene_index == -1 else f"## Scene {scene.scene_index + 1}"
        lines.append(heading)
        lines.append(f"*Visual:* {scene.visual}")
        lines.append(f"*Query:* {scene.query or '-'}")
        if not scene.assets:
            lines.append("_No topical match — scene stays text-only._")
        for asset in scene.assets:
            credit = asset.artist or "unknown artist"
            lines.append(
                f"- [{asset.file}]({asset.page_url})"
                f" — {asset.caption or 'no caption'}"
                f" ({asset.kind}, {asset.license or 'license?'}; {credit})"
            )
        lines.append("")
    return "\n".join(lines)


def _load_theory(settings, slug: str) -> Theory:
    """Fetch from the catalog; story slugs get a lightweight stand-in."""

    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slug)
    if theory is None:
        story_script = Path(settings.output_dir) / slug / "short.json"
        if not story_script.exists():
            _fail(f"No theory '{slug}' in the catalog and no story either.")
        now = datetime.now(UTC)
        return Theory(
            slug=slug,
            name=slug.replace("-", " "),
            first_seen=now,
            last_updated=now,
        )
    return theory  # type: ignore[return-value]


def _save(output_dir: Path, slug: str, name: str, content: str) -> Path:
    directory = Path(output_dir) / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _print_brief(brief: ResearchBrief) -> None:
    console.print(Panel(brief.one_line, title=brief.title, border_style="cyan"))
    console.print(f"[bold]Origin:[/bold] {brief.origin}")
    console.print(f"[bold]Background:[/bold] {brief.background}")
    console.print("[bold]Claimed:[/bold]")
    for claim in brief.claims:
        console.print(f"  • {claim}")
    console.print("[bold]Why it's compelling:[/bold]")
    for item in brief.why_compelling:
        console.print(f"  • {item}")


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)
