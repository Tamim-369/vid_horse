"""CLI command implementations."""

from __future__ import annotations

import json
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
from providers.sources.factory import build_connectors

console = Console()

_FORMATS: tuple[ScriptFormat, ...] = ("short", "long")


def register(app: typer.Typer) -> None:
    """Attach all commands to ``app``."""

    app.command(name="discover")(discover_cmd)
    app.command(name="theories")(theories_cmd)
    app.command(name="research")(research_cmd)
    app.command(name="script")(script_cmd)
    app.command(name="assets")(assets_cmd)
    app.command(name="voice")(voice_cmd)
    app.command(name="render")(render_cmd)
    app.command(name="horror-pull")(horror_pull_cmd)
    app.command(name="sfx-pull")(sfx_pull_cmd)
    app.command(name="story")(story_cmd)


# -- discovery --------------------------------------------------------------
def discover_cmd(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max theories per source."),
    metrics: bool = typer.Option(
        True, "--metrics/--no-metrics", help="Fetch pageviews (slower)."
    ),
    sources: str | None = typer.Option(
        None, "--sources", help="Comma-separated connector names to use."
    ),
) -> None:
    """Discover conspiracy theories and merge them into the local catalog."""

    settings = get_settings()
    connectors = build_connectors(settings)
    if sources:
        wanted = {name.strip() for name in sources.split(",")}
        connectors = [c for c in connectors if c.name in wanted]

    with TheoryCatalog(settings.catalog_path) as catalog:
        with console.status("Discovering theories..."):
            report = discover(
                catalog, connectors, limit=limit, with_metrics=metrics, settings=settings
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


def theories_cmd(
    limit: int = typer.Option(25, "--limit", "-l"),
    tier: str | None = typer.Option(None, "--tier", help="mainstream|popular|moderate|underground"),
    order: str = typer.Option("popularity", "--order", "-o", help="popularity|size|name"),
) -> None:
    """List catalogued theories."""

    settings = get_settings()
    with TheoryCatalog(settings.catalog_path) as catalog:
        rows = catalog.list(limit=limit, order=order, tier=tier)

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
    format: str = typer.Option("both", "--format", "-f", help="short|long|both"),
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
    with console.status(f"Researching '{theory.name}'..."):
        brief = pipeline.research(theory, max_chars=chars)
    _save(settings.output_dir, slug, "brief.json", brief.model_dump_json(indent=2))

    for fmt in formats:
        seconds = (
            settings.script_short_seconds
            if fmt == "short"
            else settings.script_long_seconds
        )
        with console.status(f"Writing {fmt} script..."):
            script = pipeline.script(brief, fmt, target_seconds=seconds)
        path = _save(settings.output_dir, slug, f"{fmt}.json", script.model_dump_json(indent=2))
        _save(settings.output_dir, slug, f"{fmt}.md", render_markdown(script))
        console.print(
            f"[green]{script.title}[/green] "
            f"({fmt}, {len(script.scenes)} scenes) -> {path}"
        )


# -- helpers ----------------------------------------------------------------
def assets_cmd(
    slug: str = typer.Argument(..., help="Catalog slug, e.g. 'baron-52'."),
    format: str = typer.Option("both", "--format", "-f", help="short|long|both"),
) -> None:
    """Match topical wiki media to the saved script's scenes."""

    settings = get_settings()
    theory = _load_theory(settings, slug)

    brief_path = Path(settings.output_dir) / slug / "brief.json"
    if not brief_path.exists():
        _fail(f"No brief for '{slug}'. Run 'videoai research {slug}' first.")
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
            _fail(f"No {fmt} script for '{slug}'. Run 'videoai script {slug}' first.")
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
    slug: str = typer.Argument(..., help="Catalog slug, e.g. 'baron-52'."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Narrate the saved script with the cloned voice; cached line by line."""

    from voice.tts import VoiceOver

    settings = get_settings()
    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")
    script_path = Path(settings.output_dir) / slug / f"{format}.json"
    if not script_path.exists():
        _fail(f"No {format} script for '{slug}'. Run 'videoai script {slug}' first.")
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    stem = "voice" if format == "short" else f"voice-{format}"
    vo = VoiceOver()
    with console.status(f"Narrating with the cloned voice ({format})..."):
        path = vo.render_script(script, Path(settings.output_dir) / slug / "audio", stem=stem)
    sfx_path = _design_sfx(settings, slug, script)
    console.print("[green]Voiceover ready[/green] -> " + str(path))
    if sfx_path:
        console.print(f"[green]Sound design[/green] -> {sfx_path}")


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


def render_cmd(
    slug: str = typer.Argument(..., help="Catalog slug, e.g. 'baron-52'."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    music: str = typer.Option(
        "", "--music", help="Bed override, e.g. 'swell-tension.mp3'."
    ),
) -> None:
    """Assemble the final vertical mp4 from script, assets and voice tracks."""

    from render.assemble import render

    settings = get_settings()
    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")
    directory = Path(settings.output_dir) / slug
    assets_path = directory / f"assets-{format}.json"
    script_path = directory / f"{format}.json"
    if not script_path.exists():
        _fail(f"No {format} script for '{slug}'. Run 'videoai script {slug}' first.")
    if not assets_path.exists():
        _fail(f"No {format} assets for '{slug}'. Run 'videoai assets {slug}' first.")
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))
    genre = None
    brief_path = directory / "brief.json"
    if brief_path.exists():
        genre = ResearchBrief.model_validate_json(
            brief_path.read_text(encoding="utf-8")
        ).genre

    stem = "voice" if format == "short" else f"voice-{format}"
    out_path = directory / f"{slug}-{format}.mp4"
    with console.status(f"Assembling the {format} video..."):
        try:
            render(
            script, pack, directory / "audio", out_path,
            stem=stem, genre=genre, music=music,
        )
        except FileNotFoundError as error:
            _fail(str(error).split(";")[0])
    console.print("[green]Video ready[/green] -> " + str(out_path))


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
        _fail("No horror corpus. Run 'videoai horror-pull' first.")
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
        f"Next: videoai assets {slug} -f short && videoai voice {slug} && "
        f"videoai render {slug}"
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
