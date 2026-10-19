"""CLI command implementations."""

from __future__ import annotations

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
            pack = pipeline.assets(theory, brief, script)
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


def _render_assets(pack: AssetPack) -> str:
    """Human-readable scene-by-scene gallery with attribution."""

    header = pack.article_title
    if pack.article_url:
        header = f"Source article: [{pack.article_title}]({pack.article_url})"
    lines = [f"# Media — {pack.topic}", "", header, ""]
    for scene in pack.scenes:
        actor_index = scene.scene_index
        lines.append(f"## Scene {actor_index + 1}")
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
    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slug)
    if theory is None:
        _fail(f"No theory '{slug}' in the catalog. Try 'videoai theories'.")
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
