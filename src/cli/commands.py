"""CLI command implementations for Nebula video factory - Book-chunk workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer
from rich.console import Console

from config import get_settings
from core.models import AssetPack, ScriptFormat, VideoScript
from blocks.music import annotate_music

console = Console()

_FORMATS: tuple[ScriptFormat, ...] = ("short", "long")


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def register(app: typer.Typer) -> None:
    """Attach all commands to ``app``."""

    # Book-chunk production pipeline (core workflow)
    app.command(name="book-scan")(book_scan_cmd)
    app.command(name="book-lessons")(book_lessons_cmd)
    app.command(name="lesson-pick")(lesson_pick_cmd)
    app.command(name="lesson-script")(lesson_script_cmd)
    app.command(name="lesson-assets")(lesson_assets_cmd)
    app.command(name="lesson-voice")(lesson_voice_cmd)
    app.command(name="lesson-render")(lesson_render_cmd)
    app.command(name="lesson-produce")(lesson_produce_cmd)

    # Asset management
    app.command(name="generate-images")(generate_images_cmd)
    app.command(name="organize")(organize_cmd)

    # Single-stage re-runs
    app.command(name="voice-regen")(voice_regen_cmd)
    app.command(name="render-regen")(render_regen_cmd)
    app.command(name="publish")(publish_cmd)


# ============================================================================
# BOOK-CHUNK PRODUCTION PIPELINE (CORE WORKFLOW)
# ============================================================================

def book_scan_cmd(
    book_dir: str = typer.Argument(..., help="Path to book directory under content/psychology_books/"),
    force: bool = typer.Option(False, "--force", help="Re-extract even if manifest exists."),
) -> None:
    """Extract book into lesson chunks with manifest (OCR + structure detection)."""
    import subprocess
    console.print(f"[cyan]Scanning book:[/cyan] {book_dir}")
    cmd = ["uv", "run", "python", "experiments/build_psychology_book_content.py", book_dir]
    if force:
        cmd.append("--force")
    subprocess.run(cmd)


def book_lessons_cmd(
    book_dir: str = typer.Argument(..., help="Book directory name under content/psychology_books/"),
    status: bool = typer.Option(False, "--status", help="Show lesson status (taken/available)."),
) -> None:
    """List all lessons in a book with their status."""
    from utils.ledger import TakenLedger
    from blocks.lesson_picker import lesson_files

    book_path = Path("content/psychology_books") / book_dir
    if not book_path.exists():
        _fail(f"Book not found: {book_path}")

    ledger = TakenLedger()
    lessons = lesson_files(book_path)

    if status:
        console.print(f"[cyan]Lessons in {book_dir}:[/cyan]")
        for i, lesson in enumerate(lessons, 1):
            taken = ledger.is_taken(lesson)
            status_str = "[red]TAKEN[/red]" if taken else "[green]AVAILABLE[/green]"
            console.print(f"  {i:3d}. {lesson.name} {status_str}")
    else:
        for lesson in lessons:
            console.print(lesson.name)


def lesson_pick_cmd(
    book_dir: str = typer.Argument(..., help="Book directory name or path."),
    lesson_file: str = typer.Argument(..., help="Lesson filename (e.g., part_01_chapter/001_lesson.txt)."),
    context: int = typer.Option(1, "--context", "-c", help="Number of neighboring chunks for context."),
) -> None:
    """Extract one teachable lesson from a book chunk."""
    from providers.llm import create_llm_provider
    from blocks.lesson_picker import pick_lesson, neighbours
    from utils.ledger import TakenLedger
    from config import get_settings

    settings = get_settings()
    llm = create_llm_provider(settings)
    ledger = TakenLedger()

    book_path = Path(book_dir)
    if not book_path.is_absolute():
        book_path = Path("content/psychology_books") / book_dir
    lesson_path = book_path / lesson_file

    if not lesson_path.exists():
        _fail(f"Lesson not found: {lesson_path}")

    if ledger.is_taken(lesson_path):
        _fail(f"Lesson already taken: {lesson_file}")

    context_paths = neighbours(lesson_path, count=max(context, 3), ledger=ledger)

    console.print(f"[cyan]Picking lesson from:[/cyan] {lesson_file}")
    idea = pick_lesson(llm, lesson_path, context_paths=context_paths, ledger=ledger)

    console.print(f"[green]Topic:[/green] {idea.topic}")
    console.print(f"[green]Lesson:[/green] {idea.lesson}")
    console.print(f"[green]Source chunks:[/green] {len(idea.source_chunks)}")
    console.print(f"[green]Context notes:[/green] {idea.context_notes or 'none'}")


def lesson_script_cmd(
    book_dir: str = typer.Argument(..., help="Book directory name."),
    lesson_file: str = typer.Argument(..., help="Lesson filename."),
    seconds: int = typer.Option(200, "--seconds", "-s", help="Max duration in seconds (uses only what it needs)."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    context: int = typer.Option(1, "--context", "-c", help="Neighboring chunks for context."),
    output: str = typer.Option("", "--output", "-o", help="Output slug (default: auto)."),
) -> None:
    """Generate a dark-themed script from a book lesson using the ScriptEngineBlock."""
    if format not in _FORMATS:
        _fail(f"Unknown format '{format}'. Choose short or long.")

    from providers.llm import create_llm_provider
    from blocks.script_engine.block import ScriptEngineBlock
    from blocks.lesson_picker import pick_lesson, neighbours
    from utils.ledger import TakenLedger
    from config import get_settings

    settings = get_settings()
    llm = create_llm_provider(settings)
    ledger = TakenLedger()
    engine = ScriptEngineBlock(llm, target_seconds=seconds)

    book_path = Path("content/psychology_books") / book_dir
    lesson_path = book_path / lesson_file

    if not lesson_path.exists():
        _fail(f"Lesson not found: {lesson_path}")

    if ledger.is_taken(lesson_path):
        _fail(f"Lesson already taken: {lesson_file}")

    context_paths = neighbours(lesson_path, count=max(context, 3), ledger=ledger)

    console.print(f"[cyan]Picking lesson...[/cyan]")
    idea = pick_lesson(llm, lesson_path, context_paths=context_paths, ledger=ledger)

    console.print(f"[cyan]Writing script...[/cyan]")
    script = engine.run(
        idea.topic,
        source_chunks=idea.source_chunks,
        context_note=idea.context_notes or "",
        format=format,
    )
    script = script.model_copy(update={"title": idea.topic})

    # Generate output slug
    if not output:
        import re
        lesson_slug = lesson_file.replace(".txt", "")
        output = f"{book_dir[:40]}--{lesson_slug}"

    out_dir = Path(settings.output_dir) / output
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / f"{format}.json"
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    console.print(f"[green]Script saved to[/green] {script_path}")
    console.print(f"Title: {script.title}")
    console.print(f"Hook: {script.hook}")
    console.print(f"Scenes: {len(script.scenes)}")
    console.print(f"Outro: {script.outro}")


def lesson_assets_cmd(
    slug: str = typer.Argument(..., help="Output slug (directory under output/)."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Select visual assets for a lesson script using the manifest."""
    from pathlib import Path
    from blocks.sentence_assets import build_asset_pack
    from config import get_settings

    settings = get_settings()
    out_dir = Path(settings.output_dir) / slug
    script_path = out_dir / f"{format}.json"

    if not script_path.exists():
        _fail(f"Script not found: {script_path}")

    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    console.print(f"[cyan]Selecting assets for[/cyan] {slug}")
    pack = build_asset_pack(script, slug=slug)

    assets_path = out_dir / f"assets-{format}.json"
    assets_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    total = sum(len(s.assets) for s in pack.scenes)
    console.print(f"[green]Assets saved to[/green] {assets_path} ({total} visuals)")


def lesson_voice_cmd(
    slug: str = typer.Argument(..., help="Output slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Generate narration for a lesson script."""
    from pathlib import Path
    from voice.tts import VoiceOver
    from config import get_settings

    settings = get_settings()
    out_dir = Path(settings.output_dir) / slug
    script_path = out_dir / f"{format}.json"
    audio_dir = out_dir / "audio"

    if not script_path.exists():
        _fail(f"No {format} script at {script_path}")

    from core.models import VideoScript
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    vo = VoiceOver()
    with console.status(f"Narrating {slug} ({format})..."):
        path = vo.render_script(script, audio_dir, stem="voice" if format == "short" else f"voice-{format}")
    console.print(f"[green]Voiceover ready[/green] -> {path}")


def lesson_render_cmd(
    slug: str = typer.Argument(..., help="Output slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    music: str = typer.Option("", "--music", help="Music bed override."),
) -> None:
    """Render the black-style video for a lesson."""
    from pathlib import Path
    from render.black_style import render_black_style
    from config import get_settings

    settings = get_settings()
    out_dir = Path(settings.output_dir) / slug
    assets_path = out_dir / f"assets-{format}.json"
    script_path = out_dir / f"{format}.json"
    audio_dir = out_dir / "audio"

    if not script_path.exists():
        _fail(f"No {format} script at {script_path}")
    if not assets_path.exists():
        _fail(f"No {format} assets at {assets_path}. Run 'nebula lesson-assets' first.")

    from core.models import VideoScript, AssetPack
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))

    stem = "voice" if format == "short" else f"voice-{format}"
    out_path = out_dir / f"{slug}-{format}-black.mp4"

    with console.status(f"Rendering {format} video..."):
        render_black_style(script, pack, audio_dir, out_path, stem=stem, music=music, slug=slug)
    console.print(f"[green]Video ready[/green] -> {out_path}")


def lesson_produce_cmd(
    book_dir: str = typer.Argument(..., help="Book directory name."),
    lesson_file: str = typer.Argument(..., help="Lesson filename."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    seconds: int = typer.Option(200, "--seconds", "-s", help="Max duration in seconds (uses only what it needs)."),
    context: int = typer.Option(1, "--context", "-c", help="Neighboring chunks for context."),
    output: str = typer.Option("", "--output", "-o", help="Output slug (default: auto)."),
) -> None:
    """Full lesson pipeline: pick lesson → script → assets → voice → render."""
    from providers.llm import create_llm_provider
    from blocks.lesson_picker import pick_lesson, neighbours
    from blocks.script_engine.block import ScriptEngineBlock
    from blocks.music import annotate_music
    from blocks.sentence_assets import build_asset_pack
    from voice.tts import VoiceOver
    from render.black_style import render_black_style
    from utils.ledger import TakenLedger
    from config import get_settings

    settings = get_settings()
    llm = create_llm_provider(settings)
    ledger = TakenLedger()
    engine = ScriptEngineBlock(llm, target_seconds=seconds)

    book_path = Path("content/psychology_books") / book_dir
    lesson_path = book_path / lesson_file

    if not lesson_path.exists():
        _fail(f"Lesson not found: {lesson_path}")

    if ledger.is_taken(lesson_path):
        _fail(f"Lesson already taken: {lesson_file}")

    context_paths = neighbours(lesson_path, count=max(context, 3), ledger=ledger)

    if not output:
        import re
        lesson_slug = lesson_file.replace(".txt", "")
        output = f"{book_dir[:40]}--{lesson_slug}"

    out_dir = Path(settings.output_dir) / output
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Producing video from lesson:[/bold cyan] {lesson_file}")

    # 1. Pick lesson
    console.print("[cyan]1/5 Picking lesson...[/cyan]")
    idea = pick_lesson(llm, lesson_path, context_paths=context_paths, ledger=ledger)

    # 2. Script
    console.print("[cyan]2/5 Writing script...[/cyan]")
    script = engine.run(
        idea.topic,
        source_chunks=idea.source_chunks,
        context_note=idea.context_notes or "",
        format=format,
    )
    script = script.model_copy(update={"title": idea.topic})
    script = annotate_music(llm, script)

    script_path = out_dir / f"{format}.json"
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    # 3. Assets
    console.print("[cyan]3/5 Selecting assets...[/cyan]")
    pack = build_asset_pack(script, slug=output)
    assets_path = out_dir / f"assets-{format}.json"
    assets_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")

    # 4. Voice
    console.print("[cyan]4/5 Generating narration...[/cyan]")
    vo = VoiceOver()
    with console.status("Generating narration..."):
        vo.render_script(script, audio_dir, stem="voice" if format == "short" else f"voice-{format}")

    # 5. Render
    console.print("[cyan]5/5 Rendering video...[/cyan]")
    stem = "voice" if format == "short" else f"voice-{format}"
    out_path = out_dir / f"{output}-{format}-black.mp4"
    with console.status("Rendering video..."):
        render_black_style(script, pack, audio_dir, out_path, stem=stem, music="", slug=output)

    # Mark lesson as taken
    ledger.mark(str(lesson_path), script_dir=str(out_dir), note=idea.lesson)

    console.print(f"[bold green]Done![/bold green] {out_path}")


# ============================================================================
# ASSET MANAGEMENT
# ============================================================================

def generate_images_cmd(
    category: str = typer.Option("", "--category", "-c", help="Generate only specific category."),
    limit: int = typer.Option(0, "--limit", "-l", help="Limit images per category (0 = all)."),
) -> None:
    """Generate images via Pollinations for all categories."""
    import subprocess
    console.print("[cyan]Generating images via Pollinations...[/cyan]")
    cmd = ["uv", "run", "python", "experiments/generate_psychology_images.py"]
    if category:
        cmd.extend(["--category", category])
    if limit:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(cmd)


def organize_cmd() -> None:
    """Organize flat images into category dirs and build manifest."""
    import subprocess
    console.print("[cyan]Organizing images and building manifest...[/cyan]")
    subprocess.run(["uv", "run", "python", "experiments/organize_and_manifest.py"])


# ============================================================================
# SINGLE-STAGE RE-RUNS
# ============================================================================

def voice_regen_cmd(
    slug: str = typer.Argument(..., help="Output slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
) -> None:
    """Re-generate narration for an existing slug."""
    from pathlib import Path
    from voice.tts import VoiceOver
    from config import get_settings

    settings = get_settings()
    out_dir = Path(settings.output_dir) / slug
    script_path = out_dir / f"{format}.json"
    audio_dir = out_dir / "audio"

    if not script_path.exists():
        _fail(f"No {format} script at {script_path}")

    from core.models import VideoScript
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))

    vo = VoiceOver()
    with console.status(f"Re-generating narration for {slug}..."):
        path = vo.render_script(script, audio_dir, stem="voice" if format == "short" else f"voice-{format}")
    console.print(f"[green]Voiceover ready[/green] -> {path}")


def render_regen_cmd(
    slug: str = typer.Argument(..., help="Output slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    music: str = typer.Option("", "--music", help="Music bed override."),
) -> None:
    """Re-render video for an existing slug."""
    from pathlib import Path
    from render.black_style import render_black_style
    from config import get_settings

    settings = get_settings()
    out_dir = Path(settings.output_dir) / slug
    assets_path = out_dir / f"assets-{format}.json"
    script_path = out_dir / f"{format}.json"
    audio_dir = out_dir / "audio"

    if not script_path.exists():
        _fail(f"No {format} script at {script_path}")
    if not assets_path.exists():
        _fail(f"No {format} assets at {assets_path}. Run 'nebula lesson-assets' first.")

    from core.models import VideoScript, AssetPack
    script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))

    stem = "voice" if format == "short" else f"voice-{format}"
    out_path = out_dir / f"{slug}-{format}-black.mp4"

    with console.status(f"Re-rendering {format} video..."):
        render_black_style(script, pack, audio_dir, out_path, stem=stem, music=music, slug=slug)
    console.print(f"[green]Video ready[/green] -> {out_path}")


def publish_cmd(
    slug: str = typer.Argument(..., help="Output slug."),
    format: str = typer.Option("short", "--format", "-f", help="short|long"),
    video: Path | None = typer.Option(None, "--video", help="Override video path."),
    push: bool = typer.Option(False, "--push", help="Upload for real even with AUTO_PUBLISH off."),
) -> None:
    """Upload the rendered video to YouTube (respects AUTO_PUBLISH)."""
    settings = get_settings()
    out_dir = Path(settings.output_dir) / slug
    video_path = video or (out_dir / f"{slug}-{format}-black.mp4")

    if not video_path.exists():
        _fail(f"No video at {video_path}. Render the video first.")

    title_path = out_dir / f"{format}.title.json"
    if title_path.exists():
        from blocks.title_description import TitlePack
        pack = TitlePack.model_validate_json(title_path.read_text(encoding="utf-8"))
    else:
        from blocks.title_description import TitleDescriptionBlock
        from providers.llm import create_llm_provider

        script_path = out_dir / f"{format}.json"
        if not script_path.exists():
            _fail(f"No {format} script at {script_path}")
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        pack = TitleDescriptionBlock(create_llm_provider(settings)).run(script, topic=script.title)

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


# ============================================================================
# AUTO PRODUCTION (fully automatic book -> video pipeline)
# ============================================================================

def auto_cmd(
    count: int = typer.Option(2, "--count", "-c", help="Number of videos to produce"),
    duration: int = typer.Option(200, "--duration", "-d", help="Max duration per video in seconds (uses only what it needs)"),
    book: str = typer.Option(None, "--book", "-b", help="Specific book directory (optional)"),
    sequential: bool = typer.Option(True, "--sequential/--parallel", help="Run videos sequentially (default) or in parallel"),
    gap: int = typer.Option(0, "--gap", "-g", help="Minimum gap between videos in seconds"),
    no_upload: bool = typer.Option(False, "--no-upload", help="Never upload to YouTube (ignore AUTO_PUBLISH setting)"),
) -> None:
    """Automatically produce videos from available book lessons.

    Scans all books in content/psychology_books/, finds available (not taken) lessons,
    and produces videos from them automatically.

    Example:
        nebula auto --count 5 --seconds 45
        nebula auto --count 3 --seconds 60 --book 2015_223078_the-crowd
        nebula auto -c 1 -s 30 -j 1
    """
    import time
    from pathlib import Path
    from utils.ledger import TakenLedger
    from blocks.lesson_picker import next_untaken, neighbours
    from blocks.lesson_picker import pick_lesson
    from blocks.script_engine.block import ScriptEngineBlock
    from blocks.music import annotate_music
    from blocks.sentence_assets import build_asset_pack
    from voice.tts import VoiceOver
    from render.black_style import render_black_style
    from providers.llm import create_llm_provider
    from config import get_settings

    settings = get_settings()
    ledger = TakenLedger()
    llm = create_llm_provider(settings)
    engine = ScriptEngineBlock(llm, target_seconds=duration)

    produced = 0

    # Temporarily disable auto_publish if --no-upload is set
    original_auto_publish = settings.auto_publish
    if no_upload:
        settings.auto_publish = False

    # Find books to process
    books_root = Path("content/psychology_books")
    if book:
        book_paths = [Path("content/psychology_books") / book]
    else:
        book_paths = sorted([p for p in books_root.iterdir() if p.is_dir()])

    for book_path in book_paths:
        if produced >= count:
            break

        console.print(f"\n[bold cyan]Processing book: {book_path.name}[/bold cyan]")

        while produced < count:
            lesson_path = next_untaken(book_path, ledger)
            if lesson_path is None:
                console.print(f"  No more available lessons in {book_path.name}")
                break

            # Get lesson info
            rel_lesson = lesson_path.relative_to(book_path)
            console.print(f"\n[cyan]Producing: {rel_lesson}[/cyan]")

            # Generate output slug
            lesson_slug = lesson_path.stem
            book_name = book_path.name
            output_slug = f"{book_name[:40]}--{lesson_slug}"
            out_dir = Path(settings.output_dir) / output_slug
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_dir = out_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Pick lesson
                console.print(f"  [1/5] Picking lesson...")
                idea = pick_lesson(llm, lesson_path, context_paths=neighbours(lesson_path, ledger=ledger), ledger=ledger)

                # 2. Write script
                console.print(f"  [2/5] Writing script...")
                script = engine.run(
                    idea.topic,
                    source_chunks=idea.source_chunks,
                    context_note=idea.context_notes or "",
                    format="short",
                )
                script = script.model_copy(update={"title": idea.topic})
                script = annotate_music(llm, script)

                script_path = out_dir / "short.json"
                script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

                # 3. Select assets
                console.print(f"  [3/5] Selecting assets...")
                pack = build_asset_pack(script, slug=output_slug)
                assets_path = out_dir / "assets-short.json"
                assets_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")

                # 4. Voice
                console.print(f"  [4/5] Generating narration...")
                vo = VoiceOver()
                with console.status("Generating narration..."):
                    vo.render_script(script, audio_dir, stem="voice")

                # 5. Render
                console.print(f"  [5/5] Rendering video...")
                stem = "voice"
                out_path = out_dir / f"{output_slug}-short-black.mp4"
                with console.status("Rendering video..."):
                    render_black_style(script, pack, audio_dir, out_path, stem=stem, music="", slug=output_slug)

                # Mark lesson as taken
                ledger.mark(str(lesson_path), script_dir=str(out_dir), note=idea.lesson)

                console.print(f"[bold green]Done![/bold green] {out_path}")
                produced += 1

                # Gap between videos
                if produced < count and gap > 0:
                    console.print(f"  Waiting {gap}s before next video...")
                    time.sleep(gap)

            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")
                # Don't mark as taken if failed - allow retry
                import traceback
                traceback.print_exc()

    console.print(f"\n[bold cyan]{'='*60}")
    console.print(f"Completed: {produced}/{count} videos")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")


# ============================================================================
# REGISTER ALL COMMANDS
# ============================================================================

def register(app: typer.Typer) -> None:
    """Attach all commands to ``app``."""

    # Book-chunk production pipeline (core workflow)
    app.command(name="book-scan")(book_scan_cmd)
    app.command(name="book-lessons")(book_lessons_cmd)
    app.command(name="lesson-pick")(lesson_pick_cmd)
    app.command(name="lesson-script")(lesson_script_cmd)
    app.command(name="lesson-assets")(lesson_assets_cmd)
    app.command(name="lesson-voice")(lesson_voice_cmd)
    app.command(name="lesson-render")(lesson_render_cmd)
    app.command(name="lesson-produce")(lesson_produce_cmd)

    # Auto production (fully automatic)
    app.command(name="auto")(auto_cmd)

    # Asset management
    app.command(name="generate-images")(generate_images_cmd)
    app.command(name="organize")(organize_cmd)

    # Single-stage re-runs
    app.command(name="voice-regen")(voice_regen_cmd)
    app.command(name="render-regen")(render_regen_cmd)
    app.command(name="publish")(publish_cmd)