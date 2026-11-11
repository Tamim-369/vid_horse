#!/usr/bin/env python3
"""Auto-produce videos from available lessons - fully automatic."""

from __future__ import annotations

import typer
from pathlib import Path
from rich.console import Console

from utils.ledger import TakenLedger
from blocks.lesson_picker import lesson_files, next_untaken
from blocks.lesson_picker import pick_lesson, neighbours
from blocks.script_engine.block import ScriptEngineBlock
from blocks.music import annotate_music
from blocks.sentence_assets import build_asset_pack
from voice.tts import VoiceOver
from render.black_style import render_black_style
from providers.llm import create_llm_provider
from config import get_settings

console = Console()


def auto_produce(count: int = 2, seconds: int = 45, book_dir: str = None):
    """Automatically produce videos from available lessons."""
    
    settings = get_settings()
    ledger = TakenLedger()
    llm = create_llm_provider(settings)
    engine = ScriptEngineBlock(llm, target_seconds=seconds)
    
    # Find books to process
    books_root = Path("content/psychology_books")
    if book_dir:
        book_paths = [Path("content/psychology_books") / book_dir]
    else:
        book_paths = sorted([p for p in books_root.iterdir() if p.is_dir()])
    
    produced = 0
    
    for book_path in book_paths:
        if produced >= count:
            break
            
        print(f"\n{'='*60}")
        print(f"Processing book: {book_path.name}")
        print(f"{'='*60}")
        
        # Find available lessons in this book
        while produced < count:
            lesson_path = next_untaken(book_path, ledger)
            if lesson_path is None:
                print(f"  No more available lessons in {book_path.name}")
                break
            
            # Get lesson info
            rel_lesson = lesson_path.relative_to(book_path)
            print(f"\n  Producing: {rel_lesson}")
            
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
                print(f"  1/5 Picking lesson...")
                idea = pick_lesson(llm, lesson_path, context_paths=neighbours(lesson_path, count=1), ledger=ledger)
                
                # 2. Write script
                print(f"  2/5 Writing script...")
                script = engine.run(
                    idea.topic,
                    source_chunks=idea.source_chunks,
                    context_note=idea.context_notes or "",
                    format="short",
                )
                script = script.model_copy(update={"title": idea.topic})
                
                # Add music annotation
                from blocks.music import annotate_music
                script = annotate_music(llm, script)
                
                script_path = out_dir / "short.json"
                script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
                
                # 3. Select assets
                print(f"  3/5 Selecting assets...")
                pack = build_asset_pack(script, slug=output_slug)
                assets_path = out_dir / "assets-short.json"
                assets_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
                
                # 4. Voice
                print(f"  4/5 Generating narration...")
                vo = VoiceOver()
                with console.status("Generating narration..."):
                    vo.render_script(script, audio_dir, stem="voice")
                
                # 5. Render
                print(f"  5/5 Rendering video...")
                stem = "voice"
                out_path = out_dir / f"{output_slug}-short-black.mp4"
                with console.status("Rendering video..."):
                    from render.black_style import render_black_style
                    render_black_style(script, pack, audio_dir, out_path, stem=stem, music="", slug=output_slug)
                
                # Mark as taken
                ledger.mark(str(lesson_path), script_dir=str(out_dir), note=idea.lesson)
                
                print(f"  ✓ Done: {out_path}")
                produced += 1
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # Don't mark as taken if failed - allow retry
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Completed: {produced}/{count} videos")
    print(f"{'='*60}")


if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def main(
        count: int = typer.Option(2, "--count", "-c", help="Number of videos to produce"),
        seconds: int = typer.Option(45, "--seconds", "-s", help="Target duration per video"),
        book: str = typer.Option(None, "--book", "-b", help="Specific book directory (optional)"),
    ):
        """Auto-produce videos from available lessons."""
        # Import console here to avoid circular import
        from rich.console import Console
        global console
        console = Console()
        
        auto_produce(count=count, seconds=seconds, book_dir=book)


if __name__ == "__main__":
    app()