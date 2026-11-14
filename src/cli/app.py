"""Command line entrypoint."""

from __future__ import annotations

import sys

import typer

from cli.commands import register, auto_cmd
from config import get_settings, setup_logging

# Load .env file explicitly so os.getenv() works for all components
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = typer.Typer(
    name="nebula",
    help=(
        "Nebula - AI-powered vertical video factory.\n\n"
        "Generate psychology/educational shorts from book lessons to rendered MP4.\n\n"
        "Quick start:\n"
        "  nebula --target 10 --no-upload          # Generate 10 videos, no YouTube upload\n"
        "  nebula --target 5 --duration 60         # 5 videos, 60 seconds each\n"
        "  nebula --book 2015_223078_the-crowd     # Use specific book\n\n"
        "Production flow:\n"
        "  nebula script \"Your Topic\"                    # Generate script from topic\n"
        "  nebula assets SCRIPT.json                     # Select visuals from manifest\n"
        "  nebula voice SCRIPT.json                      # Generate narration (Chatterbox TTS)\n"
        "  nebula render SCRIPT.json                     # Render black-style MP4\n"
        "  nebula produce \"Your Topic\"                   # Full pipeline: script → assets → voice → render\n\n"
        "Asset management:\n"
        "  nebula generate-images [--category CAT]       # Generate images via Pollinations\n"
        "  nebula organize                               # Organize images + build manifest\n\n"
        "Single-stage re-runs for existing slug:\n"
        "  nebula voice SLUG                             # Re-generate narration\n"
        "  nebula render SLUG                            # Re-render video\n"
        "  nebula publish SLUG --push                    # Upload to YouTube\n"
    ),
    no_args_is_help=False,
    add_completion=False,
)

register(app)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    target: int = typer.Option(2, "--target", "-t", help="Number of videos to generate"),
    duration: int = typer.Option(200, "--duration", "-d", help="Max duration per video in seconds (uses only what it needs)"),
    book: str = typer.Option(None, "--book", "-b", help="Specific book directory (optional)"),
    gap: int = typer.Option(0, "--gap", "-g", help="Minimum gap between videos in seconds"),
    no_upload: bool = typer.Option(False, "--no-upload", help="Never upload to YouTube (ignore AUTO_PUBLISH)"),
) -> None:
    """Run the CLI. If --target is provided, automatically runs the auto production pipeline."""
    from config import get_settings, setup_logging
    
    setup_logging(get_settings().log_level)
    
    # If target is provided but no subcommand was invoked, run auto production
    if ctx.invoked_subcommand is None:
        # Import here to avoid circular imports
        from cli.commands import auto_cmd
        # Run auto production with the provided parameters
        auto_cmd(
            count=ctx.params.get("target", 2),
            duration=ctx.params.get("duration", 45),
            book=ctx.params.get("book"),
            gap=ctx.params.get("gap", 0),
            no_upload=ctx.params.get("no_upload", False),
        )
    else:
        # Subcommand was invoked, let it run normally
        pass


if __name__ == "__main__":
    app()

def main():
    """Entry point for CLI."""
    app()