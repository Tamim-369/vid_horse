"""Command line entrypoint."""

from __future__ import annotations

import sys

import typer

from cli.commands import register
from config import get_settings, setup_logging

# Load .env file explicitly so os.getenv() works for all components
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = typer.Typer(
    name="ash_speaks",
    help=(
        "AI video making automation.\n\n"
        "Typical flow:\n"
        "  ash_speaks discover                 # crawl topics into the catalog (once)\n"
        "  ash_speaks theories -s ufo          # find a topic slug\n"
        "  ash_speaks make -n 3                # make 3 videos end-to-end\n"
        "  ash_speaks make --only voice,render # redo just voice + render\n"
        "  ash_speaks run-all x y z            # resource-budgeted overnight queue\n"
        "  ash_speaks publish SLUG --push      # upload a rendered video\n\n"
        "Stage commands (research, script, assets, voice, render, title) "
        "each run one stage for one slug.\n\n"
        "One-command faceless pipeline:\n"
        "  ash_speaks --batch --target 3\n"
        "       # research -> script -> assets -> voice -> render -> publish\n"
        "  ash_speaks TOPIC --script-only      # just topics + scripts for one topic\n"
        "  ash_speaks --list-voices            # list registered voices\n"
    ),
    no_args_is_help=True,
    add_completion=False,
)

register(app)

_TYPER_COMMANDS = {
    "discover",
    "theories",
    "research",
    "script",
    "music",
    "assets",
    "title",
    "publish",
    "voice",
    "render",
    "render-black",
    "run-all",
    "make",
    "horror-pull",
    "sfx-pull",
    "shelf-seed",
    "story",
    "help",
    "--help",
    "-h",
}


def main() -> None:
    """Run the CLI."""

    setup_logging(get_settings().log_level)
    argv = sys.argv[1:]
    first = argv[0] if argv else ""
    # videoai-style invocation (positional topic or one of its flags) goes
    # straight to the one-command pipeline; typer subcommands go to app().
    if argv and first not in _TYPER_COMMANDS:
        from cli.videoai import main as videoai_main

        videoai_main(argv)
        return
    app()


if __name__ == "__main__":
    main()
