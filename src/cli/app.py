"""Command line entrypoint."""

from __future__ import annotations

import typer

from cli.commands import register
from config import get_settings, setup_logging

app = typer.Typer(
    name="videoai",
    help="AI video making automation.",
    no_args_is_help=True,
    add_completion=False,
)

register(app)


def main() -> None:
    """Run the CLI."""

    setup_logging(get_settings().log_level)
    app()
