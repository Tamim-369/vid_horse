"""Render a :class:`VideoScript` as readable Markdown."""

from __future__ import annotations

from core.models import VideoScript


def render_markdown(script: VideoScript) -> str:
    """Return a human-friendly Markdown rendering of ``script``."""

    lines = [
        f"# {script.title}",
        "",
        f"**Format:** {script.format} (~{script.target_seconds}s)"
        f" · **Scenes:** {len(script.scenes)}",
        "",
        f"**Hook:** {script.hook}",
        "",
    ]
    for index, scene in enumerate(script.scenes, start=1):
        lines += [
            f"## Scene {index} ({scene.seconds}s)",
            "",
            f"*Visual:* {scene.visual}",
            "",
            scene.narration,
            "",
        ]
    lines += [f"**Outro:** {script.outro}"]
    if script.source_notes:
        lines += ["", "## Source notes", *[f"- {note}" for note in script.source_notes]]
    return "\n".join(lines)
