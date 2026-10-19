"""Parse the model's labelled plain-text script into the domain model.

The format is intentionally forgiving: small models drift in small ways
(extra blank lines, a renamed label, a stray quote). We accept reasonable
variation, clean artifacts, and raise :class:`ScriptParseError` only when the
result would be unusable — which lets the block ask the model to retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from blocks.textparse import clean, first_sentence, label_of, lines, value
from core.errors import BlockError
from core.models import ScriptFormat, ScriptPlan, ScriptScene, VideoScript

_MIN_SCENE_SECONDS = 3
_MAX_SCENE_SECONDS = 600
_BEAT_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_SCRIPT_LABELS = {
    "TITLE",
    "HOOK",
    "ONE-LINE",
    "ONELINE",
    "LOGLINE",
    "OUTRO",
    "SCENE",
    "VISUAL",
    "NARRATION",
    "BEAT",
}


def _script_label(line: str) -> str:
    """Return the script label for ``line``, tolerating markdown and numbering."""

    key = label_of(line)
    base = key.rstrip("0123456789").rstrip("-")
    return base if base in _SCRIPT_LABELS else ""


class ScriptParseError(BlockError):
    """Raised when model output cannot be parsed into a script."""


_HOOK_MAX_WORDS = 12
_HOOK_BANNED_CHARS = {'"', "\u201c", "\u201d", "[", "]", "(", ")"}
_HOOK_CLICHES = (
    "you will never see",
    "you won't believe",
    "what happens next",
    "what happened next",
    "will shock you",
    "no one could explain",
    "holds its breath",
    "holds their breath",
    "held its breath",
    "held their breath",
    "holding its breath",
    "holding their breath",
    "fate haunts",
)


def hook_problems(hook: str) -> list[str]:
    """Return the reasons ``hook`` breaks the hook rules (empty if it is clean)."""

    text = hook.strip()
    if not text:
        return ["the hook is empty"]

    problems: list[str] = []
    words = text.split()
    if len(words) > _HOOK_MAX_WORDS:
        problems.append(
            f"the hook is {len(words)} words; keep it to {_HOOK_MAX_WORDS} or fewer"
        )
    if any(char in text for char in _HOOK_BANNED_CHARS):
        problems.append(
            "the hook uses quotation marks or brackets, which do not survive voiceover"
        )
    lowered = text.lower()
    for cliche in _HOOK_CLICHES:
        if cliche in lowered:
            problems.append(f'the hook uses the cliché "{cliche}"')
    return problems


def _checked_hook(hook: str, fallback: str) -> str:
    """Clean the hook (or its fallback) and reject it if it breaks the rules."""

    candidate = clean(hook) or clean(fallback)
    problems = hook_problems(candidate)
    if problems:
        raise ScriptParseError("hook rejected: " + "; ".join(problems))
    return candidate


@dataclass
class _Scanned:
    title: str
    hook: str
    outro: str
    blocks: list[dict[str, list[str]]]


def _scan(text: str) -> _Scanned:
    title, hook, outro = "", "", ""
    blocks: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    section: str | None = None

    def new_scene() -> None:
        nonlocal current
        current = {"visual": [], "narration": []}
        blocks.append(current)

    for raw in lines(text):
        line = raw.strip()
        if not line:
            if current is not None and section == "narration":
                current["narration"].append("")
            continue

        key = _script_label(line)
        if key == "TITLE":
            title = value(line) or title
            section = None
        elif key in ("HOOK", "ONE-LINE", "ONELINE", "LOGLINE"):
            hook = value(line)
            section = None
        elif key == "OUTRO":
            outro = value(line)
            section = "outro"
            current = None
        elif key == "SCENE":
            new_scene()
            section = None
        elif key == "VISUAL":
            if current is None:
                new_scene()
            current["visual"].append(value(line))
            section = "visual"
        elif key == "NARRATION":
            if current is None:
                new_scene()
            current["narration"].append(value(line))
            section = "narration"
        elif section == "outro":
            outro = f"{outro} {line}".strip()
        elif current is not None and section in ("visual", "narration"):
            current[section].append(line)

    return _Scanned(title=title, hook=hook, outro=outro, blocks=blocks)


def parse_scenes(text: str, *, words_per_second: float = 2.5) -> list[ScriptScene]:
    """Parse just the scene blocks (used when narrating long-form chunks)."""

    scenes = _build_scenes(_scan(text).blocks, words_per_second)
    if not scenes:
        raise ScriptParseError("no scenes found in model output")
    return scenes


def parse_beats(text: str) -> list[str]:
    """Parse a plain list of BEAT lines (used to extend a long-form outline)."""

    beats: list[str] = []
    for raw in lines(text):
        line = raw.strip()
        if not line:
            continue
        if _script_label(line) in (
            "TITLE",
            "HOOK",
            "ONE-LINE",
            "ONELINE",
            "LOGLINE",
            "OUTRO",
        ):
            continue
        beat = _beat_text(line)
        if beat is not None:
            beats.append(beat)
    if not beats:
        raise ScriptParseError("no beats found in model output")
    return beats


def parse_hooks(text: str) -> list[str]:
    """Parse a numbered list of hook candidates."""

    hooks: list[str] = []
    seen: set[str] = set()
    for raw in lines(text):
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        if label_of(line) == "HOOK":
            line = value(line)
        else:
            match = _BEAT_PREFIX.match(line)
            if match:
                line = line[match.end() :]
        hook = clean(line)
        if len(hook.split()) < 3:
            continue
        key = hook.lower()
        if key in seen:
            continue
        seen.add(key)
        hooks.append(hook)

    if not hooks:
        raise ScriptParseError("no hooks found in model output")
    return hooks


_SCORE_LINE = re.compile(r"^\s*(\d{1,3})\s*[|:.\-–—]\s*(.+?)\s*$")


def parse_hook_scores(text: str) -> list[tuple[int, str]]:
    """Parse ``<score> | <hook>`` lines into (score, hook), best first."""

    scored: list[tuple[int, str]] = []
    for raw in lines(text):
        match = _SCORE_LINE.match(raw)
        if not match:
            continue
        score = max(0, min(100, int(match.group(1))))
        hook = clean(match.group(2))
        if len(hook.split()) < 3:
            continue
        scored.append((score, hook))

    if not scored:
        raise ScriptParseError("no scored hooks found in model output")
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def parse_plan(
    text: str,
    *,
    fallback_title: str,
    min_beats: int = 3,
) -> ScriptPlan:
    """Parse a labelled plain-text outline into a :class:`ScriptPlan`."""

    title, hook, outro = fallback_title, "", ""
    beats: list[str] = []
    section: str | None = None

    for raw in lines(text):
        line = raw.strip()
        if not line:
            continue

        key = _script_label(line)
        if key == "TITLE":
            title = clean(value(line)) or title
            section = None
            continue
        if key in ("HOOK", "ONE-LINE", "ONELINE", "LOGLINE"):
            hook = clean(value(line))
            section = None
            continue
        if key == "OUTRO":
            outro = clean(value(line))
            section = "outro"
            continue

        beat = _beat_text(line)
        if beat is not None:
            beats.append(beat)
            section = "beat"
            continue

        if section == "outro":
            outro = clean(f"{outro} {line}")
        elif section == "beat" and beats:
            beats[-1] = clean(f"{beats[-1]} {line}")

    if len(beats) < min_beats:
        raise ScriptParseError(f"outline needs at least {min_beats} beats, got {len(beats)}")

    try:
        return ScriptPlan(
            title=title or fallback_title,
            hook=_checked_hook(hook, first_sentence(beats[0])),
            beats=beats,
            outro=outro or "Thanks for watching.",
        )
    except ValidationError as exc:
        raise ScriptParseError(str(exc)) from exc


def parse_script(
    text: str,
    *,
    format: ScriptFormat,
    target_seconds: int,
    fallback_title: str,
    words_per_second: float = 2.5,
) -> VideoScript:
    """Parse labelled plain text into a :class:`VideoScript`."""

    scanned = _scan(text)
    scenes = _build_scenes(scanned.blocks, words_per_second)
    if not scenes:
        raise ScriptParseError("no scenes found in model output")

    try:
        return VideoScript(
            title=clean(scanned.title) or fallback_title,
            hook=_checked_hook(scanned.hook, first_sentence(scenes[0].narration)),
            format=format,
            target_seconds=target_seconds,
            scenes=scenes,
            outro=clean(scanned.outro) or "Thanks for watching.",
        )
    except ValidationError as exc:
        raise ScriptParseError(str(exc)) from exc


def _beat_text(line: str) -> str | None:
    """Return beat text if ``line`` starts a beat, otherwise ``None``."""

    if _script_label(line) in ("BEAT", "SCENE") and ":" in line:
        return clean(value(line)) or None
    match = _BEAT_PREFIX.match(line)
    if match:
        return clean(line[match.end() :]) or None
    return None


def _build_scenes(
    blocks: list[dict[str, list[str]]], words_per_second: float
) -> list[ScriptScene]:
    scenes: list[ScriptScene] = []
    for block in blocks:
        narration = clean(" ".join(part for part in block["narration"] if part))
        if not narration:
            continue
        visual = clean(" ".join(block["visual"])) or "—"
        seconds = _estimate_seconds(narration, words_per_second)
        try:
            scenes.append(ScriptScene(narration=narration, visual=visual, seconds=seconds))
        except ValidationError as exc:
            raise ScriptParseError(f"scene text rejected: {exc}") from exc
    return scenes


def _estimate_seconds(narration: str, words_per_second: float) -> int:
    words = len(narration.split())
    estimate = round(words / words_per_second) if words else _MIN_SCENE_SECONDS
    return max(_MIN_SCENE_SECONDS, min(_MAX_SCENE_SECONDS, estimate))

