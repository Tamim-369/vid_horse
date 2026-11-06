"""Context-driven subtitle emphasis: visual meaning for no-audio viewing.

Max 2 colors (neutral + one accent). ALL-CAPS words in cues render as
SHOUTED: bigger, bolder, fully capitalized. Emphasis is by communicative
role in context, not fixed keyword lists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from core.interfaces import LLMMessage, LLMProvider
from voice.subtitles import Cue

logger = logging.getLogger(__name__)

# 2-color system: neutral (base style) + accent. Shout = accent + max size.
ACCENT = r"{\b1\fs88\c&H0000F0FF&\3c&H50000000&}"   # electric yellow accent
SHOUT = r"{\b1\fs104\c&H0000F0FF&\3c&H60000000&}"   # ALL-CAPS words: bigger/bolder
RESET = r"{\r}"
WORD_RE = re.compile(r"[A-Za-z0-9']+")
ALLCAPS_RE = re.compile(r"\b([A-Z][A-Z0-9']{1,})\b")

SYSTEM = (
    "You style short subtitle cues so a viewer with NO AUDIO can understand "
    "the meaning. Read each cue in its sentence/lesson context and mark only "
    "the word(s) that carry the point of THIS moment (the command, the trap, "
    "the reveal/name, the turn). Do NOT use fixed keyword lists — decide from "
    "context. Emphasize at most 1 span per cue; many cues get none.\n"
    "Rules: never emphasize filler. 'text' must be words that appear in that "
    "cue. Output JSON only: {\"cues\": [{\"i\": <index>, \"spans\": [{\"text\": \"...\"}]}]}"
)


def _style_caps(text: str) -> str:
    """ALL-CAPS tokens render SHOUTED (big/bold/accent, kept uppercase)."""

    def repl(m: re.Match[str]) -> str:
        return SHOUT + m.group(1).upper() + RESET

    if "{\\" not in text:
        return ALLCAPS_RE.sub(repl, text)
    parts = re.split(r"(\{\\.*?\})", text)
    out = []
    styled_depth = 0
    for part in parts:
        if part.startswith("{\\"):
            out.append(part)
            if part == RESET or "\\r" in part:
                styled_depth = 0
            else:
                styled_depth += 1
        elif styled_depth > 0:
            out.append(part)  # already inside a styled span
        else:
            out.append(ALLCAPS_RE.sub(repl, part))
    return "".join(out)


def _apply_spans(text: str, spans: list[dict]) -> str:
    if not spans:
        return _style_caps(text)
    styled = text
    for span in sorted(spans, key=lambda s: -len((s.get("text") or ""))):
        frag = (span.get("text") or "").strip()
        if not frag:
            continue
        pat = re.compile(r"\b(" + re.escape(frag) + r")\b", re.IGNORECASE)
        m = pat.search(styled)
        if not m:
            pat = re.compile(re.escape(frag), re.IGNORECASE)
            m = pat.search(styled)
        if m and "{\\" not in styled[max(0, m.start()-2):m.start()]:
            inner = m.group(0)
            # if the source span is ALL-CAPS, use SHOUT treatment
            tag = SHOUT if inner.isupper() and len(inner) > 1 else ACCENT
            styled = styled[:m.start()] + tag + inner.upper() if tag == SHOUT else styled[:m.start()] + tag + inner + RESET + styled[m.end():]
            if tag == SHOUT:
                styled = styled  # already built
    # rebuild cleanly to avoid the ternary mess
    return _apply_spans_clean(text, spans)


def _apply_spans_clean(text: str, spans: list[dict]) -> str:
    styled = text
    for span in sorted(spans, key=lambda s: -len((s.get("text") or ""))):
        frag = (span.get("text") or "").strip()
        if not frag:
            continue
        pat = re.compile(r"\b(" + re.escape(frag) + r")\b", re.IGNORECASE)
        m = pat.search(styled)
        if not m:
            pat = re.compile(re.escape(frag), re.IGNORECASE)
            m = pat.search(styled)
        if not m:
            continue
        # don't double-style
        before = styled[max(0, m.start() - 12):m.start()]
        if "{\\" in before and r"{\r}" not in before:
            continue
        inner = m.group(0)
        is_shout = inner.isupper() and len(re.sub(r"[^A-Z]", "", inner)) > 1
        tag = SHOUT if is_shout else ACCENT
        display = inner.upper() if is_shout else inner
        styled = styled[:m.start()] + tag + display + RESET + styled[m.end():]
    return _style_caps(styled)


def emphasize_cues(
    llm: LLMProvider | None,
    lines: list[str],
    cues: list[Cue],
    *,
    beat_context: list[str] | None = None,
) -> list[Cue]:
    """Return cues with ASS-styled emphasis (2 colors max)."""

    if not cues:
        return cues
    if llm is None:
        return _fallback_emphasis(lines, cues, beat_context)
    cue_payload = [{"i": i, "text": c.text} for i, c in enumerate(cues)]
    beat_info = ""
    if beat_context:
        beat_info = "\nBeat roles by line index: " + ", ".join(
            f"{i}={b}" for i, b in enumerate(beat_context)
        )
    user = (
        "Script lines (hook, scenes..., outro):\n"
        + "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines))
        + beat_info
        + "\n\nSubtitle cues (timed, in order):\n"
        + json.dumps(cue_payload, ensure_ascii=False)
        + "\n\nReturn emphasis spans per cue index. JSON only."
    )
    try:
        raw = llm.complete(
            [LLMMessage(role="system", content=SYSTEM),
             LLMMessage(role="user", content=user)],
            temperature=0.2,
            max_tokens=1200,
        )
        raw = re.sub(r"[\x00-\x1f]+", " ", raw)
        # lenient JSON grab
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return _fallback_emphasis(lines, cues, beat_context)
        blob = m.group(0)
        # fix common trailing commas
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = json.loads(blob, strict=False)
        by_index: dict[int, list[dict]] = {}
        for item in data.get("cues", []):
            idx = item.get("i")
            if isinstance(idx, int):
                by_index[idx] = item.get("spans") or []
        out: list[Cue] = []
        for i, cue in enumerate(cues):
            out.append(replace(cue, text=_apply_spans_clean(cue.text, by_index.get(i, []))))
        if any(a.text != b.text for a, b in zip(cues, out)):
            return out
        return _fallback_emphasis(lines, cues, beat_context)
    except Exception as exc:  # noqa: BLE001
        logger.warning("subtitle emphasis failed (%s); using fallback", exc)
        return _fallback_emphasis(lines, cues, beat_context)


# Structural/context heuristic fallback (not a flat keyword list).
_CONTRAST = re.compile(r"\b(but|instead|yet|though|unless)\b", re.I)
_NEGATE = re.compile(r"\b(don't|dont|not|never|no)\b", re.I)
_IMPERATIVE = re.compile(r"^(demand|ask|show|stop|watch|look|notice|check|wait|hold|prove|remember)\b", re.I)
_NAMING = re.compile(r"\b(called|means|is when|that is|this is|named)\b", re.I)


def _cue_line_index(cue_idx: int, total_cues: int, total_lines: int) -> int:
    if total_cues <= 0 or total_lines <= 0:
        return 0
    return min(int(cue_idx * total_lines / max(total_cues, 1)), total_lines - 1)


def _fallback_emphasis(
    lines: list[str], cues: list[Cue], beat_context: list[str] | None = None
) -> list[Cue]:
    out: list[Cue] = []
    for i, cue in enumerate(cues):
        text = cue.text
        li = _cue_line_index(i, len(cues), len(lines))
        is_hook = li == 0
        is_outro = li == len(lines) - 1
        span = None
        m = _CONTRAST.search(text)
        if m:
            span = m.group(0)
        if not span:
            m = _NEGATE.search(text)
            if m and not is_hook:
                span = m.group(0)
        if not span and not is_outro and li >= max(1, len(lines) // 2):
            m = _IMPERATIVE.search(text)
            if m:
                span = m.group(0)
        if not span:
            m = _NAMING.search(text)
            if m:
                after = text[m.end():].strip().split()
                if after:
                    span = " ".join(after[:2]).rstrip(".,!?")
                else:
                    span = m.group(0)
        if not span and (is_hook or li == len(lines) - 2):
            words = [w for w in WORD_RE.findall(text) if len(w) > 3]
            if words:
                span = max(words, key=len)
        spans = [{"text": span}] if span else []
        out.append(replace(cue, text=_apply_spans_clean(text, spans)))
    return out
