"""Context-driven subtitle emphasis: style words by communicative role.

No fixed keyword lists. An LLM reads each line in its lesson context and
decides which spans carry meaning (command, danger, reveal, contrast, key)
so a muted/deaf viewer can follow visually. Falls back to neutral captions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from core.interfaces import LLMMessage, LLMProvider
from voice.subtitles import Cue

logger = logging.getLogger(__name__)

# ASS inline overrides (BGR color). \r resets to base Cue style.
ROLES: dict[str, str] = {
    "action": r"{\b1\fs86\c&H00F2FF00&\3c&H40101010&}",   # cyan-green: do this
    "danger": r"{\b1\fs86\c&H004466FF&\3c&H40101010&}",   # orange-red: trap/warning
    "reveal": r"{\b1\fs90\c&H0000FFFF&\3c&H40101010&}",   # yellow: name/aha
    "contrast": r"{\b1\fs84\c&H00FF88AA&\3c&H40101010&}", # magenta: flip/turn
    "key": r"{\b1\fs84\c&H0000D7FF&\3c&H40101010&}",      # gold: core concept
}
RESET = r"{\r}"
WORD_RE = re.compile(r"[A-Za-z0-9']+")

SYSTEM = (
    "You style short subtitle cues so a viewer with NO AUDIO can understand "
    "the meaning. You do NOT pick fixed 'power words' — you read the sentence "
    "in context and decide what this moment DOES (warn, command, reveal, "
    "contrast, name the key idea) and emphasize only the words that carry "
    "that meaning.\n\n"
    "Roles (use sparingly, 0-2 spans per cue, most cues get 0 or 1):\n"
    "- action: the move to take / what to do now (imperative force in context)\n"
    "- danger: the trap, pressure, manipulation, or threat being exposed\n"
    "- reveal: the tactic named, the 'aha', the reframe that clicks\n"
    "- contrast: the turn/flip where meaning changes (only when it IS the turn)\n"
    "- key: the core concept to remember (when nothing else fits)\n\n"
    "Rules: never emphasize filler. Never emphasize more than 2 words in a "
    "3-4 word cue. If a cue is just connective, return no spans. Preserve the "
    "cue's words; 'text' must be words that appear in that cue.\n"
    "Output JSON only: {\"cues\": [{\"i\": <cue index>, \"spans\": [{\"text\": \"...\", \"role\": \"action|danger|reveal|contrast|key\"}]}]}"
)


def _apply_spans(text: str, spans: list[dict]) -> str:
    if not spans:
        return text
    styled = text
    # apply longer spans first to avoid partial overlaps
    for span in sorted(spans, key=lambda s: -len(s.get("text", ""))):
        role = span.get("role", "")
        tag = ROLES.get(role)
        frag = (span.get("text") or "").strip()
        if not tag or not frag:
            continue
        # case-insensitive replace of the first occurrence as a word-ish match
        pat = re.compile(r"\b(" + re.escape(frag) + r")\b", re.IGNORECASE)
        m = pat.search(styled)
        if not m:
            # try without word boundary for punctuation-adjacent tokens
            pat = re.compile(re.escape(frag), re.IGNORECASE)
            m = pat.search(styled)
        if m and "{\\" not in m.group(0):  # don't double-style
            styled = styled[:m.start()] + tag + m.group(0) + RESET + styled[m.end():]
    return styled


def emphasize_cues(
    llm: LLMProvider | None,
    lines: list[str],
    cues: list[Cue],
    *,
    beat_context: list[str] | None = None,
) -> list[Cue]:
    """Return cues with ASS-styled emphasis based on communicative context."""

    if not cues or llm is None:
        return cues
    # Map cues to their source line context (best-effort: group by time order
    # isn't perfect, but we pass all lines + cue texts so the LLM has context).
    cue_payload = [{"i": i, "text": c.text} for i, c in enumerate(cues)]
    beat_info = ""
    if beat_context:
        beat_info = "\nBeat roles by line index: " + ", ".join(
            f"{i}={b}" for i, b in enumerate(beat_context)
        )
    user = (
        "Script lines (in order: hook, scenes..., outro):\n"
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
        # sanitize control chars that break JSON (common with local models)
        raw = re.sub(r"[\x00-\x1f]+", " ", raw)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return _fallback_emphasis(lines, cues, beat_context)
        blob = m.group(0)
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
            spans = by_index.get(i, [])
            out.append(replace(cue, text=_apply_spans(cue.text, spans)))
        if any(c.text != o.text for c, o in zip(cues, out)):
            return out
        return _fallback_emphasis(lines, cues, beat_context)
    except Exception as exc:  # noqa: BLE001
        logger.warning("subtitle emphasis failed (%s); using fallback", exc)
        return _fallback_emphasis(lines, cues, beat_context)


# Structural/context heuristic fallback (used when LLM emphasis is unavailable).
# Emphasis follows sentence function/position, not a flat keyword-pop list.
_CONTRAST = re.compile(r"\b(but|instead|yet|though|unless|however)\b", re.I)
_NEGATE = re.compile(r"\b(don't|dont|not|never|no|nothing|nobody)\b", re.I)
_IMPERATIVE = re.compile(r"^(demand|ask|show|stop|watch|look|notice|check|wait|hold|say|think|prove|name|remember)\b", re.I)
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
        role = span = None
        m = _CONTRAST.search(text)
        if m:
            role, span = "contrast", m.group(0)
        if not span:
            m = _NEGATE.search(text)
            if m and not is_hook:
                role, span = "danger", m.group(0)
        if not span and is_outro is False and li >= len(lines) // 2:
            m = _IMPERATIVE.search(text)
            if m:
                role, span = "action", m.group(0)
        if not span:
            m = _NAMING.search(text)
            if m:
                # emphasize the word(s) after the naming marker when present
                after = text[m.end():].strip().split()
                if after:
                    role, span = "reveal", " ".join(after[:2]).rstrip(".,!?")
                else:
                    role, span = "reveal", m.group(0)
        if not span and (is_hook or li == max(1, len(lines) - 2)):
            # hook/payoff: lift the most content-bearing word (longest)
            words = [w for w in WORD_RE.findall(text) if len(w) > 3]
            if words:
                role, span = ("reveal" if is_outro or li == len(lines) - 2 else "key"), max(words, key=len)
        if span and role:
            out.append(replace(cue, text=_apply_spans(text, [{"text": span, "role": role}])))
        else:
            out.append(cue)
    return out
