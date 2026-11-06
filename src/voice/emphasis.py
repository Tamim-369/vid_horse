"""Context-driven subtitle emphasis: match visual vibe to sentence meaning.

Uses LLM (complete_json) to assign communicative roles per cue in context;
falls back to structural heuristics. 2 colors max (white base + muted accent),
vibe expressed through size/weight by role. ALL-CAPS words always SHOUT.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from pydantic import BaseModel, Field

from core.interfaces import LLMMessage, LLMProvider
from voice.subtitles import Cue

logger = logging.getLogger(__name__)

# Bright white emphasis on dimmer white base (dark psych, high-contrast pop).
ACCENT = "&H00FFFFFF"  # bright white for emphasized words
OUTLINE = "&H90000000"
SHOUT_OUTLINE = "&H90000000"
ROLE_FS = {
    "action": 74,    # direct, protective move
    "danger": 76,    # threat/trap exposed
    "reveal": 80,    # tactic named / aha lands
    "contrast": 72,  # turn/flip
    "key": 70,       # core idea (subtle)
}
SHOUT_FS = 86  # ALL-CAPS: biggest
RESET = r"{\r}"
WORD_RE = re.compile(r"[A-Za-z0-9']+")
# ALL-CAPS token: at least 2 letters, no lowercase. Avoid matching inside ASS tags.
ALLCAPS_RE = re.compile(r"\b([A-Z]{2,}[A-Z0-9']*)\b")
ASS_TAG_RE = re.compile(r"\{[^}]*\}")


class SpanMark(BaseModel):
    text: str = Field(description="Word(s) present in the cue to emphasize")
    role: str = Field(default="key", description="action|danger|reveal|contrast|key")


class CueMark(BaseModel):
    i: int
    spans: list[SpanMark] = Field(default_factory=list)


class EmphasisPlan(BaseModel):
    cues: list[CueMark] = Field(default_factory=list)


SYSTEM = (
    "Style subtitle cues so a muted/deaf viewer FEELS the sentence's vibe. "
    "For each cue read its meaning in the full lesson context and mark 1-2 spans "
    "(the specific word(s) that carry the vibe) — can be more than one word in a "
    "cue when they together carry the moment. No fixed keyword lists — decide from "
    "what the sentence DOES.\n"
    "Roles (pick the one matching the moment's vibe):\n"
    "- danger: a trap, pressure, manipulation, threat, or 'don't get pulled in'\n"
    "- action: the protective move/command to take now (do this)\n"
    "- reveal: the tactic named or the realization that clicks\n"
    "- contrast: the turn/flip where meaning changes (the BUT/instead moment)\n"
    "- key: the core idea to remember (sparing)\n"
    "Calm/neutral explanatory cues get NO spans. Never emphasize filler. "
    "'text' must be word(s) that appear in that cue."
)


def _tag(role: str | None, *, shout: bool = False) -> str:
    if shout:
        fs = SHOUT_FS
        ol = SHOUT_OUTLINE
    else:
        fs = ROLE_FS.get((role or "key").lower(), ROLE_FS["key"])
        ol = OUTLINE
    # regular weight (\b0) overall; emphasis pops via bright white + size
    return r"{\b0\fs" + str(fs) + r"\c" + ACCENT + r"\3c" + ol + "&}"


def _strip_tags(text: str) -> str:
    return ASS_TAG_RE.sub("", text)


def _style_caps(text: str) -> str:
    """Style ALL-CAPS tokens as SHOUT, but never touch existing ASS tags."""

    def repl(m: re.Match[str]) -> str:
        return _tag(None, shout=True) + m.group(1).upper() + RESET

    if "{" not in text:
        return ALLCAPS_RE.sub(repl, text)
    parts = ASS_TAG_RE.split(text)
    tags = ASS_TAG_RE.findall(text)
    out: list[str] = []
    styled = False
    for i, part in enumerate(parts):
        # A segment is "inside a styled span" if the preceding tag opened
        # styling and wasn't a reset. Track by scanning tags seen so far.
        if styled:
            out.append(part)
        else:
            out.append(ALLCAPS_RE.sub(repl, part))
        if i < len(tags):
            tag = tags[i]
            out.append(tag)
            if tag == RESET or r"\r" in tag:
                styled = False
            elif r"\fs" in tag or r"\b1" in tag or r"\c" in tag:
                styled = True
            # other tags (pos/karaoke etc., unlikely here) leave state as-is
    return "".join(out)


def _already_styled_before(text: str, pos: int) -> bool:
    window = text[max(0, pos - 40) : pos]
    last_open = window.rfind("{")
    last_reset = window.rfind(RESET)
    if last_open == -1:
        return False
    # styled if an opening tag appeared after the last reset
    return last_open > last_reset


def _apply(text: str, spans: list[SpanMark]) -> tuple[str, list[str]]:
    styled = text
    pops: list[str] = []
    for span in sorted(spans, key=lambda s: -len(s.text or "")):
        frag = (span.text or "").strip()
        if not frag:
            continue
        m = re.search(r"\b(" + re.escape(frag) + r")\b", styled, re.I)
        if not m:
            m = re.search(re.escape(frag), styled, re.I)
        if not m:
            continue
        if _already_styled_before(styled, m.start()):
            continue
        inner = m.group(0)
        role = (span.role or "key").lower()
        # Shout when source/ASR token is ALL-CAPS, or for strong command beats.
        strong = role in {"action", "danger", "reveal"} and len(inner) >= 4
        is_shout = (inner.isupper() and len(re.sub(r"[^A-Z]", "", inner)) >= 2) or (
            inner.lower() in {"demand", "prove", "stop", "never", "don't", "dont"}
        )
        use_shout = is_shout
        tag = _tag(role, shout=use_shout)
        display = inner.upper() if (use_shout or strong) else inner
        styled = styled[: m.start()] + tag + display + RESET + styled[m.end() :]
        pops.append(display.upper() if len(display.strip()) <= 24 else display.upper())
    styled = _style_caps(styled)
    # pops for any ALL-CAPS tokens that became SHOUT in-text but weren't from spans
    return styled, pops[:3]


def emphasize_cues(
    llm: LLMProvider | None,
    lines: list[str],
    cues: list[Cue],
    *,
    beat_context: list[str] | None = None,
) -> list[Cue]:
    if not cues:
        return cues
    if llm is None:
        return _fallback(lines, cues)
    try:
        beat_info = ""
        if beat_context:
            beat_info = "\nBeat roles by line: " + ", ".join(
                f"{i}={b}" for i, b in enumerate(beat_context)
            )
        user = (
            "Lesson lines (hook, scenes..., outro):\n"
            + "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines))
            + beat_info
            + "\n\nTimed subtitle cues:\n"
            + "\n".join(f'{i}: "{c.text}"' for i, c in enumerate(cues))
            + "\n\nReturn vibe-matching spans per cue index as JSON."
        )
        plan = llm.complete_json(
            [
                LLMMessage(role="system", content=SYSTEM),
                LLMMessage(role="user", content=user),
            ],
            EmphasisPlan,
            temperature=0.2,
            max_tokens=1400,
        )
        by_i = {m.i: m.spans for m in plan.cues}
        out: list[Cue] = []
        for i, c in enumerate(cues):
            text, pops = _apply(c.text, by_i.get(i, []))
            out.append(replace(c, text=text, pops=pops))
        if any(a.text != b.text for a, b in zip(cues, out)):
            return out
        return _fallback(lines, cues)
    except Exception as exc:  # noqa: BLE001
        logger.warning("subtitle emphasis failed (%s); using fallback", exc)
        return _fallback(lines, cues)


# --- structural fallback: vibe from sentence function/position ---
_CONTRAST = re.compile(r"\b(but|instead|yet|unless|however)\b", re.I)
_NEGATE = re.compile(r"\b(don't|doesn't|isn't|won't|can't|not|never|no)\b", re.I)
_ACTION = re.compile(
    r"\b(demand|ask|show|stop|watch|look|check|hold|prove|remember|test|pause|wait|step|walk|verify|question)\b",
    re.I,
)
_TRAP = re.compile(
    r"\b(trap|trick|pressure|push|force|hook|rush|panic|scare|fake|lie|lies|bait|manipulat|control)\w*\b",
    re.I,
)
_NAMING = re.compile(r"\b(called|means|is when|this is|that is|named)\b", re.I)
# cues that are plain scene-setting questions/explanations → no styling
_CALM = re.compile(
    r"^(it'?s easy|imagine|think about|what if|does that mean|if (most|someone|they)|like that)\b",
    re.I,
)


def _li(cue_i: int, n_cues: int, n_lines: int) -> int:
    return min(int(cue_i * n_lines / max(n_cues, 1)), n_lines - 1) if n_cues else 0


def _fallback(lines: list[str], cues: list[Cue]) -> list[Cue]:
    """Restrained, vibe-aware fallback: style only moments with a clear function."""

    out: list[Cue] = []
    n = len(lines)
    for i, cue in enumerate(cues):
        li = _li(i, len(cues), n)
        text = cue.text
        plain = _strip_tags(text)
        role: str | None = None
        frag: str | None = None
        calm = bool(_CALM.search(plain)) or li == 0

        # Priority by communicative vibe (not keywords-as-style): reveal/name,
        # explicit trap/threat, turn/contrast, protective action, firm negation.
        m = _NAMING.search(plain)
        if m and not calm:
            after = plain[m.end() :].strip().split()
            role, frag = "reveal", (
                " ".join(after[:2]).rstrip(".,!?") if after else m.group(0)
            )
        if not frag:
            m = _TRAP.search(plain)
            if m:
                role, frag = "danger", m.group(0)
        if not frag:
            m = _CONTRAST.search(plain)
            if m:
                role, frag = "contrast", m.group(0)
        if not frag and not calm:
            m = _ACTION.search(plain)
            if m and (li >= max(1, n // 2) or m.group(0).lower() in {"demand", "prove", "stop", "question", "verify"}):
                role, frag = "action", m.group(0)
        if not frag and not calm:
            m = _NEGATE.search(plain)
            if m and li > 0:
                role, frag = "danger", m.group(0)

        # Calm/scene-setting cues stay unstyled (vibe = neutral). Only ALL-CAPS
        # words get SHOUT via _apply/_style_caps. Allow a 2nd vibe word when clear.
        spans: list[SpanMark] = []
        if frag:
            spans.append(SpanMark(text=frag, role=role or "key"))
            if role in {"action", "danger"} and frag.lower() not in plain.lower():
                pass
            elif role in {"action", "danger"}:
                # add a supporting verb/noun if present (e.g. "DEMAND proof")
                m2 = _ACTION.search(plain) if role == "action" else _TRAP.search(plain)
                if m2 and m2.group(0).lower() != frag.lower().split()[0].lower():
                    spans.append(SpanMark(text=m2.group(0), role=role))
        text, pops = _apply(plain, spans)
        out.append(replace(cue, text=text, pops=pops))
    return out
