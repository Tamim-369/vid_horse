"""Context-driven subtitle emphasis: match visual vibe to sentence meaning.

Uses LLM (complete_json) when available, merged with a coverage-guaranteed
content-salience baseline so main meaning words (incl. multi-word phrases)
are never missed. No-audio/deaf-viewer clarity, dark-psych tone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from pydantic import BaseModel, Field

from core.interfaces import LLMMessage, LLMProvider
from voice.subtitles import Cue

logger = logging.getLogger(__name__)

# Bright white emphasis (rendered by subtitles module's palette/kinetics).
ACCENT = "&H00FFFFFF"
OUTLINE = "&H90000000"
ROLE_FS = {"action": 74, "danger": 76, "reveal": 80, "contrast": 72, "key": 70}
SHOUT_FS = 86
RESET = r"{\r}"
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", (s or "").lower())


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
    "Style subtitle cues so a muted/deaf viewer FEELS the sentence's vibe and "
    "never misses the MAIN meaning. For each cue, mark 1-3 spans: the core "
    "meaning-carrying word(s) (can be a multi-word phrase like 'demand proof', "
    "'gain control', 'don't believe'). Do not skip the main point of the cue.\n"
    "Roles (pick the vibe):\n"
    "- danger: trap, pressure, manipulation, threat, or 'don't get pulled in'\n"
    "- action: the protective move/command to take now\n"
    "- reveal: tactic named or the realization that clicks\n"
    "- contrast: the turn/flip where meaning changes\n"
    "- key: the core idea/fact to remember\n"
    "Avoid pure filler (a, the, it, is, to, and). Prefer content words/phrases "
    "that carry meaning. 'text' must be word(s) that appear in that cue. JSON."
)


def _tag(role: str | None, *, shout: bool = False) -> str:
    if shout:
        fs, ol = SHOUT_FS, OUTLINE
    else:
        fs = ROLE_FS.get((role or "key").lower(), ROLE_FS["key"])
        ol = OUTLINE
    return r"{\b0\fs" + str(fs) + r"\c" + ACCENT + r"\3c" + ol + "&}"


def _strip_tags(text: str) -> str:
    return ASS_TAG_RE.sub("", text)


def _style_caps(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return _tag(None, shout=True) + m.group(1).upper() + RESET

    if "{" not in text:
        return ALLCAPS_RE.sub(repl, text)
    parts = ASS_TAG_RE.split(text)
    tags = ASS_TAG_RE.findall(text)
    out: list[str] = []
    styled = False
    for i, part in enumerate(parts):
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
    return "".join(out)


def _already_styled_before(text: str, pos: int) -> bool:
    window = text[max(0, pos - 40) : pos]
    last_open = window.rfind("{")
    last_reset = window.rfind(RESET)
    if last_open == -1:
        return False
    return last_open > last_reset


STRONG_CMD = {"demand", "prove", "stop", "never", "don't", "dont", "verify", "question", "refuse", "check"}


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
        strong = role in {"action", "danger", "reveal"} and len(inner) >= 4
        is_shout = (inner.isupper() and len(re.sub(r"[^A-Z]", "", inner)) >= 2) or (
            _norm(inner.split()[0]) in STRONG_CMD
        )
        tag = _tag(role, shout=is_shout)
        display = inner.upper() if (is_shout or strong) else inner
        styled = styled[: m.start()] + tag + display + RESET + styled[m.end() :]
        pops.append(display.upper() if len(display.strip()) <= 24 else display.upper())
    styled = _style_caps(styled)
    return styled, pops[:4]


# --- vibe/term patterns ---
_CONTRAST = re.compile(r"\b(but|instead|yet|unless|however|still)\b", re.I)
_NEGATE = re.compile(r"\b(don't|doesn't|isn't|won't|can't|cannot|not|never|no|without)\b", re.I)
_ACTION = re.compile(
    r"\b(demand|ask|show|stop|watch|look|check|hold|prove|proof|remember|test|pause|wait|step|verify|question|notice|choose|decide|compare|refuse|resist|think)\b",
    re.I,
)
_TRAP = re.compile(
    r"\b(trap|trick|tricks|pressure|pressured|push|pushed|force|forced|hook|rush|rushed|panic|scare|scared|fake|lie|lies|lying|bait|manipulat\w*|control\w*|guilt\w*|shame\w*|fear\w*|doubt\w*|blame\w*|threat\w*|obey\w*|fool\w*|cheat\w*|hide\w*|secret\w*)\b",
    re.I,
)
_NAMING = re.compile(r"\b(called|means|is when|this is|that is|named)\b", re.I)

STOP = {
    "a","an","the","is","are","was","were","be","been","being","it","its","it's","that",
    "this","these","those","to","of","in","on","for","with","as","at","by","from","or",
    "and","if","so","do","does","did","will","would","can","could","should","may","might",
    "you","your","we","our","they","their","them","he","she","his","her","i","me","my",
    "just","than","then","into","about","up","out","get","got","have","has","had","some",
    "any","all","more","most","very","also","like","what","when","how","who","why",
    "there","here","because","before","after","really",
}


def _role_for(text: str) -> str:
    t = text
    if _NAMING.search(t):
        return "reveal"
    if _TRAP.search(t):
        return "danger"
    if _CONTRAST.search(t):
        return "contrast"
    if _ACTION.search(t):
        return "action"
    if _NEGATE.search(t):
        return "danger"
    return "key"


def _is_strong(text: str) -> bool:
    return bool(_TRAP.search(text) or _ACTION.search(text) or _NAMING.search(text) or _CONTRAST.search(text))


def _extract_spans(plain: str, limit: int = 3) -> list[SpanMark]:
    """Coverage-guaranteed: pick the main meaning-carrying word(s)/phrases."""

    text = (plain or "").strip()
    if not text:
        return []
    tokens = [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    if not tokens:
        return []

    spans: list[SpanMark] = []
    seen: set[str] = set()

    def add(frag: str, role: str | None = None) -> None:
        frag = " ".join(frag.split()).strip(" .,!?:;\"'")
        nt = _norm(frag)
        if not frag or not nt or nt in seen:
            return
        if any(nt in s or s in nt for s in seen if len(s) > 2 and len(nt) > 2):
            # prefer the longer/more specific phrase
            for s in list(seen):
                if s in nt and len(nt) > len(s):
                    seen.discard(s)
                    spans[:] = [sp for sp in spans if _norm(sp.text) != s]
        seen.add(nt)
        spans.append(SpanMark(text=frag, role=role or _role_for(frag)))

    # 1) named tactic / reveal phrase ("called ...", "this is ...")
    m = _NAMING.search(text)
    if m:
        after = text[m.end() :].strip()
        words = re.findall(r"[A-Za-z0-9']+", after)
        if words:
            add(" ".join(words[:3]), "reveal")
        else:
            add(m.group(0), "reveal")

    # 2) multi-word strong phrases (adjacent important terms), e.g. "demand proof", "gain control", "don't believe"
    norms = [_norm(t) for t, _s, _e in tokens]
    for i in range(len(tokens) - 1):
        a, b = norms[i], norms[i + 1]
        pair = f"{tokens[i][0]} {tokens[i+1][0]}"
        if (_is_strong(tokens[i][0]) and (b not in STOP or b in {"not", "no", "don't", "dont", "never"})) or (
            _is_strong(tokens[i][0]) and _is_strong(tokens[i+1][0])
        ) or (a in {"gain", "take", "get", "feel", "make"} and b not in STOP):
            add(pair, _role_for(pair))
        # "don't/never/not/just + believe/agree/accept..." pattern
        if a in {"don't", "dont", "never", "not", "no", "just"} and b not in STOP:
            add(pair, "danger" if a in {"don't", "dont", "never", "not", "no"} else "key")
        if len(spans) >= limit:
            return spans[:limit]

    # 3) individual strong terms
    for tok, _s, _e in tokens:
        if _is_strong(tok):
            add(tok, _role_for(tok))
        if len(spans) >= limit:
            return spans[:limit]

    # 4) content-word fallback (ensures main words never missed)
    if len(spans) < limit:
        scored = []
        for tok, _s, _e in tokens:
            nt = _norm(tok)
            if not nt or nt in STOP or len(nt) < 3:
                continue
            score = len(nt)
            if _KEY.search(tok):
                score += 5
            if tok[0].isupper() and _s > 0:
                score += 1
            scored.append((score, tok))
        scored.sort(key=lambda x: -x[0])
        for _sc, tok in scored:
            add(tok, "key")
            if len(spans) >= limit:
                break

    # 5) last resort: longest token
    if not spans:
        tok = max((t for t, _s, _e in tokens), key=lambda t: len(_norm(t)), default="")
        if tok:
            add(tok, "key")
    return spans[:limit]


_KEY = re.compile(
    r"\b(truth|real|really|fact|facts|proof|evidence|reason|choice|power|right|wrong|believe|agreement|agree|everyone|always|nothing|control|pressure|trick|trap)\b",
    re.I,
)


def _fallback(lines: list[str], cues: list[Cue]) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        plain = _strip_tags(cue.text)
        spans = _extract_spans(plain, limit=3)
        text, pops = _apply(plain, spans)
        out.append(replace(cue, text=text, pops=pops))
    return out


def _merge_spans(primary: list[SpanMark], extra: list[SpanMark], limit: int = 3) -> list[SpanMark]:
    out: list[SpanMark] = []
    seen: set[str] = set()
    for sp in [*primary, *extra]:
        t = (sp.text or "").strip()
        nt = _norm(t)
        if not t or not nt or nt in seen:
            continue
        if any(nt in s or s in nt for s in seen if len(s) > 2 and len(nt) > 2):
            continue
        seen.add(nt)
        out.append(SpanMark(text=t, role=sp.role or _role_for(t)))
        if len(out) >= limit:
            break
    return out


def emphasize_cues(
    llm: LLMProvider | None,
    lines: list[str],
    cues: list[Cue],
    *,
    beat_context: list[str] | None = None,
) -> list[Cue]:
    if not cues:
        return cues
    baseline = _fallback(lines, cues)  # coverage-guaranteed
    if llm is None:
        return baseline
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
            + "\n\nMark 1-3 main meaning-carrying spans per cue (multi-word ok). JSON."
        )
        plan = llm.complete_json(
            [
                LLMMessage(role="system", content=SYSTEM),
                LLMMessage(role="user", content=user),
            ],
            EmphasisPlan,
            temperature=0.2,
            max_tokens=1800,
        )
        by_i = {m.i: m.spans for m in plan.cues}
        base_spans = {i: _extract_spans(_strip_tags(c.text), limit=3) for i, c in enumerate(cues)}
        out: list[Cue] = []
        for i, c in enumerate(cues):
            merged = _merge_spans(by_i.get(i, []), base_spans.get(i, []), limit=3)
            text, pops = _apply(c.text, merged)
            out.append(replace(c, text=text, pops=pops))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("subtitle emphasis failed (%s); using baseline", exc)
        return baseline
