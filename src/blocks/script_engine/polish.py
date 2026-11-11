"""Whole-script coherence polish: one LLM pass over all lines.

The per-scene writer optimizes line-by-line and can leave telegraph-style
fragments ("Crowd swears miracles; you demand repeat") and abrupt seams
between lines. This pass rewrites every line ONCE as flowing spoken prose
— same order, same count, roughly the same length — so the narration
sounds like a person talking, not a robot filling slots.

Failure-safe: any validation problem returns the original lines unchanged.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a dialogue editor for a spoken short-form video (ONE narrator "
    "talking to ONE viewer — 'you'). You receive the script's lines in "
    "order. Rewrite EACH line so the whole thing reads and SOUNDS like "
    "natural spoken English by one person: contractions, normal sentence "
    "rhythm, lines that connect to what came before (small bridges like "
    "'And that is exactly why...' or 'Here is the strange part —' are "
    "welcome). Fix telegraph-style fragments into complete sentences. Keep "
    "the plain dark edge of the voice — never soften a line into friendly "
    "self-help coaching, never add hype or exclamation marks.\n"
    "Keep: the meaning of each line, the line order, the line COUNT, and "
    "each line's approximate length (±30%). Do NOT add new facts, names, "
    "numbers, or examples not already present. No headings, no stage "
    "directions, no quotation marks around lines. Output ONLY JSON: "
    '{"lines": ["...", "..."]} with exactly as many lines as you received.'
)


class _PolishedScript(BaseModel):
    lines: list[str]


_WORD_RE = re.compile(r"[A-Za-z']+")


def _is_placeholder(text: str) -> bool:
    stripped = (text or "").strip()
    return (
        len(_WORD_RE.findall(stripped)) < 3
        or re.match(r"^(?:narration|visual|scene|text|query)\b", stripped, re.I) is not None
    )


def _within_tolerance(original: str, polished: str) -> bool:
    """Length sanity: ±45% of the original, and at least 3 real words."""

    lo = max(3, int(len(original.split()) * 0.55))
    hi = max(6, int(len(original.split()) * 1.45) + 2)
    return lo <= len(polished.split()) <= hi


def polish_script_lines(
    llm: LLMProvider,
    lines: list[str],
    *,
    context_note: str = "",
) -> list[str]:
    """Rewrite all lines as natural spoken prose; returns the polished list.

    On any failure the ORIGINAL list is returned unchanged — the polish
    pass is an enhancement, never a risk to script availability.
    """

    if not lines or not any(lines):
        return lines

    context = f"\n(lesson context: {context_note})" if context_note else ""
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=f"Script lines:\n{numbered}{context}\nPolish every line. Output JSON.",
        ),
    ]
    try:
        out = llm.complete_json(msgs, _PolishedScript, temperature=0.4)
    except (LLMError, BlockError, Exception) as exc:  # noqa: BLE001
        logger.warning("polish pass skipped: %s", exc)
        return lines

    got = [ln.strip() for ln in out.lines]
    if len(got) != len(lines) or any(_is_placeholder(ln) for ln in got):
        logger.warning(
            "polish pass rejected (%d lines back, expected %d)", len(got), len(lines)
        )
        return lines

    result: list[str] = []
    for original, polished in zip(lines, got):
        p = polished.strip().strip('"')
        # normalize typography the TTS/validators expect
        p = p.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", "").replace("\u201d", "")
        if not _within_tolerance(original, p):
            logger.info("polish kept original (length drift): %r", p[:60])
            p = original
        result.append(p)
    return result
