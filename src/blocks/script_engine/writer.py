"""Per-scene line writing."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

# Kept deliberately SHORT: a 4B writer drowns in long prompt essays and
# then ignores half of it. Hard rules that matter live in the validators
# (word count, question policy, repeats) — those are enforced mechanically
# after generation, not asked for politely here.
SYSTEM = (
    "You write ONE spoken line of a dark-psychology short video built on a "
    "real book's idea, talking to ONE viewer. Voice: small everyday words, "
    "short sentences, calm and unnerving — like showing the viewer the "
    "strings other people pull on them. Never theatrical, never hype, no "
    "semicolons, no em-dash chains. Ground lines in real, well-known "
    "psychology (Milgram's obedience experiments, Asch's conformity lines, "
    "cult love-bombing, foot-in-the-door sales). NEVER invent statistics or "
    "numbers; never say 'studies show'. Every sentence COMPLETE, second "
    "person only. No headings, bullets or quotes.\n"
    "ATTENTION SHAPE: the video's first line is a short QUESTION aimed at "
    "the viewer's own life (their relationships, blind spots, or an "
    "unfairness nobody talks about). The line right after it may ask ONE "
    "more question. Every later line is a declarative statement that adds "
    "NEW information — a real example, the mechanism, the stake, or the "
    "counter-move. Never repeat an earlier line.\n"
    "OUTPUT CONTRACT: output EXACTLY two lines, nothing else —\n"
    "NARRATION: <the spoken text>\n"
    "VISUAL: <one concept key copied from the provided concept list>\n"
    "No other labels. A reply without a VISUAL line is a failure."
)


def write_scene(
    llm: LLMProvider,
    beat: str,
    fact_sheet: FactSheet,
    previous: list[str],
    target_words: int,
    source_text: str = "",
    concepts: list[str] | None = None,
) -> str:
    prev = "\n".join(previous) if previous else "(none)"
    source = ""
    if source_text:
        source = (
            "\nSOURCE BOOK EXCERPT (draw the line's idea from this; say "
            "what it says, in your own speaking voice):\n"
            f"'''\n{source_text[:2600]}\n'''\n"
        )
    vocab = ""
    if concepts:
        vocab = "CONCEPT LIST (VISUAL must be exactly one of these keys): " + (
            ", ".join(concepts)
        ) + "\n"
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Beat: {beat}\n"
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"{vocab}"
                f"Previous lines (never repeat these):\n{prev}\n"
                f"{source}"
                f"Write ONE line of about {target_words} words (±2). "
                "Prefer a CONCRETE detail from the source excerpt (a named "
                "example, a practice, a scene you can picture) over "
                "abstract talk. "
                "If this is the video's FIRST line: ONE question, 8-12 "
                "words, ending with a question mark, nothing after it, no "
                "ellipses. If it directly follows the hook: you may ask ONE "
                "more question, or state a fact. Every later line: ONE "
                "declarative statement, no question mark. "
                "Format: NARRATION: <text>\nVISUAL: <one concept key>"
            ),
        ),
    ]
    try:
        raw = llm.complete(msgs, temperature=0.6, max_tokens=120)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Scene write failed: {exc}") from exc


def condense_line(
    llm: LLMProvider,
    text: str,
    target_words: int,
    *,
    statement: bool = True,
) -> str:
    """Compress an overlong line to the target word count.

    Rewriting an existing line to N words is a transformation task small
    models handle far better than free generation, so this rescues lines
    the writer blew past the budget. Returns "" on any failure (callers
    keep the original line then).
    """

    rule = "declarative sentence with no question mark" if statement else "spoken line"
    msgs = [
        LLMMessage(
            role="system",
            content=(
                "You compress ONE spoken line without changing its meaning. "
                "Keep the most concrete, unsettling detail. Small words. "
                "Output ONLY the line — no quotes, no labels, no explanation."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Rewrite this as ONE {rule} of about {target_words} words "
                f"({max(1, target_words - 1)} to {target_words + 2} words):\n{text}"
            ),
        ),
    ]
    try:
        out = llm.complete(msgs, temperature=0.3, max_tokens=100).strip()
    except (LLMError, BlockError):
        return ""
    return out.strip("\"'` ")
