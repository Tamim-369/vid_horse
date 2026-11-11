"""Per-scene line writing."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

# Prompt kept focused: clear voice, clear arc, clear contract.
SYSTEM = (
    "You write ONE spoken line of a dark-psychology short video built on a "
    "real book's idea, talking to ONE viewer. Voice: small everyday words, "
    "short sentences, calm, confident, quietly gripping — like a sharp "
    "friend letting the viewer in on a secret about the strings other "
    "people pull on them. Translate bookish words into everyday speech: "
    "'individuals' → 'people', 'do not' → 'don't', 'utilize' → 'use', "
    "'numerous' → 'many', 'conduct' → 'behave'. Never theatrical, never "
    "hype, NEVER fear-mongering or ominous. No semicolons, no em-dash "
    "chains. Ground lines in real, well-known psychology (Milgram's "
    "obedience experiments, Asch's conformity lines, cult love-bombing, "
    "foot-in-the-door sales). NEVER invent statistics or numbers; never "
    "say 'studies show'. Every sentence COMPLETE, second person only.\n"
    "FLOW: the whole script is ONE connected argument. Each new line must "
    "follow naturally from the line before it — continue the thread, add "
    "the next concrete point, and carry the viewer forward. Never jump "
    "topics, never restate.\n"
    "ATTENTION SHAPE: line 1 is a short QUESTION aimed at the viewer's own "
    "life. Line 2 may ask one more question or state a fact. Every later "
    "line is a declarative statement that adds NEW information — a real "
    "example, the mechanism, the stake, or the counter-move.\n"
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
    *,
    plan: list[str] | None = None,
    beat_index: int | None = None,
    model: str | None = None,
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
    arc = ""
    if plan:
        total = len(plan)
        if beat_index is not None and 0 <= beat_index < total:
            outline = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(plan))
            arc = (
                f"FULL SCRIPT PLAN ({total} lines, you are writing line {beat_index + 1}):\n"
                f"{outline}\n"
                "Stay true to this plan and keep the script as one connected argument.\n"
            )
        else:
            outline = "\n".join(f"  - {b}" for b in plan)
            arc = f"SCRIPT PLAN:\n{outline}\n"
    prev_note = "Script so far (continue naturally from the LAST line):" if previous else "Script so far:"
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"{arc}"
                f"Beat for this line: {beat}\n"
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"Counter-move: {fact_sheet.counter_move}\n"
                f"{vocab}"
                f"{prev_note}\n{prev}\n"
                f"{source}"
                f"Write ONE line of about {target_words} words (±2). "
                "Prefer a CONCRETE detail from the source excerpt (a named "
                "example, a practice, a scene you can picture) over "
                "abstract talk. Continue smoothly from the previous line. "
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
        raw = llm.complete(msgs, temperature=0.6, max_tokens=140, model=model)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Scene write failed: {exc}") from exc


def condense_line(
    llm: LLMProvider,
    text: str,
    target_words: int,
    *,
    statement: bool = True,
    previous: str = "",
    model: str | None = None,
) -> str:
    """Compress an overlong line to the target word count.

    Rewriting an existing line to N words is a transformation task small
    models handle far better than free generation, so this rescues lines
    the writer blew past the budget. Returns "" on any failure (callers
    keep the original line then).
    """

    rule = "declarative sentence with no question mark" if statement else "spoken line"
    flow = ""
    if previous:
        flow = (
            f"\nPrevious line (the rewrite must flow naturally after this):\n"
            f"{previous}\n"
        )
    msgs = [
        LLMMessage(
            role="system",
            content=(
                "You compress ONE spoken line without changing its meaning. "
                "Keep the most concrete, compelling detail and the same "
                "connection to the previous line. Small words. "
                "Output ONLY the line — no quotes, no labels, no explanation."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Rewrite this as ONE {rule} of about {target_words} words "
                f"({max(1, target_words - 1)} to {target_words + 2} words):"
                f"{flow}\nLine to compress:\n{text}"
            ),
        ),
    ]
    try:
        out = llm.complete(msgs, temperature=0.2, max_tokens=100, model=model).strip()
    except (LLMError, BlockError):
        return ""
    return out.strip("\"'` ")
