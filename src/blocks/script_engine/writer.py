"""Per-scene line writing."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write ONE short-form scene of a mini-lesson teaching a real book's "
    "idea. You are a sharp, calm dark-psychology teacher talking to ONE smart "
    "viewer. Voice: clear-eyed, a little unsettling because you see the "
    "pattern plainly — like someone quietly pointing out the trap before it "
    "snaps. Never theatrical, never a horror narrator. Grim honesty, simple "
    "words. Spoken aloud: short, natural sentences, one idea per scene. "
    "You may draw on the book's ideas and phrasing (this lesson teaches that "
    "book), but rewrite for speech — no dense archaic sentences read aloud "
    "verbatim. Do not drift to internet-psychology clichés not in the "
    "provided facts. "
    "Every sentence must be COMPLETE — no trailing off, no fragments, no "
    "unfinished thoughts (never end on 'when...', 'and you...', 'because...'). "
    "Use concrete everyday examples with generic roles (a colleague, a "
    "bystander, a friend, someone) — NEVER personal names. "
    "No headings, no bullets, no quotes. Second person 'you/your' only; NEVER "
    "'I/me/my/we/us/our'. Assets assigned separately — VISUAL is only a hint. "
    "Output only narration and a VISUAL line.\n"
    "    VISUAL: (hint, 2 words)"
)


def write_scene(
    llm: LLMProvider,
    beat: str,
    fact_sheet: FactSheet,
    previous: list[str],
    target_words: int,
) -> str:
    prev = "\n".join(previous) if previous else "(none)"
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Beat: {beat}\n"
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"Counter-move: {fact_sheet.counter_move}\n"
                f"Previous lines:\n{prev}\n"
                f"Write the scene in {target_words} words (±2). "
                "End with a natural advantage line, no filler tags. "
                "Format: NARRATION: <text>\nVISUAL: <query>"
            ),
        ),
    ]
    try:
        raw = llm.complete(msgs, temperature=0.6, max_tokens=200)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Scene write failed: {exc}") from exc
