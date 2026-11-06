"""Per-scene line writing."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write ONE short-form scene. Spoken aloud, simple words, one idea. "
    "No headings, no bullets, no quotes. Use second person 'you' only; never mix 'I' or 'my'. "
    "Output only narration and a VISUAL line.\n    VISUAL: 2-4 search words."
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
