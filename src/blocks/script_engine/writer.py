"""Per-scene line writing."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write ONE short-form scene of a mini-lesson. You are a patient, warm "
    "teacher talking to ONE viewer who needs it explained simply, like a "
    "curious 10-year-old. Spoken aloud: simple everyday words, one idea per "
    "scene, short sentences a child could follow. Use a concrete example from "
    "school, family or daily life — use generic roles (a classmate, a teacher, "
    "a friend, someone) and NEVER invent personal names like Sarah, Liam or "
    "Alex. Encouraging, never condescending; no jargon. "
    "TONE: dark, calm and calculative — a quiet strategist who has seen how "
    "people really work. Grim honesty is welcome; menace lives in what you "
    "imply, not in big words. Think a wise elder warning you about a trap, "
    "never a horror narrator. Still simple: every sentence a 10-year-old "
    "understands. "
    "No headings, no bullets, no quotes. Use second person 'you' and 'your' "
    "only; NEVER write 'I', 'me', 'my', 'we', 'us' or 'our'. "
    "Assets are assigned separately by context — the VISUAL is only a hint. "
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
