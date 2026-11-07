"""Beat sheet generation: one function per scene."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import BeatPlan, FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a sharp dark-psychology teacher planning a 6-beat mini-lesson, "
    "explained simply (10-year-old clear). Your voice: calm, clear-eyed, a "
    "little unsettling because you see the pattern plainly — never theatrical, "
    "never a horror narrator. NEVER copy or closely paraphrase the source/book "
    "and do NOT mimic its tone; bring your own original teacher voice. "
    "Beats: 1 hook (a complete, gripping question about something felt), "
    "2 situation (a concrete moment — generic roles like a classmate, a "
    "teacher, a friend; NEVER personal names like Sarah/Liam/Alex), 3 mechanism "
    "(why it works, plain words), 4 turn (name what it is, plainly), 5 action "
    "(one simple thing to say or do), 6 payoff (closes the hook). "
    "Every beat must be a COMPLETE thought/sentence, never cut off. "
    "One idea per beat. Output JSON only."
)


def generate_beats(llm: LLMProvider, fact_sheet: FactSheet) -> list[str]:
    facts = "\n".join(f"- {f}" for f in fact_sheet.facts)
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"Source: {fact_sheet.source}\n"
                f"Counter-move: {fact_sheet.counter_move}\n"
                f"Facts:\n{facts}\n"
                "Generate 6 beats."
            ),
        ),
    ]
    try:
        plan = llm.complete_json(msgs, BeatPlan, temperature=0.4)
        return list(plan.beats)
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Beat plan failed: {exc}") from exc
