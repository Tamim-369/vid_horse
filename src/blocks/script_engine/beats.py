"""Beat sheet generation: one function per scene."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import BeatPlan, FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a teacher planning a 6-beat mini-lesson for viewers who need it "
    "explained simply, like teaching a curious 10-year-old. Your tone is dark, "
    "calm and calculative: you teach how influence and manipulation really "
    "work so the viewer can see it coming. Beats: 1 hook (open question about "
    "something the viewer has felt), 2 situation (a concrete moment they "
    "recognise — use generic roles like a classmate, a teacher, a friend; "
    "NEVER invent personal names like Sarah, Liam or Alex), 3 mechanism (why "
    "it works, in plain everyday words), 4 turn (oh-that-is-what-it-is-called), "
    "5 action (one simple thing to say or do), 6 payoff (closes the hook). "
    "Each beat is one idea a child could follow. Output JSON only."
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
