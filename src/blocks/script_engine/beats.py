"""Beat sheet generation: one function per scene."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import BeatPlan, FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You plan a 6-beat dark-psychology mini-video from a book's idea, "
    "for ONE viewer. Use the SIMPLEST everyday words — never copy the "
    "book's style or wording. Voice: calm, confident, quietly dark — a "
    "real-life truth about how people act, said like a friend sharing a "
    "secret. Short sentences, one idea per beat. No hype, never scary or "
    "preachy. Stay true to the book's meaning, but say it in plain, "
    "common words with real-life examples. "
    "Beats: 1 hook — an OPENING QUESTION about a concrete real-life "
    "moment that shows this pattern, asking WHY it happens (a scene from "
    "work, friends, family, or a crowd the viewer knows). Ask about that "
    "moment, NOT about the pattern's name. "
    "2 real-life example — a moment from everyday life, or a well-known "
    "event like Milgram's shock test or Asch's line test, explained in "
    "simple words (NEVER invent numbers). "
    "3 mechanism — why it happens, in plain words. "
    "4 the pattern — name what is going on in simple words. "
    "5 action — one simple thing to say or do. "
    "6 payoff — closes the hook; the viewer walks away seeing this in "
    "their own life. Use everyday roles (a friend, a boss, a stranger) — "
    "never invented personal names. Every beat must be a COMPLETE "
    "thought, never cut off. Write each beat as PLAIN PROSE — no "
    "curly-brace labels, no prefixes, no bullet marks. Output JSON only."
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
