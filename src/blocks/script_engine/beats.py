"""Beat sheet generation: one function per scene."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import BeatPlan, FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You plan a 6-beat dark-psychology mini-video from a real book's idea, "
    "for ONE viewer. Goal: FOMO — every beat pushes the feeling that the "
    "viewer needs this knowledge NOW or they are missing out while others "
    "use these tricks on them. Voice: dark but plain. Small words, short "
    "sentences, one idea per beat — never complex sentence structures. The "
    "darkness is in the content (what is being revealed about people), not "
    "in theatrical or horror language. Stay faithful to the book's "
    "argument: every beat must be an idea the fact sheet's sources support. "
    "Beats: 1 hook (an OPENING LINE, not a question — a callout that makes "
    "not-knowing feel dangerous; state what the viewer is missing or what "
    "is being done to them), 2 real-world example (a REAL famous "
    "experiment, practice, or event — e.g. Milgram's obedience "
    "experiments, Asch's conformity lines, cult love-bombing, "
    "foot-in-the-door sales; describe it in simple words, NEVER invent "
    "statistics or numbers), 3 mechanism (why it works, plain words), "
    "4 turn (name the pattern plainly, the way the book does), 5 action "
    "(one simple thing to say or do), 6 payoff (closes the hook; the "
    "viewer walks away knowing something others don't). Use generic roles "
    "for characters (a colleague, a bystander, a friend) — never invented "
    "personal names; real public experiment or event names are fine. "
    "Every beat must be a COMPLETE thought/sentence, never cut off. "
    "Write each beat as PLAIN PROSE — no curly-brace labels like "
    "{hook}, no prefixes, no bullet marks. Output JSON only."
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
