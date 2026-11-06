"""Fact sheet generation: grounded facts for a topic."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a patient teacher preparing to explain a psychology concept to a "
    "10-year-old-level audience (children and adults who want it simple). Your "
    "tone is dark, calm and calculative: you expose how influence and "
    "manipulation really work, without drama. Extract the lesson of the topic: "
    "what it is, how it works, why people fall for it, and one concrete thing "
    "a listener can say or do about it. Use only facts you can ground in the "
    "source material; no invented statistics or citations. Prefer everyday "
    "examples over academic wording. Output JSON only matching the schema."
)


def generate_fact_sheet(
    llm: LLMProvider, topic: str, *, source_chunks: list[str] | None = None
) -> FactSheet:
    grounding = ""
    if source_chunks:
        grounding = (
            "\nSOURCE MATERIAL (ground every fact in these passages; if a fact "
            "is not supported by them, leave it out — never invent):\n"
            + "\n---\n".join(c[:600] for c in source_chunks[:6])
        )
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user", content=f"Topic: {topic}\n{grounding}\nGenerate the fact sheet."
        ),
    ]
    try:
        fs = llm.complete_json(msgs, FactSheet, temperature=0.3)
        if not fs.source or fs.source.lower() in {"none", ""}:
            fs = fs.model_copy(update={"source": "none"})
        return fs
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Fact sheet failed: {exc}") from exc
