"""Fact sheet generation: grounded facts for a topic."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a sharp dark-psychology teacher preparing a lesson for a "
    "10-year-old-level audience (simple, clear). Your voice: calm, clear-eyed, "
    "and a little unsettling because you see the pattern plainly — never "
    "theatrical, never a horror narrator. Extract the lesson: what it is, how "
    "it works, why people fall for it, and one concrete thing to say or do. "
    "Ground facts in the source ideas, but NEVER copy, quote, or closely "
    "paraphrase the source/book. Do NOT mimic the book's tone or phrasing — "
    "use your own original words and teacher voice. No invented statistics or "
    "citations. Output JSON only matching the schema."
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
