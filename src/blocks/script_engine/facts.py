"""Fact sheet generation: grounded facts for a topic."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 2400

SYSTEM = (
    "You are a sharp dark-psychology teacher preparing a lesson from a REAL "
    "BOOK EXCERPT. Audience: a smart general viewer. Voice: calm, clear-eyed, "
    "a little unsettling because you see the pattern plainly — never "
    "theatrical, never a horror narrator. "
    "Extract ONLY what the excerpt supports: what the author says, how it "
    "works, and one concrete takeaway consistent with the author's argument. "
    "Grounding rules: it is GOOD to closely reflect the book's ideas and "
    "phrasing (the video teaches this book); do NOT transplant unrelated "
    "internet-psychology clichés; do NOT invent statistics, studies, or "
    "citations the excerpt doesn't contain. Output JSON only matching the "
    "schema."
)


def generate_fact_sheet(
    llm: LLMProvider,
    topic: str,
    *,
    source_chunks: list[str] | None = None,
    context_note: str = "",
) -> FactSheet:
    grounding = ""
    if source_chunks:
        grounding = (
            "\nSOURCE MATERIAL (the lesson MUST come from this; facts not "
            "supported by it must be left out — never invent):\n"
            + "\n---\n".join(c[:MAX_CHARS_PER_CHUNK] for c in source_chunks[:6])
        )
    context = ""
    if context_note:
        context = f"\n(book: {context_note})\n"
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=f"Idea/topic: {topic}{context}{grounding}\nGenerate the fact sheet.",
        ),
    ]
    try:
        fs = llm.complete_json(msgs, FactSheet, temperature=0.3)
        if not fs.source or fs.source.lower() in {"none", ""}:
            fs = fs.model_copy(update={"source": "none"})
        return fs
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Fact sheet failed: {exc}") from exc
