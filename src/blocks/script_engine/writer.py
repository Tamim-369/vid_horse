"""Script writing: one grounded LLM call from a lesson brief."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)


class FullScript(BaseModel):
    hook: str = Field(
        description=(
            "Line 1: a short question about a concrete real-life moment that shows this pattern, "
            "asking WHY it happens (ask about the moment, not the pattern's name). 8-14 words."
        )
    )
    scene1: str = Field(description="Line 2: follows the hook with a concrete fact or everyday example (question or statement).")
    scene2: str = Field(description="Line 3: statement with a real-life example or why it happens in plain words.")
    scene3: str = Field(description="Line 4: statement showing what is at stake or the pattern in plain words.")
    scene4: str = Field(description="Line 5: statement with one simple thing to notice or do.")
    outro: str = Field(description="Line 6: closing statement that lands the point.")


FULL_SYSTEM = (
    "You write a short dark-psychology video script for ONE viewer. It reveals the quiet, "
    "hidden truth behind an everyday behavior from a real book — like a friend leaning in "
    "to tell you what is really going on. This is NOT a lesson, lecture, or research paper.\n\n"
    "Voice: calm, confident, quietly dark and intimate. Make the viewer recognize themselves. "
    "Never theatrical, never hype, never scary or preachy.\n"
    "Language: SIMPLEST everyday spoken words only (how you talk to a friend). Short, direct "
    "sentences. Say \"you\", not \"people\" or \"individuals\". Never copy the book's style or "
    "wording. No semicolons, no em-dash chains.\n"
    "Banned tone/phrasing: no academic or textbook talk — never say \"this concept\", "
    "\"this phenomenon\", \"research shows\", \"studies show\", \"it can be understood as\", "
    "\"in other words\", \"this means that\", \"this teaches\", \"the lesson is\". Do not define "
    "terms; show the moment and the hidden pull behind it.\n"
    "Truth: stay true to the TARGET PASSAGE and the brief. Use concrete real-life moments "
    "(work, friends, family, online, crowds). NEVER invent numbers, statistics, studies, or "
    "facts the text does not support.\n\n"
    "Shape (6 lines, one connected thread):\n"
    "1. hook: a short question about a concrete real-life moment that shows this pattern, "
    "asking WHY it happens. Ask about the moment, NOT the pattern's name. (8-14 words)\n"
    "2. scene1: follows the hook — pull the viewer into that moment or state the hidden truth "
    "behind it, plainly (question or statement).\n"
    "3. scene2: statement — show why you do it, in plain words, through a real moment. No question mark.\n"
    "4. scene3: statement — the quiet cost or what it really does to you. No question mark.\n"
    "5. scene4: statement — one simple thing to notice or do next time. No question mark.\n"
    "6. outro: a short closing line that lands the uneasy truth. No question mark.\n\n"
    "Every line must be complete and spoken. Do not repeat the same idea. "
    "Do not mention 'the book', 'the author', or these instructions. Output JSON only."
)


def _join_quotes(quotes: list[str] | None) -> str:
    if not quotes:
        return ""
    cleaned = [q.strip() for q in quotes if q and q.strip()]
    if not cleaned:
        return ""
    return "\n".join(f"- {q}" for q in cleaned[:4])


def write_full_script(
    llm: LLMProvider,
    *,
    topic: str,
    lesson: str,
    hook_moment: str = "",
    why_plain: str = "",
    takeaway: str = "",
    key_quotes: list[str] | None = None,
    target_text: str = "",
    context_text: str = "",
    target_words: int = 22,
    model: str | None = None,
    flexible: bool = False,
) -> FullScript:
    """Write the complete 6-line script in ONE call from a grounded lesson brief."""

    brief_lines = [f"Topic (context only, do not name it in the script): {topic}"]
    if lesson:
        brief_lines.append(f"Hidden truth to reveal: {lesson}")
    if hook_moment:
        brief_lines.append(f"Real-life moment (use for the hook): {hook_moment}")
    if why_plain:
        brief_lines.append(f"Why you do it (plain words): {why_plain}")
    if takeaway:
        brief_lines.append(f"What to notice or do: {takeaway}")
    brief = "\n".join(brief_lines)

    quotes = _join_quotes(key_quotes)
    quotes_block = f"\nKey passages from the target text:\n{quotes}\n" if quotes else ""

    target = (target_text or "").strip()
    target_block = ""
    if target:
        target_block = (
            "\nTARGET PASSAGE (teach this — stay true to its meaning, say it simply):\n"
            f"\"\"\"\n{target[:3500]}\n\"\"\"\n"
        )

    ctx = (context_text or "").strip()
    ctx_block = ""
    if ctx:
        ctx_block = (
            "\nSURROUNDING CONTEXT (for understanding only, do not take the lesson from here):\n"
            f"\"\"\"\n{ctx[:1200]}\n\"\"\"\n"
        )

    if flexible:
        length = (
            f"Length: up to ~{target_words} words per line (use only as many as needed; "
            "shorter is fine). A line may use one or two short sentences if it helps."
        )
    else:
        length = f"Length: about {target_words} words per line."

    msgs = [
        LLMMessage(role="system", content=FULL_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"{brief}\n"
                f"{quotes_block}"
                f"{target_block}"
                f"{ctx_block}\n"
                f"{length}\n"
                "Write the full 6-line script now as JSON (hook, scene1, scene2, scene3, scene4, outro)."
            ),
        ),
    ]
    try:
        per_line = target_words * 4 + 30 if flexible else target_words * 3 + 25
        max_tokens = max(350, 6 * per_line)
        return llm.complete_json(
            msgs, FullScript, temperature=0.5, max_tokens=max_tokens, model=model
        )
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Full script write failed: {exc}") from exc

