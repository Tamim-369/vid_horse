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
    "You write a short dark-psychology video script for ONE viewer. It puts the viewer inside "
    "a real everyday moment and quietly reveals the hidden pull behind what they do — like a "
    "friend leaning in to tell you what is really going on. This is NOT a lesson, lecture, "
    "essay, or research paper.\n\n"
    "Voice: calm, confident, quietly dark and intimate. The viewer should recognize themselves. "
    "Never theatrical, never hype, never scary or preachy.\n"
    "Language: SIMPLEST everyday spoken words only (how you talk to a friend). Short, direct "
    "sentences. Say \"you/your\", not \"people\" or \"individuals\". Never copy the book's style "
    "or wording. No semicolons, no em-dash chains.\n"
    "Spoken clarity: use punctuation lightly. Do not rely on commas or pauses to carry meaning, "
    "because a pause can make a line sound like it means the opposite. Make meaning unmistakable "
    "in the words: use clear connectors like \"that\" or \"so\", and if a thought could be heard "
    "two ways, rewrite it or split it into two short, clear sentences.\n"
    "Every sentence must end on a complete thought. Never let a sentence trail off into a "
    "pronoun or fragment that has nothing to answer to (a line like \"before anyone truly "
    "understands them yourself.\" is broken — cut the dangling tail or finish the thought).\n"
    "Never join two complete thoughts with a comma, a semicolon, or a dash (a comma splice). "
    "When a thought is complete, end it with a period and start the next sentence. "
    "\"...or spread, instinct takes over...\" is a splice — write \"...or spread. Instinct "
    "takes over...\" instead.\n"
    "Do NOT explain or summarize like a paper. Banned phrasing: \"this concept\", "
    "\"this phenomenon\", \"research shows\", \"studies show\", \"it can be understood as\", "
    "\"in other words\", \"this means that\", \"this teaches\", \"the lesson is\", "
    "\"people often\", \"tends to\". Do not define terms or draw abstract conclusions — "
    "stay inside the moment and show the hidden pull and its quiet cost.\n"
    "Truth: stay true to the TARGET PASSAGE and the brief. Use concrete real-life moments "
    "(work, friends, family, online, crowds). NEVER invent numbers, statistics, studies, or "
    "facts the text does not support.\n\n"
    "Shape (6 lines, one flowing thread that stays with the same moment):\n"
    "1. hook: put \"you\" in a concrete real-life moment and ask WHY you do it. Ask about the "
    "moment, NOT the pattern's name. (8-14 words, a question)\n"
    "2. scene1: stay in that moment — show what you do or feel right there (question or statement).\n"
    "3. scene2: statement — the hidden reason pulling you, shown in plain words. No question mark.\n"
    "4. scene3: statement — the quiet cost, or how it steers you without you noticing. No question mark.\n"
    "5. scene4: statement — one small thing to notice next time you are in it. No question mark.\n"
    "6. outro: a short line that lands the uneasy truth. No question mark.\n\n"
    "Every line must be complete and spoken, and move the same thread forward (no generic summaries). "
    "Write clean, readable plain text only: no control characters, no garbled text, no repeated fragments.\n"
    "Do not mention 'the book', 'the author', or these instructions. Output JSON only."
)

# Shape-of-the-day rotation: every video follows the same quality rules
# (why-question hook, declarative lines, "you" voice) but progresses
# differently, so the channel never reads as one template. The shape is
# chosen deterministically per video and handed to the writer as the
# scene-by-scene plan for lines 2-6 (line 1 is always the same hook rule).
_SHAPES: dict[str, str] = {
    "mechanism": (
        "SHAPE OF THIS VIDEO — mechanism: hook (the moment + why-question). "
        "scene1: stay in the moment, show what you do or feel right there. "
        "scene2: the hidden reason pulling you, in plain words. "
        "scene3: what that reason quietly costs you or where it steers you. "
        "scene4: one small thing to notice next time. "
        "outro: land the uneasy truth in one short line."
    ),
    "mirror": (
        "SHAPE OF THIS VIDEO — mirror: hook (the moment + why-question). "
        "scene1: YOUR side — what you do or feel in that moment. "
        "scene2: the OTHER side — what is really happening around you that you barely register. "
        "scene3: the connecting truth — how the two sides feed each other without anyone deciding to. "
        "scene4: one small thing to notice next time. "
        "outro: land the uneasy truth in one short line."
    ),
    "funnel": (
        "SHAPE OF THIS VIDEO — funnel: hook (a tiny everyday habit of yours + why-question about it). "
        "scene1: the habit itself, smaller and more specific than anyone would admit. "
        "scene2: why the brain likes it — what it quietly gives you. "
        "scene3: what it builds into when it repeats, without you noticing. "
        "scene4: one small thing to notice next time. "
        "outro: land the uneasy truth in one short line."
    ),
    "earn-it": (
        "SHAPE OF THIS VIDEO — earn-it: hook (the moment + why-question). "
        "scene1: open on the counterintuitive point as a plain statement — something that sounds wrong at first. "
        "scene2: the everyday evidence that it is true, from the same moment. "
        "scene3: why it is true — the plain mechanism under it. "
        "scene4: one small thing to notice next time. "
        "outro: land the uneasy truth in one short line."
    ),
}

_SHAPE_ORDER = list(_SHAPES)


def pick_shape(seed: str) -> tuple[str, str]:
    """Deterministic shape (name, instruction) for one video."""

    import hashlib

    digest = int.from_bytes(hashlib.sha256(str(seed).encode("utf-8")).digest()[:4], "big")
    name = _SHAPE_ORDER[digest % len(_SHAPE_ORDER)]
    return name, _SHAPES[name]


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
    shape: str = "",
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
                + (f"{shape}\n" if shape else "")
                + "Write the full 6-line script now as JSON (hook, scene1, scene2, scene3, scene4, outro)."
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

