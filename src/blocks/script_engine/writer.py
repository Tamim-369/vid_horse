"""Per-scene line writing."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

# Prompt kept focused: clear voice, clear arc, clear contract.
SYSTEM = (
    "You write ONE spoken line of a dark-psychology short video from a "
    "book's idea, talking to ONE viewer. Use the SIMPLEST everyday words "
    "only — the kind you say to a friend. Never copy the book's style or "
    "wording. If a word feels fancy, swap it for a plain one ('people' "
    "not 'individuals', 'don't' not 'do not', 'use' not 'utilize', "
    "'show' not 'demonstrate'). Short sentences. Voice: calm, confident, "
    "quietly dark — like a friend sharing a secret about how people "
    "really act. Never theatrical, never hype, never scary or preachy. "
    "No semicolons, no em-dash chains. Use real-life examples from "
    "everyday life (work, friends, family, online, crowds) and, when "
    "fitting, well-known moments like Milgram's shock test or Asch's "
    "line test — explained in plain words. NEVER invent numbers or "
    "statistics; never say 'studies show'. Every sentence COMPLETE, "
    "second person only.\n"
    "FLOW: the whole script is ONE connected argument. Each new line must "
    "follow naturally from the line before it — continue the thread, add "
    "the next concrete point, and carry the viewer forward. Never jump "
    "topics, never restate.\n"
    "ATTENTION SHAPE: line 1 is the HOOK — a short QUESTION about a "
    "concrete real-life moment that shows this pattern, asking WHY it "
    "happens (example style: 'Why do you laugh when everyone laughs, "
    "even if it's not funny?' — ask about the moment, NOT about the "
    "pattern's name). Line 2 may ask one more question or state a fact. "
    "Every later line is a declarative statement that adds NEW "
    "information — a real-life example, why it happens, what's at stake, "
    "or what to do.\n"
    "OUTPUT CONTRACT: output EXACTLY two lines, nothing else —\n"
    "NARRATION: <the spoken text>\n"
    "VISUAL: <one concept key copied from the provided concept list>\n"
    "No other labels. A reply without a VISUAL line is a failure."
)


def write_scene(
    llm: LLMProvider,
    beat: str,
    fact_sheet: FactSheet,
    previous: list[str],
    target_words: int,
    source_text: str = "",
    concepts: list[str] | None = None,
    *,
    plan: list[str] | None = None,
    beat_index: int | None = None,
    model: str | None = None,
    flexible: bool = False,
) -> str:
    prev = "\n".join(previous) if previous else "(none)"
    source = ""
    if source_text:
        source = (
            "\nSOURCE BOOK EXCERPT (draw the line's idea from this; say "
            "what it says, in your own speaking voice):\n"
            f"'''\n{source_text[:4500]}\n'''\n"
        )
    vocab = ""
    if concepts:
        vocab = "CONCEPT LIST (VISUAL must be exactly one of these keys): " + (
            ", ".join(concepts)
        ) + "\n"
    arc = ""
    if plan:
        total = len(plan)
        if beat_index is not None and 0 <= beat_index < total:
            outline = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(plan))
            arc = (
                f"FULL SCRIPT PLAN ({total} lines, you are writing line {beat_index + 1}):\n"
                f"{outline}\n"
                "Stay true to this plan and keep the script as one connected argument.\n"
            )
        else:
            outline = "\n".join(f"  - {b}" for b in plan)
            arc = f"SCRIPT PLAN:\n{outline}\n"
    prev_note = "Script so far (continue naturally from the LAST line):" if previous else "Script so far:"
    if beat_index == 0:
        line_rule = (
            "HOOK (line 1): ask ONE question about a concrete real-life "
            "moment that shows this pattern and makes the viewer wonder "
            "WHY it happens. Ask about that moment (something you do, see, "
            "or feel in daily life), NOT the pattern's name or the idea in "
            "general. 8-12 words, end with a question mark, nothing after it."
        )
    elif beat_index == 1:
        line_rule = (
            "Line 2 (right after the hook): keep it concrete and connected "
            "to line 1. You may ask one more short question, or give a "
            "clear fact with a real-life example."
        )
    else:
        if flexible:
            line_rule = (
                "This line: ONE or MORE complete sentences (a declarative "
                "statement). Explain this part clearly using real-life "
                "examples in plain words — take the space you need. No "
                "question mark."
            )
        else:
            line_rule = (
                "Middle/late line: ONE declarative statement, no question "
                "mark. Add a real-life example, explain why it happens in "
                "plain words, show what's at stake, or give a simple thing "
                "to do."
            )
    if flexible and beat_index not in (0, 1):
        length_rule = (
            f"Write up to {target_words} words. Use only as many words as "
            "you need to explain this clearly — shorter is fine, do not pad."
        )
    else:
        length_rule = f"Write ONE line of about {target_words} words (±2)."
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"{arc}"
                f"Beat for this line: {beat}\n"
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"Counter-move: {fact_sheet.counter_move}\n"
                f"{vocab}"
                f"{prev_note}\n{prev}\n"
                f"{source}"
                f"{length_rule} "
                "Use the simplest common words. Do NOT copy the book's "
                "style. Prefer concrete real-life examples over abstract "
                "talk. Continue smoothly from the previous line.\n"
                f"{line_rule}\n"
                "Format: NARRATION: <text>\nVISUAL: <one concept key>"
            ),
        ),
    ]
    try:
        # token budget scaled to the line length: a 4B model given a big
        # token budget will happily ramble 130+ words for an 18-word line
        max_tokens = max(60, target_words * 3 + 20)
        raw = llm.complete(msgs, temperature=0.6, max_tokens=max_tokens, model=model)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Scene write failed: {exc}") from exc


# Batch system prompt: the middle lines are written TOGETHER in one call
# so they form one connected passage instead of four orphaned one-liners.
MIDDLE_SYSTEM = (
    "You write the MIDDLE lines of a dark-psychology short video script "
    "from a book's idea, talking to ONE viewer. Use the SIMPLEST everyday "
    "words only — the kind you say to a friend. Never copy the book's "
    "style or wording. If a word feels fancy, swap it for a plain one. "
    "Short sentences. Voice: calm, confident, quietly dark — like a "
    "friend sharing a secret about how people really act. Use real-life "
    "examples from everyday life (work, friends, family, online, crowds). "
    "Never scary or preachy. No semicolons, no em-dash chains. NEVER "
    "invent numbers or statistics; never say 'studies show'. Every "
    "sentence COMPLETE, second person only.\n"
    "FLOW: the lines together are ONE connected argument — they follow a "
    "hook about a real-life moment and build point by point (real "
    "example, why it happens, what's at stake, what to do), then hand "
    "off to a closing line. Each line follows naturally from the one "
    "before. Never repeat an earlier line, never restate it. No question "
    "marks in these lines.\n"
    "OUTPUT CONTRACT: exactly ONE numbered line per beat, nothing else —\n"
    "2. <the spoken text for beat 2>\n"
    "3. <the spoken text for beat 3>\n"
    "(and so on). Numbers only, no labels, no headings, no extra words "
    "before or after."
)


def write_middle(
    llm: LLMProvider,
    beats: list[str],
    fact_sheet: FactSheet,
    hook_line: str,
    payoff_line: str,
    target_words: int,
    source_text: str = "",
    concepts: list[str] | None = None,
    *,
    model: str | None = None,
    flexible: bool = False,
) -> str:
    """Write ALL middle lines in ONE call.

    Writing them together is both cheaper (one round-trip instead of one
    per line — a real cost on CPU-bound local models) and far more
    coherent: the model composes a single passage that flows from the
    hook into the payoff, rather than four isolated one-liners.
    """
    outline = "\n".join(
        f"  {i + 2}. {beat}" for i, beat in enumerate(beats)
    )
    source = ""
    if source_text:
        source = (
            "\nSOURCE BOOK EXCERPT (draw every line's idea from this; say "
            "what it says, in your own speaking voice):\n"
            f"'''\n{source_text[:4500]}\n'''\n"
        )
    if flexible:
        length_rule = (
            f"up to {target_words} words each — use only as many as you need "
            "to explain clearly (shorter is fine). You may use one or two "
            "short sentences per line if it helps explain the idea."
        )
    else:
        length_rule = f"about {target_words} words each (±2)"
    msgs = [
        LLMMessage(role="system", content=MIDDLE_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                "FULL SCRIPT PLAN (hook and closing payoff are already "
                f"written; you write ONLY the numbered middle lines):\n"
                f"  1. {hook_line}   <- hook (already written, do NOT "
                "rewrite or repeat it)\n"
                f"{outline}\n"
                f"  {len(beats) + 2}. {payoff_line}   <- closing payoff "
                "(already written; your last line must hand off "
                "naturally to it)\n"
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"Counter-move: {fact_sheet.counter_move}\n"
                f"{source}"
                f"Write ONLY lines 2 to {len(beats) + 1}, {length_rule}, "
                "in order, connected as one flowing passage. Use the "
                "simplest common words, not the book's style. Use real-life "
                "examples from everyday life with a quiet dark vibe. Start "
                "your reply with '2.'."
            ),
        ),
    ]
    try:
        # enough tokens for N lines of narration
        per_line = target_words * 4 + 40 if flexible else target_words * 3 + 30
        max_tokens = len(beats) * per_line + 60
        raw = llm.complete(msgs, temperature=0.6, max_tokens=max_tokens, model=model)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Batch scene write failed: {exc}") from exc


class FullScript(BaseModel):
    hook: str = Field(description="Line 1: question about a real-life moment asking why it happens (8-14 words).")
    scene1: str = Field(description="Line 2: follows the hook with a concrete fact or example.")
    scene2: str = Field(description="Line 3: statement with a real-life example or why it happens.")
    scene3: str = Field(description="Line 4: statement showing what's at stake or the pattern in plain words.")
    scene4: str = Field(description="Line 5: statement with a simple thing to do or notice.")
    outro: str = Field(description="Line 6: closing statement that lands the point.")


FULL_SYSTEM = (
    "You write a complete dark-psychology short video script from a book's "
    "idea, for ONE viewer. Write the whole script as ONE connected "
    "argument — each line follows naturally from the line before it.\n"
    "Voice: calm, confident, quietly dark — like a friend sharing a "
    "secret about how people really act. Use the SIMPLEST everyday words "
    "only. Never copy the book's style or wording. Use concrete real-life "
    "examples (work, friends, family, online, crowds). Never theatrical, "
    "never hype, never scary. No semicolons, no em-dash chains. NEVER "
    "invent numbers or statistics; never say 'studies show'.\n"
    "Rules:\n"
    "- hook: a short question about a concrete real-life moment that "
    "shows the pattern, asking WHY it happens. Ask about the moment, "
    "NOT the pattern's name.\n"
    "- scene1: follows the hook with a concrete fact or example (may be "
    "a question or a statement).\n"
    "- scene2, scene3, scene4: statements (no question marks) that build "
    "the argument with real-life examples, why it happens in plain words, "
    "what's at stake, or a simple thing to do. Each adds something new.\n"
    "- outro: a strong closing statement (no question mark).\n"
    "Every line must be a complete, clear sentence in plain everyday "
    "words. No vague 'they found'. Stay true to the book's meaning but "
    "say it simply. Output JSON only."
)


def write_full_script(
    llm: LLMProvider,
    fact_sheet: FactSheet,
    beats: list[str],
    target_words: int,
    *,
    source_text: str = "",
    model: str | None = None,
    flexible: bool = False,
) -> FullScript:
    """Write the complete 6-line script in ONE call for maximum coherence."""
    outline = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(beats[:6]))
    source = ""
    if source_text:
        source = (
            "\nSOURCE EXCERPT (stay true to its meaning, but say it in "
            "simple everyday words and real-life examples):\n"
            f"'''\n{source_text[:4500]}\n'''\n"
        )
    if flexible:
        length = (
            f"Each line: up to ~{target_words} words. Use only as many words "
            "as you need to explain clearly — shorter is fine. You may use "
            "one or two short sentences in a line if it helps explain."
        )
    else:
        length = f"Each line: about {target_words} words."
    msgs = [
        LLMMessage(role="system", content=FULL_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Topic: {fact_sheet.concept}\n"
                f"Why it happens: {fact_sheet.mechanism}\n"
                f"What to do: {fact_sheet.counter_move}\n"
                f"Beat plan (for meaning only, do not copy its wording):\n{outline}\n"
                f"{source}"
                f"{length}\n"
                "Write the full script now. Output JSON."
            ),
        ),
    ]
    try:
        per_line = target_words * 4 + 30 if flexible else target_words * 3 + 25
        max_tokens = max(350, 6 * per_line)
        return llm.complete_json(msgs, FullScript, temperature=0.6, max_tokens=max_tokens, model=model)
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Full script write failed: {exc}") from exc


def condense_line(
    llm: LLMProvider,
    text: str,
    target_words: int,
    *,
    statement: bool = True,
    previous: str = "",
    model: str | None = None,
) -> str:
    """Compress an overlong line to the target word count.

    Rewriting an existing line to N words is a transformation task small
    models handle far better than free generation, so this rescues lines
    the writer blew past the budget. Returns "" on any failure (callers
    keep the original line then).
    """

    rule = "declarative sentence with no question mark" if statement else "spoken line"
    flow = ""
    if previous:
        flow = (
            f"\nPrevious line (the rewrite must flow naturally after this):\n"
            f"{previous}\n"
        )
    msgs = [
        LLMMessage(
            role="system",
            content=(
                "You compress ONE spoken line without changing its meaning. "
                "Use the simplest everyday words — never fancy or bookish. "
                "Keep the most concrete real-life detail and the same "
                "connection to the previous line. "
                "Output ONLY the line — no quotes, no labels, no explanation."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Rewrite this as ONE {rule} of about {target_words} words "
                f"({max(1, target_words - 1)} to {target_words + 2} words):"
                f"{flow}\nLine to compress:\n{text}"
            ),
        ),
    ]
    try:
        out = llm.complete(
            msgs, temperature=0.2, max_tokens=max(45, target_words * 3), model=model,
        ).strip()
    except (LLMError, BlockError):
        return ""
    return out.strip("\"'` ")
