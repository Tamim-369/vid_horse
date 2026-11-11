"""Per-scene line writing."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import FactSheet

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write ONE short-form scene of a dark-psychology video built on a "
    "real book's idea. You talk to ONE viewer. Your goal is FOMO: the "
    "viewer must feel they NEED to know this or they are missing out while "
    "other people quietly use these tricks on them. "
    "The dark power is in WHAT you say and HOW you say it — not in fancy "
    "words. HARD RULE: small everyday words, short sentences, one idea per "
    "sentence, no complex sentence structures. No semicolons, no nested "
    "clauses, no em-dash chains. Say the uncomfortable thing calmly, like "
    "someone showing the viewer the strings other people pull on them. "
    "Never theatrical, never a horror narrator, never hype. "
    "REAL-WORLD EXAMPLES: ground scenes in real, well-known psychology — "
    "named experiments, common practices, or events (e.g. Milgram's "
    "obedience experiments, Asch's conformity lines, the Stanford prison "
    "experiment, door-to-door foot-in-the-door sales, cult love-bombing, "
    "hotel-towel social-proof signs). Describe them in simple words. NEVER "
    "invent statistics, percentages, or exact numbers; say any number in "
    "plain words the source supports or drop the number. Never say "
    "'studies show' or 'research shows'. "
    "Every sentence must be COMPLETE — no trailing off, no fragments "
    "(never end on 'when...', 'and you...', 'because...'). Never end a "
    "line on a filler tag like 'watch', 'notice', 'beware', 'stay alert'. "
    "Second person 'you/your' only; NEVER 'I/me/my/we/us/our'. No headings, "
    "no bullets, no quotes. Assets are assigned separately.\n"
    "OUTPUT CONTRACT: output EXACTLY two lines, nothing else —\n"
    "NARRATION: <the spoken text>\n"
    "VISUAL: <one concept key>\n"
    "Never output any other label or field (no 'FOMO:', no 'HOOK:', no "
    "'SCENE:', no 'GOAL:'). The words 'fear of missing out' belong inside "
    "the narration only if they fit it naturally.\n"
    "VISUAL RULE (REQUIRED): the last line is 'VISUAL: ' plus ONE concept "
    "key copied from the provided concept list — the key whose image best "
    "fits this scene. A reply without a VISUAL line is a failure. When it "
    "fits naturally, also use the concept's plain phrase inside the "
    "narration (e.g. 'social proof', 'foot in the door', 'gaslighting', "
    "'missing out')."
)


HOOK_RULES = (
    "\nHOOK MODE — you are writing the VERY FIRST line of the video. The "
    "first three seconds decide whether the viewer stays or swipes. "
    "Craft principles (apply them, never mention them):\n"
    "1. Open a loop the viewer NEEDS closed: name something they believe "
    "or do right now that they can't explain, or a hidden move someone is "
    "running on them at this exact moment.\n"
    "2. Direct callout beats a generic question. 'You are being persuaded "
    "right now.' beats 'Have you ever wondered...'.\n"
    "3. Concrete over abstract: one real experiment, one specific moment, "
    "one scene you can picture. If a named real example fits in a few "
    "words, use it.\n"
    "4. State the stake: what it costs them RIGHT NOW — being used, "
    "fooled, controlled, left behind while others quietly exploit this.\n"
    "5. BANNED openings (auto-fail): 'Have you ever', 'What if I told "
    "you', 'Did you know', 'Everyone knows', 'Let me ask you', 'Ever "
    "wonder', 'Imagine this', 'You won't believe'. No warm-up, no "
    "throat-clearing.\n"
    "6. ONE complete sentence, 8-12 words, small plain words, second "
    "person. No hype, no theater, no fragments. Say it calmly — that is "
    "what makes it land.\n"
)


def write_scene(
    llm: LLMProvider,
    beat: str,
    fact_sheet: FactSheet,
    previous: list[str],
    target_words: int,
    source_text: str = "",
    concepts: list[str] | None = None,
    mode: str = "scene",
    model: str | None = None,
) -> str:
    prev = "\n".join(previous) if previous else "(none)"
    source = ""
    if source_text:
        source = (
            "\nSOURCE BOOK EXCERPT (draw the scene's idea from this; say "
            "what it says, in your own speaking voice):\n"
            f"'''\n{source_text[:1600]}\n'''\n"
        )
    vocab = ""
    if concepts:
        vocab = "CONCEPT LIST (VISUAL must be exactly one of these keys): " + (
            ", ".join(concepts)
        ) + "\n"
    system = SYSTEM + (HOOK_RULES if mode == "hook" else "")
    if mode == "hook":
        ask = (
            f"Beat: {beat}\n"
            f"Concept: {fact_sheet.concept}\n"
            f"Mechanism: {fact_sheet.mechanism}\n"
            f"Payoff the video builds toward: {previous[0] if previous else ''}\n"
            f"{vocab}"
            f"{source}"
            f"Write ONLY the opening hook line in {target_words} words "
            "(±2) following the HOOK MODE principles above. The hook must "
            "promise the payoff without giving it away. "
            "Your last line MUST be a VISUAL line with one concept key "
            "from the list. "
            "Format: NARRATION: <text>\nVISUAL: <one concept key>"
        )
    else:
        ask = (
            f"Beat: {beat}\n"
            f"Concept: {fact_sheet.concept}\n"
            f"Mechanism: {fact_sheet.mechanism}\n"
            f"Counter-move: {fact_sheet.counter_move}\n"
            f"{vocab}"
            f"Previous lines:\n{prev}\n"
            f"{source}"
            f"Write the scene in {target_words} words (±2). "
            "Make the viewer feel they are missing out if they look "
            "away. End with a natural advantage line, no filler tags. "
            "Your last line MUST be a VISUAL line with one concept key "
            "from the list. "
            "Format: NARRATION: <text>\nVISUAL: <one concept key>"
        )
    msgs = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=ask),
    ]
    try:
        raw = llm.complete(msgs, temperature=0.6, max_tokens=200, model=model)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Scene write failed: {exc}") from exc
