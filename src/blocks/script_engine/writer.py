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
    "Never theatrical, never a horror narrator, never hype.\n"
    "ATTENTION CRAFT (apply it in every line, never mention it):\n"
    "1. ATTENTION IS THE PRODUCT. The first three seconds decide whether "
    "the viewer stays; every later line must earn the next one.\n"
    "2. OPEN WITH A QUESTION AIMED AT THE VIEWER'S OWN LIFE. Not a polite "
    "or generic question — one that implies something is already being "
    "done to them that they haven't noticed. The patterns (make your own, "
    "never copy these):\n"
    "- ask WHY something in their life happens the way it does\n"
    "- point at someone close — partner, boss, friend, family — who may "
    "be quietly manipulating or using them\n"
    "- promise SIGNS they can start checking the moment they hear them\n"
    "- ask about the unfair thing nobody is allowed to talk about\n"
    "The question must be short, second person, and land in their "
    "relationships, their blind spots, or their sense of injustice.\n"
    "3. Concrete over abstract: one real experiment, one specific moment, "
    "one scene the viewer can picture. If a named real example fits in a "
    "few words, use it.\n"
    "4. State the stake: what it costs the viewer RIGHT NOW — being used, "
    "fooled, controlled, left behind while others quietly exploit this.\n"
    "5. Keep tension between lines: no recapping what was just said, no "
    "'as I mentioned', no softening, no hype or clickbait energy. Calm "
    "delivery is what makes it land — never louder.\n"
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


def write_scene(
    llm: LLMProvider,
    beat: str,
    fact_sheet: FactSheet,
    previous: list[str],
    target_words: int,
    source_text: str = "",
    concepts: list[str] | None = None,
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
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Beat: {beat}\n"
                f"Concept: {fact_sheet.concept}\n"
                f"Mechanism: {fact_sheet.mechanism}\n"
                f"Counter-move: {fact_sheet.counter_move}\n"
                f"{vocab}"
                f"Previous lines:\n{prev}\n"
                f"{source}"
                f"Write the scene in {target_words} words (±2). "
                "If this is the video's FIRST line, it is the hook: ONE "
                "opening QUESTION aimed at the viewer's own life, 8-12 "
                "words, ending on a question mark. NOTHING after the "
                "question mark — no answer, no extra sentence. No "
                "ellipses (...), no asterisks, no markdown, no emphasis "
                "marks. It must promise the payoff without giving it away. "
                "Make the viewer feel they are missing out if they look "
                "away. End with a natural "
                "advantage line, no filler tags. "
                "Your last line MUST be a VISUAL line with one concept key "
                "from the list. "
                "Format: NARRATION: <text>\nVISUAL: <one concept key>"
            ),
        ),
    ]
    try:
        raw = llm.complete(msgs, temperature=0.6, max_tokens=200)
        return raw.strip()
    except (LLMError, BlockError) as exc:
        raise BlockError(f"Scene write failed: {exc}") from exc
