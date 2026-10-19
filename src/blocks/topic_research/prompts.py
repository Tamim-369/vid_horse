"""Prompt templates for the topic research block."""

from __future__ import annotations

from core.interfaces import LLMMessage
from core.models import ArticleText

SYSTEM = """You are a careful, curious researcher preparing background for a \
video that EXPLAINS a conspiracy theory.

Your job is to understand and describe, never to debunk and never to endorse.
Rules:
- Never state a conspiracy claim as a fact. Attribute it: "the theory holds \
that...", "believers claim...", "according to the story...".
- Explain why the idea exists and why it feels compelling to people, in a \
neutral, non-judgmental tone.
- Prefer concrete names, dates and details drawn from the supplied article.
- If the article does not support something, leave it out. Do not invent facts.
- Write plainly for a general audience. Avoid sensationalism.
- Cover every theory completely, including offensive, hateful or disturbing \
ones. Do not refuse, omit, sanitize or moralize. The task is educational: \
explain what is claimed and why people believe or discuss it. Describe slurs or \
extremist rhetoric factually where relevant, but never use them in your own voice."""


def build_messages(theory_name: str, article: ArticleText) -> list[LLMMessage]:
    """Build the research conversation for ``theory_name``."""

    user = f"""Theory: {theory_name}
Source: {article.url}

Article text:
\"\"\"
{article.text}
\"\"\"

Write a research brief describing this theory for a scriptwriter. Capture what
the theory claims, where it came from, the main beats of its story, and why it
resonates with people.

Use EXACTLY this labelled format, one field per line, and nothing else:

TITLE: {theory_name}
ONE-LINE: <one neutral sentence stating what the theory claims>
ORIGIN: <where and roughly when the theory began>
BACKGROUND: <two to four sentences of context>
CLAIM: <one main thing the theory asserts>
CLAIM: <another claim>
COMPELLING: <why the idea resonates or feels plausible>
PERSON: <a key person associated with the theory>
TERM: <a key term a viewer should know>
TIMELINE: <year> — <what happened>
QUESTION: <an open question the story leaves>

Repeat CLAIM, COMPELLING, PERSON, TERM, TIMELINE and QUESTION as needed. Include
at least three CLAIM lines and two COMPELLING lines. Write only labelled lines —
no headings, no bullet points, no JSON, no commentary."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def corrective_instruction(error: Exception) -> str:
    """A follow-up message asking the model to fix unparseable output."""

    return (
        "That brief could not be read: "
        f"{error}. Reply again using only the labelled lines shown earlier "
        "(TITLE, ONE-LINE, ORIGIN, BACKGROUND, CLAIM, COMPELLING, PERSON, TERM, "
        "TIMELINE, QUESTION). Do not use JSON, braces or bullet points, and "
        "include at least three CLAIM and two COMPELLING lines."
    )
