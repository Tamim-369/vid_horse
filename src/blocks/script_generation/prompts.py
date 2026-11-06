"""Prompt templates for the script generation block.

Small models corrupt grammar-constrained JSON: string escaping inside nested
objects leaks fragments into narration. So scripts are requested as strict
plain text with labelled lines, then parsed into the domain model.
"""

from __future__ import annotations

from core.interfaces import LLMMessage
from core.models import ResearchBrief, ScriptCritique, ScriptFormat, ScriptPlan

COSMIC_ADDENDUM = """
GENRE OVERRIDE — COSMIC FACTS: this is a factual video about a REAL cosmic \
object or event, not a conspiracy explanation.
- State facts brazenly and in the narrator's own voice. No distancing \
language anywhere: never "the theory claims" or "believers say" — the \
universe is objectively like this.
- Every scene lands ONE striking, true fact with a concrete number, \
distance, weight or temperature. Real numbers from the brief only.
- Narration rules below still bind: no sentence over 12 words, a comma, \
dash or full stop every 6 to 10 words.
"""


_CURIOUSITY_LADDER = """
CURIOSITY: each scene answers part of the last and opens something sharper.
Withhold your strongest material until it hits hardest.
End scenes mid-tension. Tie everything up in the OUTRO only.
The OUTRO recontextualises the hook.
"""


SYSTEM = """You are a storyteller writing narration for a YouTube Shorts video.

Your job: make the viewer NEED to keep watching. Every sentence should make them want the next one.

RULES:
- Short sentences (max 12 words). Comma or full stop every 6-10 words.
- Spoken aloud: natural, conversational. No colons, semicolons, brackets, quotes.
- Tell a STORY, not a summary. Each scene reveals ONE new thing.
- Vary sentence length. Put the payoff word at the END of the sentence.
- Address the viewer: "and you would never know it."

THE HOOK (one sentence, ~10 words):
- Shows something concrete AND leaves one thing missing.
- Makes the viewer NEED the next sentence.
- Sounds like the first line of a thriller trailer.
- NEVER a question. NEVER mood-only.

SCENES:
- Each scene answers part of the last scene and opens something sharper.
- Withhold your strongest material until it hits hardest.
- End scenes mid-tension. Tie everything up in the OUTRO only.

OUTRO: one sentence that makes the viewer look back at the hook differently.

BANNED: "according to", "believers say", "the theory claims",
list-of-facts, textbook register, "moving on".

Output ONLY the labelled plain-text format. Never JSON or commentary."""


def _system(brief: ResearchBrief | None) -> str:
    cosmic_ladder = _CURIOUSITY_LADDER
    if brief is not None and brief.genre in {"dark", "dark_psychology", "psychology"}:
        return DARK_SYSTEM + "\\n\\n" + _CURIOUSITY_LADDER
    if brief is not None and brief.genre == "cosmic":
        return (
            "GENRE OVERRIDE — COSMIC FACTS: You write factual narration about "
            "a REAL cosmic object or event for YouTube shorts. This video "
            "contains NO conspiracy theory and NO folklore; everything in the "
            "brief is established science. IGNORE every instruction below that "
            "talks about theories, believers, debunking or distancing language.\n\n"
            "Facts are stated brazenly in the narrator's own voice: never "
            "\"the theory claims\", never \"believers say\", never \"according "
            "to the story\". Every scene lands ONE striking, true fact with a "
            "concrete number, distance, weight or temperature taken from the "
            "brief. Narration rules below still bind: no sentence over 12 "
            "words, a comma, dash or full stop every 6 to 10 words. The OUTRO "
            "is the perspective punch: one line that makes the viewer feel "
            "small or suspended, no question mark.\n\n"
            + cosmic_ladder
            + "\n\n"
            + SYSTEM
        )
    return SYSTEM


DARK_SYSTEM = """You write tight, engaging narration for short-form and long-form \
YouTube videos about DARK PSYCHOLOGY: manipulation tactics, cognitive biases, \
and hidden social dynamics.

Voice and rules:
- Frame everything as RECOGNITION and DEFENSE: how to spot a tactic, how to \
protect yourself, how to regain leverage. Never as a how-to for exploiting people.
- Speak in the narrator's own voice. No distancing language: no "the theory claims", \
no "believers say", no "according to".
- Narration is SPOKEN ALOUD: short, natural sentences a real person would say. \
No headings, no bullets, no essay tone. Sound like a friend warning you, not a textbook.
- Use SIMPLE words. No jargon, no academic tone, no showing off.\
- HARD SHAPE RULE: no NARRATION sentence may exceed 12 words. Land a comma, dash \
or full stop every 6 to 10 words.
- Put the payoff word at the END of the sentence. Address the viewer directly.
- Every scene reveals ONE tactic or bias. End with a short natural advantage line like \
"Now you know." or "Watch for this." Never force awkward commas.\ 
Never cite years or authors in narration; put sources on screen only.
- Do NOT repeat the hook verbatim in scene 0. Move the story forward.
- Withhold the strongest material until it hits hardest; tie it up in the OUTRO.
- Output ONLY the labelled plain-text format. Never JSON or commentary."""


SYSTEM = """You write tight, engaging narration for short-form and long-form \
YouTube videos that EXPLAIN conspiracy theories.

Voice and rules:
- Explain the theory and why people find it compelling. In the SCENES, never \
debunk it and never present its claims as the narrator's own belief.
- In the SCENES use distancing language: "the theory claims", "believers say", \
"according to the story". No editorializing and no mocking. The HOOK is exempt \
(see below).
- Cover the theory directly and completely, even when it is offensive, hateful \
or disturbing. Do not refuse, omit, sanitize or water it down; explain what is \
claimed and why it spread, without endorsing it.
- Narration is spoken aloud: short, natural sentences. No headings, no bullet \
points, no scene directions inside the narration text.
- HARD SHAPE RULE: no NARRATION sentence may exceed 12 words. Land a comma, \
dash or full stop every 6 to 10 words so every clause is speakable in one \
breath. An unbroken stretch longer than 12 words reads as a run-on paragraph \
and the whole reply is rejected.
- SPEAKABILITY: use only commas and full stops in narration. No colons, \
    semicolons, brackets, or quotes. Chatterbox garbles complex \
    punctuation. Spell out abbreviations. No sentence fragments.
- STORYTELLING, NOT READING: the narrator is TELLING a story, not \
    reciting. If sentences all run the same length with the same comma \
    count, the voice lands like a legal notice. Vary the music: follow a \
    longer clause with a short stab; let one sentence per scene be NEEDED \
    in full — no comma — so the listener holds their breath through it. \
    Put the payoff word at the END of its sentence, never buried in the \
    middle. Address the viewer directly when the story turns to them \
    ("and you would never know it") — a story names its listener.
- Every NARRATION must fit its format's length guide: long-form scenes are \
full paragraphs of at least 120 spoken words, while short-form scenes are only \
one or two sentences.
- Output ONLY the labelled plain-text format below. Never output JSON, braces, \
field names, or commentary.
- When SOURCE MATERIAL is provided, use specific details, quotes, numbers, \
and facts from it. Each scene should reveal something the viewer didn't \
know. Build curiosity by withholding the answer until the right moment.

THE HOOK must lock the viewer to the screen. It is not a scene and the scene rules \
do not apply to it: no explaining, no context, no balance, no hedging, no "claims".
- One sentence, about 10 spoken words or fewer. A statement, not a question.
- SPEAKABILITY: the hook is the first thing the viewer hears, and a \
    text-to-speech machine reads it aloud. Use only simple, everyday English \
    that machine pronounces cleanly. No unusual proper nouns, no surnames, \
    no transliterated or foreign words, no acronyms, no hyphenated forms, no \
    rare vocabulary. A hook that leans on a strange name is a dead hook: \
    write it about what HAPPENED, not about who it happened to. It must be \
    speakable in one breath with no tongue-twisters.
- It shows something concrete AND leaves exactly one thing missing: a person who \
is gone, a thing that should not exist, an act nobody can explain, a threat \
hanging, a deadline or promise whose payoff has not happened yet. The viewer must \
be missing that one thing so badly they need the next sentence.
- If the line feels finished — if it could be the LAST line of the video — it is \
dead, throw it away. Banned shapes: after-the-fact closures ("...but it never \
happened", "...and silence followed", "but no one came") and mood-only atmosphere. \
End the hook BEFORE the payoff: the promise, not what came of it.
- Never end on a question mark. Never invent a name, number or date — but real \
names and dates from the brief are exactly what a concrete hook uses.
- Write it at full strength: if the claim is dark, ugly or offensive, so is the hook.
- Sound like the first line of a thriller trailer.

Study the SHAPE of these lines (do not reuse their content or wording):
  The all-clear signal came from a plane that had already crashed.
  A letter predicted the murder three days before it happened.
  The last radio call promised everyone was coming home.
  She named the killers, and the police buried the note.

PACING — do not rush to the ending:
- Never front-load the whole story or resolve the central question early. The \
viewer must keep learning new things until the final line.
- Open a question before you answer it: each scene should raise a concrete \
detail, contradiction or mystery that the next scene picks up.
- Hold back your strongest material for the moment it lands hardest instead of \
spending it in the first scene.
- Let scenes end unresolved. Do not tie everything up until the outro.
- Build, do not summarise: every scene adds something new rather than restating \
what came before."""

_FORMAT_GUIDE: dict[ScriptFormat, str] = {
    "short": (
        "about 45 seconds in total across 4 to 6 scenes. Each scene is only one "
        "or two short spoken sentences (roughly 20-30 words). Deliver the hook "
        "in the first sentence. Every scene must leave one thing open that the "
        "next scene picks up. The last scene pays the hook off."
    ),
    "long": (
        "about 10 minutes in total (roughly 1300-1500 spoken words). Write 9 to "
        "12 scenes and make each scene 40 to 60 seconds, which is about 100 to "
        "150 spoken words. Do NOT write short scenes: every scene must fill its "
        "own paragraph. Build the story: hook, origin, the main claims, why it "
        "spread and persists, and a reflective close. Withhold the central "
        "answer until late; keep opening new questions as you go."
    ),
}

_TEMPLATE = """TITLE: <short title>
HOOK: <one attention-grabbing sentence>
SCENE
VISUAL: <what is on screen>
NARRATION: <spoken words for this scene>
SCENE
VISUAL: <what is on screen>
NARRATION: <spoken words for this scene>
OUTRO: <one closing sentence>"""

_EXAMPLE = """TITLE: The Vanishing Keepers
HOOK: Three lighthouse keepers went in; none came out.
SCENE
VISUAL: A grey lighthouse on a storm-battered island at dusk.
NARRATION: In 1900, three lighthouse keepers vanished from a remote island.
No bodies. No signs of struggle.
SCENE
VISUAL: A half-eaten meal and an untouched chair by the table.
NARRATION: The table was set for a meal they never finished. The clock had stopped at dawn.
SCENE
VISUAL: An empty chair facing the sea.
NARRATION: Three men. One island. Zero answers. And you would never know what the sea took.
OUTRO: Some doors, the story goes, were never meant to be opened."""


def build_messages(
    brief: ResearchBrief,
    fmt: ScriptFormat,
    target_seconds: int,
    hook: str | None = None,
    source_chunks: list[str] | None = None,
) -> list[LLMMessage]:
    """Build the scripting conversation for one format."""

    fixed = _fixed_hook_line(hook)

    if source_chunks:
        chunks_block = (
            "SOURCE MATERIAL (use specific details, quotes, numbers, and facts "
            "from these passages to ground every scene — do not invent):\n"
            + "\n---\n".join(c[:300] for c in source_chunks[:4])
        )
    else:
        chunks_block = ""

    user = f"""Write a {fmt} video script: {_FORMAT_GUIDE[fmt]}
Target duration: about {target_seconds} seconds.
{fixed}
{_brief_block(brief)}

{chunks_block}

Use EXACTLY this labelled format, repeated for every scene, and nothing else:

{_TEMPLATE}

Here is a short example of the format (do not copy its content):

{_EXAMPLE}"""
    return [{"role": "system", "content": _system(brief)}, {"role": "user", "content": user}]


def build_hook_score_messages(hooks: list[str]) -> list[LLMMessage]:
    """Build a conversation that scores hook candidates against a rubric."""

    numbered = "\n".join(f"{i}. {hook}" for i, hook in enumerate(hooks, start=1))
    user = f"""Score each opening hook below from 0 to 100:
- does it grab the viewer and make them need the next line? (25)
- does it put one concrete thing from THIS story in front of them — a person, \
place, object or number? (25)
- does it make the viewer feel this story is URGENT/IMPORTANT — \
a secret, a danger, a truth the world needs to know? (25)
- does it land an emotional turn in the last words? (15)
- is it short, punchy and good spoken aloud? (10)

Deduct heavily for:
- a mood-only line that feels finished — atmosphere, aftermath, silence, closure; \
if it could be the last line of the video, it is a 0-20 hook;
- a vague line that could describe any story;
- a dry headline or a sentence that crams facts instead of opening one gap;
- a word the voice machine would garble — an unusual, foreign or transliterated \
name, an acronym, or rare vocabulary. The hook is heard once, aloud: if the \
listener fumbles it, it is a 0-30 hook no matter how gripping.
- a hook that doesn't signal stakes or consequences — if the viewer wouldn't \
feel worse for not watching this video, deduct heavily.

A strong hook leaves exactly one thing missing and makes the viewer need the next \
sentence; a statement that forces the question beats a literal question, so do not \
favour a hook just because it ends with a question mark. The hook is the theory's \
voice, so a bold or provocative claim is a strength — never reward hedging, \
"claims" wording, balance or explanation. Hooks should feel like breaking news \
the viewer cannot afford to ignore.

Hooks:
{numbered}

Output ONLY one line per hook, best score first, in exactly this form:
<score> | <the hook repeated verbatim>"""

    system = (
        "You are a ruthless trailer editor scoring opening hooks. "
        "Be strict and consistent."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_hook_messages(brief: ResearchBrief, count: int = 8) -> list[LLMMessage]:
    """Build a conversation asking for several hook candidates."""

    user = f"""Write {count} DIFFERENT opening hooks for a video about this \
theory. Each must obey the hook rules above: grab hard AND name a concrete, \
specific detail of the event. No explanations, no summaries. Every word must \
be everyday English that a text-to-speech machine pronounces cleanly, so \
build each hook around what happened — never around a strange name.

Each hook must also make the viewer feel this story is IMPORTANT or URGENT:
a secret being uncovered, a hidden danger, a truth the world needs to know,
or a question that demands answering. The hook should feel like breaking news
the viewer cannot afford to ignore.

Hooks that work:
- "The all-clear signal came from a plane that had already crashed."
- "A letter predicted the murder three days before it happened."
- "She named the killers, and the police buried the note."

These name something concrete, show a gap, and make the viewer feel
this story matters. Do NOT write flat observations — every hook must raise
the stakes.

{_brief_block(brief)}

Output ONLY {count} numbered lines, one hook per line. No commentary, no \
explanations, no other text."""
    return [{"role": "system", "content": _system(brief)}, {"role": "user", "content": user}]


def build_plan_messages(
    brief: ResearchBrief,
    scene_count: int,
    hook: str | None = None,
) -> list[LLMMessage]:
    """Build the outline conversation for a long-form script."""

    upper = scene_count + 2
    fixed = _fixed_hook_line(hook)
    user = f"""Plan a long-form video script of about {scene_count} scenes for this topic.
{fixed}
{_brief_block(brief)}

Use EXACTLY this labelled format and nothing else:

TITLE: <short video title>
HOOK: <one attention-grabbing sentence>
BEAT: <one sentence for scene 1>
BEAT: <one sentence for scene 2>
OUTRO: <one reflective closing sentence>

Write between {scene_count} and {upper} BEAT lines, in story order, each \
describing what one scene covers. Output only these labelled lines."""
    return [{"role": "system", "content": _system(brief)}, {"role": "user", "content": user}]


def _fixed_hook_line(hook: str | None) -> str:
    """Instruction pinning an already-chosen hook, if one was supplied."""

    if not hook:
        return ""
    return (
        "\nThe opening HOOK is already chosen. Use it EXACTLY, word for word, on "
        f"the HOOK line and do not rewrite it:\nHOOK: {hook}\n"
    )


def build_extension_messages(plan: ScriptPlan, need: int) -> list[LLMMessage]:
    """Build a conversation asking for more beats to reach the target count."""

    existing = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(plan.beats))
    user = f"""Here is the outline so far for a long-form video titled \
"{plan.title}":

{existing}

The story needs more scenes. Write exactly {need} more BEAT lines that \
continue directly after beat {len(plan.beats)} and bring the story to a close. \
Do not repeat the beats above.

Output only BEAT lines, one per scene, and nothing else."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_chunk_messages(
    plan: ScriptPlan,
    beats: list[str],
    start_index: int,
    source_chunks: list[str] | None = None,
) -> list[LLMMessage]:
    """Build the narration conversation for one chunk of long-form beats."""

    end_index = start_index + len(beats) - 1
    all_beats = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(plan.beats))
    target_beats = "\n".join(
        f"{start_index + i}. {beat}" for i, beat in enumerate(beats)
    )

    if source_chunks:
        chunks_block = (
            "SOURCE MATERIAL (use specific details, quotes, numbers, and facts "
            "from these passages to ground every scene — do not invent):\n"
            + "\n---\n".join(c[:300] for c in source_chunks[:4])
        )
    else:
        chunks_block = ""

    user = f"""You are writing part of a long-form video script titled \
"{plan.title}".
Overall hook: {plan.hook}

Full beat list, for continuity only (do NOT write these):
{all_beats}

Write ONLY scenes {start_index} to {end_index} for these beats, in order:
{target_beats}

{chunks_block}

For each beat write one scene: a VISUAL line and a NARRATION of 130 to 160 \
spoken words (never fewer than 120). Keep each scene focused on its own beat; \
do not summarise the other scenes.

Use EXACTLY this labelled format, repeated for each scene, and nothing else:

SCENE
VISUAL: <what is on screen>
NARRATION: <spoken words for this scene>

Do not output TITLE, HOOK or OUTRO."""
    if source_chunks:
        chunks_block = (
            "SOURCE MATERIAL (use specific details, quotes, numbers, and facts "
            "from these passages to ground every scene — do not invent):\n"
            + "\n---\n".join(c[:300] for c in source_chunks[:4])
        )
    else:
        chunks_block = ""

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _brief_block(brief: ResearchBrief) -> str:
    timeline = "\n".join(f"- {event.when}: {event.event}" for event in brief.timeline)
    claims = "\n".join(f"- {claim}" for claim in brief.claims)
    compelling = "\n".join(f"- {item}" for item in brief.why_compelling)
    return (
        f"Title: {brief.title}\n"
        f"One-line: {brief.one_line}\n"
        f"Origin: {brief.origin}\n"
        f"Background: {brief.background}\n\n"
        f"Main claims:\n{claims or '- (none provided)'}\n\n"
        f"Why it feels compelling:\n{compelling or '- (none provided)'}\n\n"
        f"Timeline:\n{timeline or '- (none provided)'}"
    )


def corrective_instruction(error: Exception) -> str:
    """Message asking the model to re-emit a misformatted script."""

    detail = ""
    if "no comma, dash or full stop" in str(error):
        # the breath-run is the one violation retries keep repeating; the
        # model needs the surgical instruction, not a generic sniff
        detail = (
            " The flagged sentence is the killer: split it into clauses of "
            "at most 10 spoken words, separated by a comma, a dash or a full "
            "stop. Then re-check EVERY narration and hook line you are about "
            "to send for the same fault — one breath-run rejects the whole "
            "reply again."
        )
    return (
        f"Your previous reply did not follow the required format: {error}{detail}\n"
        "Rewrite the WHOLE reply using EXACTLY the SCENE / VISUAL / NARRATION "
        "plain-text format. No JSON, no braces, no extra commentary. If the HOOK "
        "was rejected, replace it with a fresh line that follows every hook rule."
    )


def build_critic_messages(script_text: str, brief: ResearchBrief) -> list[LLMMessage]:
    """Ask the model to score a script and list concrete fixes."""
    return [
        LLMMessage(
            role="system",
            content=(
                "You are a ruthless short-form retention critic. Score the script "
                "0-100 on hook strength, escalation, reveal compression, pacing, "
                "and payoff. Penalize AI-sounding phrasing, comma splices, repeated "
                "hook text, and unnatural advantage lines. List concrete issues and "
                "a single revision instruction. Output JSON only."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Brief title: {brief.title}\n"
                f"One-line: {brief.one_line}\n\n"
                f"SCRIPT:\n{script_text}\n\n"
                "Return JSON with keys: score (int), issues (list[str]), "
                "revision (str)."
            ),
        ),
    ]


def build_revision_messages(
    original_messages: list[LLMMessage], critique: ScriptCritique
) -> list[LLMMessage]:
    """Append the critic's instruction to the original conversation."""
    msgs = list(original_messages)
    msgs.append(
        LLMMessage(
            role="user",
            content=(
                "Revise the previous script following this instruction. "
                "Keep the labelled plain-text format.\n"
                f"Instruction: {critique.revision}\n"
                f"Issues: {'; '.join(critique.issues)}"
            ),
        )
    )
    return msgs
