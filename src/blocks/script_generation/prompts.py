"""Prompt templates for the script generation block.

Small models corrupt grammar-constrained JSON: string escaping inside nested
objects leaks fragments into narration. So scripts are requested as strict
plain text with labelled lines, then parsed into the domain model.
"""

from __future__ import annotations

from core.interfaces import LLMMessage
from core.models import ResearchBrief, ScriptFormat, ScriptPlan

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
CURIOSITY FIRST, ANSWERS LATER — the viewer must feel they are discovering \
this, not receive a lecture:
- Do not explain everything up front. Each scene answers only part of what \
the last scene left open, and opens something sharper. An answer is always \
also the setup of the next question.
- Scene 2 should make the viewer think they now understand the thing. \
Scene 3 should quietly prove them wrong.
- Before a big fact, let the viewer form a simple guess — "you'd expect...", \
"surely that just..." — then break that guess with the real number.
- Withhold your strongest, largest or strangest number until the moment it \
hits hardest. Naming it early is the difference a taper makes: light held \
back lands; light spent early is wallpaper.
- When scale finally lands, translate it to something a human can feel — \
a road trip, a heartbeat, a full water glass — and never earlier than that.
- End scenes mid-tension where it is honest: the unfinished thing is what \
drags the viewer to the next line. Tie everything up in the OUTRO only.
- The OUTRO recontextualises the hook: the viewer should look back at the \
first sentence and see it differently now.
- Banned: list-of-facts narration ("and it is also...", "next, ..."), \
textbook register ("this is defined as..."), and scene transitions that \
say "moving on" or "another".
"""


def _system(brief: ResearchBrief | None) -> str:
    cosmic_ladder = _CURIOUSITY_LADDER
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
- Every NARRATION must fit its format's length guide: long-form scenes are \
full paragraphs of at least 120 spoken words, while short-form scenes are only \
one or two sentences.
- Output ONLY the labelled plain-text format below. Never output JSON, braces, \
field names, or commentary.

THE HOOK must lock the viewer to the screen. It is not a scene and the scene rules \
do not apply to it: no explaining, no context, no balance, no hedging, no "claims".
- One sentence, about 10 spoken words or fewer. A statement, not a question.
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
        "in the first sentence and end on a memorable line. Every scene must "
        "leave one thing open that the next scene picks up; the last scene "
        "pays the hook off, and not one scene earlier."
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
SCENE
VISUAL: A half-eaten meal and an untouched chair by the table.
NARRATION: The theory claims something came ashore and took them before morning.
OUTRO: Some doors, the story goes, were never meant to be opened."""


def build_messages(
    brief: ResearchBrief,
    fmt: ScriptFormat,
    target_seconds: int,
    hook: str | None = None,
) -> list[LLMMessage]:
    """Build the scripting conversation for one format."""

    fixed = _fixed_hook_line(hook)
    user = f"""Write a {fmt} video script: {_FORMAT_GUIDE[fmt]}
Target duration: about {target_seconds} seconds.
{fixed}
{_brief_block(brief)}

Use EXACTLY this labelled format, repeated for every scene, and nothing else:

{_TEMPLATE}

Here is a short example of the format (do not copy its content):

{_EXAMPLE}"""
    return [{"role": "system", "content": _system(brief)}, {"role": "user", "content": user}]


def build_hook_score_messages(hooks: list[str]) -> list[LLMMessage]:
    """Build a conversation that scores hook candidates against a rubric."""

    numbered = "\n".join(f"{i}. {hook}" for i, hook in enumerate(hooks, start=1))
    user = f"""Score each opening hook below from 0 to 100:
- does it grab the viewer and make them need the next line? (30)
- does it put one concrete thing from THIS story in front of them — a person, \
place, object or number? (25)
- does it land an emotional turn in the last words? (25)
- is it short, punchy and good spoken aloud? (10)
- is it fresh rather than a cliché? (10)

Deduct heavily for:
- a mood-only line that feels finished — atmosphere, aftermath, silence, closure; \
if it could be the last line of the video, it is a 0-20 hook;
- a vague line that could describe any story;
- a dry headline or a sentence that crams facts instead of opening one gap.

A strong hook leaves exactly one thing missing and makes the viewer need the next \
sentence; a statement that forces the question beats a literal question, so do not \
favour a hook just because it ends with a question mark. The hook is the theory's \
voice, so a bold or provocative claim is a strength — never reward hedging, \
"claims" wording, balance or explanation.

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
specific detail of the event. No explanations, no summaries.

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
) -> list[LLMMessage]:
    """Build the narration conversation for one chunk of long-form beats."""

    end_index = start_index + len(beats) - 1
    all_beats = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(plan.beats))
    target_beats = "\n".join(
        f"{start_index + i}. {beat}" for i, beat in enumerate(beats)
    )

    user = f"""You are writing part of a long-form video script titled \
"{plan.title}".
Overall hook: {plan.hook}

Full beat list, for continuity only (do NOT write these):
{all_beats}

Write ONLY scenes {start_index} to {end_index} for these beats, in order:
{target_beats}

For each beat write one scene: a VISUAL line and a NARRATION of 130 to 160 \
spoken words (never fewer than 120). Keep each scene focused on its own beat; \
do not summarise the other scenes.

Use EXACTLY this labelled format, repeated for each scene, and nothing else:

SCENE
VISUAL: <what is on screen>
NARRATION: <spoken words for this scene>

Do not output TITLE, HOOK or OUTRO."""
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
