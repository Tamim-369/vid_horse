"""Write original horror stories as video scripts.

A story is drafted from a mined style pack
(see :mod:`blocks.horror_style`) — never copied from the corpus — and shaped
as a :class:`VideoScript` so the whole voice -> assets -> render pipeline
works untouched. The hook is the opener, scenes are escalation beats, and
the outro is the twist landing.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from core.interfaces import LLMMessage, LLMProvider
from core.models import VideoScript

logger = logging.getLogger(__name__)


_SCHEMA_NOTE = (
    "hook: the OPENING line, <= 15 spoken words, instant dread, no context "
    "needed. scenes: narration / visual / seconds story beats that escalate "
    "one step each; narration 1-2 short spoken sentences, present tense, "
    "written the way a person tells a story out loud; visual = what is on "
    "screen, filmable and eerie (a real place, a real object, no text in "
    "frame). outro: the TWIST line, <= 12 spoken words, lands cold, does not "
    "explain itself."
)

_STORY_SYSTEM = """You are a horror short-form writer for narration videos. \
You use proven patterns (premises, hook shapes, twist moves) ONLY as \
inspiration — every story you write must be new: new entity, new setting, \
new twist combo. Never retell a known story, creepypasta or urban legend.

Story rules:
- One continuous story, told in escalating beats. Fear builds per sentence.
- Concrete and specific: real-feeling places, times, objects. Specifics \
are what make it scary.
- One unexplained detail left hanging — the unanswered thing is scarier \
than an explanation.
- The twist recontextualises an earlier beat; plant it quietly mid-story.
- No gore-porn, no monster reveal on camera. What is NOT shown is scarier.
- Spoken-word style: short clauses, natural rhythm, no semicolons, no \
parentheses, no emoji, no hashtags.
"""

_STORY_USER_TEMPLATE = """Write an original horror story as a \
{seconds}-second narration video with {scenes} scenes.

USE THESE PATTERNS AS INSPIRATION (do not copy any specific story):
premises: {premises}
hook shapes: {hooks}
twist moves: {twists}
techniques: {techniques}
{theme_line}
""" + _SCHEMA_NOTE


class StoryDraft(BaseModel):
    """LLM-facing story schema, converted to a :class:`VideoScript` after."""

    title: str = Field(description="Short evocative title, 2-5 words.")
    hook: str
    scenes: list[StorySceneDraft] = Field(min_length=4, max_length=9)
    outro: str = Field(description="The twist line.")


class StorySceneDraft(BaseModel):
    narration: str
    visual: str
    seconds: int = Field(ge=4, le=10)


class HorrorStoryBlock:
    """Write an original horror story from a mined style pack."""

    name = "horror_story"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        temperature: float = 0.9,
        scenes: int = 6,
        target_seconds: int = 45,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._scene_count = scenes
        self._target_seconds = target_seconds

    def run(self, style_pack, *, theme: str | None = None) -> VideoScript:
        """Write one story; ``theme`` steers the genre flavour if given."""

        messages: list[LLMMessage] = [
            {"role": "system", "content": _STORY_SYSTEM},
            {
                "role": "user",
                "content": _STORY_USER_TEMPLATE.format(
                    seconds=self._target_seconds,
                    scenes=self._scene_count,
                    premises=_bullets(style_pack.premises),
                    hooks=_bullets(style_pack.hook_patterns),
                    twists=_bullets(style_pack.twists),
                    techniques=_bullets(style_pack.techniques),
                    theme_line=f"\nTHEME (must fit it): {theme}"
                    if theme
                    else "",
                ),
            },
        ]
        draft = self._llm.complete_json(
            messages,
            StoryDraft,
            temperature=self._temperature,
        )
        return self._to_script(draft)

    def _to_script(self, draft: StoryDraft) -> VideoScript:
        from core.models import ScriptScene

        scenes = [
            ScriptScene(
                narration=scene.narration,
                visual=scene.visual,
                seconds=max(4, min(10, scene.seconds)),
            )
            for scene in draft.scenes
        ]
        return VideoScript(
            title=draft.title,
            hook=draft.hook,
            format="short",
            target_seconds=self._target_seconds,
            scenes=scenes,
            outro=draft.outro,
            source_notes=["original horror story; patterns mined from CC BY-SA corpus"],
        )


def _bullets(items) -> str:
    return "\n".join(f"- {item}" for item in items[:10])
