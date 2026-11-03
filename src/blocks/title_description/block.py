"""Title and description generator for YouTube Shorts."""

from __future__ import annotations

from blocks.base import Block
from core.errors import BlockError, LLMError
from core.interfaces import LLMProvider
from core.models import VideoScript

from .prompts import TITLE_SYSTEM
from .schema import TitlePack


class TitleDescriptionBlock(Block):
    name = "title_description"

    def __init__(self, llm: LLMProvider, *, race: int = 0) -> None:
        self._llm = llm
        self._race = race

    def run(self, script: VideoScript, topic: str = "") -> TitlePack:

        messages = [
            {"role": "system", "content": TITLE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic or script.title}\n"
                    f"Hook: {script.hook}\n"
                    f"Scenes:\n"
                    + "\n".join(f"- {s.narration}" for s in script.scenes)
                    + f"\nOutro: {script.outro}\n"
                    f"Format: {script.format}\n"
                    "Produce the title, description, tags, hashtags and pinned comment."
                ),
            },
        ]
        try:
            return self._llm.complete_json(messages, TitlePack)
        except LLMError as exc:
            raise BlockError(f"Title generation failed: {exc}") from exc
