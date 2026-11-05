"""Topic generation: turn a seed into a structured ResearchBrief via LLM."""

from __future__ import annotations

import logging

from core.errors import BlockError, LLMError
from core.interfaces import LLMProvider
from core.models import ResearchBrief

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a dark-psychology researcher. Given a topic, produce a structured "
    "research brief. Frame everything as recognition and defense. Output JSON only, "
    "matching the provided schema. No markdown, no commentary."
)


class TopicGenerationBlock:
    name = "topic_generation"

    def __init__(self, llm: LLMProvider, *, temperature: float = 0.5) -> None:
        self._llm = llm
        self._temperature = temperature

    def run(self, topic: str) -> ResearchBrief:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Topic: {topic}\nGenerate the brief."},
        ]
        # convert dict messages to LLMMessage? providers expect Sequence[LLMMessage]
        from core.interfaces import LLMMessage

        msgs = [LLMMessage(role=m["role"], content=m["content"]) for m in messages]
        try:
            return self._llm.complete_json(msgs, ResearchBrief, temperature=self._temperature)
        except (LLMError, BlockError) as exc:
            raise BlockError(f"Topic generation failed: {exc}") from exc
