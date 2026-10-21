"""Mine a horror style pack from the creepypasta study corpus.

The corpus (CC BY-SA, see :mod:`pipeline.horror_corpus`) is *input* to the
writer agents only. This block samples openings and endings, then asks the
LLM to extract reusable patterns — premises, hook shapes, twist types and
pacing techniques. Nothing copied reaches a script.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.errors import BlockError
from core.interfaces import LLMMessage, LLMProvider
from core.models import HorrorStylePack

logger = logging.getLogger(__name__)

_SAMPLE_STORIES = 12
_OPENING_CHARS = 1200
_ENDING_CHARS = 900

_SYSTEM = """You are a horror-writing analyst. You read a sample of amateur \
horror stories and extract what makes them work as PATTERNS, not content.

Extraction rules:
- premises: recurring story situations ("an object that shouldn't exist", \
"a nightly sound that keeps moving closer"). One line each.
- hook_patterns: shapes of opening lines that hook instantly — describe the \
shape, optionally with a skeleton, e.g. "timeline + normal life: 'I've \
lived here for N years, and last night that changed'". Do NOT quote lines.
- twists: ending moves that recontextualise everything ("the narrator IS \
the threat", "the listener is already involved"). One line each.
- techniques: pacing and voice moves (present tense, escalation per \
sentence, the unanswered detail left unexplained, etc.). One line each.

Be concrete and usable as writing instructions."""

_USER_TEMPLATE = """Here are openings and endings from {count} horror stories \
(opening excerpt, story ending excerpt).

STORIES:
{stories}

Extract the patterns into the JSON schema. 6-10 items per field, concrete \
enough that a writer who has never read these stories could use them."""


def _story_fields(text: str) -> str:
    opening = text[:_OPENING_CHARS].strip()
    ending = text[-_ENDING_CHARS:].strip()
    return f"--- BEGINNING ---\n{opening}\n--- ENDING ---\n{ending}"


class HorrorStyleBlock:
    """Extract reusable horror-writing patterns from a story corpus."""

    name = "horror_style"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def run(
        self,
        corpus_path: Path,
        *,
        sample: int = _SAMPLE_STORIES,
        seed: int | None = None,
    ) -> HorrorStylePack:
        """Mine ``corpus_path`` (JSONL) into a style pack."""

        stories: list[dict[str, Any]] = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                stories.append(json.loads(line))
        if len(stories) < 3:
            raise BlockError(
                f"Corpus too small: {len(stories)} stories in {corpus_path}."
            )
        rng = random.Random(seed)
        chosen = rng.sample(stories, min(sample, len(stories)))
        digest = "\n\n".join(
            f"[{story['title']}]\n{_story_fields(story['text'])}" for story in chosen
        )
        messages: Sequence[LLMMessage] = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(count=len(chosen), stories=digest),
            },
        ]
        pack = self._llm.complete_json(messages, HorrorStylePack)
        logger.info(
            "Style pack: %d premises, %d hooks, %d twists, %d techniques",
            len(pack.premises),
            len(pack.hook_patterns),
            len(pack.twists),
            len(pack.techniques),
        )
        return pack
