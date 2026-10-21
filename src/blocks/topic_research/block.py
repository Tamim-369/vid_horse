"""Topic research: turn a catalog theory into a structured research brief."""

from __future__ import annotations

import logging

from blocks.base import Block, generate_parsed
from blocks.topic_research.parser import parse_brief
from blocks.topic_research.prompts import build_messages, corrective_instruction
from core.errors import BlockError, ProviderError
from core.interfaces import ArticleFetcher, LLMProvider
from core.models import ResearchBrief, Theory, TheorySource

logger = logging.getLogger(__name__)


class TopicResearchBlock(Block):
    """Fetch a theory's detail and distil it into a :class:`ResearchBrief`."""

    name = "topic_research"

    def __init__(
        self,
        llm: LLMProvider,
        fetcher: ArticleFetcher,
        *,
        max_chars: int = 6000,
        temperature: float = 0.4,
        max_attempts: int = 2,
    ) -> None:
        self._llm = llm
        self._fetcher = fetcher
        self._max_chars = max_chars
        self._temperature = temperature
        self._max_attempts = max(1, max_attempts)

    def run(self, theory: Theory, *, max_chars: int | None = None) -> ResearchBrief:
        """Research ``theory`` and return a descriptive brief."""

        source = self._select_source(theory)
        title = source.title if source is not None else theory.name
        url = source.url if source is not None else None
        budget = max_chars or self._max_chars

        try:
            article = self._fetcher.fetch(title, url=url, max_chars=budget)
        except ProviderError as exc:
            raise BlockError(f"Could not fetch detail for '{theory.name}': {exc}") from exc

        logger.info(
            "Researching '%s' (%d chars%s)",
            theory.name,
            len(article.text),
            ", truncated" if article.truncated else "",
        )
        brief = generate_parsed(
            self._llm,
            build_messages(theory.name, article),
            lambda raw: parse_brief(raw, fallback_title=theory.name),
            attempts=self._max_attempts,
            temperature=self._temperature,
            corrective=corrective_instruction,
            label="research brief",
        )
        return brief.model_copy(
            update={
                "title": theory.name,
                "genre": "cosmic" if "space" in theory.categories else "",
            }
        )

    @staticmethod
    def _select_source(theory: Theory) -> TheorySource | None:
        """Prefer a Wikipedia write-up; fall back to the first known source."""

        for source in theory.sources:
            if "wikipedia.org" in source.url:
                return source
        return theory.sources[0] if theory.sources else None
