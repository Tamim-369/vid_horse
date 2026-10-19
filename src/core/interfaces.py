"""Structural interfaces shared across layers.

Blocks and pipelines depend on these protocols, never on concrete providers.
That inversion keeps the LLM/search backends swappable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict, TypeVar, runtime_checkable

from pydantic import BaseModel

from core.models import ArticleText

TModel = TypeVar("TModel", bound=BaseModel)


class LLMMessage(TypedDict):
    """A single chat message."""

    role: str
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface every language-model backend must satisfy."""

    name: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        """Return the assistant's text for ``messages``."""
        ...

    def complete_json(
        self,
        messages: Sequence[LLMMessage],
        schema: type[TModel],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> TModel:
        """Return a validated ``schema`` instance parsed from the response."""
        ...


class SearchResult(BaseModel):
    """A single web search hit."""

    title: str
    url: str
    snippet: str = ""


@runtime_checkable
class SearchProvider(Protocol):
    """Interface for optional web-search grounding."""

    name: str

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Return up to ``max_results`` results for ``query``."""
        ...


@runtime_checkable
class ArticleFetcher(Protocol):
    """Interface for fetching the plain-text detail of a theory."""

    name: str

    def fetch(
        self,
        title: str,
        *,
        url: str | None = None,
        max_chars: int | None = None,
    ) -> ArticleText:
        """Return the article text for ``title``."""
        ...
