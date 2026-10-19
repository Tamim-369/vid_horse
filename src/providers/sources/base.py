"""Base helpers for discovery source connectors.

A *connector* yields lightweight :class:`SourceTheory` hits: a topic, a URL to
its detailed write-up, and cheap metadata. It never returns article text.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import httpx

from config.settings import Settings
from core.models import SourceTheory


@runtime_checkable
class SourceConnector(Protocol):
    """Interface every discovery source must satisfy."""

    name: str

    def discover(
        self,
        *,
        limit: int | None = None,
        with_metrics: bool = False,
    ) -> Iterator[SourceTheory]:
        """Yield discovered theories, optionally enriching popularity metrics."""
        ...


def build_http_client(settings: Settings) -> httpx.Client:
    """Create an httpx client with the configured user agent."""

    return httpx.Client(
        headers={"User-Agent": settings.http_user_agent},
        timeout=settings.http_timeout,
        follow_redirects=True,
    )


def chunked(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield ``items`` in fixed-size chunks."""

    for start in range(0, len(items), size):
        yield items[start : start + size]
