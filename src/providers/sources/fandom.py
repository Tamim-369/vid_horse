"""Fandom (MediaWiki) discovery connector.

Pulls from MediaWiki-based conspiracy wikis, which document theories that never
clear Wikipedia's notability bar.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import quote

import httpx

from core.models import SourceTheory, TheoryMetrics
from providers.sources.mediawiki import MediaWikiClient


class FandomConnector:
    """Discover theories from a Fandom wiki."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        wiki: str,
        categories: tuple[str, ...] = ("Conspiracy theories",),
        seed_pages: tuple[str, ...] = ("List of Conspiracy Theories",),
        delay: float = 0.2,
        max_depth: int = 1,
    ) -> None:
        self.name = f"fandom:{wiki}"
        self._wiki = wiki
        self._categories = categories
        self._seed_pages = seed_pages
        self._max_depth = max_depth
        self._mw = MediaWikiClient(f"https://{wiki}/api.php", client=client, delay=delay)

    def discover(
        self,
        *,
        limit: int | None = None,
        with_metrics: bool = False,  # noqa: ARG002 - kept for interface parity
    ) -> Iterator[SourceTheory]:
        titles = self._collect_titles(limit)
        infos = self._mw.page_info(titles)
        for title in titles:
            yield SourceTheory(
                source=self.name,
                name=title,
                url=f"https://{self._wiki}/wiki/{quote(title.replace(' ', '_'))}",
                metrics=TheoryMetrics(size_bytes=infos.get(title, {}).get("length")),
            )

    def _collect_titles(self, limit: int | None) -> list[str]:
        seen: list[str] = []
        seen_set: set[str] = set()

        def add(title: str) -> None:
            if title not in seen_set:
                seen_set.add(title)
                seen.append(title)

        queue = [f"Category:{category}" for category in self._categories]
        for _ in range(self._max_depth):
            if not queue:
                break
            next_queue: list[str] = []
            for category in queue:
                for title in self._mw.category_members(category, namespace=0):
                    add(title)
                next_queue.extend(self._mw.category_members(category, namespace=14))
            queue = next_queue

        for page in self._seed_pages:
            for title in self._mw.links(page):
                add(title)

        return seen[:limit] if limit is not None else seen
