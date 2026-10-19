"""Wikipedia discovery connector.

Seeds from category trees and list articles, and can enrich each theory with
average monthly pageviews (Wikimedia's own API, not a search engine).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from urllib.parse import quote

import httpx

from core.models import SourceTheory, TheoryMetrics
from providers.sources.mediawiki import MediaWikiClient

_DEFAULT_MAX_DEPTH = 2


class WikipediaConnector:
    """Discover theories from a Wikipedia language edition."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        language: str = "en",
        categories: tuple[str, ...] = ("Conspiracy theories",),
        seed_pages: tuple[str, ...] = ("List of conspiracy theories",),
        delay: float = 0.2,
        pageviews_months: int = 12,
        max_depth: int = _DEFAULT_MAX_DEPTH,
    ) -> None:
        self.name = f"{language}wiki"
        self._language = language
        self._project = f"{language}.wikipedia"
        self._categories = categories
        self._seed_pages = seed_pages
        self._pageviews_months = pageviews_months
        self._max_depth = max_depth
        self._client = client
        self._mw = MediaWikiClient(
            f"https://{language}.wikipedia.org/w/api.php", client=client, delay=delay
        )

    def discover(
        self,
        *,
        limit: int | None = None,
        with_metrics: bool = False,
    ) -> Iterator[SourceTheory]:
        titles = self._collect_titles(limit)
        infos = self._mw.page_info(titles)
        categories = self._mw.categories(titles)
        qids = self._mw.wikidata_ids(titles)

        for title in titles:
            metrics = TheoryMetrics(size_bytes=infos.get(title, {}).get("length"))
            if with_metrics:
                views = self.pageviews(title)
                if views is not None:
                    metrics.avg_monthly_views = views
            yield SourceTheory(
                source=self.name,
                name=title,
                url=self.article_url(title),
                wikidata_id=qids.get(title),
                categories=categories.get(title, []),
                metrics=metrics,
            )

    def article_url(self, title: str) -> str:
        return f"https://{self._language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    def pageviews(self, title: str) -> float | None:
        """Return average monthly views over the configured window, if any."""

        end = date.today()
        start = end - timedelta(days=30 * self._pageviews_months)
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{self._project}/all-access/user/{quote(title.replace(' ', '_'))}/monthly/"
            f"{start:%Y%m%d}00/{end:%Y%m%d}00"
        )
        try:
            response = self._client.get(url)
            response.raise_for_status()
            items = response.json().get("items", [])
        except (httpx.HTTPError, ValueError):
            return None
        if not items:
            return None
        return sum(item["views"] for item in items) / len(items)

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
                if limit is not None and len(seen) >= limit:
                    return seen[:limit]
            queue = next_queue

        for page in self._seed_pages:
            for title in self._mw.links(page):
                add(title)
                if limit is not None and len(seen) >= limit:
                    return seen[:limit]

        return seen[:limit] if limit is not None else seen
