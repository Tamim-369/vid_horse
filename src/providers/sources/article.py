"""Fetch the plain-text detail of a topic from a MediaWiki wiki.

Discovery connectors deliberately return only a pointer; this fetcher turns the
pointer back into text when a theory is actually being researched.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx

from core.errors import ProviderError
from core.models import ArticleText
from providers.sources.mediawiki import MediaWikiClient
from utils.text import clean_article_text


class MediaWikiArticleFetcher:
    """Fetch article text from a Wikipedia (or compatible) language edition."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        language: str = "en",
        delay: float = 0.2,
        max_chars: int = 6000,
    ) -> None:
        self.name = f"{language}wiki"
        self._language = language
        self._max_chars = max_chars
        self._mw = MediaWikiClient(
            f"https://{language}.wikipedia.org/w/api.php", client=client, delay=delay
        )

    def article_url(self, title: str) -> str:
        return f"https://{self._language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    def fetch(
        self,
        title: str,
        *,
        url: str | None = None,
        max_chars: int | None = None,
    ) -> ArticleText:
        """Return cleaned, length-bounded text for ``title``."""

        budget = max_chars or self._max_chars
        try:
            text = clean_article_text(self._mw.extract(title) or "")
            source_url = url or self.article_url(title)
        except ProviderError:
            # Wikipedia rate-limited/blocked: fall back to DuckDuckGo
            # instant answer so research never hard-fails on a 429
            text, source_url = self._ddg_fallback(title, url)
        truncated = len(text) > budget
        return ArticleText(
            title=title,
            url=source_url,
            text=text[:budget],
            truncated=truncated,
        )

    def _ddg_fallback(self, title: str, url: str | None) -> tuple[str, str]:
        """Fetch a short abstract from DuckDuckGo when Wikipedia is blocked."""

        import json as _json

        q = title.replace(" ", "+")
        endpoint = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        try:
            resp = self._mw._client.get(endpoint, timeout=15.0)
            resp.raise_for_status()
            data = resp.json() if hasattr(resp, "json") else _json.loads(resp.text)
        except Exception:  # noqa: BLE001
            return "", url or self.article_url(title)
        abstract = data.get("AbstractText") or data.get("Answer") or ""
        # DDG sometimes returns related topics with text
        if not abstract:
            topics = data.get("RelatedTopics") or []
            for t in topics:
                if isinstance(t, dict) and t.get("Text"):
                    abstract = t["Text"]
                    break
        text = clean_article_text(abstract or "")
        source_url = data.get("AbstractURL") or url or self.article_url(title)
        return text, source_url
