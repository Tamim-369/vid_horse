"""Fetch the plain-text detail of a topic from a MediaWiki wiki.

Discovery connectors deliberately return only a pointer; this fetcher turns the
pointer back into text when a theory is actually being researched.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx

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
        text = clean_article_text(self._mw.extract(title) or "")
        truncated = len(text) > budget
        return ArticleText(
            title=title,
            url=url or self.article_url(title),
            text=text[:budget],
            truncated=truncated,
        )
