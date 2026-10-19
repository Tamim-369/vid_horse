"""Build discovery connectors from settings."""

from __future__ import annotations

import httpx

from config.settings import Settings, get_settings
from providers.sources.base import SourceConnector, build_http_client
from providers.sources.fandom import FandomConnector
from providers.sources.wikidata import WikidataConnector
from providers.sources.wikipedia import WikipediaConnector


def build_connectors(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
) -> list[SourceConnector]:
    """Create the configured set of discovery connectors."""

    settings = settings or get_settings()
    http = client or build_http_client(settings)

    connectors: list[SourceConnector] = [
        WikipediaConnector(
            client=http,
            language=settings.wikipedia_language,
            categories=tuple(settings.discovery_categories),
            seed_pages=tuple(settings.discovery_seed_pages),
            delay=settings.request_delay,
            pageviews_months=settings.pageviews_months,
        ),
        WikidataConnector(client=http, delay=settings.request_delay),
    ]
    connectors.extend(
        FandomConnector(client=http, wiki=wiki, delay=settings.request_delay)
        for wiki in settings.fandom_wikis
    )
    return connectors
