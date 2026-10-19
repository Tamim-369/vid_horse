"""Wikidata discovery connector.

Seeds the canonical catalog: every item that is (a subclass of) a conspiracy
theory, with its English Wikipedia link and language count.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx

from core.errors import ProviderError
from core.models import SourceTheory, TheoryMetrics

CONSPIRACY_THEORY_QID = "Q159535"

_QUERY = """
SELECT ?item ?itemLabel ?enwiki ?sitelinks WHERE {
  ?item wdt:P31/wdt:P279* wd:%s .
  OPTIONAL { ?item wikibase:sitelinks ?sitelinks }
  OPTIONAL { ?enwiki schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
"""


class WikidataConnector:
    """Discover theories from the Wikidata Query Service."""

    name = "wikidata"
    endpoint = "https://query.wikidata.org/sparql"

    def __init__(
        self,
        *,
        client: httpx.Client,
        limit: int = 5000,
        delay: float = 0.2,
        class_qid: str = CONSPIRACY_THEORY_QID,
    ) -> None:
        self._client = client
        self._limit = limit
        self._delay = delay
        self._class_qid = class_qid

    def discover(
        self,
        *,
        limit: int | None = None,
        with_metrics: bool = False,  # noqa: ARG002 - kept for interface parity
    ) -> Iterator[SourceTheory]:
        query = _QUERY % (self._class_qid, limit or self._limit)
        try:
            response = self._client.post(
                self.endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Wikidata query failed: {exc}") from exc
        if self._delay:
            time.sleep(self._delay)

        bindings = response.json().get("results", {}).get("bindings", [])
        for binding in bindings:
            qid = binding["item"]["value"].rsplit("/", 1)[-1]
            name = binding.get("itemLabel", {}).get("value", qid)
            url = binding.get("enwiki", {}).get("value") or f"https://www.wikidata.org/wiki/{qid}"
            sitelinks = binding.get("sitelinks", {}).get("value")
            metrics = (
                TheoryMetrics(language_count=int(sitelinks)) if sitelinks else TheoryMetrics()
            )
            yield SourceTheory(
                source=self.name,
                name=name,
                url=url,
                wikidata_id=qid,
                metrics=metrics,
            )
