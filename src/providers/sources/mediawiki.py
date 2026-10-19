"""Generic MediaWiki Action API client.

Shared by Wikipedia and any MediaWiki-based wiki (e.g. Fandom), since they all
expose the same ``api.php`` surface.
"""

from __future__ import annotations

import time

import httpx

from core.errors import ProviderError
from providers.sources.base import chunked
from utils.text import strip_html


class MediaWikiClient:
    """Thin, polite wrapper around the MediaWiki Action API."""

    def __init__(self, api_url: str, *, client: httpx.Client, delay: float = 0.2) -> None:
        self._api_url = api_url
        self._client = client
        self._delay = delay

    def _get(self, params: dict) -> dict:
        query = {**params, "format": "json", "formatversion": 2}
        try:
            response = self._client.get(self._api_url, params=query)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"MediaWiki request failed: {exc}") from exc
        if self._delay:
            time.sleep(self._delay)
        return response.json()

    def category_members(
        self,
        category: str,
        *,
        namespace: int = 0,
        limit: int | None = None,
    ) -> list[str]:
        """Return page titles in ``category`` (``namespace`` 0=articles, 14=categories)."""

        title = category if category.lower().startswith("category:") else f"Category:{category}"
        titles: list[str] = []
        cont: str | None = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": title,
                "cmnamespace": namespace,
                "cmlimit": "500",
            }
            if cont:
                params["cmcontinue"] = cont
            data = self._get(params)
            titles.extend(m["title"] for m in data.get("query", {}).get("categorymembers", []))
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont or (limit is not None and len(titles) >= limit):
                break
        return titles[:limit] if limit is not None else titles

    def search(self, query: str, *, limit: int = 20) -> list[str]:
        data = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(limit, 500),
            }
        )
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def page_info(self, titles: list[str]) -> dict[str, dict]:
        """Return ``{title: {pageid, length}}`` for existing pages."""

        result: dict[str, dict] = {}
        for chunk in chunked(titles, 50):
            data = self._get(
                {"action": "query", "titles": "|".join(chunk), "prop": "info"}
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing"):
                    continue
                result[page["title"]] = {
                    "pageid": page.get("pageid"),
                    "length": page.get("length"),
                }
        return result

    def links(self, title: str, *, namespace: int = 0, limit: int = 500) -> list[str]:
        """Return outgoing links from ``title``."""

        links: list[str] = []
        cont: str | None = None
        while True:
            params = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "plnamespace": namespace,
                "pllimit": "500",
            }
            if cont:
                params["plcontinue"] = cont
            data = self._get(params)
            for page in data.get("query", {}).get("pages", []):
                links.extend(link["title"] for link in page.get("links", []))
            cont = data.get("continue", {}).get("plcontinue")
            if not cont or len(links) >= limit:
                break
        return links[:limit]

    def categories(self, titles: list[str]) -> dict[str, list[str]]:
        """Return ``{title: [categories]}`` (hidden categories excluded)."""

        result: dict[str, list[str]] = {}
        for chunk in chunked(titles, 50):
            data = self._get(
                {
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "categories",
                    "cllimit": "500",
                    "clshow": "!hidden",
                }
            )
            for page in data.get("query", {}).get("pages", []):
                result[page["title"]] = [
                    cat["title"].removeprefix("Category:")
                    for cat in page.get("categories", [])
                ]
        return result

    def wikidata_ids(self, titles: list[str]) -> dict[str, str]:
        """Return ``{title: QID}`` from page props, where available."""

        result: dict[str, str] = {}
        for chunk in chunked(titles, 50):
            data = self._get(
                {
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "pageprops",
                    "ppprop": "wikibase_item",
                }
            )
            for page in data.get("query", {}).get("pages", []):
                qid = page.get("pageprops", {}).get("wikibase_item")
                if qid:
                    result[page["title"]] = qid
        return result

    def extract(self, title: str) -> str | None:
        """Return plain text (``prop=extracts``) for ``title``, or ``None``."""

        data = self._get(
            {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
            }
        )
        for page in data.get("query", {}).get("pages", []):
            if page.get("missing"):
                return None
            return page.get("extract")
        return None

    def page_files(self, titles: list[str], *, limit: int = 100) -> dict[str, list[str]]:
        """Return ``{page title: [file titles]}`` for media used on the pages."""

        result: dict[str, list[str]] = {}
        for chunk in chunked(titles, 10):
            data = self._get(
                {
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "images",
                    "imlimit": str(max(limit, 1)),
                    "redirects": 1,
                }
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing"):
                    continue
                result[page["title"]] = [
                    image["title"] for image in page.get("images", [])
                ]
        return result

    def image_info(
        self, files: list[str], *, thumb_width: int = 1280
    ) -> list[dict]:
        """Return normalized metadata for ``files`` (File: subjects).

        Each dict has: file, url, thumburl, page_url, mime, width, height,
        description, license, artist. Missing files are skipped.
        """

        infos: list[dict] = []
        for chunk in chunked(files, 20):
            data = self._get(
                {
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": str(thumb_width),
                    "redirects": 1,
                }
            )
            for page in data.get("query", {}).get("pages", []):
                info_items = page.get("imageinfo") or []
                if page.get("missing") or not info_items:
                    continue
                meta = info_items[0].get("extmetadata") or {}
                infos.append(self._info_dict(page, meta))
        return infos

    def search_media(
        self, query: str, *, limit: int = 10, thumb_width: int = 1280
    ) -> list[dict]:
        """Search media by query and return the same dicts as :meth:`image_info`."""

        data = self._get(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(max(limit, 1)),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": str(thumb_width),
            }
        )
        hits = [
            page
            for page in data.get("query", {}).get("pages", [])
            if page.get("imageinfo")
        ]
        hits.sort(key=lambda page: page.get("index", 10**9))
        return [
            self._info_dict(page, page.get("imageinfo", [{}])[0].get("extmetadata") or {})
            for page in hits
        ]

    def _info_dict(self, page: dict, meta: dict) -> dict:
        """Normalize one imageinfo payload into the shared dict shape."""

        info = page.get("imageinfo", [{}])[0]
        return {
            "file": page["title"],
            "url": info.get("url", ""),
            "thumburl": info.get("thumburl") or info.get("url", ""),
            "page_url": info.get("descriptionurl", ""),
            "mime": info.get("mime", ""),
            "width": info.get("width"),
            "height": info.get("height"),
            "description": _meta_text(meta, "ImageDescription"),
            "license": _meta_text(meta, "LicenseShortName"),
            "artist": _meta_text(meta, "Artist") or None,
        }


def _meta_text(meta: dict, key: str) -> str:
    """Return cleaned plain text for an extmetadata field, or ``""``."""

    value = meta.get(key, {}).get("value", "")
    return strip_html(str(value))[:300] if value else ""
