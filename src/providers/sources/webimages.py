"""Web image search — direct scraping of engine result pages, no API keys.

Two engines that historically tolerate polite fetching without JavaScript:
Bing's HTML image grid (results carry the full-resolution ``murl``, the
hosting page ``purl`` and the engine thumbnail ``turl`` inside ``iusc``
attributes) and DuckDuckGo's ``i.js`` JSON endpoint (needs one page fetch
up front for the ``vqd`` token).

The client never raises: an engine that changes shape, blocks us or times
out yields empty results and the matching pipeline falls back to the next
source. These results freeload off the global web and carry no machine
readable license — the caller knows what that means.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import re
import time

import httpx

from core.errors import ProviderError

logger = logging.getLogger(__name__)

_IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.IGNORECASE)
_BING_IUSC = re.compile(r'class="iusc"[^>]*\sm="([^"]+)"')
_DDG_VQD = re.compile(r'vqd(?:["\u2019]+)?\s*[=:]\s*["\u2019]([\d-]+)["\u2019]')
_MIN_WIDTH = 400
_TIMEOUT = 20.0


class WebImagesClient:
    """Image hits scraped directly from search-engine result pages.

    Every hit uses the asset-research shared dict shape (file/url/
    thumburl/page_url/width/height/mime...) with ``license: "unknown"``
    (the engine chip filters the pool but does not identify each hit).
    ``engine`` records which one agreed to talk. ``license_image`` rides
    the engine's own license chip — DDG's grammar is
    ``f={time},{size},{color},{type},{layout},{license}`` and the chip
    vocabulary is ``any``, ``Public``, ``Share``, ``ShareCommercially``,
    ``Modify``, ``ModifyCommercially`` (verified live: Public returns a
    publicdomainpictures/goodfreephotos pool, ShareCommercially returns
    Wikimedia and Flickr dynamic hits).
    """

    def __init__(
        self,
        *,
        client: httpx.Client,
        delay: float = 1.5,
        engine: str = "ddg",
        license_image: str = "ShareCommercially",
    ) -> None:
        self._client = client
        self._delay = delay
        self._engine = engine
        self._license_image = license_image

    def search(self, query: str, *, limit: int = 10) -> list[dict]:
        """Return candidate image hits for ``query`` (never raises).

        The configured engine runs first; when it yields nothing usable
        the other one gets a shot, so a markup quirk or a bot-challenge
        cannot blank a scene.
        """

        engines = [self._engine] + [e for e in ("ddg", "bing") if e != self._engine]
        hits: list[dict] = []
        for engine in engines:
            if hits:
                break
            try:
                hits = self._bing(query) if engine == "bing" else self._ddg(query)
            except (
                httpx.HTTPError,
                ProviderError,
                KeyError,
                AttributeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                logger.warning(
                    "Web image search failed for %r via %s: %s", query, engine, exc
                )
        wanted: list[dict] = []
        for hit in hits:
            if hit["width"] and hit["width"] < _MIN_WIDTH:
                continue
            if not _IMAGE_EXT.search(hit["url"] or ""):
                continue
            wanted.append(hit)
            if len(wanted) >= limit:
                break
        return wanted

    def _bing(self, query: str) -> list[dict]:
        response = self._client.get(
            "https://www.bing.com/images/search",
            params={"q": query, "mkt": "en-US", "count": "35"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        if self._delay:
            time.sleep(self._delay)
        hits: list[dict] = []
        for encoded in _BING_IUSC.findall(response.text):
            try:
                data = json.loads(html_mod.unescape(encoded))
            except json.JSONDecodeError:
                continue
            url = (data.get("murl") or "").strip()
            if not url:
                continue
            title = (data.get("t") or "").strip() or query
            hits.append(
                {
                    "file": f"Bing: {html_mod.unescape(title)}",
                    "url": url,
                    "thumburl": (data.get("turl") or "").strip() or url,
                    "page_url": (data.get("purl") or "").strip(),
                    "description": html_mod.unescape(title),
                    "license": "unknown",
                    "artist": None,
                    "width": _first_int(_dsize(data)),
                    "height": None,
                    "mime": "image/jpeg",
                }
            )
        return hits

    def _ddg(self, query: str) -> list[dict]:
        page = self._client.get(
            "https://duckduckgo.com/",
            params={"q": query, "iax": "images", "ia": "images"},
            timeout=_TIMEOUT,
        )
        page.raise_for_status()
        match = _DDG_VQD.search(page.text)
        if not match:
            return []
        # grammar: f={time},{size},{color},{type},{layout},{license}
        # exactly six chips; empties filter nothing
        size_chip = "size:Large"
        license_chip = (
            f"license:{self._license_image}" if self._license_image else ""
        )
        data = self._client.get(
            "https://duckduckgo.com/i.js",
            params={
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": match.group(1),
                "f": f",,{size_chip},,,{license_chip}",
                "p": "1",
            },
            timeout=_TIMEOUT,
        )
        data.raise_for_status()
        if self._delay:
            time.sleep(self._delay)
        hits: list[dict] = []
        for result in data.json().get("results", []):
            url = (result.get("image") or "").strip()
            if not url:
                continue
            title = (result.get("title") or "").strip() or query
            hits.append(
                {
                    "file": f"Duck: {title}",
                    "url": url,
                    "thumburl": (result.get("thumbnail") or "").strip() or url,
                    "page_url": (result.get("url") or "").strip(),
                    "description": title,
                    "license": "unknown",
                    "artist": None,
                    "width": result.get("width"),
                    "height": result.get("height"),
                    "mime": "image/jpeg",
                }
            )
        return hits


def _dsize(data: dict) -> object:
    """Bing hides a decoded ``W|H|F`` size inside ``md5`` — string or dict."""

    md5 = data.get("md5")
    if isinstance(md5, dict):
        return md5.get("dsize", "")
    return ""


def _first_int(dsize: object) -> int | None:
    """Bing packs the decoded size as ``'W|H|F'`` inside md5... sometimes."""

    if isinstance(dsize, str) and "|" in dsize:
        parts = dsize.split("|")
        if len(parts) >= 2 and parts[0].isdigit():
            return int(parts[0])
    return None
