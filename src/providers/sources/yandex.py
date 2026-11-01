"""Yandex image search scraper — no API key, no rate limits, scrapes public results."""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass

import httpx

from core.errors import ProviderError

logger = logging.getLogger(__name__)


@dataclass
class YandexImageResult:
    url: str
    thumb_url: str
    caption: str
    source_page: str
    width: int | None = None
    height: int | None = None


class YandexImagesClient:
    """Scrape Yandex Images for free, no-API-key results."""

    BASE_URL = "https://yandex.com/images/search"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        client: httpx.Client | None = None,
        delay: float = 1.0,
        max_results: int = 20,
    ) -> None:
        self._client = client or httpx.Client(
            headers=self.HEADERS, timeout=30.0, follow_redirects=True
        )
        self._delay = delay
        self._max_results = max_results

    def search(self, query: str, *, limit: int = 20) -> list[YandexImageResult]:
        """Search Yandex Images and return parsed results."""
        params = {"text": query, "type": "photo", "size": "large"}
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Yandex request failed: {exc}") from exc

        time.sleep(self._delay)
        return self._parse(resp.text)[:limit]

    def _parse(self, html_text: str) -> list[YandexImageResult]:
        """Parse Yandex search results HTML.

        Yandex embeds image data in the HTML as escaped JSON.
        We extract img_href URLs and associated metadata directly
        from the HTML using regex patterns.
        """
        results = []
        try:
            import html

            # Extract all img_href URLs from the HTML
            # Pattern: img_href&quot;:&quot;URL
            img_hrefs = re.findall(r'img_href&quot;:&quot;([^&]+)', html_text)

            # Extract captions/snippets - they appear near the img_href in the JSON
            # Look for snippet/title fields near each img_href
            # For simplicity, use a broader extraction
            captions = re.findall(r'snippet&quot;:&quot;([^&]+)', html_text)
            titles = re.findall(r'title&quot;:&quot;([^&]+)', html_text)
            detail_urls = re.findall(r'detail_url&quot;:&quot;([^&]+)', html_text)

            # Also extract preview/thumb URLs
            # Look for preview array URLs
            thumb_urls = []
            preview_matches = re.findall(r'"preview"\s*:\s*\[([^\]]+)\]', html_text)
            for pm in preview_matches:
                urls = re.findall(r'"url"\s*:\s*"([^"]+)"', pm)
                thumb_urls.extend(urls)

            # Fallback: extract any image URLs from the page
            if not thumb_urls:
                thumb_urls = re.findall(
                    r'"url"\s*:\s*"([^"]+\.(?:jpg|jpeg|png|webp|gif))"', html_text
                )

            # Build results by pairing img_hrefs with available metadata
            for i, img_url in enumerate(img_hrefs):
                if len(results) >= self._max_results:
                    break

                # Unescape HTML entities
                img_url = html.unescape(img_url)
                # Get caption (prefer snippet, then title)
                caption = ""
                if i < len(captions):
                    caption = html.unescape(captions[i])
                elif i < len(titles):
                    caption = html.unescape(titles[i])

                # Get source page
                source = ""
                if i < len(detail_urls):
                    source = html.unescape(detail_urls[i])

                # Get thumbnail
                thumb_url = ""
                if i < len(thumb_urls) or i < len(thumb_urls):
                    thumb_url = html.unescape(thumb_urls[i])

                if not thumb_url and i < len(thumb_urls):
                    thumb_url = html.unescape(thumb_urls[i])

                # Fallback: use the image URL itself as thumb
                if not thumb_url:
                    thumb_url = img_url

                results.append(
                    YandexImageResult(
                        url=img_url,
                        thumb_url=thumb_url,
                        caption=caption[:500] if caption else "",
                        source_page=source,
                        width=None,
                        height=None,
                    )
                )

        except Exception as exc:
            logger.warning("Yandex parse error: %s", exc)
        return results[: self._max_results]


def build_yandex_client(
    client: httpx.Client | None = None,
    delay: float = 1.0,
    max_results: int = 20,
) -> YandexImagesClient:
    return YandexImagesClient(client=client, delay=delay, max_results=max_results)
