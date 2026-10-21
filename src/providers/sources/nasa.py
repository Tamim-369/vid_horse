"""NASA image archive client — public-domain imagery, no key needed."""

from __future__ import annotations

import time

import httpx

from core.errors import ProviderError

API_URL = "https://images-api.nasa.gov/search"


class NasaImagesClient:
    """Thin, polite wrapper over ``images-api.nasa.gov``."""

    def __init__(self, *, client: httpx.Client, delay: float = 0.4) -> None:
        self._client = client
        self._delay = delay

    def search_images(self, query: str, *, limit: int = 6) -> list[dict]:
        """Return image hits for ``query`` in the shared media-info shape.

        Shape matches the MediaWiki ``image_info`` entries so the asset
        block can score both without caring about the origin: ``file``,
        ``url``, ``thumburl``, ``page_url``, ``description``, ``license``,
        ``artist``, ``width``/``height``.
        """

        params = {"q": query, "media_type": "image", "page_size": limit}
        try:
            response = self._client.get(API_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"NASA image search failed: {exc}") from exc
        if self._delay:
            time.sleep(self._delay)

        hits: list[dict] = []
        items = response.json().get("collection", {}).get("items", [])
        for item in items[:limit]:
            data = next(
                (d for d in item.get("data", []) if d.get("media_type") == "image"),
                None,
            )
            if data is None:
                continue
            thumb = next(
                (
                    link.get("href")
                    for link in item.get("links", [])
                    if link.get("rel") == "preview"
                ),
                None,
            )
            title = data.get("title") or " NASA image"
            hits.append(
                {
                    "file": f"NASA: {title}",
                    "url": thumb or "",
                    "thumburl": thumb,
                    "page_url": f"https://images.nasa.gov/details/{data.get('nasa_id', '')}",
                    "description": data.get("description") or "",
                    "license": "public domain",
                    "artist": data.get("photographer") or data.get("center") or None,
                    "width": 1920,
                    "height": 1080,
                    "mime": "image/jpeg",
                }
            )
        return [hit for hit in hits if hit["url"]]
