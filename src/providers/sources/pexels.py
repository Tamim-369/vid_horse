"""Pexels client — free licensed stock photography via the public API.

The Pexels license permits commercial use and needs no attribution.
A missing or rejected key yields zero hits so fallback keeps working.
"""

from __future__ import annotations

import os
import time

import httpx

from core.errors import ProviderError

API_URL = "https://api.pexels.com/v1/search"


def _api_key() -> str | None:
    key = os.getenv("PEXELS_API_KEY") or os.getenv("PIXELS_API_KEY")
    return key.strip() if key else None


class PexelsClient:
    """Thin wrapper over the Pexels image search API."""

    def __init__(
        self, *, client: httpx.Client | None = None, api_key: str | None = None, delay: float = 0.5
    ) -> None:
        self._client = client or httpx.Client(timeout=20.0, follow_redirects=True)
        self._owns_client = client is None
        self._api_key = api_key or _api_key()
        self._delay = delay

    def search(self, query: str, *, per_page: int = 8, limit: int | None = None) -> list[dict]:
        """Image hits in the shared media-info shape."""
        if not self._api_key:
            return []
        n = limit or per_page
        try:
            response = self._client.get(
                API_URL,
                params={"query": query, "per_page": n},
                headers={"Authorization": self._api_key},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # 401/403/429 -> yield nothing so fallback keeps working
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403, 429}:
                return []
            raise ProviderError(f"Pexels search failed for {query!r}: {exc}") from exc
        if self._delay:
            time.sleep(self._delay)

        hits: list[dict] = []
        for photo in response.json().get("photos", [])[:n]:
            src = photo.get("src") or {}
            url = src.get("large2x") or src.get("large") or src.get("original") or ""
            if not url:
                continue
            alt = (photo.get("alt") or "").strip() or f"Pexels photo {photo.get('id','')}"
            thumb = src.get("small") or src.get("tiny") or url
            hits.append(
                {
                    "file": f"Pexels: {alt}",
                    "url": url,
                    "thumburl": thumb,
                    "page_url": photo.get("url") or "",
                    "description": alt,
                    "license": "Pexels (free, no attribution required)",
                    "artist": photo.get("photographer"),
                    "width": photo.get("width"),
                    "height": photo.get("height"),
                    "mime": "image/jpeg",
                }
            )
        return hits
