"""Openverse client — aggregated CC-licensed imagery (Flickr, museums, news)."""

from __future__ import annotations

import time

import httpx

from core.errors import ProviderError

API_URL = "https://api.openverse.org/v1/images/"


class OpenverseClient:
    """Thin, polite wrapper over the Openverse imagery API (anonymous)."""

    def __init__(self, *, client: httpx.Client, delay: float = 0.6) -> None:
        self._client = client
        self._delay = delay

    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        """Return image hits in the shared media-info shape.

        See :meth:`providers.sources.nasa.NasaImagesClient.search_images`
        for the key contract: the same ``image_info``-compatible dict so the
        asset block scores every source uniformly.
        """

        params = {"q": query, "page_size": limit, "license_type": "all"}
        try:
            response = self._client.get(API_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Openverse search failed: {exc}") from exc
        if self._delay:
            time.sleep(self._delay)

        hits: list[dict] = []
        for result in response.json().get("results", [])[:limit]:
            title = result.get("title") or "Openverse image"
            creator = result.get("creator")
            license_tag = result.get("license") or "cc"
            version = result.get("license_version")
            hits.append(
                {
                    "file": f"Openverse: {title}",
                    "url": result.get("url") or "",
                    "thumburl": result.get("thumbnail") or result.get("url"),
                    "page_url": result.get("foreign_landing_url") or "",
                    "description": title,
                    "license": f"{license_tag} {version}".strip(),
                    "artist": creator,
                    "width": result.get("width"),
                    "height": result.get("height"),
                    "mime": "image/jpeg",
                }
            )
        return [hit for hit in hits if hit["url"]]
