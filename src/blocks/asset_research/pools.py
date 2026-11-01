"""Metadata pools: article files, entity portraits, claim sheets."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.errors import ProviderError
from core.models import (
    ResearchBrief,
    Theory,
    VisualAsset,
)
from utils import computing

from .claims import _Claims
from .media import info_kind, is_media_junk, is_video_junk, to_asset

logger = logging.getLogger(__name__)




class PoolsMixin:
    """Mixin: metadata pools, attached to AssetResearchBlock."""

    def _page_pool(self, titles: list[str]) -> list[dict]:
        """Media metadata for the files used on ``titles``."""

        files = [f for entries in self._wiki.page_files(titles).values() for f in entries]
        if not files:
            return []
        return self._wiki.image_info(files, thumb_width=self._options.thumb_width)

    def _entity_pool(self, brief: ResearchBrief) -> dict[str, VisualAsset]:
        """First usable portrait for each notable person named in the brief."""

        names = [
            name.split(",")[0] for name in brief.key_people[: self._options.entity_limit]
        ]
        if not names:
            return {}
        pool: dict[str, VisualAsset] = {}
        try:
            per_person = self._wiki.page_files(names)
            active = {person: files[:15] for person, files in per_person.items() if files}
            if len(active) > 1:
                workers = min(computing.variant_workers(), len(active))
                with ThreadPoolExecutor(max_workers=workers) as workers_pool:
                    futures = {
                        person: workers_pool.submit(
                            self._wiki.image_info, files, thumb_width=self._options.thumb_width
                        )
                        for person, files in active.items()
                    }
                for person, future in futures.items():
                    for info in future.result():
                        if self._usable(info):
                            pool[person.lower()] = to_asset(info)
                            break
            else:
                for person, files in active.items():
                    infos = self._wiki.image_info(
                        files, thumb_width=self._options.thumb_width
                    )
                    for info in infos:
                        if self._usable(info):
                            pool[person.lower()] = to_asset(info)
                            break
        except ProviderError as exc:
            logger.debug("Entity portraits skipped: %s", exc)
        return pool

    # -- per scene -------------------------------------------------------------

    @staticmethod
    def _article_source(theory: Theory):
        for source in theory.sources:
            if "wikipedia.org" in source.url:
                return source
        return theory.sources[0] if theory.sources else None

    def _usable(self, info: dict) -> bool:
        # Allow Yandex results with unknown width to pass initial filter
        # Vision model will do the real relevance check
        is_yandex = info.get("source") == "yandex"
        width = info.get("width")
        if is_yandex and width is None:
            width = 400  # assume sufficient for junk filter
        if is_media_junk(info["file"], info.get("mime", ""), width):
            return False
        if info_kind(info) == "video":
            return self._video_slots > 0 and not is_video_junk(info)
        return True

    def _blocked(self, info: dict, used: _Claims) -> bool:
        """File used, or its extensionless twin already in play."""

        return used.blocked(info["file"])

    def _take(
        self,
        info: dict,
        chosen: list[VisualAsset],
        used: _Claims,
        trigger: str = "",
    ) -> bool:
        """Atomically claim the file, then append; False when already in play."""

        if not used.claim(info["file"]):
            return False
        asset = to_asset(info)
        if asset.kind == "video":
            with used.lock:
                self._video_slots = max(0, self._video_slots - 1)
        asset.trigger = trigger
        chosen.append(asset)
        if self._shelf is not None and self._clip is not None and self._http is not None:
            # everything the gauntlet approved goes on the shelf: later
            # research passes match subjects locally instead of re-walking
            # Commons. Takes without a stamped CLIP score land at 0.0 —
            # they still count (their search path passed its own gates),
            # and the route re-verifies every hit with analyze() anyway.
            try:
                embed = self._clip.image_embed(
                    info.get("thumburl") or info.get("url"), self._http
                )
                if embed is not None:
                    score = info.get("_clip_score") or asset.clip_score or 0.0
                    self._shelf.store(
                        info,
                        embed if isinstance(embed, np.ndarray)
                        else np.asarray(embed, dtype=np.float32),
                        float(score),
                    )
            except Exception as exc:  # noqa: BLE001 — shelfing is best-effort
                logger.debug("shelfing %s failed: %s", info.get("file"), exc)
        return True
