"""Pack settlement: race dedupe, adjacency rule, blank donation."""

from __future__ import annotations

import logging
from collections import Counter

from core.models import (
    VisualAsset,
)

from .claims import _Claims

logger = logging.getLogger(__name__)


class PackingMixin:
    """Mixin: claim-race dedupe, adjacency rule and blank-card donation."""

    def resolve_races(self, scenes, tasks, brief, entity_pool, article_pool, used):
        """Re-air every entry whose lead visual a parallel worker double-won."""
        claimed: set[str] = set()
        for entry in scenes:
            if not entry.assets:
                continue
            first = entry.assets[0].url
            if first in claimed:
                task = next(t for t in tasks if t[0] == entry.scene_index)
                _index, scene, query, actions, target = task
                entry.assets = self._assets_for_scene(
                    scene, query, actions, target, brief,
                    entity_pool, article_pool, used,
                )
            else:
                claimed.add(first)

    def unique_within(self, scenes):
        """Drop repeats inside one card: a card never shows one url twice."""
        for entry in scenes:
            per_entry: set[str] = set()
            unique: list[VisualAsset] = []
            for asset in entry.assets:
                if asset.url in per_entry:
                    continue
                per_entry.add(asset.url)
                unique.append(asset)
            entry.assets = unique

    def settle_adjacency(self, scenes, tasks, brief, entity_pool, article_pool):
        """No image may lead two adjacent scenes back to back."""
        
        scenes.sort(key=lambda entry: entry.scene_index)
        task_map = {t[0]: t for t in tasks}
        for _pass_index in range(len(scenes) + 1):
            progress = False
            for previous, current in zip(scenes, scenes[1:], strict=False):
                if not (previous.assets and current.assets):
                    continue
                lead = current.assets[0].url
                if previous.assets[0].url != lead:
                    continue
                swapped = False
                own_alt = next(
                    (a for a in current.assets[1:] if a.url != lead), None
                )
                if own_alt is not None:
                    swap_at = current.assets.index(own_alt)
                    current.assets.insert(0, current.assets.pop(swap_at))
                    swapped = True
                else:
                    donor_alt = next(
                        (a for a in previous.assets[1:] if a.url != previous.assets[0].url),
                        None,
                    )
                    if donor_alt is not None:
                        previous.assets.remove(donor_alt)
                        current.assets.insert(
                            0, donor_alt.model_copy(update={"trigger": ""})
                        )
                        swapped = True
                if not swapped:
                    freq = Counter(a.url for e in scenes for a in e.assets)
                    donated = None
                    donated_count = None
                    for donor in scenes:
                        distance = abs(donor.scene_index - current.scene_index)
                        if donor is current or distance <= 1 or not donor.assets:
                            continue
                        for cand in donor.assets:
                            if cand.url == lead or cand.url == previous.assets[0].url:
                                continue
                            count = freq.get(cand.url, 0)
                            if donated is None or count < donated_count:
                                donated, donated_count = cand, count
                    if donated is not None:
                        current.assets.insert(
                            0, donated.model_copy(update={"trigger": ""})
                        )
                        swapped = True
                if not swapped:
                    task = task_map.get(current.scene_index)
                    if task is not None:
                        relaxed = _Claims()
                        for elsewhere in scenes:
                            if elsewhere is current:
                                continue
                            for held in elsewhere.assets:
                                relaxed.claim(held.file)
                        _index, scene, query, actions, target = task
                        current.assets = self._assets_for_scene(
                            scene, query, actions, target, brief,
                            entity_pool, article_pool, relaxed,
                        )
                        swapped = True
                progress = progress or swapped
            if not progress:
                break


    def _vibe_terms(self, text: str) -> list[str]:
        """Extract atmospheric/vibe search terms from a scene description."""
        # mood keywords that suggest good image searches
        mood_map = {
            "dark": "dark vintage photograph",
            "night": "night scene dark",
            "old": "old vintage photograph",
            "ancient": "ancient manuscript illustration",
            "scary": "dark atmospheric image",
            "mysterious": "mysterious dark scene",
            "space": "cosmic nebula dark",
            "cosmic": "cosmic space image",
            "book": "old book page",
            "text": "manuscript page old",
            "church": "gothic cathedral dark",
            "religious": "religious art painting",
            "psychology": "vintage psychology experiment",
            "mind": "abstract dark concept",
            "death": "dark memorial image",
            "shadow": "shadow dark atmosphere",
            "light": "dramatic lighting dark",
        }
        terms = []
        lowered = text.lower()
        for keyword, vibe in mood_map.items():
            if keyword in lowered and vibe not in terms:
                terms.append(vibe)
        if not terms:
            terms = ["dark atmospheric image", "vintage photograph"]
        return terms[:3]

    def fill_blanks(self, scenes):
        """Fill empty scenes: reuse topic images if enough exist, else vibe-search.

        Strategy:
        1. Count topic-related images across all scenes.
        2. If more than 5, reuse a distant-scene image (not back-to-back).
        3. If 5 or fewer, search for a vibe-matching image that fits the
           sentence's mood (may not be topic-related).
        """
        freq = Counter(asset.url for entry in scenes for asset in entry.assets)

        for entry in scenes:
            if entry.assets:
                continue

            # Strategy 1: reuse if topic images exist (not back-to-back)
            if True:  # always try reuse
                best = None
                best_count = None
                # Only reuse from non-adjacent scenes (no back-to-back)
                for donor in scenes:
                    if donor is entry or not donor.assets:
                        continue
                    if abs(donor.scene_index - entry.scene_index) <= 1:
                        continue
                    for cand in donor.assets:
                        count = freq.get(cand.url, 0)
                        if best is None or count < best_count:
                            best, best_count = cand, count
                        count = freq.get(cand.url, 0)
                        if best is None or count < best_count:
                            best, best_count = cand, count
                if best is not None:
                    freq[best.url] = best_count + 1
                    entry.assets.append(best.model_copy(update={"trigger": ""}))
                    continue

            # Strategy 2: vibe search for a mood-matching image
            try:
                text_scene = f"{entry.visual} {entry.query}"
                vibe_terms = self._vibe_terms(text_scene)
                for term in vibe_terms:
                    if entry.assets:
                        break
                    for score, info in self._search_ranked(term, text_scene):
                        if score < 2:
                            break
                        if self._blocked(info, _Claims()):
                            continue
                        # d3 judge: only keep vibe-matching images
                        if self._vision is not None:
                            relevant, _prob = self._vision.d3_judge(
                                info.get("description") or info.get("file", ""), text_scene
                            )
                            if not relevant:
                                continue
                        self._take(info, entry.assets, _Claims())
                        break
            except Exception:  # noqa: BLE001
                pass

            # Strategy 3: leave empty — render uses blank card
