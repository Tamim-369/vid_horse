"""Per-scene staging: search cascades, deep search, repair waves."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from blocks.base import generate_json_parsed
from core.errors import BlockError, ProviderError
from core.models import (
    ResearchBrief,
    ScriptScene,
    VisualAsset,
)
from utils import computing

from .claims import _Claims
from .media import _CLIP_FLOOR, content_tokens

logger = logging.getLogger(__name__)




class StagingMixin:
    """Mixin: per-scene staging, attached to AssetResearchBlock."""

    def _target_for(self, scene: ScriptScene) -> int:
        """1-3 assets, scaled by how much the scene has to say."""

        words = len(scene.narration.split()) + len(scene.visual.split())
        return min(self._options.assets_max, max(1, round(words / self._options.words_per_asset)))

    def _assets_for_scene(
        self,
        scene: ScriptScene,
        query: str,
        actions,
        target: int,
        brief: ResearchBrief,
        entity_pool: dict[str, VisualAsset],
        article_pool: list[dict],
        used: _Claims,
    ) -> list[VisualAsset]:
        chosen: list[VisualAsset] = []
        text_scene = f"{scene.visual} {scene.narration}"

        # the directed agents' source-routed searches come first: they name
        # the exact moment, entity-pool portraits are backup only
        for action in actions or []:
            if len(chosen) >= target:
                break
            unclaimed = len(chosen)
            self._route_query(action, text_scene, chosen, target, used)
            # one image per subject: a route that caught several frames must
            # not starve the next subject of its own shot
            del chosen[unclaimed + 1 :]
            for asset in chosen[unclaimed:]:
                asset.trigger = action.trigger

        for person, asset in entity_pool.items():
            if len(chosen) >= target:
                break
            words = person.split()
            surname = words[-1].lower()
            verifiable = asset.caption and surname in asset.caption.lower()
            if (
                verifiable
                and all(word in text_scene.lower() for word in words)
                and used.claim(asset.file)
            ):
                chosen.append(asset)

        searches: list[str] = []
        if len(chosen) < target and self._yandex is not None:
            # Yandex Images (search engine) is the primary source: unlimited
            # search, no API keys, no rate limits like Wikipedia/NASA.
            routed = [item.query for item in (actions or [])]
            for candidate in [*routed, query, brief.title]:
                candidate = (candidate or "").strip()
                if not candidate or len(chosen) >= target:
                    continue
                self._route_yandex(candidate, text_scene, chosen, target, used)

        if len(chosen) < target and self._shelf is not None:
            # local shelf: approved-by-us candidates from earlier research
            # passes — zero network when the shelf is warm
            self._route_shelf(query or brief.title, text_scene, chosen, target, used)

        if len(chosen) < target:  # fallback: mine the scene's own phrases
            # Vibe-based: search for the topic title + key person.
            searches = [query] if query else []
            if brief.title.lower() not in {s.lower() for s in searches}:
                searches.append(brief.title)
            for search_query in searches:
                if len(chosen) >= target:
                    break
                for score, info in self._search_ranked(search_query, text_scene):
                    if len(chosen) >= target or score < self._options.min_score:
                        break
                    if self._blocked(info, used):
                        continue
                    self._take(info, chosen, used)

        if not chosen:  # topic-guaranteed filler from the theory's own article
            for info in article_pool:
                if self._blocked(info, used):
                    continue
                self._take(info, chosen, used)
                break

        if not chosen and self._web is not None:  # global web: scraped engines
            self._route_web(searches[-1] or query or brief.title, text_scene, chosen, target, used)

        if not chosen:  # last resort: the topic title itself, looseness ok
            for score, info in self._search_ranked(brief.title, text_scene):
                if score < 2:
                    break
                self._take(info, chosen, used)
                break

        # One score for every pick on one scale: what the shot depicts
        # matters across paths (routed, entity, filler), not just inside
        # whichever candidate pool it came from.
        # Deep search is for scenes that route/target more than the normal
        # quota; a one-asset scene with its asset must not pay for it
        _judge = getattr(self._options, "judge", "vision")
        if _judge != "d3" and len(chosen) < min(self._MIN_TOPIC_ASSETS, target):
            self._deep_search(text_scene, brief, query, chosen, used)
        if _judge != "d3" and self._clip is not None and self._http is not None and chosen:
            with ThreadPoolExecutor(max_workers=computing.clip_workers()) as pool:
                results = list(
                    pool.map(
                        lambda asset: self._clip.analyze(
                            text_scene, asset.thumb_url or asset.url, self._http
                        ),
                        chosen,
                    )
                )
            for asset, (similarity, _collage_like, _texty) in zip(
                chosen, results, strict=True
            ):
                asset.clip_score = similarity
            ranked = sorted(
                zip(chosen, results, strict=True),
                key=lambda pair: pair[1][0] if pair[1][0] is not None else -1.0,
                reverse=True,
            )
            healthy: list[VisualAsset] = []
            for slot, (asset, (sim, collage_like, texty)) in enumerate(ranked):
                if collage_like or texty:
                    continue
                if sim is not None:
                    # the strongest subject may clear the normal floor;
                    # every EXTRA subject must prove itself clearly (a
                    # 0.19 marginal "tennis court telescope" barely
                    # sliding over the floor used to pair with real
                    # content and read on screen as visual noise)
                    floor = _CLIP_FLOOR if slot == 0 else _CLIP_FLOOR + 0.05
                    if sim < floor:
                        continue
                healthy.append(asset)
            # relative floor: keep only assets within 0.12 of the best
            # match to kill random/off-topic images (bowl, grandpa, etc.)
            if healthy:
                best_sim = max(a.clip_score or 0.0 for a in healthy)
                healthy = [a for a in healthy if (a.clip_score or 0.0) >= best_sim - 0.12]
            # the filter's verdict must actually replace the picks below
            # the floor were staged before scoring and only this pass can
            # veto them; an empty healthy list means AIR NOTHING (a junk
            # pile must never leak out as served scenes)
            chosen = healthy

        # the vision model is the judge what CLIP could not confidently
        # read: each marginal pick is looked at in actual pixels. A veto
        # opens the slot for the deeper re-search underneath
        _j = getattr(self._options, "judge", "vision")
        if _j == "d3" and self._vision is not None and chosen:
            kept = []
            for asset in chosen:
                relevant, _prob = self._vision.d3_judge(asset.caption or asset.file, text_scene)
                if relevant:
                    kept.append(asset)
            # Accept at least 1 image even if d3 rejects all
            if not kept and chosen:
                kept = [chosen[0]]
            chosen = kept[:target]
        elif self._vision is not None and chosen:
            certain = [a for a in chosen if not self._vision_policy(a.clip_score)]
            marginal = [a for a in chosen if self._vision_policy(a.clip_score)]
            visioned: list[VisualAsset] = []
            rescued: list[VisualAsset] = []
            for asset in marginal:
                info = {
                    "url": asset.url,
                    "thumburl": asset.thumb_url,
                    "title": asset.file,
                    "description": asset.caption,
                    "_clip_score": asset.clip_score,
                }
                if self._vision_vet(info, text_scene, brief.title):
                    visioned.append(asset)
                    # carry the verdict into the pack for the renderer
                    asset.vision_relevant = info.get("_vision_relevant")
                    asset.vision_confidence = info.get("_vision_confidence")
                    asset.vision_reason = info.get("_vision_reason")
                else:
                    # the model rejected it: search for something better.
                    # _deep_search FILLS the list it is handed; its fresh
                    # picks must survive the rebind below or the rescue
                    # silently evaporates
                    search_list = certain + visioned + rescued
                    before = len(search_list)
                    self._deep_search(text_scene, brief, query, search_list, used)
                    rescued.extend(search_list[before:])
            chosen = certain + visioned + rescued
            if len(chosen) > target:
                chosen = chosen[:target]
        self._timing_for(chosen, scene.narration)
        return chosen


    # -- deep search -----------------------------------------------------------
    _MIN_TOPIC_ASSETS = 3

    def _reject_summary(self, picks: list[dict]) -> list[str]:
        """Evidence about what failed, for the repair agent.

        ``picks`` are the staged-but-unshippable candidates of this run:
        every entry tells the agent (title, caption, why staged).
        """

        lines: list[str] = []
        for info in picks[:12]:
            caption = (info.get("description") or info.get("caption") or "")[:80]
            lines.append(f"- {info.get('file', '')[:60]!r} (caption: {caption!r})")
        return lines

    def _think_queries(
        self, text_scene: str, brief: ResearchBrief, tried: list[str], rejects: list[str]
    ) -> list[str]:
        """Ask the agent for fresh query ideas after weak search waves.

        The agent reasons about what the narration moment literally shows
        and how a stock photo would depict it, given everything tried so
        far, then proposes better search phrases.
        """

        if self._llm is None:
            return []
        prompt = (
            "You fix image searches that came back with irrelevant results.\n"
            f"Video topic: {brief.title}\n"
            f"Narration moment: {json.dumps(text_scene, ensure_ascii=False)}\n"
            f"Search phrases already tried: {json.dumps(tried, ensure_ascii=False)}\n"
            "Captions of the images those searches returned, all judged"
            " irrelevant or junk:\n"
            f"{chr(10).join(rejects) if rejects else '(none)'}\n\n"
            "Think about what the moment literally shows and how stock and"
            " press photographers would shoot it. Propose 3 NEW search"
            " phrases most likely to surface real photographs:\n"
            "- prefer concrete, famous, well-photographed subjects (proper"
            " nouns first: the person, object, vehicle, place, artefact)\n"
            "- never repeat a tried phrase; never search for abstract ideas\n"
            "- 1-3 catalog keywords, lowercase, no punctuation\n\n"
            'Answer JSON: {"queries": ["...", "..."]}'
        )
        messages = [{"role": "user", "content": prompt}]

        class FreshQueries(BaseModel):
            queries: list[str]

        try:
            answer = generate_json_parsed(
                self._llm,
                messages,
                FreshQueries,
                attempts=2,
                race=0,
                temperature=0.6,
                label="query repair wave",
            )
        except (BlockError, ProviderError) as exc:
            logger.warning("query repair wave failed: %s", exc)
            return []
        seen = {q.lower() for q in tried}
        fresh = []
        for query in answer.queries:
            token = query.strip().lower()
            if token and token not in seen:
                seen.add(token)
                fresh.append(token)
        return fresh[:3]

    def _deep_search(
        self,
        text_scene: str,
        brief: ResearchBrief,
        query: str,
        chosen: list[VisualAsset],
        used: _Claims,
    ) -> None:
        """Run extra search waves until we have >= 3 relevant assets."""

        # first wave: deterministic noun combos + the routed query
        nouns = sorted(content_tokens(text_scene), key=lambda w: -len(w))
        extras: list[str] = []
        if query and query.lower() not in {q.lower() for q in [brief.title] + nouns[:3]}:
            extras.append(query)
        for noun in nouns[:3]:
            combo = f"{brief.title} {noun}".strip()
            if combo.lower() not in {e.lower() for e in extras}:
                extras.append(combo)
            if noun.lower() not in {e.lower() for e in extras}:
                extras.append(noun)

        # everything staged-but-unshipped accumulates here: the evidence
        # the repair agent gets to reason over
        rejects: list[dict] = []
        for extra_q in extras:
            if len(chosen) >= self._MIN_TOPIC_ASSETS:
                break
            try:
                extra_results = self._search_ranked(extra_q, text_scene)
            except (ProviderError, BlockError) as exc:
                logger.warning("extra search %r failed: %s", extra_q, exc)
                continue
            for score, info in extra_results:
                if len(chosen) >= self._MIN_TOPIC_ASSETS:
                    break
                if score < self._options.min_score:
                    break
                if self._blocked(info, used):
                    rejects.append({**info, "why": "already in another scene"})
                    continue
                # CLIP-verify with the same floor logic as the main flow
                similarity = None
                if self._clip is not None and self._http is not None:
                    similarity, collage_like, texty = self._clip.analyze(
                        text_scene, info.get("thumburl") or info.get("url"), self._http
                    )
                    if collage_like or texty:
                        rejects.append({**info, "why": "collage or text-heavy"})
                        continue
                    if similarity is not None and similarity < _CLIP_FLOOR:
                        rejects.append({**info, "why": f"clip {similarity:.2f}"})
                        continue
                    else:
                        info["_clip_score"] = similarity
                if not self._take(info, chosen, used):
                    rejects.append({**info, "why": "claim lost the race"})

        # still short? the think wave: the agent reads the evidence (what
        # was tried and why candidates fell) and rewrites the search itself
        if len(chosen) < self._MIN_TOPIC_ASSETS:
            tried = [query, text_scene.strip(), *extras]
            rejected = self._reject_summary(rejects)
            for idea in self._think_queries(text_scene, brief, tried, rejected):
                if len(chosen) >= self._MIN_TOPIC_ASSETS:
                    break
                try:
                    idea_results = self._search_ranked(idea, text_scene)
                except (ProviderError, BlockError) as exc:
                    logger.warning("think search %r failed: %s", idea, exc)
                    continue
                for score, info in idea_results:
                    if len(chosen) >= self._MIN_TOPIC_ASSETS:
                        break
                    if score < self._options.min_score:
                        break
                    if self._blocked(info, used):
                        continue
                    similarity = None
                    if self._clip is not None and self._http is not None:
                        similarity, collage_like, texty = self._clip.analyze(
                            text_scene,
                            info.get("thumburl") or info.get("url"),
                            self._http,
                        )
                        if collage_like or texty:
                            continue
                        if similarity is not None and similarity < _CLIP_FLOOR:
                            continue
                        else:
                            info["_clip_score"] = similarity
                    self._take(info, chosen, used)

    def _timing_for(self, chosen: list[VisualAsset], narration: str) -> None:
        """Pin each visual to its narration trigger; order and stagger the rest."""

        words = re.findall(r"[a-z0-9']+", narration.lower())
        if len(chosen) <= 1:
            if chosen:
                chosen[0].position = 0.0
            return

        n = max(len(words), 1)
        fractions: list[float | None] = []
        for asset in chosen:
            trigger = re.findall(r"[a-z0-9']+", asset.trigger.lower())
            window = max(len(trigger), 1)
            best, at = 0, None
            if not trigger or window > n:
                match = None
            else:
                for start in range(n - window + 1):
                    overlap = len(trigger) - len(set(trigger) - set(words[start : start + window]))
                    if overlap > best:
                        best, at = overlap, start
                needed = (len(trigger) + 1) // 2
                match = at / n if best >= needed and at is not None else None
            fractions.append(match)

        order = sorted(
            range(len(chosen)),
            key=lambda index: (
                1.0 if fractions[index] is None else fractions[index]
            ),
        )
        for slot, index in enumerate(order):
            fraction = fractions[index]
            if fraction is None:
                fraction = slot / len(chosen)
            chosen[index].position = round(min(fraction, 1.0), 3)
