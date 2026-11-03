"""Search routing: routed queries across the source catalogue."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from core.models import ResearchBrief, ScriptScene, VisualAsset
from utils import computing

from .claims import _Claims
from .media import (
    _CLIP_FLOOR,
    _STOPWORDS,
    _VIDEO_BONUS,
    _variants,
    content_tokens,
    info_kind,
    proper_phrases,
    score_media,
)

logger = logging.getLogger(__name__)


class RoutingMixin:
    """Mixin: search routing, attached to AssetResearchBlock."""

    def _search_ranked(self, query: str, scene: str) -> list[tuple[int, dict]]:
        """Commons search for ``query`` sorted by strict topical score.

        Fiction scripts bonus dark frames: a candidate's thumbnail luminance
        can add up to +2 to its score.
        """

        hits = self._commons.search_media(
            query, limit=self._options.search_limit, thumb_width=self._options.thumb_width
        )
        scored: list[tuple[float, dict]] = []
        for info in hits:
            if not self._usable(info):
                continue
            score = float(score_media(info, query, scene))
            if info_kind(info) == "video":
                # a real motion clip beats a pan over a still when either
                # passes, but never so much that a weaker clip outranks a
                # clearly better photo
                score += _VIDEO_BONUS
            if self._fiction:
                light = self._luminance(info)
                if light is not None:
                    score += min(max(0.0, (110 - light) / 30), 2.0)
            scored.append((score, info))
        return sorted(scored, key=lambda pair: pair[0], reverse=True)

    def _route_query(self, action, scene, chosen, target, used) -> None:
        """One directed search: nasa, openverse or commons.

        ``action.query`` is what we SEARCH with (the agent crafted it for the
        catalogue); ``intent`` only joins the topical scoring gate — long
        search strings dilute Commons's ranked search.
        """

        search_words = action.query
        relevance = f"{action.intent} {search_words}".strip()

        if action.source in ("nasa", "openverse"):
            self._route_direct(action.source, relevance, search_words, chosen, target, used)
            return

        # Commons: the agent's query first, then richer variants, ranked
        # together as one pool.
        variants: list[str] = [search_words] if search_words.strip() else []
        strong = [
            (index, word)
            for index, word in enumerate(search_words.split())
            if word.lower() not in _STOPWORDS
        ]
        compact = " ".join(word for _, word in strong)
        if compact and compact.lower() not in {known.lower() for known in variants}:
            variants.append(compact)
        ultimate = sorted(strong, key=lambda item: -len(item[1]))[:2]
        if len(ultimate) == 2:
            tight = " ".join(word for _, word in sorted(ultimate))
            if tight.lower() not in {known.lower() for known in variants}:
                variants.append(tight)

        ranked: list[tuple[int, dict]] = []
        seen: set[str] = set()
        if len(variants) > 1:
            with ThreadPoolExecutor(
                max_workers=min(computing.variant_workers(), len(variants))
            ) as pool:
                variant_lists = list(
                    pool.map(lambda v: self._search_ranked(v, scene), variants)
                )
        else:
            variant_lists = [self._search_ranked(variants[0], scene)] if variants else []
        for variant_hits in variant_lists:
            for score, info in variant_hits:
                if info["file"] not in seen:
                    seen.add(info["file"])
                    ranked.append((score, info))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        eligible: list[tuple[int, dict]] = []
        for score, info in ranked:
            if len(eligible) >= self._chooser_pool:
                break
            if len(chosen) + len(eligible) >= target or score < self._options.min_score:
                break
            if self._blocked(info, used):
                continue
            description = (info.get("description") or "").strip()
            if not description and len(info.get("file", "").split()) > 2:
                # nothing to verify identity against: a name match in the
                # file title alone has already shipped wrong people
                continue
            eligible.append((score, info))

        # d3 single-judge path: no CLIP, no chooser LLM. Take the top
        # unclaimed candidates by source score; the caption judge filters
        # them afterwards.
        for _score, info in eligible:
            if self._blocked(info, used):
                continue
            if self._take(info, chosen, used):
                break

    def _route_shelf(
        self, query: str, scene: str, chosen: list[VisualAsset], target: int, used: _Claims
    ) -> None:
        """Local shelf match — the fast path for subjects research has seen."""

        _j = getattr(self._options, "judge", "vision")
        if _j == "d3" or len(chosen) >= target or self._clip is None or self._http is None:
            return
        text_embed = self._clip.text_embed(query)
        if text_embed is None:
            return
        hits = self._shelf.match(
            text_embed, limit=max(target - len(chosen), 6)
        )
        if not hits:
            return
        logger.info("Shelf matched %d candidate(s) for %r", len(hits), query[:50])
        for info in hits:
            if len(chosen) >= target:
                break
            if self._blocked(info, used) or not self._usable(info):
                continue
            # same contract the network path faces: pixels must depict
            # the scene, nothing collages/postery/texty survives
            sim, collage_like, texty = self._clip.analyze(
                scene, info.get("thumburl") or info.get("url"), self._http
            )
            if collage_like or texty:
                continue
            info["_clip_score"] = sim
            if sim is not None and sim < _CLIP_FLOOR:
                continue
            if self._take(info, chosen, used):
                self._shelf.touch(info["file"])

    def _route_web(
        self, query: str, scene: str, chosen: list[VisualAsset], target: int, used: _Claims
    ) -> None:
        """Scraped-engine fallback: fires when the RSS sources starve a scene.

        Engine results carry no machine-readable license and cost one
        HTTP trip per candidate, so they run only here — after commons,
        direct sources (nasa/openverse) and the scene-phrase mining all
        came up empty. The same CLIP floor applies: the pixels must
        depict the named subject; a score below the floor is a news-
        channel opener with nothing to say.
        """

        _j = getattr(self._options, "judge", "vision")
        if _j == "d3" or self._clip is None or self._http is None or not query.strip():
            return

        candidates: list[dict] = []
        try:
            _web_hits = list(self._web.search(query, limit=self._options.search_limit))
        except Exception:  # noqa: BLE001
            _web_hits = []
        for info in _web_hits:
            if not self._usable(info) or self._blocked(info, used):
                continue
            candidates.append(info)
            if len(candidates) >= self._chooser_pool:
                break
        if not candidates:
            return

        if getattr(self._options, "judge", "vision") == "d3":
            for info in candidates:
                if self._blocked(info, used):
                    continue
                if self._take(info, chosen, used):
                    break
            return
        subject_text = query
        clip_scored: list[tuple[float, int, dict]] = []
        with ThreadPoolExecutor(max_workers=computing.clip_workers()) as pool:
            results = list(
                pool.map(
                    lambda info: self._clip.analyze(
                        subject_text,
                        info.get("thumburl") or info.get("url"),
                        self._http,
                    ),
                    candidates,
                )
            )
        for index, (similarity, collage_like, texty) in enumerate(results):
            if collage_like or texty:
                continue
            if similarity is not None and similarity < _CLIP_FLOOR:
                continue
            if similarity is None:
                similarity = -1.0 + 0.001 * index
            else:
                info = candidates[index]
                info["_clip_score"] = similarity
            clip_scored.append((similarity, index, candidates[index]))
        clip_scored.sort(key=lambda pair: (-pair[0], pair[1]))
        if not clip_scored:
            return

        pick = self._choose(
            scene,
            query,
            [(0.0, info) for _sim, _idx, info in clip_scored],
            confirm_only=True,
        )
        if pick is None:
            return
        self._take(pick, chosen, used)

    def _route_yandex(
        self, query: str, scene: str, chosen: list[VisualAsset], target: int, used: _Claims
    ) -> None:
        if getattr(self._options, "judge", "vision") == "d3":
            return
        """Yandex Images search, CLIP-ranked, vision-verified picks.

        Yandex has no per-query rate limit, so it is the unlimited source
        for scenes the curated sources starve. Candidates are CLIP-scored
        first (cheap); the top pick is then verified by the local vision
        model (qwen3-vl:2b, one slow call) — a rejected pick falls through
        to the next CLIP-ranked candidate.
        """

        if len(chosen) >= target or self._yandex is None or not query.strip():
            return
        try:
            results = self._yandex.search(query, limit=self._options.search_limit)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Yandex search failed: %s", exc)
            return
        if not results:
            return

        candidates: list[dict] = []
        for result in results:
            if len(candidates) >= self._chooser_pool * 2:
                break
            mime = "image/webp" if result.url.lower().endswith(".webp") else (
                "image/png" if result.url.lower().endswith(".png") else "image/jpeg"
            )
            info = {
                "file": result.url,
                "url": result.url,
                "thumburl": result.thumb_url,
                "description": result.caption,
                "source": "yandex",
                "mime": mime,
                "width": result.width,
                "height": result.height,
            }
            if not self._usable(info) or self._blocked(info, used):
                continue
            candidates.append(info)
        if not candidates:
            return

        # d3 mode: skip CLIP/vision/chooser, take first unclaimed.
        if getattr(self._options, "judge", "vision") == "d3":
            for info in candidates:
                if self._blocked(info, used):
                    continue
                if self._take(info, chosen, used):
                    break
            return
        # CLIP first: cheap, catches off-topic pixels and collage/poster junk.
        ranked: list[tuple[float, dict]] = []
        if self._clip is not None and self._http is not None:
            with ThreadPoolExecutor(max_workers=computing.clip_workers()) as pool:
                analyzed = list(
                    pool.map(
                        lambda info: self._clip.analyze(
                            query,
                            info.get("thumburl") or info.get("url"),
                            self._http,
                        ),
                        candidates,
                    )
                )
            for (similarity, collage_like, texty), info in zip(analyzed, candidates, strict=True):
                if collage_like or texty:
                    continue
                if similarity is not None and similarity < _CLIP_FLOOR:
                    continue
                info["_clip_score"] = similarity if similarity is not None else 0.0
                ranked.append((similarity if similarity is not None else 0.0, info))
            ranked.sort(key=lambda pair: -pair[0])
        else:
            ranked = [(0.0, info) for info in candidates]

        # Vision-verify picks, falling to the next candidate on a reject.
        while ranked and len(chosen) < target:
            pick = self._choose(
                scene,
                query,
                [(_score, info) for _score, info in ranked],
                confirm_only=True,
            )
            if pick is None or (
                self._vision_policy(pick) and not self._vision_vet(
                    pick, scene, self._run_title
                )
            ):
                ranked.pop(0)
                continue
            if self._take(pick, chosen, used):
                ranked.pop(0)
            else:
                ranked.pop(0)

    def _route_direct(
        self, source, relevance, search_words, chosen, target, used
    ) -> None:
        """A straight-to-source search: acceptable results are taken as given."""

        if source == "nasa":
            search = self._nasa.search_images if self._nasa else None
        else:
            search = self._openverse.search if self._openverse else None
        if search is None:
            return
        limit = self._options.nasa_limit if source == "nasa" else self._options.search_limit
        tokens = {
            variant
            for token in content_tokens(relevance)
            for variant in _variants(token)
        }
        bar = max(2, len(tokens) - 2)
        passing: list[dict] = []
        try:
            hits = list(search(search_words, limit=limit))
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s search failed: %s", source, exc)
            hits = []
        for info in hits:
            if len(passing) >= target * 2:
                break
            if not self._usable(info) or self._blocked(info, used):
                continue
            text = {
                variant
                for token in content_tokens(info["file"])
                for variant in _variants(token)
            }
            # NASA's catalogue is source-scoped already; openverse titles lie
            # too easily, so they must share more than one token.
            if source != "nasa" and len(tokens & text) < bar:
                continue
            passing.append(info)
        _j = getattr(self._options, "judge", "vision")
        if _j != "d3" and passing and self._clip is not None and self._http is not None:
            subject_text = f"{relevance} {search_words}".strip()
            clip_scored: list[tuple[float, int, dict]] = []
            with ThreadPoolExecutor(max_workers=computing.clip_workers()) as pool:
                results = list(
                    pool.map(
                        lambda info: self._clip.analyze(
                            subject_text,
                            info.get("thumburl") or info.get("url"),
                            self._http,
                        ),
                        passing,
                    )
                )
            for index, (similarity, collage_like, texty) in enumerate(results):
                if collage_like or texty:
                    # multi-panel sheets read as clutter; text posters
                    # violate the no-text-as-image rule
                    continue
                if similarity is not None and similarity < _CLIP_FLOOR:
                    # pixels that don't depict the subject at all: a metric
                    # noun matching "MPC" must not become a bridge sign
                    continue
                if similarity is None:
                    similarity = -1.0 + 0.001 * index
                else:
                    passing[index]["_clip_score"] = similarity
                clip_scored.append((similarity, index, passing[index]))
            clip_scored.sort(key=lambda pair: (-pair[0], pair[1]))
            for _sim, _idx, info in clip_scored:
                if len(chosen) >= target:
                    break
                self._take(info, chosen, used)
        else:
            for info in passing:
                if len(chosen) >= target:
                    break
                self._take(info, chosen, used)

    # -- helpers ---------------------------------------------------------------

    def _query_for(self, scene: ScriptScene, brief: ResearchBrief) -> str:
        phrases = proper_phrases(scene.narration)
        if phrases:
            return phrases[0]
        nouns = sorted(
            content_tokens(scene.narration),
            key=lambda w: -len(w),
        )
        if not nouns:
            return brief.title
        return f"{brief.title} {nouns[0]}".strip()
