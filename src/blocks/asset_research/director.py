"""The two asset director blocks: visual intent, then queries."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from core.interfaces import LLMProvider
from core.models import (
    IntentPack,
    QueryPack,
    ResearchBrief,
    VideoScript,
)

from .generation import (
    _fallback_intents,
    _is_fiction,
    _one_scene_intents,
    _one_scene_queries,
)
from .prompts import (
    DIRECTOR_SYSTEM,
    HORROR_DIRECTOR_NOTES,
    HORROR_QUERY_NOTES,
    QUERY_SYSTEM,
)
from .quality import _repair_queries, _repair_triggers

logger = logging.getLogger(__name__)


class VisualDirectorBlock:
    """Agent 1: script narration -> per-scene visual subjects."""

    name = "visual_direction"

    def __init__(self, llm: LLMProvider, race: int = 0) -> None:
        self._llm = llm
        self._race = race

    def run(self, theory, script: VideoScript) -> IntentPack:
        payload = [
            {
                "scene_index": -1,
                "visual": "(hook line)",
                "narration": script.hook,
            },
            *[
                {
                    "scene_index": index,
                    "visual": scene.visual,
                    "narration": scene.narration,
                }
                for index, scene in enumerate(script.scenes)
            ],
        ]
        system = DIRECTOR_SYSTEM + (HORROR_DIRECTOR_NOTES if _is_fiction(script) else "")
        narrations = {
            item["scene_index"]: f"{item['narration']} {item['visual']}"
            for item in payload
        }
        jobs = [
            (
                item["scene_index"],
                item["narration"],
                item["visual"],
            )
            for item in payload
        ]
        workers = max(1, min(len(jobs), self._race or 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _one_scene_intents,
                    self._llm,
                    system,
                    theory.name,
                    index,
                    narration,
                    visual,
                    narrations,
                )
                for index, narration, visual in jobs
            ]
            scenes = []
            for future in futures:
                try:
                    scenes.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("visual direction failed: %s", exc)
                    scenes.append(None)
        # filter out subjects that are visual forms or filler
        from blocks.asset_research.quality import _STOPWORDS_GATE, _VISUAL_FORM_WORDS
        _BAD_SUBJECTS = _VISUAL_FORM_WORDS | {"according", "claims", "believers", "theory", "story"}
        filtered_scenes = []
        for scene in scenes:
            if scene is None or not getattr(scene, "subjects", None):
                filtered_scenes.append(_fallback_intents(script, 0))
                continue
            good_subjects = [
                subj for subj in scene.subjects
                if not any(w in _BAD_SUBJECTS for w in subj.subject.lower().split())
                and any(
                    w not in _STOPWORDS_GATE and len(w) >= 3
                    for w in subj.subject.lower().split()
                )
            ]
            if not good_subjects:
                filtered_scenes.append(_fallback_intents(script, 0))
            else:
                filtered_scenes.append(scene.model_copy(update={"subjects": good_subjects}))
        scenes = filtered_scenes
        return IntentPack(
            slug=theory.slug,
            topic=theory.name,
            scenes=_repair_triggers(scenes, script),
            fiction=_is_fiction(script),
        )


class QueryBuilderBlock:
    """Agent 2: subjects -> keywords + source routing (commons/nasa/openverse)."""

    name = "query_routing"

    def __init__(self, llm: LLMProvider, race: int = 0) -> None:
        self._llm = llm
        self._race = race

    def run(self, brief: ResearchBrief, intents: IntentPack) -> QueryPack:
        system = QUERY_SYSTEM + (HORROR_QUERY_NOTES if intents.fiction else "")
        scene_text = {
            scene.scene_index: " ".join(
                s.intent for s in scene.subjects
            )
            for scene in intents.scenes
        }
        workers = max(1, min(len(intents.scenes), self._race or 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _one_scene_queries,
                    self._llm,
                    system,
                    brief.title,
                    brief.one_line,
                    intent,
                    scene_text,
                )
                for intent in intents.scenes
            ]
            scenes = []
            for future in futures:
                try:
                    scenes.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("query routing failed: %s", exc)
                    scenes.append(None)
        return QueryPack(
            slug=intents.slug,
            topic=intents.topic,
            scenes=_repair_queries([sc for sc in scenes if sc is not None]),
            fiction=intents.fiction,
        )
