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
            scenes = [future.result() for future in futures]
        scenes = [
            scene if scene.subjects else _fallback_intents(script, scene.scene_index)
            for scene in scenes
        ]
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
            scenes = [future.result() for future in futures]
        return QueryPack(
            slug=intents.slug,
            topic=intents.topic,
            scenes=_repair_queries(scenes),
            fiction=intents.fiction,
        )
