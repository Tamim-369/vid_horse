"""One-scene director generation: races, quality gates, fallbacks."""

from __future__ import annotations

import json
import logging
import re

from blocks.base import generate_json_parsed
from core.errors import BlockError
from core.interfaces import LLMProvider
from core.models import (
    AssetSource,
    SceneIntents,
    SceneQueries,
    SceneQuery,
    Subject,
    VideoScript,
)

from .prompts import _DIRECTOR_OUTPUT, _QUERY_OUTPUT, _TEMPERATURE
from .quality import _intent_problems, _query_problems
from .schema import StrictSceneIntents, StrictSceneQueries

logger = logging.getLogger(__name__)


def _is_fiction(script: VideoScript) -> bool:
    return any("original horror story" in note.lower() for note in script.source_notes)


def _scene_intent_quality(
    index: int, answer: StrictSceneIntents, narrations: dict[int, str] | None = None
) -> list[str]:
    """Per-scene gate: right index, clean subjects, usable triggers."""

    problems = _intent_problems(
        [
            SceneIntents(
                scene_index=answer.scene_index,
                subjects=[
                    Subject(subject=s.subject, intent=s.intent, trigger=s.trigger)
                    for s in answer.subjects
                ],
            )
        ],
        narrations,
    )
    if answer.scene_index != index:
        problems.insert(0, f"scene_index must be {index}, got {answer.scene_index}")
    return problems


def _scene_query_quality(
    index: int, answer: StrictSceneQueries, narrations: dict[int, str] | None = None
) -> list[str]:
    """Per-scene gate: right index, clean keywords, legal source."""

    problems = _query_problems(
        [
            SceneQueries(
                scene_index=answer.scene_index,
                queries=[
                    SceneQuery(
                        intent=q.intent, source=q.source, query=q.query, trigger=q.trigger
                    )
                    for q in answer.queries
                ],
            )
        ],
        narrations,
    )
    if answer.scene_index != index:
        problems.insert(0, f"scene_index must be {index}, got {answer.scene_index}")
    return problems


def _fallback_intents(script: VideoScript, index: int) -> SceneIntents:
    """Deterministic last resort for one scene the model keeps failing."""

    narration = script.hook if index < 0 else script.scenes[index].narration
    words = [w.strip(".,;:!?\u2014\"'") for w in narration.split()]
    subject = " ".join(words[:3]) or "documentary scene"
    return SceneIntents(
        scene_index=index,
        subjects=[Subject(subject=subject[:40], intent=subject[:60], trigger="")],
    )


def _fallback_queries(intent: SceneIntents) -> SceneQueries:
    """Deterministic last resort: route each subject name as a query."""

    space = (
        "space", "planet", "star", "moon", "sun", "galaxy", "orbit",
        "nasa", "cosmos", "asteroid", "comet",
    )
    queries: list[SceneQuery] = []
    for subject in intent.subjects:
        name = " ".join(
            re.sub(r"[^a-z0-9 ]", " ", subject.subject.lower()).split()[:4]
        )
        if not name:
            continue
        source: AssetSource = (
            "nasa" if any(word in name.split() for word in space) else "commons"
        )
        queries.append(
            SceneQuery(
                intent=subject.intent,
                source=source,
                query=name,
                trigger=subject.trigger,
            )
        )
    return SceneQueries(scene_index=intent.scene_index, queries=queries)


def _one_scene_intents(
    llm: LLMProvider,
    system: str,
    theory_name: str,
    index: int,
    narration: str,
    visual: str,
    narrations: dict[int, str] | None = None,
) -> SceneIntents:
    """Direct one scene (small answer, so small models cannot truncate)."""

    payload = {"scene_index": index, "visual": visual, "narration": narration}
    messages = [
        {"role": "system", "content": system + _DIRECTOR_OUTPUT},
        {
            "role": "user",
            "content": (
                f"Theory: {theory_name}\n"
                f"Direct this ONE scene: {json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        answer = generate_json_parsed(
            llm,
            messages,
            StrictSceneIntents,
            quality=lambda parsed: _scene_intent_quality(index, parsed, narrations),
            attempts=3,
            race=0,
            temperature=_TEMPERATURE,
            label=f"scene {index} direction",
        )
    except BlockError as exc:
        logger.warning("scene %d direction fell back: %s", index, exc)
        return SceneIntents(scene_index=index, subjects=[])
    return SceneIntents(
        scene_index=index,
        subjects=[
            Subject(subject=s.subject, intent=s.intent, trigger=s.trigger)
            for s in answer.subjects
        ],
    )


def _one_scene_queries(
    llm: LLMProvider,
    system: str,
    topic: str,
    summary: str,
    intent: SceneIntents,
    narrations: dict[int, str] | None = None,
) -> SceneQueries:
    """Route one scene's subjects into searches."""

    payload = {
        "scene_index": intent.scene_index,
        "subjects": [s.model_dump() for s in intent.subjects],
    }
    messages = [
        {"role": "system", "content": system + _QUERY_OUTPUT},
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\nSummary: {summary}\n"
                f"Route this ONE scene: {json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        answer = generate_json_parsed(
            llm,
            messages,
            StrictSceneQueries,
            quality=lambda parsed: _scene_query_quality(intent.scene_index, parsed, narrations),
            attempts=3,
            race=0,
            temperature=_TEMPERATURE,
            label=f"scene {intent.scene_index} routing",
        )
    except BlockError as exc:
        logger.warning("scene %d routing fell back: %s", intent.scene_index, exc)
        return _fallback_queries(intent)
    return SceneQueries(
        scene_index=intent.scene_index,
        queries=[
            SceneQuery(intent=q.intent, source=q.source, query=q.query, trigger=q.trigger)
            for q in answer.queries
        ],
    )
