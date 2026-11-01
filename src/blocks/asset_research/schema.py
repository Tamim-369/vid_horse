"""Strict Pydantic shapes + quote clamping for small-model output."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


_QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019`"
_SPLIT_MERGED = re.compile(r"[\"\u201c\u201d]\s*[,;]\s*[\"\u201c\u201d]")


def _clamp(value: str, *, words: int) -> str:
    """Strip quote debris and clamp a model-written field to ``words``."""

    text = value.replace("\\n", " ")
    for char in _QUOTE_CHARS:
        text = text.replace(char, "")
    text = re.sub(r"\s+", " ", text).strip(" ,;:.-")
    text = re.sub(r"^(subject|intent|visual|query|trigger)\s*:\s*", "", text, flags=re.I)
    return " ".join(text.split()[:words])


class StrictSubject(BaseModel):
    """Subject shape the small model fills without optional-field drift."""

    subject: str
    intent: str
    trigger: str = ""


class StrictSceneIntents(BaseModel):
    """One scene of visual direction (all fields required)."""

    scene_index: int
    subjects: list[StrictSubject]

    @model_validator(mode="after")
    def _normalize(self) -> StrictSceneIntents:
        """Split subjects the model merged into one quoted string."""

        cleaned: list[StrictSubject] = []
        for item in self.subjects:
            intent = _clamp(item.intent, words=12)
            trigger = _clamp(item.trigger, words=6)
            parts = _SPLIT_MERGED.split(item.subject) or [item.subject]
            for part in parts:
                name = _clamp(part, words=6)
                if name:
                    cleaned.append(
                        StrictSubject(subject=name, intent=intent, trigger=trigger)
                    )
        self.subjects = cleaned[:3]
        return self


class StrictQuery(BaseModel):
    """One routed search (all fields required)."""

    intent: str
    source: str
    query: str
    trigger: str = ""


class StrictSceneQueries(BaseModel):
    """One scene of routed searches (all fields required)."""

    scene_index: int
    queries: list[StrictQuery]

    @model_validator(mode="after")
    def _normalize(self) -> StrictSceneQueries:
        """Clean query text and coerce the source to a legal value."""

        legal = {"commons", "nasa", "openverse"}
        fixed: list[StrictQuery] = []
        for item in self.queries:
            query = _clamp(item.query, words=6)
            if not query:
                continue
            source = item.source.strip().lower()
            fixed.append(
                StrictQuery(
                    intent=_clamp(item.intent, words=12),
                    source=source if source in legal else "commons",
                    query=query,
                    trigger=_clamp(item.trigger, words=6),
                )
            )
        self.queries = fixed
        return self
