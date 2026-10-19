"""Domain models shared across the pipeline.

Two families live here:

- ``StrictModel`` is the base for models produced *by* the language model;
  rejecting unknown fields turns silent schema drift into a loud, repairable
  error.
- The ``Theory*`` models describe the local conspiracy-theory catalog. They
  store only a topic, a pointer (URL) to where its detail lives, and cheap
  metadata (size, popularity) — never the article text itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PopularityTier = Literal["mainstream", "popular", "moderate", "underground", "unknown"]
ScriptFormat = Literal["short", "long"]

_LEAKED_KEYS = (
    '"narration"',
    '"visual"',
    '"seconds"',
    '"scenes"',
    '"hook"',
    '"outro"',
    '"one_line"',
    '"origin"',
    '"background"',
    '"claims"',
    '"why_compelling"',
    '"timeline"',
    '"key_people"',
    '"key_terms"',
    '"open_questions"',
    '"event"',
)


def _plain_prose(value: str) -> str:
    """Reject values where the model echoed JSON structure instead of prose."""

    normalized = (
        value.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    lowered = normalized.lower()
    if "{" in normalized or "}" in normalized or any(
        key in lowered for key in _LEAKED_KEYS
    ):
        raise ValueError("must be plain prose, without JSON keys or braces")
    return value


class StrictModel(BaseModel):
    """Base model for structured LLM outputs."""

    model_config = ConfigDict(extra="forbid")


class TheoryMetrics(BaseModel):
    """Cheap, source-derived size/interest signals for a theory."""

    size_bytes: int | None = Field(default=None, description="Article length in bytes.")
    language_count: int | None = Field(
        default=None, description="Number of languages with an article."
    )
    avg_monthly_views: float | None = Field(
        default=None, description="Average monthly pageviews."
    )
    word_count: int | None = Field(default=None)
    section_count: int | None = Field(default=None)
    reference_count: int | None = Field(default=None)


class TheorySource(BaseModel):
    """Provenance: where a theory was found and where its detail lives."""

    source: str = Field(description="Connector id, e.g. 'enwiki' or 'wikidata'.")
    url: str = Field(description="URL giving the detailed write-up.")
    title: str = Field(description="Title as it appears in that source.")
    retrieved_at: datetime


class SourceTheory(BaseModel):
    """A single raw discovery hit from a connector."""

    source: str
    name: str
    url: str
    wikidata_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    metrics: TheoryMetrics = Field(default_factory=TheoryMetrics)


class Theory(BaseModel):
    """A canonical, merged catalog entry."""

    id: int | None = None
    slug: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    wikidata_id: str | None = None
    theory_type: str | None = None
    popularity_tier: PopularityTier = "unknown"
    metrics: TheoryMetrics = Field(default_factory=TheoryMetrics)
    categories: list[str] = Field(default_factory=list)
    sources: list[TheorySource] = Field(default_factory=list)
    first_seen: datetime
    last_updated: datetime


class ArticleText(BaseModel):
    """Plain-text detail fetched for a theory, ready to feed the model."""

    title: str
    url: str
    text: str
    truncated: bool = Field(
        default=False, description="Whether text was cut to fit the budget."
    )


class TimelineEvent(StrictModel):
    """One dated beat in a theory's story."""

    when: str = Field(description="Year or rough date, e.g. '1947' or 'late 1980s'.")
    event: str = Field(description="One sentence describing what happened.")

    _validate_text = field_validator("when", "event")(_plain_prose)


class ResearchBrief(StrictModel):
    """Structured understanding of a theory, produced by the research block.

    The brief is descriptive: it records what the theory claims and why it is
    told, without asserting any of it as fact.
    """

    title: str
    one_line: str = Field(description="One neutral sentence stating the claim.")
    origin: str = Field(description="Where and roughly when the theory began.")
    background: str = Field(description="Two to four sentences of context.")
    claims: list[str] = Field(description="The main things the theory asserts.")
    why_compelling: list[str] = Field(
        description="Why the idea resonates or feels plausible to people."
    )
    timeline: list[TimelineEvent] = Field(default_factory=list)
    key_people: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    _validate_prose = field_validator("one_line", "origin", "background")(_plain_prose)

    @field_validator(
        "claims", "why_compelling", "key_people", "key_terms", "open_questions"
    )
    @classmethod
    def _validate_items(cls, items: list[str]) -> list[str]:
        return [_plain_prose(item) for item in items]


class ScriptScene(StrictModel):
    """A single narrated beat of a video script."""

    narration: str = Field(description="Voice-over text, spoken naturally.")
    visual: str = Field(description="What is shown on screen during the line.")
    seconds: int = Field(ge=1, le=600, description="Estimated duration.")

    _validate_text = field_validator("narration", "visual")(_plain_prose)


class VideoScript(StrictModel):
    """A scene-by-scene video script for one format."""

    title: str
    hook: str = Field(description="The opening line that grabs attention.")
    format: ScriptFormat
    target_seconds: int = Field(ge=5, le=1800)
    scenes: list[ScriptScene] = Field(min_length=1)
    outro: str = Field(description="Closing line.")
    source_notes: list[str] = Field(default_factory=list)

    _validate_hook = field_validator("hook", "outro")(_plain_prose)


class ScriptPlan(StrictModel):
    """An outline for a long script: ordered beats, one per scene."""

    title: str
    hook: str
    beats: list[str] = Field(
        min_length=3,
        max_length=20,
        description="One short sentence per scene, in order.",
    )
    outro: str

    _validate_fields = field_validator("title", "hook", "outro")(_plain_prose)

    @field_validator("beats")
    @classmethod
    def _validate_beats(cls, beats: list[str]) -> list[str]:
        return [_plain_prose(beat) for beat in beats]


AssetKind = Literal["image", "video"]


class VisualAsset(BaseModel):
    """One topical media file fetched from a MediaWiki wiki."""

    kind: AssetKind
    file: str = Field(description="File title, e.g. 'File:Chichen Itza 2011.jpg'.")
    url: str = Field(description="Full-size image/audio/video URL.")
    thumb_url: str | None = Field(default=None)
    page_url: str = Field(description="Description page URL.")
    caption: str = Field(default="", description="Cleaned description or caption.")
    license: str = Field(default="")
    artist: str | None = Field(default=None)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)


class SceneAssets(BaseModel):
    """Assets matched to one script scene."""

    scene_index: int = Field(description="0-based index into ``VideoScript.scenes``.")
    visual: str = Field(description="The scene's visual line, echoed for reference.")
    query: str = Field(default="", description="The search query used.")
    assets: list[VisualAsset] = Field(default_factory=list)


class AssetPack(BaseModel):
    """Media assets for a whole video, scene by scene."""

    slug: str = Field(description="Theory slug.")
    topic: str = Field(description="Theory name.")
    article_title: str = Field(description="Wiki article the pack was built from.")
    article_url: str = Field(default="", description="Source article URL.")
    scenes: list[SceneAssets] = Field(default_factory=list)

