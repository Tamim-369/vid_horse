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
    genre: str = Field(
        default="",
        description='Presentation style: "" (conspiracy) or "cosmic" (space facts).',
    )
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


class SfxCue(StrictModel):
    """One sound effect placed on the video timeline."""

    scene_index: int = Field(ge=0, description="Scene the effect belongs to.")
    effect: str = Field(description="Effect keyword, e.g. 'creak' or 'impact'.")
    at: float = Field(ge=0, le=60, description="Offset in seconds within the scene.")
    gain: float = Field(default=0.5, ge=0.1, le=1.2, description="Volume 0-1-ish.")


class SfxPlan(StrictModel):
    """Sound design for a short video: at most a handful of effects."""

    cues: list[SfxCue] = Field(max_length=8)


class MusicPlan(StrictModel):
    """Role-labelled music choice: one base bed, rare per-scene overrides.

    Roles resolve through the rack table in ``assets/music/music.json`` —
    threat, big-event, mystery, discovery, vast-reveal, dread-mission and
    the neutral default. ``scene_roles`` keys are scene indexes (as stored
    in JSON, string-y) whose content demands a different track than the
    base for that scene's duration.
    """

    base_role: str = "default"
    scene_roles: dict[str, str] = Field(default_factory=dict)


class HorrorStylePack(StrictModel):
    """Patterns mined from a corpus of horror stories — writer's ammunition."""

    premises: list[str] = Field(
        description="Story premises that recur in the corpus, one line each."
    )
    hook_patterns: list[str] = Field(
        description="Opening-line shapes that pull the reader in immediately."
    )
    twists: list[str] = Field(
        description="Ending twists that the corpus uses again and again."
    )
    techniques: list[str] = Field(
        description="Pacing and voice techniques worth imitating."
    )

    @field_validator(
        "premises", "hook_patterns", "twists", "techniques"
    )
    @classmethod
    def _validate_lists(cls, items: list[str]) -> list[str]:
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
    music: MusicPlan | None = Field(
        default=None,
        description="Role-based music plan; absent means legacy genre routing.",
    )

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
    trigger: str = Field(default="", description="Narration phrase that aires it.")
    position: float = Field(
        default=0.0,
        description="0-1 scene fraction when this visual starts showing.",
    )
    clip_score: float | None = Field(
        default=None,
        description="CLIP similarity to the scene subject; what the shot depicts.",
    )


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


AssetSource = Literal["commons", "nasa", "openverse"]


class Subject(BaseModel):
    """One visual subject the director wants to see in a scene."""

    subject: str = Field(description="2-4 word name of the thing to show.")
    intent: str = Field(description="Concrete image description, max 9 words.")
    trigger: str = Field(
        default="",
        description="Short phrase copied from the narration that fires it.",
    )


class SceneIntents(BaseModel):
    """The director's image intents for one scene."""

    scene_index: int = Field(description="0-based script scene index.")
    subjects: list[Subject] = Field(
        default_factory=list, description="One per distinct subject, in speech order."
    )


class IntentPack(BaseModel):
    """Visual direction for a whole script, scene by scene."""

    slug: str = Field(description="Theory slug.")
    topic: str = Field(description="Theory name.")
    scenes: list[SceneIntents] = Field(default_factory=list)
    fiction: bool = Field(
        default=False,
        description="True when the script is an original horror story, not a documentary.",
    )


class SceneQuery(BaseModel):
    """One directed image search: intent, chosen source, and the keywords."""

    intent: str = Field(description="The image intent being searchd for.")
    source: AssetSource = Field(description="Where to search: commons|nasa|openverse.")
    query: str = Field(description="Search keywords, no punctuation.")
    trigger: str = Field(default="", description="Narration phrase that fires this image.")


class SceneQueries(BaseModel):
    """Directed searches for one scene."""

    scene_index: int = Field(description="0-based script scene index.")
    queries: list[SceneQuery] = Field(default_factory=list)


class QueryPack(BaseModel):
    """Source-routed search queries for a whole script, scene by scene."""

    slug: str = Field(description="Theory slug.")
    topic: str = Field(description="Theory name.")
    scenes: list[SceneQueries] = Field(default_factory=list)
    fiction: bool = Field(default=False)

