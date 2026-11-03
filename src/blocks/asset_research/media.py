"""Candidate metadata: cleaning, ranking and mechanical rejection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from core.models import (
    VisualAsset,
)
from utils.text import strip_html

logger = logging.getLogger(__name__)


_MAGICAL_WORDS = (
    "a an and are as at be but by during for from graphic his her into is it "
    "its like near no not of on or over photograph photo picture scene shot "
    "showing shows still the their them then there this to under up view "
    "while with within without"
    " close closeup aerial film poster overlay overlaid blend blended "
    "montage intercut beside night dawn dusk morning evening washes rendering "
    "dramatized"
)

_STOPWORDS = frozenset(_MAGICAL_WORDS.split())


_ALLOWED_MIME = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
        # motion footage: used when a topical clip exists on Commons
        "video/webm",
        "video/ogg",
        "video/mp4",
    }
)

_WORD_SPLIT = re.compile(r"[^A-Za-z0-9']+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")
_EDGE_PUNCT = ".,:;!?\"'“”‘’()[]"

# Below this CLIP similarity a thumbnail does not depict the scene's
# subject well enough to appear — dark space shots score ~0.2+, a railway
# sign that matched some metric noun scores ~0.17, junk scores ~0.0.
_CLIP_FLOOR = 0.18
# picks below this CLIP score are not confident enough to trust blindly;
# the vision model gets the final word on them
_CONFIDENT_CLIP = 0.30


def content_tokens(text: str) -> set[str]:
    """Lowercased content words of ``text`` (stopwords and single letters out)."""

    text = text.replace("‑", "-")
    words = _WORD_SPLIT.split(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def proper_phrases(text: str) -> list[str]:
    """Capitalized phrases of ``text``, best first (e.g. ``['Chichén Itzá']``).

    A phrase is dropped when all its tokens are stopwords, and phrases
    are ranked so multi-word real subjects beat bare single words. No
    word-level bans: whether ``Archive`` yields shelves or the story's
    subject is a judgment for the chooser and vision judge, encoded in
    the query, not in a hardcoded list.
    """

    text = text.replace("‑", "-")
    phrases: list[str] = []
    chunk: list[str] = []
    for word in text.split():
        cleaned = word.strip(_EDGE_PUNCT)
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned[:1].isupper():
            chunk.append(cleaned)
        elif chunk:
            phrases.append(" ".join(chunk))
            chunk = []
    if chunk:
        phrases.append(" ".join(chunk))

    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    for phrase in phrases:
        phrase = phrase.strip(_EDGE_PUNCT)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        tokens = {w for w in _WORD_SPLIT.split(phrase.lower()) if w and len(w) > 1}
        strong = tokens - _STOPWORDS
        if not strong:
            continue
        scored.append((len(strong), sum(len(w) for w in strong), phrase))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [phrase for _, _, phrase in scored]


def is_media_junk(file_title: str, mime: str, width: int | None) -> bool:
    """Mechanical unusability only: mime format and resolution floor.

    Content judgments (UI chrome, text-images, posters, diagrams) are
    never title-keyword bans -- the chooser editor and the vision judge
    decide what the image actually shows.
    """

    if mime not in _ALLOWED_MIME:
        return True
    return bool(mime.startswith("image/") and (width is None or width < 400))


def is_video_junk(info: dict) -> bool:
    """Whether a motion candidate is too heavy, too long, or too tiny.

    Videos ride the same topical filters as stills; this adds only the
    physical caps: below 4s (or beyond 90) and above 60MB they are more
    download than card.
    """

    size = info.get("size") or 0
    duration = info.get("duration") or 0.0
    width = info.get("width") or 0
    if size <= 0 or size > _VIDEO_MAX_BYTES:
        return True
    if not (4.0 <= float(duration) <= 90.0):
        return True
    return width < 400


_VIDEO_MAX_BYTES = 60_000_000
# motion clips are shelved for now: the gate below rejects every video
# candidate while this is zero, but the whole selection + render path
# stays wired so it can be switched back on by flipping this to 2
_VIDEO_SLOTS_PER_VIDEO = 0
_VIDEO_BONUS = 0.035


def _stem_of(file_title: str) -> str:
    """Normalised title without extension, so png/tiff twins collapse."""

    base = file_title.rsplit(".", 1)[0].lower()
    return " ".join(sorted(_WORD_SPLIT.split(base)))

def media_text(info: dict) -> set[str]:
    """Content words for scoring, from the file title + cleaned caption."""

    combined = _CAMEL_SPLIT.sub(" ", f"{info['file']} {info.get('description') or ''}")
    return content_tokens(combined)


def _variants(token: str) -> set[str]:
    """Match simple stem variants, mainly "Maya" vs "Mayan"."""

    return {token, token[:-1]} if token.endswith("n") and len(token) > 4 else {token}


def score_media(info: dict, query: str, scene: str) -> int:
    """Strict topical score; a single-word query needs 2+ scene-word hits."""

    pool = media_text(info)
    if not pool:
        return 0
    pool_variants = {variant for token in pool for variant in _variants(token)}
    query_hit = {
        token for token in content_tokens(query) if _variants(token) & pool_variants
    }
    scene_hit = {
        token for token in content_tokens(scene) if _variants(token) & pool_variants
    }
    if not query_hit or not scene_hit:
        return 0
    min_scene = 1 if len(query_hit) > 1 else 2
    if len(scene_hit) < min_scene:
        return 0
    return 3 * len(query_hit) + len(scene_hit) if query_hit or scene_hit else 0


def info_kind(info: dict) -> str:
    """Return ``"video"`` or ``"image"`` based on the mime type."""

    return "video" if info.get("mime", "").startswith("video/") else "image"


def to_asset(info: dict) -> VisualAsset:
    """Build a :class:`VisualAsset` from raw API info."""

    artist = info.get("artist")
    return VisualAsset(
        kind=info_kind(info),  # type: ignore[arg-type]
        file=info["file"],
        url=info["url"],
        thumb_url=info.get("thumburl"),
        page_url=info.get("page_url", ""),
        caption=strip_html(info.get("description") or ""),
        license=info.get("license") or "",
        artist=strip_html(artist) if artist else None,
        width=info.get("width"),
        height=info.get("height"),
        duration=info.get("duration"),
        clip_score=info.get("_clip_score"),
        vision_relevant=info.get("_vision_relevant"),
        vision_confidence=info.get("_vision_confidence"),
        vision_reason=info.get("_vision_reason"),
    )

@dataclass(frozen=True)
class AssetOptions:
    """Tuning knobs for :class:`AssetResearchBlock`."""

    assets_max: int = 3
    judge: str = "d3"
    words_per_asset: int = 18
    search_limit: int = 12
    nasa_limit: int = 6
    entity_limit: int = 5
    thumb_width: int = 1280
    min_score: int = 3
