"""Asset research: match topical Commons/wiki media to script scenes.

No language model is involved: the article's own media plus Commons search
results are scored against the scene's visual line, and a scene only receives
assets that share at least one content word with it. Scenes that match nothing
stay empty — an off-topic image is never accepted.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from core.errors import BlockError, ProviderError
from core.models import (
    AssetPack,
    ResearchBrief,
    SceneAssets,
    ScriptScene,
    Theory,
    VideoScript,
    VisualAsset,
)
from providers.sources.mediawiki import MediaWikiClient
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

_JUNK_MARKERS = (
    "logo",
    "icon",
    "edit-",
    "padlock",
    "question book",
    "ambox",
    "stub",
    "letter w",
    "wikipe",
    "wikimed",
    "disambig",
    "symbol",
    "dagger",
    "arrow",
    "blank",
    "placeholder",
    "copyright",
    "public domain",
    "creative commons",
    "gnu",
    "upload",
    "sound",
    "speaker",
)

_ALLOWED_MIME = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
        "video/webm",
        "video/mp4",
    }
)

_WORD_SPLIT = re.compile(r"[^A-Za-z0-9']+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")
_EDGE_PUNCT = ".,:;!?\"'“”‘’()[]"


def content_tokens(text: str) -> set[str]:
    """Lowercased content words of ``text`` (stopwords and single letters out)."""

    text = text.replace("‑", "-")
    words = _WORD_SPLIT.split(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def proper_phrases(text: str) -> list[str]:
    """Capitalized phrases of ``text``, best first (e.g. ``['Chichén Itzá']``).

    Sentence-initial camera words (``Night``, ``Close‑up``,
    ``Film poster``...) never survive: a phrase is dropped when all its
    tokens are stopwords, and phrases are ranked so multi-word real
    subjects beat bare single words.
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
    """Whether the file is UI chrome, too small, or an unusable type."""

    lowered = file_title.lower()
    if mime not in _ALLOWED_MIME:
        return True
    if mime.startswith("image/") and (width is None or width < 400):
        return True
    return any(marker in lowered for marker in _JUNK_MARKERS)


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
    )


@dataclass(frozen=True)
class AssetOptions:
    """Tuning knobs for :class:`AssetResearchBlock`."""

    assets_per_scene: int = 4
    search_limit: int = 8
    entity_limit: int = 5
    thumb_width: int = 1280
    min_score: int = 3


class AssetResearchBlock:
    """Build an :class:`AssetPack` of topical media for every script scene."""

    name = "asset_research"

    def __init__(
        self,
        wiki: MediaWikiClient,
        commons: MediaWikiClient,
        *,
        options: AssetOptions | None = None,
    ) -> None:
        self._wiki = wiki
        self._commons = commons
        self._options = options or AssetOptions()

    def run(
        self,
        theory: Theory,
        brief: ResearchBrief,
        script: VideoScript,
    ) -> AssetPack:
        """Fetch and match media for every scene of ``script``."""

        source = self._article_source(theory)
        article_title = source.title if source is not None else theory.name
        article_url = source.url if source is not None else ""
        try:
            article_pool = [info for info in self._page_pool([article_title]) if self._usable(info)]
            entity_pool = self._entity_pool(brief)
        except ProviderError as exc:
            raise BlockError(f"Could not fetch media for '{theory.name}': {exc}") from exc

        used: set[str] = set()
        scenes: list[SceneAssets] = []
        for index, scene in enumerate(script.scenes):
            query = self._query_for(scene, brief)
            assets = self._assets_for_scene(scene, query, entity_pool, article_pool, used)
            scenes.append(
                SceneAssets(
                    scene_index=index,
                    visual=scene.visual,
                    query=query,
                    assets=assets,
                )
            )

        return AssetPack(
            slug=theory.slug,
            topic=theory.name,
            article_title=article_title,
            article_url=article_url,
            scenes=scenes,
        )

    # -- pools -----------------------------------------------------------------
    def _page_pool(self, titles: list[str]) -> list[dict]:
        """Media metadata for the files used on ``titles``."""

        files = [f for entries in self._wiki.page_files(titles).values() for f in entries]
        if not files:
            return []
        return self._wiki.image_info(files, thumb_width=self._options.thumb_width)

    def _entity_pool(self, brief: ResearchBrief) -> dict[str, VisualAsset]:
        """First usable portrait for each notable person named in the brief."""

        names = [
            name.split(",")[0] for name in brief.key_people[: self._options.entity_limit]
        ]
        if not names:
            return {}
        pool: dict[str, VisualAsset] = {}
        try:
            per_person = self._wiki.page_files(names)
            for person, files in per_person.items():
                infos = self._wiki.image_info(
                    files[:15], thumb_width=self._options.thumb_width
                )
                for info in infos:
                    if self._usable(info):
                        pool[person.lower()] = to_asset(info)
                        break
        except ProviderError as exc:
            logger.debug("Entity portraits skipped: %s", exc)
        return pool

    # -- per scene -------------------------------------------------------------
    def _assets_for_scene(
        self,
        scene: ScriptScene,
        query: str,
        entity_pool: dict[str, VisualAsset],
        article_pool: list[dict],
        used: set[str],
    ) -> list[VisualAsset]:
        chosen: list[VisualAsset] = []
        text_scene = f"{scene.visual} {scene.narration}"

        for person, asset in entity_pool.items():
            if len(chosen) >= self._options.assets_per_scene:
                break
            words = person.split()
            if all(word in text_scene.lower() for word in words) and asset.file not in used:
                chosen.append(asset)
                used.add(asset.file)

        if query:
            hits = self._commons.search_media(
                query, limit=self._options.search_limit, thumb_width=self._options.thumb_width
            )
            scored = sorted(
                (
                    (score_media(info, query, text_scene), info)
                    for info in hits
                    if self._usable(info)
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            for score, info in scored:
                if len(chosen) >= self._options.assets_per_scene or score < self._options.min_score:
                    break
                if info["file"] in used:
                    continue
                chosen.append(to_asset(info))
                used.add(info["file"])

        for info in article_pool:  # topic-guaranteed filler from the theory's own article
            if len(chosen) >= self._options.assets_per_scene:
                break
            if info["file"] in used:
                continue
            chosen.append(to_asset(info))
            used.add(info["file"])

        return chosen

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _article_source(theory: Theory):
        for source in theory.sources:
            if "wikipedia.org" in source.url:
                return source
        return theory.sources[0] if theory.sources else None

    def _usable(self, info: dict) -> bool:
        return not is_media_junk(info["file"], info.get("mime", ""), info.get("width"))

    def _query_for(self, scene: ScriptScene, brief: ResearchBrief) -> str:
        phrases = proper_phrases(scene.visual)
        if phrases:
            return phrases[0]
        nouns = sorted(content_tokens(scene.visual), key=lambda w: -len(w))
        if not nouns:
            return ""
        return f"{brief.title} {nouns[0]}".strip()
