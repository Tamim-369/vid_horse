"""Asset research: match topical Commons/wiki media to script scenes.

No language model is involved in the fetch: the article's own media plus
Commons-search results are scored against the scene, a scene only receives
assets that share content words with it, and every asset is timed to the
narration phrase that triggered it. Scenes that match nothing stay empty —
an off-topic image is never accepted.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel

from core.errors import BlockError, ProviderError
from core.models import (
    AssetPack,
    QueryPack,
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

_CAMERA_WORDS = frozenset({
    "archive", "archives", "footage", "monitor", "monitors",
    "timeline", "catalog", "catalogs", "catalogue", "catalogues",
})

CHOOSER_SYSTEM = """You are the image editor who makes the final pick for
one narration moment of a video. One image is selected from the candidates
by reading the file titles and descriptions alone.

Pick the candidate that actually shows the SUBJECT OF THE QUERY as part of
the narration moment. Reject everything else. REJECT (empty file) when:
- the image shows a DIFFERENT thing than the query subject — especially:
  for a person query where the description does not clearly identify the
  SAME person (right name AND era/profession, not just a shared name),
  or 'no caption' portraits whose identity cannot be verified;
- a group of people, a conference, a portrait of somebody else, when the
  subject is a thing these people invented, used or worked on;
- the image is mostly TEXT — poster, cover, infographic, diagram, chart,
  comic, screenshot, map, scanned page, award banner;
- it is a grave, tomb or cemetery shot when the subject is a living
  contribution, or any image that would look silly or tone-breaking on
  a serious narration channel (gags, parody, tongue-out poses);
- the connection is only a shared word, not the same referent.

When several qualify, prefer the one showing the subject itself, clearly,
in a photograph or faithful artistic depiction, with no text overlay.
"""

class PickSchema(BaseModel):
    """The chooser's verdict: one file title, or "" to reject all."""

    file: str = ""


_JUNK_PHRASES = (
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
    "diagram",
    "graph of",
    "chart",
    "infographic",
    "schematic",
    "orbit of",
    "orbit diagram",
    "elliptical orbit",
    "trajectory",
    "orbit animation",
    "text in this image",
    "transcribed and included",
    "replace this template",
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

# Word-boundary bans: files that are mostly *text* (posters, covers, scans,
# cartoons, diagrams) or otherwise unreadable as footage.
_JUNK_WORDS = frozenset(
    {
        "poster",
        "posters",
        "cover",
        "covers",
        "bookcover",
        "book",
        "meme",
        "screenshot",
        "screencap",
        "tongue",
        "caricature",
        "cartoon",
        "comics",
        "grave",
        "graves",
        "graveyard",
        "cemetery",
        "tomb",
        "tombstone",
        "tombstones",
        "burial",
        "diagram",
        "chart",
        "charts",
        "graph",
        "infographic",
        "banner",
        "titlecard",
        "intertitle",
        "caption",
        "codex",
        "codices",
        "manuscript",
        "manuscripts",
        "folio",
        "palimpsest",
        "papyrus",
        "cartoons",
        "comic",
        "caricatures",
        "drawing",
        "sketch",
        "sketches",
        "timetable",
        "document",
        "documents",
        "price",
        "sheet",
        "movie",
        "movies",
        "film",
        "films",
        "premiere",
        "mtv",
        "award",
        "awards",
        "celebrity",
        "celebrities",
        "hollywood",
        "actor",
        "actress",
        "screensaver",
        "wallpaper",
        "headlines",
        "headline",
        "newspaper",
        "news",
        "sentiment",
        "statistics",
        "tables",
        "table",
        "frontpage",
        "magazine",
        "coverpage",
        "typography",
        "map",
        "atlas",
        "chartography",
        "planispheres",
    }
)

_ALLOWED_MIME = frozenset(
    {
        # stills only: the render treats anything else as a blank slot
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
)

_WORD_SPLIT = re.compile(r"[^A-Za-z0-9']+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")
_EDGE_PUNCT = ".,:;!?\"'“”‘’()[]"

# Below this CLIP similarity a thumbnail does not depict the scene's
# subject well enough to appear — dark space shots score ~0.2+, a railway
# sign that matched some metric noun scores ~0.17, junk scores ~0.0.
_CLIP_FLOOR = 0.18


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
    subjects beat bare single words. Storyboard nouns (``Archive``,
    ``Timeline``, ``Catalog``...) are camera words too: on Commons they
    return page scans and library shelves ("Archive of Contemporary
    Music", "Card catalog card") instead of the subject.
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
        if not strong or strong & _CAMERA_WORDS:
            continue
        scored.append((len(strong), sum(len(w) for w in strong), phrase))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [phrase for _, _, phrase in scored]


def is_media_junk(file_title: str, mime: str, width: int | None) -> bool:
    """Whether the file is UI chrome, a text-image, too small, or unusable."""

    lowered = file_title.lower()
    if mime not in _ALLOWED_MIME:
        return True
    if mime.startswith("image/") and (width is None or width < 400):
        return True
    raw_tokens = frozenset(re.findall(r"[a-z0-9]+", lowered))
    if raw_tokens & _JUNK_WORDS:
        return True
    return any(marker in lowered for marker in _JUNK_PHRASES)


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
        clip_score=info.get("_clip_score"),
    )


@dataclass(frozen=True)
class AssetOptions:
    """Tuning knobs for :class:`AssetResearchBlock`."""

    assets_max: int = 3
    words_per_asset: int = 18
    search_limit: int = 12
    nasa_limit: int = 6
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
        nasa=None,
        openverse=None,
        llm=None,
        http=None,
        clip=None,
    ) -> None:
        self._wiki = wiki
        self._commons = commons
        self._options = options or AssetOptions()
        self._nasa = nasa
        self._openverse = openverse
        self._llm = llm
        self._http = http
        self._clip = clip
        self._fiction = False
        self._chooser_pool = 6

    def run(
        self,
        theory: Theory,
        brief: ResearchBrief,
        script: VideoScript,
        queries: QueryPack | None = None,
    ) -> AssetPack:
        """Fetch and match media for every scene of ``script``.

        With ``queries`` the directed agents + nasa/openverse sources drive
        the search; without them the block falls back to its own phrase
        mining and Commons-only results.
        """

        source = self._article_source(theory)
        article_title = source.title if source is not None else theory.name
        article_url = source.url if source is not None else ""
        self._fiction = any(
            "original horror story" in note.lower() for note in script.source_notes
        )
        try:
            article_pool = [info for info in self._page_pool([article_title]) if self._usable(info)]
            entity_pool = self._entity_pool(brief)
        except ProviderError as exc:
            raise BlockError(f"Could not fetch media for '{theory.name}': {exc}") from exc

        query_map = {
            scene.scene_index: scene.queries for scene in (queries.scenes if queries else [])
        }
        used: set[str] = set()
        scenes: list[SceneAssets] = []

        # the hook is a sentence too: its own query, its own asset
        hook_scene = ScriptScene(
            narration=script.hook, visual=f"hook: {script.hook}", seconds=6
        )
        hook_assets = self._assets_for_scene(
            hook_scene,
            self._query_for(hook_scene, brief),
            query_map.get(-1),
            self._target_for(hook_scene),
            brief,
            entity_pool,
            article_pool,
            used,
        )
        scenes.append(
            SceneAssets(
                scene_index=-1,
                visual="hook",
                query=script.hook,
                assets=hook_assets,
            )
        )

        for index, scene in enumerate(script.scenes):
            query = self._query_for(scene, brief)
            actions = query_map.get(index)
            target = len(actions) if actions else self._target_for(scene)
            assets = self._assets_for_scene(
                scene,
                query,
                actions,
                target,
                brief,
                entity_pool,
                article_pool,
                used,
            )
            scenes.append(
                SceneAssets(
                    scene_index=index,
                    visual=scene.visual,
                    query=query,
                    assets=assets,
                )
            )

        # a blank screen must be impossible: hand an underfed entry the
        # primary visual of a well-fed neighbour
        for entry in scenes:
            if entry.assets:
                continue
            for donor in scenes:
                if donor is entry or not donor.assets:
                    continue
                entry.assets.append(donor.assets[0].model_copy(update={"trigger": ""}))
                break

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
    def _target_for(self, scene: ScriptScene) -> int:
        """1-3 assets, scaled by how much the scene has to say."""

        words = len(scene.narration.split()) + len(scene.visual.split())
        return min(self._options.assets_max, max(1, round(words / self._options.words_per_asset)))

    def _luminance(self, info: dict) -> float | None:
        """Mean brightness (0-255) of the thumbnail, for horror mood ranking."""

        if not self._fiction:
            return None
        url = info.get("thumburl") or info.get("url")
        if not url:
            return None
        try:
            from io import BytesIO

            from PIL import Image

            response = self._commons.fetch_bytes(url)
            image = Image.open(BytesIO(response)).convert("L").resize((32, 32))
            pixels = list(image.getdata())
            return sum(pixels) / len(pixels)
        except Exception:  # noqa: BLE001 — mood ranking is best-effort
            return None

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
            if self._fiction:
                light = self._luminance(info)
                if light is not None:
                    score += min(max(0.0, (110 - light) / 30), 2.0)
            scored.append((score, info))
        return sorted(scored, key=lambda pair: pair[0], reverse=True)

    def _assets_for_scene(
        self,
        scene: ScriptScene,
        query: str,
        actions,
        target: int,
        brief: ResearchBrief,
        entity_pool: dict[str, VisualAsset],
        article_pool: list[dict],
        used: set[str],
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
                and asset.file not in used
            ):
                chosen.append(asset)
                used.add(asset.file)
                used.add(_stem_of(asset.file))

        if len(chosen) < target:  # fallback: mine the scene's own phrases
            phrases = [phrase.lower() for phrase in proper_phrases(scene.visual)]
            searches = [query] if query else []
            for phrase in phrases:
                if len(searches) >= 2:
                    break
                if phrase not in {existing.lower() for existing in searches}:
                    searches.append(phrase)
            if len(searches) < 2:
                nouns = sorted(content_tokens(scene.narration), key=lambda w: -len(w))
                fallback = f"{brief.title} {nouns[0]}".strip() if nouns else brief.title
                if fallback.lower() not in {existing.lower() for existing in searches}:
                    searches.append(fallback)
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

        if not chosen:  # last resort: the topic title itself, looseness ok
            for score, info in self._search_ranked(brief.title, text_scene):
                if score < 2:
                    break
                self._take(info, chosen, used)
                break

        # One score for every pick on one scale: what the shot depicts
        # matters across paths (routed, entity, filler), not just inside
        # whichever candidate pool it came from.
        if self._clip is not None and self._http is not None and chosen:
            with ThreadPoolExecutor(max_workers=4) as pool:
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
            healthy = [
                asset
                for asset, (sim, collage_like, texty) in zip(chosen, results, strict=True)
                if not (collage_like or texty)
                and (sim is None or sim >= _CLIP_FLOOR)
            ]
            if healthy:
                chosen = healthy
            else:
                # everything the scene caught was junk (a metric noun that
                # matches railway signs, off-topic posters); the topic's own
                # name is the last chance to buy a real shot
                rescue = self._rescue_visuals(text_scene, brief.title, used)
                if rescue:
                    chosen = rescue
        self._timing_for(chosen, scene.narration)
        return chosen

    def _rescue_visuals(
        self, text_scene: str, topic: str, used: set[str]
    ) -> list[VisualAsset]:
        """Last-hope search under the scene's topic name, CLIP-checked."""

        rescue: list[VisualAsset] = []
        for score, info in self._search_ranked(topic, text_scene):
            if score < self._options.min_score:
                break
            similarity = None
            if self._clip is not None and self._http is not None:
                similarity, collage_like, texty = self._clip.analyze(
                    text_scene, info.get("thumburl") or info.get("url"), self._http
                )
                if collage_like or texty:
                    continue
                if similarity is not None and similarity < _CLIP_FLOOR:
                    continue
                else:
                    info["_clip_score"] = similarity
            self._take(info, rescue, used)
            if len(rescue) >= 2:
                break
        return rescue

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

    def _choose(
        self,
        scene: str,
        search_words: str,
        candidates: list[tuple[int, dict]],
        confirm_only: bool = False,
    ) -> dict | None:
        """One editor-pick decision over a ranked shortlist.

        The chooser reads each candidate's own metadata like a human editor:
        for a person query the description must identify the right person;
        anything that cannot be verified, or a mostly-text image, is
        rejected outright. With no LLM wired in the top-ranked candidate
        passes unchanged.

        With ``confirm_only`` (CLIP already ranked by depicted content) the
        LLM may only veto the top candidate — it cannot down-shift to a
        pixel-weaker pick the way it did when it chose by caption alone.
        """

        if not candidates:
            return None
        if self._llm is None:
            return candidates[0][1]
        confirm = confirm_only
        lines = "\n".join(
            f"[{index}] file: {info.get('file', '')}\n"
            f"    description: {(info.get('description') or '').strip()[:260]}"
            for index, (_score, info) in enumerate(candidates)
        )
        if confirm:
            instruction = (
                "The FIRST candidate is the current pick. Study its own "
                "description. Reply its exact file title unchanged unless it "
                "breaks a veto rule: wrong-person or unverifiable identity, "
                "era mismatch, gag/parody pose, mostly-text, or visually "
                "irrelevant to the subject. To veto, reply file = empty string."
            )
        else:
            instruction = (
                "Pick the one candidate the viewer should see. "
                "file = its exact file title, or empty string to reject all."
            )
        messages = [
            {
                "role": "system",
                "content": CHOOSER_SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    f"NARRATION (this moment): {scene}\n"
                    f"SUBJECT QUERY: {search_words}\n"
                    f"CANDIDATES:\n{lines}\n"
                    f"{instruction}"
                ),
            },
        ]
        try:
            verdict = self._llm.complete_json(messages, PickSchema)
        except Exception as exc:
            logger.warning("Chooser call failed: %s", exc)
            return candidates[0][1]
        wanted = (verdict.file or "").strip()
        if not wanted:
            logger.info("Chooser rejected all candidates for '%s'", search_words)
            return None
        if confirm:
            # veto-only: the pixel ranking stays the master
            return candidates[0][1]
        for _score, info in candidates:
            if info.get("file", "").lower() == wanted.lower():
                return info
        return None

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
            with ThreadPoolExecutor(max_workers=min(3, len(variants))) as pool:
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
        clip_scored: list[tuple[float, int, dict]] = []
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

        # CLIP: what does the image DEPICT about the subject?
        if self._clip is not None and self._http is not None and eligible:
            subject_text = f"{action.intent} {action.query}".strip()
            clip_scored: list[tuple[float, int, dict]] = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda pair: self._clip.analyze(
                            subject_text,
                            pair[1].get("thumburl") or pair[1].get("url"),
                            self._http,
                        ),
                        eligible,
                    )
                )
            for index, (similarity, collage_like, texty) in enumerate(results):
                if collage_like or texty:
                    # multi-panel sheets read as clutter; text posters
                    # violate the no-text-as-image rule
                    continue
                if similarity is not None and similarity < _CLIP_FLOOR:
                    # pixels that don't depict the subject at all
                    continue
                if similarity is None:
                    similarity = -1.0 + 0.001 * index
                else:
                    info = eligible[index][1]
                    info["_clip_score"] = similarity
                clip_scored.append((similarity, index, eligible[index][1]))
            clip_scored.sort(key=lambda pair: (-pair[0], pair[1]))
            eligible = [(score, info) for _sim, _idx, info in clip_scored]
        # A top shot that clearly depicts the subject needs no editor: when
        # CLIP both likes the pick and separates it from the runner-up the
        # pixels settle it and the LLM's flaky veto never runs.
        margin = 1.0
        if clip_scored:
            top_sim = clip_scored[0][0]
            next_sim = clip_scored[1][0] if len(clip_scored) > 1 else -1.0
            margin = top_sim - next_sim
            confident = top_sim >= 0.30 and margin >= 0.02
        else:
            confident = False
        for rank, _entry in enumerate(eligible):
            pick = self._choose(
                scene, search_words, eligible[rank:], confirm_only=not confident
            )
            if pick is None:
                continue
            self._take(pick, chosen, used)
            break

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
        for info in search(search_words, limit=limit):
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
        if passing and self._clip is not None and self._http is not None:
            subject_text = f"{relevance} {search_words}".strip()
            clip_scored: list[tuple[float, int, dict]] = []
            with ThreadPoolExecutor(max_workers=4) as pool:
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
    @staticmethod
    def _article_source(theory: Theory):
        for source in theory.sources:
            if "wikipedia.org" in source.url:
                return source
        return theory.sources[0] if theory.sources else None

    def _usable(self, info: dict) -> bool:
        return not is_media_junk(info["file"], info.get("mime", ""), info.get("width"))

    def _blocked(self, info: dict, used: set[str]) -> bool:
        """File used, or its extensionless twin already in play."""

        return info["file"] in used or _stem_of(info["file"]) in used

    def _take(
        self, info: dict, chosen: list[VisualAsset], used: set[str], trigger: str = ""
    ) -> None:
        asset = to_asset(info)
        asset.trigger = trigger
        chosen.append(asset)
        used.add(info["file"])
        used.add(_stem_of(info["file"]))

    def _query_for(self, scene: ScriptScene, brief: ResearchBrief) -> str:
        phrases = proper_phrases(scene.visual)
        if phrases:
            return phrases[0]
        nouns = sorted(
            content_tokens(scene.visual),
            key=lambda w: -len(w),
        )
        nouns = [w for w in nouns if w not in _CAMERA_WORDS]
        if not nouns:
            return ""
        return f"{brief.title} {nouns[0]}".strip()
