"""Per-sentence asset agent: assign concept-driven visual metaphors to every spoken sentence.

Each sentence's psychology concepts drive image search, and CLIP verifies
which downloaded candidate actually depicts the sentence before it is used.
Motion style, color grade, and timing follow the concept's visual config.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path

import httpx

from blocks.concept_metaphors import (
    ConceptVisuals,
    extract_concepts_from_text,
    get_concept_visuals,
)
from core.models import AssetPack, SceneAssets, VideoScript, VisualAsset

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# CLIP similarity of a real match usually sits above ~0.18; candidates below
# this are recorded as weak in the QA metadata but still used (better than
# a blank).
CLIP_WEAK = 0.18

_STOPWORDS = frozenset(
    [
        "a", "an", "the", "this", "that", "these", "those", "you", "your",
        "they", "their", "them", "it", "its", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "should", "just", "like", "when",
        "what", "how", "why", "not", "no", "yes", "and", "but", "or", "so",
        "if", "then", "than", "because", "about", "into", "over", "under",
        "with", "from", "for", "of", "on", "at", "by", "to", "in", "out",
        "up", "down", "next", "time", "one", "two", "thing", "things",
        "really", "very", "much", "more", "most", "some", "every", "all",
        "say", "says", "said", "ask", "asks", "asking", "tell", "tells",
        "told", "where", "there", "here", "now", "always", "never",
    ]
)


def _visual_keywords(text: str, limit: int = 4) -> str:
    """Derive a short image-query from the most significant words."""
    words = re.findall(r"[a-z][a-z-]{2,}", text.lower())
    picked: list[str] = []
    for word in words:
        if word in _STOPWORDS or word in picked:
            continue
        picked.append(word)
        if len(picked) >= limit:
            break
    return " ".join(picked)


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    if not parts and text.strip():
        return [text.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(part.split()) < 4:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged or ([text.strip()] if text.strip() else [])


def _trigger(sentence: str) -> str:
    words = _WORD_RE.findall(sentence)
    return " ".join(words[:4])


# -- candidates ----------------------------------------------------------------

def _download(url_or_path: str, fetch_dir: Path, tag: str = "c") -> Path | None:
    """Download one candidate image, cached by URL hash."""
    fetch_dir.mkdir(parents=True, exist_ok=True)
    name = tag + hashlib.sha1(url_or_path.encode()).hexdigest()[:16] + ".jpg"
    raw = fetch_dir / ("raw-" + name)
    out = fetch_dir / name
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        if url_or_path.startswith("file://"):
            src = Path(url_or_path[7:])
            if not src.exists():
                return None
            raw.write_bytes(src.read_bytes())
        else:
            r = httpx.get(url_or_path, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
            raw.write_bytes(r.content)
    except Exception as exc:  # noqa: BLE001
        logger.debug("candidate download failed for %r: %s", url_or_path, exc)
        return None
    return _darken(raw, out)


def _darken(src: Path, dst: Path) -> Path:
    """Apply a darker, moodier grade (keeps detail, no bounce)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", "eq=brightness=-0.10:contrast=1.12:saturation=0.72,curves=preset=darker",
             "-frames:v", "1", str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return dst
    except Exception:  # noqa: BLE001
        return src


_Scorer = None


def _clip_scorer():
    """Lazily built CLIP scorer shared by the whole process (None if unusable)."""
    global _Scorer
    if _Scorer is not None:
        return _Scorer
    try:
        from utils.clip import ClipScorer

        scorer = ClipScorer(user_agent="videoai-sentence-assets/1.0")
        scorer._load()  # warm up; if the model can't load it stays unusable
        if not scorer._failed:
            _Scorer = scorer
    except Exception as exc:  # noqa: BLE001
        logger.debug("CLIP scorer unavailable: %s", exc)
    return _Scorer


def _clip_score(scorer, subject: str, image_path: Path) -> float | None:
    """Similarity between a narration sentence and a local candidate image."""
    if scorer is None:
        return None
    try:
        import PIL.Image

        with PIL.Image.open(image_path) as im:
            image = im.convert("RGB")
        return scorer._similarity(subject, image)
    except Exception as exc:  # noqa: BLE001
        logger.debug("clip scoring failed for %s: %s", image_path.name, exc)
        return None


def _gather_candidates(
    sent: str,
    concept_visuals: ConceptVisuals,
    fetch_dir: Path,
    max_candidates: int = 8,
) -> list[tuple[str, Path]]:
    """Download first-hit images for each candidate query: sentence keywords
    first, then the concept's metaphor queries, then fallbacks."""

    hits_dir = Path("output/_pexels-hits-cache")

    queries: list[tuple[str, str]] = []
    kws = _visual_keywords(sent)
    if len(kws.split()) >= 2:
        queries.append(("sent", kws))
    queries += [("concept", q) for q in concept_visuals.primary_queries]
    queries += [("fallback", q) for q in concept_visuals.fallback_queries]

    seen_urls: set[str] = set()
    candidates: list[tuple[str, Path]] = []
    for kind, query in queries:
        if len(candidates) >= max_candidates:
            break
        if not query:
            continue
        hits = []
        try:
            from providers.sources.pexels import PexelsClient

            client = PexelsClient(client=httpx.Client(timeout=20.0, follow_redirects=True))
            hits = client.search(query, per_page=4)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pexels search failed for %r: %s", query, exc)
        if not hits:
            # Pexels key may be missing/exhausted; Openverse needs no key
            try:
                from providers.sources.openverse import OpenverseClient

                hits = OpenverseClient(client=httpx.Client(timeout=20.0)).search(query, limit=4)
            except Exception as exc:  # noqa: BLE001
                logger.debug("openverse search failed for %r: %s", query, exc)
        for hit in hits[:3]:
            url = hit.get("url") or hit.get("thumburl")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            img = _download(url, fetch_dir, tag=f"q{len(seen_urls)}")
            if img:
                candidates.append((f"{kind}:{query}", img))
                if len(candidates) >= max_candidates:
                    break

    return candidates


def _select_concept_for_sentence(sentence: str, used_concepts: set[str]) -> ConceptVisuals:
    """Pick the most relevant unused concept for this sentence."""
    concepts = extract_concepts_from_text(sentence)
    for c in concepts:
        if c not in used_concepts:
            used_concepts.add(c)
            return get_concept_visuals(c)
    # Fallback: allow reuse if all used
    for c in concepts:
        return get_concept_visuals(c)
    return get_concept_visuals("default")


def _build_asset_from_concept(
    sent: str,
    concept_visuals: ConceptVisuals,
    fetch_dir: Path,
    position: float,
) -> VisualAsset:
    """Create a VisualAsset: best CLIP-matched candidate for this sentence."""

    chosen_path: Path | None = None
    query_used = ""
    clip = None

    try:
        candidates = _gather_candidates(sent, concept_visuals, fetch_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("candidate gathering failed: %s", exc)
        candidates = []

    if candidates:
        scorer = _clip_scorer()
        best = (None, candidates[0][1], candidates[0][0])  # (score, path, query)
        for query, path in candidates:
            s = _clip_score(scorer, sent, path)
            if s is not None and (best[0] is None or s > best[0]):
                best = (s, path, query)
        chosen_path = best[1]
        query_used = best[2]
        clip = best[0]
        if clip is not None and clip < CLIP_WEAK:
            logger.info(
                "weak visual match (%.2f) for %r [query %r]",
                clip,
                sent[:60],
                query_used,
            )

    if chosen_path:
        url = f"file://{chosen_path.resolve()}"
        page_url = f"pexels:{chosen_path.name}"
        caption = f"{concept_visuals.concept}: {query_used}"
        fname = chosen_path.name
    else:
        # Ultimate fallback - placeholder
        url = ""
        page_url = f"concept:{concept_visuals.concept}"
        caption = concept_visuals.concept
        fname = ""

    return VisualAsset(
        kind="image",
        file=fname,
        url=url,
        page_url=page_url,
        caption=caption,
        trigger=_trigger(sent),
        position=round(position, 3),
        extra={
            "concept": concept_visuals.concept,
            "motion_style": concept_visuals.motion_style,
            "grade_emphasis": concept_visuals.grade_emphasis,
            "subject_crossfade": concept_visuals.subject_crossfade,
            "card_crossfade": concept_visuals.card_crossfade,
            "music_mood": concept_visuals.music_mood,
            "clip_score": round(clip, 4) if clip is not None else None,
            "query": query_used,
        },
    )


def assign_sentence_assets(
    texts: list[str], *, used_concepts: set[str] | None = None, fetch_dir: Path | None = None
) -> list[VisualAsset]:
    """One concept-driven visual per sentence in ``texts``."""

    if used_concepts is None:
        used_concepts = set()
    all_sents: list[tuple[int, int, str]] = []
    for ti, text in enumerate(texts):
        sents = _sentences(text)
        for si, sent in enumerate(sents):
            all_sents.append((ti, si, sent))
    if not all_sents:
        return []

    assets: list[VisualAsset] = []
    for idx, (_ti, _si, sent) in enumerate(all_sents):
        concept_visuals = _select_concept_for_sentence(sent, used_concepts)

        position = idx / max(len(all_sents) - 1, 1) if len(all_sents) > 1 else 0.0
        asset = _build_asset_from_concept(sent, concept_visuals, fetch_dir, position)
        assets.append(asset)

    return assets


def build_asset_pack(script: VideoScript, *, slug: str) -> AssetPack:
    """Assign one concept-driven visual per sentence of every scene (and hook)."""

    used_concepts: set[str] = set()
    scenes: list[SceneAssets] = []
    fetch_dir = Path("output") / slug / "fetched"

    # hook gets its own visual (scene_index -1, consumed by render)
    hook_assets = assign_sentence_assets([script.hook], used_concepts=used_concepts, fetch_dir=fetch_dir)
    scenes.append(
        SceneAssets(
            scene_index=-1,
            visual=script.hook,
            query="hook",
            assets=hook_assets,
        )
    )

    for index, scene in enumerate(script.scenes):
        sent_assets = assign_sentence_assets(
            [scene.narration], used_concepts=used_concepts, fetch_dir=fetch_dir
        )
        scenes.append(
            SceneAssets(
                scene_index=index,
                visual=scene.visual,
                query=_trigger(scene.narration),
                assets=sent_assets,
            )
        )
    return AssetPack(
        slug=slug,
        topic=script.title,
        article_title="concept-driven visual metaphors",
        article_url="",
        scenes=scenes,
    )


def choose_concepts(script: VideoScript) -> list[str]:
    """Compatibility: extract primary concept per scene."""
    used: set[str] = set()
    result: list[str] = []
    for scene in script.scenes:
        concepts = extract_concepts_from_text(scene.narration)
        for c in concepts:
            if c not in used:
                used.add(c)
                result.append(c)
                break
        else:
            result.append("default")
    return result
