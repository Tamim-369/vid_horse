"""Per-sentence asset agent: assign concept-driven visual metaphors to every spoken sentence.

Replaces motif library with concept → metaphor mapping. Each sentence's
psychology concepts drive image search, motion style, color grade, and timing.
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


def _pexels_fetch(query: str, fetch_dir: Path) -> Path | None:
    name = hashlib.sha1(query.encode()).hexdigest()[:16] + ".jpg"
    raw = fetch_dir / ("raw-" + name)
    out = fetch_dir / name
    if out.exists() and out.stat().st_size > 0:
        return out
    fetch_dir.mkdir(parents=True, exist_ok=True)
    try:
        from providers.sources.pexels import PexelsClient
        client = PexelsClient(client=httpx.Client(timeout=20.0, follow_redirects=True))
        hits = client.search(query, per_page=4)
        if not hits:
            return None
        url = hits[0].get("url") or hits[0].get("thumburl")
        if not url:
            return None
        r = httpx.get(url, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        raw.write_bytes(r.content)
        return _darken(raw, out)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pexels fetch failed for %r: %s", query, exc)
        return None


def _select_concept_for_sentence(sentence: str, used_concepts: set[str]) -> ConceptVisuals | None:
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
    """Create a VisualAsset driven by the concept's visual config."""
    # Try primary queries first, then fallbacks
    fetched: Path | None = None
    query_used = ""
    for query in concept_visuals.primary_queries + concept_visuals.fallback_queries:
        fetched = _pexels_fetch(query, fetch_dir)
        if fetched:
            query_used = query
            break
    if not fetched:
        # Last resort: generic fetch
        fetched = _pexels_fetch(sent[:120], fetch_dir)
        query_used = sent[:80]

    if fetched:
        url = f"file://{fetched.resolve()}"
        page_url = f"pexels:{fetched.name}"
        caption = f"{concept_visuals.concept}: {query_used}"
        fname = fetched.name
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
        # Store concept metadata for downstream use
        extra={
            "concept": concept_visuals.concept,
            "motion_style": concept_visuals.motion_style,
            "grade_emphasis": concept_visuals.grade_emphasis,
            "subject_crossfade": concept_visuals.subject_crossfade,
            "card_crossfade": concept_visuals.card_crossfade,
            "music_mood": concept_visuals.music_mood,
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
        if concept_visuals is None:
            concept_visuals = get_concept_visuals("default")

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