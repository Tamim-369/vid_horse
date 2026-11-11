"""Per-sentence asset agent: assign visuals from manifest using LLM selection.

Images are pre-generated and cataloged in manifest.json. LLM picks the best
match for each sentence based on descriptions. No decision model needed.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from config.settings import get_settings
from core.interfaces import LLMMessage
from core.models import AssetPack, SceneAssets, VideoScript, VisualAsset
from providers.llm import create_llm_provider

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

MANIFEST_PATH = Path("assets/psychology_motifs/manifest.json")
ASSETS_ROOT = Path("assets/psychology_motifs")

# LLM selection prompt
_SELECTION_PROMPT = """You are selecting the best visual for a spoken sentence in a dark psychology teaching video.

Given the sentence and a list of available images with descriptions, pick the ONE image that best matches what the sentence is actually about. A visual metaphor is fine; an unrelated subject is not.

Return ONLY the image ID (e.g., "face_05", "crowd_12", "arch_03", "diag_14").

Sentence: "{sentence}"

Available images:
{image_list}

Best image ID:"""


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


def _load_manifest() -> dict:
    """Load the manifest with all available images."""
    if not MANIFEST_PATH.exists():
        logger.warning("Manifest not found at %s", MANIFEST_PATH)
        return {"categories": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load manifest: %s", e)
        return {"categories": {}}


def _build_image_catalog(manifest: dict) -> list[dict]:
    """Flatten manifest into a list of image entries for LLM selection."""
    catalog = []
    for cat_name, cat_data in manifest.get("categories", {}).items():
        # Handle both old format (dict with "images") and new format (list)
        images = cat_data if isinstance(cat_data, list) else cat_data.get("images", [])
        for img in images:
            img_id = img.get("id", "")
            if not img_id:
                continue
            catalog.append({
                "id": img_id,
                "category": img.get("category", cat_name),
                "name": img.get("name", img_id),
                "description": img.get("description", ""),
                "filename": img.get("filename", f"{img_id}.png"),
            })
    return catalog


def _format_catalog_for_prompt(catalog: list[dict]) -> str:
    """Format catalog for LLM prompt."""
    lines = []
    for img in catalog:
        lines.append(f"  {img['id']}: {img['name']} — {img['description'][:120]}")
    return "\n".join(lines)


_llm_cache = None


def _get_selector_llm():
    """Lazy LLM for image selection."""
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    try:
        settings = get_settings()
        _llm_cache = create_llm_provider(settings, role="assets")
        logger.info("Image selector LLM initialized")
    except Exception as e:
        logger.warning("Selector LLM unavailable: %s", e)
        _llm_cache = None
    return _llm_cache


def _select_image_for_sentence(
    sentence: str, catalog: list[dict], used_ids: set[str] | None = None
) -> dict | None:
    """Use LLM to pick the best image for a sentence."""
    llm = _get_selector_llm()
    used_ids = used_ids or set()
    if llm is None or not catalog:
        return None

    valid_ids = {img["id"].lower() for img in catalog}

    # Only offer images not already shown in this video (fall back to the
    # full catalog if we would starve the prompt).
    unused_catalog = [img for img in catalog if img["id"] not in used_ids]
    pool = unused_catalog or catalog

    # Limit pool to the top N most relevant by keyword overlap
    sentence_lower = sentence.lower()
    keywords = set(re.findall(r"[a-z]{3,}", sentence_lower)) - {
        "the", "and", "you", "your", "they", "their", "them", "this", "that",
        "with", "from", "have", "has", "had", "will", "would", "could", "should",
        "when", "what", "how", "why", "not", "are", "was", "were", "been", "being",
    }

    scored = []
    for img in pool:
        desc = (img.get("description", "") + " " + img.get("name", "")).lower()
        score = sum(1 for kw in keywords if kw in desc)
        if score > 0:
            scored.append((score, img))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_catalog = [img for _, img in scored[:15]] or pool[:15]

    image_list = _format_catalog_for_prompt(top_catalog)
    avoid = ""
    if used_ids:
        avoid_list = ", ".join(sorted(used_ids)[:20])
        avoid = f"\n\nDo NOT reuse these images (they already appeared): {avoid_list}."
    prompt = _SELECTION_PROMPT.format(sentence=sentence.strip()[:300], image_list=image_list) + avoid

    try:
        raw = llm.complete(
            [LLMMessage(role="user", content=prompt)],
            temperature=0.0,
        )
        match = re.search(r'\b([a-z]{3,6})[_\\](\d+)\b', raw.replace("\\", ""), re.IGNORECASE)
        if match:
            picked_id = f"{match.group(1)}_{match.group(2)}".lower()
            if picked_id in used_ids:
                logger.warning("LLM re-picked used image %s; falling back", picked_id)
                return None
            if picked_id in valid_ids:
                for img in catalog:
                    if img["id"].lower() == picked_id:
                        logger.info("LLM selected %s for: %s...", img["id"], sentence[:60])
                        return img
        logger.warning("LLM returned invalid ID: %s", raw[:100])
    except Exception as e:
        logger.warning("LLM selection failed: %s", e)

    return None


def _fallback_select(
    sentence: str, catalog: list[dict], used_ids: set[str],
    last_category: str | None = None,
) -> dict | None:
    """Fallback: keyword-based selection when LLM fails."""
    sentence_lower = sentence.lower()
    keywords = set(re.findall(r"[a-z]{3,}", sentence_lower)) - {
        "the", "and", "you", "your", "they", "their", "them", "this", "that",
        "with", "from", "have", "has", "had", "will", "would", "could", "should",
        "when", "what", "how", "why", "not", "are", "was", "were", "been", "being",
    }

    best = None
    best_score = -1

    for img in catalog:
        if img["id"] in used_ids:
            continue

        # Score based on keyword overlap in description + name
        desc_lower = (img.get("description", "") + " " + img.get("name", "")).lower()
        score = sum(1 for kw in keywords if kw in desc_lower)

        # Boost for category relevance
        cat = img.get("category", "")
        if "face" in cat and any(w in sentence_lower for w in ["face", "expression", "look", "stare", "eye", "smile", "tear"]):
            score += 3
        elif "crowd" in cat and any(w in sentence_lower for w in ["crowd", "people", "protest", "line", "group", "mass", "gathering"]):
            score += 3
        elif "arch" in cat and any(w in sentence_lower for w in ["leader", "politician", "ceo", "therapist", "parent", "child", "narcissist", "psychopath", "healer"]):
            score += 3
        elif "diag" in cat and any(w in sentence_lower for w in ["inequality", "gap", "wealth", "income", "discriminat", "politics", "system", "structure", "data", "chart", "statistic"]):
            score += 3
        elif "env" in cat and any(w in sentence_lower for w in ["room", "office", "building", "space", "institution", "prison", "cult"]):
            score += 2
        elif "surveillance" in cat and any(w in sentence_lower for w in ["watch", "monitor", "track", "surveil", "camera", "data", "algorithm"]):
            score += 3
        elif "manip" in cat and any(w in sentence_lower for w in ["control", "manipulat", "gaslight", "puppet", "coerce", "pressure"]):
            score += 3

        # Variety: avoid showing the same category back-to-back
        if last_category and cat == last_category:
            score -= 2

        if score > best_score:
            best_score = score
            best = img

    if best:
        logger.info("Fallback selected %s (score=%d) for: %s...", best["id"], best_score, sentence[:60])
    return best


def assign_sentence_assets(
    texts: list[str], *, used_concepts: set[str] | None = None
) -> list[VisualAsset]:
    """Assign one visual per sentence using manifest catalog + LLM selection."""

    if used_concepts is None:
        used_concepts = set()
    used_ids = used_concepts  # shared across hook + scenes so no image repeats

    all_sents: list[tuple[int, int, str]] = []
    for ti, text in enumerate(texts):
        sents = _sentences(text)
        for si, sent in enumerate(sents):
            all_sents.append((ti, si, sent))

    if not all_sents:
        return []

    manifest = _load_manifest()
    catalog = _build_image_catalog(manifest)

    if not catalog:
        logger.warning("No images in manifest!")
        return []

    last_category: str | None = None
    assets: list[VisualAsset] = []

    for idx, (_ti, _si, sent) in enumerate(all_sents):
        llm_picked = False
        # Try LLM selection first (it only sees unused images)
        img = _select_image_for_sentence(sent, catalog, used_ids=used_ids)
        if img is not None and img["id"] not in used_ids:
            llm_picked = True

        # Fallback if LLM fails
        if img is None or img["id"] in used_ids:
            img = _fallback_select(sent, catalog, used_ids, last_category=last_category)

        if img is None:
            # Last resort: any unused image
            for c in catalog:
                if c["id"] not in used_ids:
                    img = c
                    break

        if img is None:
            img = catalog[0]  # Should not happen

        used_ids.add(img["id"])
        last_category = img.get("category", last_category)

        position = idx / max(len(all_sents) - 1, 1) if len(all_sents) > 1 else 0.0

        # Find actual file path
        cat_dir = ASSETS_ROOT / img["category"]
        file_path = cat_dir / img["filename"]
        if not file_path.exists():
            # Search all subdirs
            for subdir in ASSETS_ROOT.iterdir():
                if subdir.is_dir():
                    p = subdir / img["filename"]
                    if p.exists():
                        file_path = p
                        break

        asset = VisualAsset(
            kind="image",
            file=img["filename"],
            url=f"file://{file_path.resolve()}" if file_path.exists() else "",
            page_url=f"{img['category']}:{img['filename']}",
            caption=f"{img['category']}: {img['name']}",
            trigger=_trigger(sent),
            position=round(position, 3),
            extra={
                "concept": img["category"],
                "source": "manifest",
                "llm_selected": llm_picked,
            },
        )
        assets.append(asset)

    return assets


def build_asset_pack(script: VideoScript, *, slug: str) -> AssetPack:
    """Assign one visual per sentence of every scene (and hook)."""

    used_concepts: set[str] = set()
    scenes: list[SceneAssets] = []

    # Hook gets its own visual
    hook_assets = assign_sentence_assets([script.hook], used_concepts=used_concepts)
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
            [scene.narration], used_concepts=used_concepts
        )
        scenes.append(
            SceneAssets(
                scene_index=index,
                visual=scene.visual,
                query=sent_assets[0].trigger if sent_assets else "",
                assets=sent_assets,
            )
        )

    return AssetPack(
        slug=slug,
        topic=script.title,
        article_title="manifest-driven visual selection",
        article_url="",
        scenes=scenes,
    )


def choose_concepts(script: VideoScript) -> list[str]:
    """Compatibility: extract primary concept per scene."""
    manifest = _load_manifest()
    catalog = _build_image_catalog(manifest)
    used: set[str] = set()
    result: list[str] = []

    for scene in script.scenes:
        img = _fallback_select(scene.narration, catalog, used)
        if img:
            used.add(img["id"])
            result.append(img["category"])
        else:
            result.append("face_expressions")
    return result