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
_SELECTION_PROMPT = """You are selecting visuals for a spoken sentence in a dark psychology teaching video.

Given the sentence and a list of available images with descriptions, pick {count} DIFFERENT images that best match what the sentence is talking about. Each image should relate to a different part or idea in the sentence (a visual metaphor is fine; an unrelated subject is not).

Return ONLY a comma-separated list of image IDs in the order they should appear, e.g. "face_05, crowd_12" or "arch_03, diag_14, face_01".

Sentence: "{sentence}"

Available images:
{image_list}

Best image IDs:"""


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


def _load_descriptions() -> dict:
    p = ASSETS_ROOT / "descriptions.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_image_catalog(manifest: dict) -> list[dict]:
    """Flatten manifest into a list of image entries for LLM/keyword selection."""
    descriptions = _load_descriptions()
    catalog = []
    for cat_name, cat_data in manifest.get("categories", {}).items():
        # Handle both old format (dict with "images") and new format (list)
        images = cat_data if isinstance(cat_data, list) else cat_data.get("images", [])
        for img in images:
            img_id = img.get("id", "")
            if not img_id:
                continue
            entry = {
                "id": img_id,
                "category": img.get("category", cat_name),
                "name": img.get("name", img_id),
                "description": img.get("description", ""),
                "filename": img.get("filename", f"{img_id}.png"),
            }
            rich = descriptions.get(img_id)
            if rich:
                entry["name"] = rich.get("name", entry["name"])
                entry["description"] = rich.get("description", entry["description"])
            catalog.append(entry)
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


_STOP_WORDS = {
    "the", "and", "you", "your", "they", "their", "them", "this", "that",
    "with", "from", "have", "has", "had", "will", "would", "could", "should",
    "when", "what", "how", "why", "not", "are", "was", "were", "been", "being",
    "into", "than", "then", "just", "like", "more", "some", "most", "even",
    "only", "also", "over", "such", "each", "every", "other", "about",
}


def _sentence_keywords(sentence: str) -> set[str]:
    sentence_lower = sentence.lower()
    return set(re.findall(r"[a-z]{3,}", sentence_lower)) - _STOP_WORDS


def _images_for_sentence_count(sentence: str) -> int:
    """How many images to show during one sentence (at least 2)."""
    wc = len(sentence.split())
    if wc <= 6:
        return 2
    if wc <= 12:
        return 2
    return 3


def _score_image(img: dict, keywords: set[str], sentence_lower: str) -> int:
    desc_lower = (img.get("description", "") + " " + img.get("name", "")).lower()
    score = sum(1 for kw in keywords if kw in desc_lower)
    cat = img.get("category", "")
    if "face" in cat and any(w in sentence_lower for w in ["face", "expression", "look", "stare", "eye", "smile", "tear", "feel", "emotion"]):
        score += 3
    elif "crowd" in cat and any(w in sentence_lower for w in ["crowd", "people", "protest", "line", "group", "mass", "gathering", "together"]):
        score += 3
    elif "arch" in cat and any(w in sentence_lower for w in ["leader", "politician", "ceo", "therapist", "parent", "child", "narcissist", "psychopath", "healer", "authority", "boss"]):
        score += 3
    elif "diag" in cat and any(w in sentence_lower for w in ["inequality", "gap", "wealth", "income", "discriminat", "politics", "system", "structure", "data", "chart", "statistic", "power"]):
        score += 3
    elif "env" in cat and any(w in sentence_lower for w in ["room", "office", "building", "space", "institution", "prison", "cult", "place"]):
        score += 2
    elif "surveillance" in cat and any(w in sentence_lower for w in ["watch", "monitor", "track", "surveil", "camera", "data", "algorithm", "follow"]):
        score += 3
    elif "manip" in cat and any(w in sentence_lower for w in ["control", "manipulat", "gaslight", "puppet", "coerce", "pressure", "trick", "force"]):
        score += 3
    return score


def _ranked_for_sentence(sentence: str, catalog: list[dict], used_ids: set[str]) -> list[dict]:
    sentence_lower = sentence.lower()
    keywords = _sentence_keywords(sentence)
    unused = [img for img in catalog if img["id"] not in used_ids] or catalog
    scored = [( _score_image(img, keywords, sentence_lower), img) for img in unused]
    scored.sort(reverse=True, key=lambda x: x[0])
    # prefer matches with some relevance, but keep others as tail so we never starve
    relevant = [img for s, img in scored if s > 0]
    rest = [img for s, img in scored if s <= 0]
    return relevant + rest


def _select_images_for_sentence(
    sentence: str, catalog: list[dict], used_ids: set[str], count: int,
) -> list[dict]:
    """Pick `count` different images related to parts of the sentence."""
    if not catalog or count <= 0:
        return []
    count = min(count, len(catalog))

    llm = _get_selector_llm()
    valid_ids = {img["id"].lower() for img in catalog}
    ranked = _ranked_for_sentence(sentence, catalog, used_ids)
    pool = ranked[:25] if ranked else catalog[:25]
    picks: list[dict] = []
    picked_ids: set[str] = set()

    if llm is not None and pool:
        image_list = _format_catalog_for_prompt(pool)
        avoid = ""
        if used_ids:
            avoid_list = ", ".join(sorted(used_ids)[:20])
            avoid = f"\n\nDo NOT reuse these images (they already appeared): {avoid_list}."
        prompt = _SELECTION_PROMPT.format(
            count=count, sentence=sentence.strip()[:300], image_list=image_list
        ) + avoid
        try:
            raw = llm.complete([LLMMessage(role="user", content=prompt)], temperature=0.0)
            for match in re.finditer(r"\b([a-z]{3,6})[_\\](\d+)\b", raw.replace("\\", ""), re.IGNORECASE):
                pid = f"{match.group(1)}_{match.group(2)}".lower()
                if pid in used_ids or pid in picked_ids or pid not in valid_ids:
                    continue
                for img in catalog:
                    if img["id"].lower() == pid:
                        picks.append(img)
                        picked_ids.add(pid)
                        break
                if len(picks) >= count:
                    break
            if picks:
                logger.info("LLM selected %s for: %s...", [p["id"] for p in picks], sentence[:50])
        except Exception as e:
            logger.warning("LLM multi-selection failed: %s", e)

    # fill remaining slots with ranked diverse picks
    if len(picks) < count:
        for img in ranked:
            if img["id"] in used_ids or img["id"] in picked_ids:
                continue
            # prefer category variety among the picks for this sentence
            if picks and all(img.get("category") == p.get("category") for p in picks):
                continue
            picks.append(img)
            picked_ids.add(img["id"])
            if len(picks) >= count:
                break
    # final fill: any unused
    if len(picks) < count:
        for img in catalog:
            if img["id"] in used_ids or img["id"] in picked_ids:
                continue
            picks.append(img)
            picked_ids.add(img["id"])
            if len(picks) >= count:
                break
    # absolute last resort: allow reuse rather than returning fewer than requested
    if len(picks) < count:
        for img in catalog:
            if img["id"] in picked_ids:
                continue
            picks.append(img)
            picked_ids.add(img["id"])
            if len(picks) >= count:
                break
    return picks[:count]


def _fallback_select(
    sentence: str, catalog: list[dict], used_ids: set[str],
    last_category: str | None = None,
) -> dict | None:
    """Fallback: keyword-based selection when LLM fails."""
    ranked = _ranked_for_sentence(sentence, catalog, used_ids)
    best = None
    best_score = -10**9
    sentence_lower = sentence.lower()
    keywords = _sentence_keywords(sentence)
    for img in ranked:
        if img["id"] in used_ids:
            continue
        score = _score_image(img, keywords, sentence_lower)
        if last_category and img.get("category") == last_category:
            score -= 2
        if score > best_score:
            best_score = score
            best = img
        if best_score >= 4:
            break
    if best:
        logger.info("Fallback selected %s (score=%d) for: %s...", best["id"], best_score, sentence[:60])
    return best


def assign_sentence_assets(
    texts: list[str], *, used_concepts: set[str] | None = None
) -> list[VisualAsset]:
    """Assign multiple related visuals per sentence.

    Each sentence gets 2-3 images (shown in sequence) so that different
    parts of what the sentence talks about have a matching visual.
    """

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

    plan: list[tuple[str, int]] = [
        (sent, _images_for_sentence_count(sent)) for _ti, _si, sent in all_sents
    ]
    total = sum(c for _, c in plan) or 1
    assets: list[VisualAsset] = []
    seq_idx = 0

    for sent, count in plan:
        picks = _select_images_for_sentence(sent, catalog, used_ids, count)
        if len(picks) < count:
            ranked = _ranked_for_sentence(sent, catalog, used_ids)
            picked_set = {p["id"] for p in picks}
            for img in ranked:
                if img["id"] in picked_set or img["id"] in used_ids:
                    continue
                picks.append(img)
                picked_set.add(img["id"])
                if len(picks) >= count:
                    break
        if not picks:
            picks = [catalog[0]]

        for img in picks:
            used_ids.add(img["id"])
            position = seq_idx / max(total - 1, 1) if total > 1 else 0.0
            seq_idx += 1

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
                    "llm_selected": False,
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