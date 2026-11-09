"""Per-sentence asset agent: assign concept-driven visual metaphors to every spoken sentence.

Each sentence's psychology concepts drive image search, and CLIP verifies
which downloaded candidate actually depicts the sentence before it is used.
Motion style, color grade, and timing follow the concept's visual config.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
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

_MOTIF_DIR = Path(__file__).resolve().parents[2] / "assets" / "motifs_pollinations"
_SENTENCES_DIR = Path(__file__).resolve().parents[2] / "assets" / "motifs_sentences"
_on_demand_generated = 0
_ON_DEMAND_CAP = 8
MotifIndexState = dict | None
_motif_state: MotifIndexState = None


def _darken(src: Path, dst: Path) -> Path:
    """Apply a darker, moodier grade (keeps detail, no bounce)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", "eq=brightness=-0.15:contrast=1.15:saturation=0.65,curves=preset=darker",
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


# -- motif library (assets/motifs_pollinations, taxonomy in render.motifs) ------

def _motif_state_ready() -> dict | None:
    """One-time index of all local motif images with CLIP embeddings."""
    global _motif_state
    if _motif_state is not None:
        return _motif_state or None
    try:
        from render.motifs import MOTIFS

        scorer = _clip_scorer()
        if scorer is None or (not _MOTIF_DIR.exists() and not _SENTENCES_DIR.exists()):
            _motif_state = {}
            return None
        import numpy as np
        import torch
        import PIL.Image

        entries: dict[str, dict] = {}
        # 1. curated motif library (taxonomy from render.motifs)
        if _MOTIF_DIR.exists():
            for motif in MOTIFS:
                img_path = _MOTIF_DIR / f"{motif.id}.png"
                if not img_path.exists():
                    continue
                with PIL.Image.open(img_path) as im:
                    prepared = scorer._preprocess(im.convert("RGB"))
                with torch.no_grad():
                    feat = scorer._model.encode_image(prepared.unsqueeze(0))
                feat = feat / feat.norm(dim=-1, keepdim=True)
                entries[motif.id] = {
                    "path": img_path,
                    "name": motif.name,
                    "category": motif.category,
                    "description": motif.description,
                    "terms": motif.search_terms,
                    "embed": feat,
                }
        # 2. generated sentence images (phrase-as-description)
        if _SENTENCES_DIR.exists():
            for img_path in sorted(_SENTENCES_DIR.glob("*.png")):
                slug = img_path.stem
                if slug in entries:
                    continue
                try:
                    with PIL.Image.open(img_path) as im:
                        prepared = scorer._preprocess(im.convert("RGB"))
                    with torch.no_grad():
                        feat = scorer._model.encode_image(prepared.unsqueeze(0))
                except Exception:  # noqa: BLE001
                    continue
                feat = feat / feat.norm(dim=-1, keepdim=True)
                phrase = slug.replace("_", " ")
                entries[slug] = {
                    "path": img_path,
                    "name": phrase,
                    "category": "sentence",
                    "description": phrase,
                    "terms": (phrase,),
                    "embed": feat,
                }
        if not entries:
            _motif_state = {}
            return None
        # stack for vectorized scoring
        ids = sorted(entries)
        _motif_state = {
            "entries": entries,
            "ids": ids,
            "matrix": torch.cat([entries[i]["embed"] for i in ids]).float(),
        }
        return _motif_state
    except Exception as exc:  # noqa: BLE001
        logger.debug("motif index build failed: %s", exc)
        _motif_state = {}
        return None


def _slugify(phrase: str) -> str:
    import re as _re

    slug = _re.sub(r"[^a-z0-9]+", "_", phrase.lower().strip()).strip("_")[:60]
    return slug or hashlib.sha1(phrase.encode()).hexdigest()[:10]


def _generate_on_demand(subject: str) -> str | None:
    """Generate a Pollinations image for ``subject`` into the sentences dir.

    Returns the motif slug on success (registered into the live CLIP index
    so later sentences can reuse it) or None. Capped per process.
    """

    global _on_demand_generated, _motif_state
    if _on_demand_generated >= _ON_DEMAND_CAP:
        return None
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            key = os.getenv("POLLINATIONS_API_KEY")
        except ImportError:
            pass
    if not key:
        return None
    import time
    import urllib.parse

    _SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(subject)
    out = _SENTENCES_DIR / f"{slug}.png"
    if not out.exists():
        seed = random.randint(1, 10_000_000)
        style = (
            "dark cinematic noir psychological, 9:16 vertical, near-black "
            "gradient, heavy vignette, film grain, centered subject, high "
            "contrast, shallow depth, colored accent rim light, "
            "photorealistic scene, no text, no watermark, original"
        )
        prompt = urllib.parse.quote(f"{subject}. {style}. seed {seed}.")
        url = (
            f"https://image.pollinations.ai/prompt/{prompt}"
            f"?width=1080&height=1920&model=flux&nologo=true&seed={seed}&key={key}"
        )
        try:
            r = httpx.get(url, timeout=120, follow_redirects=True)
            r.raise_for_status()
            out.write_bytes(r.content)
        except Exception as exc:  # noqa: BLE001
            logger.debug("on-demand generation failed for %r: %s", subject, exc)
            return None
    try:
        scorer = _clip_scorer()
        state = _motif_state_ready()
        if scorer is not None and state is not None:
            import torch
            import PIL.Image

            with PIL.Image.open(out) as im:
                prepared = scorer._preprocess(im.convert("RGB"))
            with torch.no_grad():
                feat = scorer._model.encode_image(prepared.unsqueeze(0))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            if slug not in state["entries"]:
                state["entries"][slug] = {
                    "path": out,
                    "name": subject,
                    "category": "sentence",
                    "description": subject,
                    "terms": (subject,),
                    "embed": feat,
                }
                state["ids"] = sorted(state["entries"])
                state["matrix"] = torch.cat(
                    [state["entries"][i]["embed"] for i in state["ids"]]
                ).float()
    except Exception as exc:  # noqa: BLE001
        logger.debug("on-demand index register failed: %s", exc)
    _on_demand_generated += 1
    print(f"[gen] on-demand motif '{slug}' from: {subject}", flush=True)
    return slug


def _best_motifs(subject: str, top: int = 3, exclude: set[str] | None = None) -> list[tuple[str, float]]:
    """Motif ids ranked by CLIP similarity between the sentence and the image.

    ``exclude`` (already-used motif ids) gets a penalty so a single video
    never repeats the same image when any alternative exists.
    """
    state = _motif_state_ready()
    if not state:
        return []
    scorer = _clip_scorer()
    if scorer is None:
        return []
    try:
        import torch

        text = scorer.text_embed(subject[:180])
        if text is None:
            return []
        sims = (state["matrix"] @ text.t()).squeeze(-1).clone()
        if exclude:
            for i, mid in enumerate(state["ids"]):
                if mid in exclude:
                    sims[i] = sims[i] - 0.15
        order = sims.argsort(descending=True)[:top]
        ids = state["ids"]
        return [(ids[i], float(sims[i])) for i in order]
    except Exception as exc:  # noqa: BLE001
        logger.debug("motif scoring failed: %s", exc)
        return []


def _download(url_or_path: str, fetch_dir: Path, tag: str = "c") -> Path | None:
    """Download one candidate image, cached by URL hash. Always returns darkened image."""
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


def _ensure_darkened(src: Path) -> Path:
    """Ensure an image has the dark grade applied (on-the-fly, no cache)."""
    import tempfile
    from pathlib import Path as PPath
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out = PPath(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", "eq=brightness=-0.15:contrast=1.15:saturation=0.65,curves=preset=darker",
             "-frames:v", "1", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return out
    except Exception:  # noqa: BLE001
        return src


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
    used_motifs: set[str] | None = None,
    used_stock_files: set[str] | None = None,
) -> VisualAsset:
    """Create a VisualAsset: best-matching image for this sentence.

    Candidates come from two pools:
      1. the local motif library (assets/motifs_pollinations) scored by CLIP
      2. stock downloads (sentence keywords + concept metaphor queries)
    The highest CLIP similarity wins; the motif library guarantees every
    sentence gets an image even when every external source fails.
    """

    if used_motifs is None:
        used_motifs = set()
    if used_stock_files is None:
        used_stock_files = set()
    chosen: tuple[str, Path, float | None, dict] | None = None  # (source, path, clip, meta)
    subject_base = sent[:180]

    # 1. motif library
    ranked = _best_motifs(subject_base, top=6, exclude=used_motifs)
    motif_state = _motif_state_ready() or {}
    entries = motif_state.get("entries", {})
    if (not ranked or (ranked[0][1] if ranked else 0.0) < CLIP_WEAK) and _on_demand_generated < _ON_DEMAND_CAP:
        phrase = " ".join(concept_visuals.primary_queries[:2] or []) or subject_base
        new_slug = _generate_on_demand(phrase)
        if new_slug:
            used_motifs.discard(new_slug)  # allow immediate use of freshly made image
            ranked = _best_motifs(subject_base, top=6)
            motif_state = _motif_state_ready() or {}
            entries = motif_state.get("entries", {})
    for motif_id, score in ranked:
        entry = entries.get(motif_id)
        if entry is None:
            continue
        chosen = (
            "motif",
            entry["path"],
            score,
            {
                "motif": motif_id,
                "motif_name": entry["name"],
                "category": entry["category"],
                "description": entry["description"],
            },
        )
        break

    # 2. stock candidates (only worth it when they can beat the motif)
    stock_best: tuple[str, Path, float | None, dict] | None = None
    try:
        candidates = _gather_candidates(subject_base, concept_visuals, fetch_dir, max_candidates=4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("candidate gathering failed: %s", exc)
        candidates = []
    if candidates:
        scorer = _clip_scorer()
        best = (None, candidates[0][1], candidates[0][0])
        for query, path in candidates:
            if path.name in used_stock_files:
                continue
            s = _clip_score(scorer, subject_base, path)
            if s is not None and (best[0] is None or s > best[0]):
                best = (s, path, query)
        stock_best = (
            "stock",
            best[1],
            best[0],
            {"query": best[2]},
        )

    # pick the pool winner; ties slightly favor the signed motif art
    if chosen and stock_best:
        motif_clip = chosen[2] if chosen[2] is not None else -1.0
        stock_clip = stock_best[2] if stock_best[2] is not None else -1.0
        winner = chosen if motif_clip >= stock_clip - 0.012 else stock_best
    else:
        winner = chosen or stock_best

    if winner:
        source, path, clip, meta = winner
        # Ensure ALL images get the dark treatment (motif, stock, generated)
        path = _ensure_darkened(path)
        if source == "motif":
            mid = meta.get("motif")
            if mid:
                used_motifs.add(mid)
        elif source == "stock":
            used_stock_files.add(path.name)
        url = f"file://{path.resolve()}"
        page_url = f"{source}:{path.name}"
        caption = f"{concept_visuals.concept}: {meta.get('motif_name') or meta.get('query', '')}"
        fname = path.name
    else:
        # unreachable while the motif library exists, but keep the tabs safe
        url = ""
        page_url = f"concept:{concept_visuals.concept}"
        caption = concept_visuals.concept
        fname = ""
        source, clip, meta = "none", None, {}

    if clip is not None and clip < CLIP_WEAK:
        logger.info(
            "weak visual match (%.2f) for %r [%s %s]",
            clip, sent[:60], source, meta.get("query", meta.get("motif", "")),
        )

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
            "source": source,
            **meta,
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

    used_motifs: set[str] = set()
    used_stock_files: set[str] = set()
    assets: list[VisualAsset] = []
    for idx, (_ti, _si, sent) in enumerate(all_sents):
        concept_visuals = _select_concept_for_sentence(sent, used_concepts)

        position = idx / max(len(all_sents) - 1, 1) if len(all_sents) > 1 else 0.0
        asset = _build_asset_from_concept(
            sent, concept_visuals, fetch_dir, position,
            used_motifs=used_motifs, used_stock_files=used_stock_files,
        )
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
