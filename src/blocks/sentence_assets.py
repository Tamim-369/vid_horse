"""Per-sentence asset agent: assign one motif image to every spoken sentence.

Splits each scene (and hook/outro) into sentences, scores motifs by content
overlap, and returns an AssetPack with multiple timed assets per scene.
The render's trigger/position alignment places each image at its sentence.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

from core.models import AssetPack, SceneAssets, VideoScript, VisualAsset
from render.motifs import MOTIFS, Motif

MOTIF_DIR = Path("assets/motifs_pollinations")
_TOKEN_STRIP = re.compile(r"[^a-z0-9 ]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS = frozenset(
    [
        "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
        "when", "while", "your", "you", "they", "them", "their", "this",
        "that", "these", "those", "what", "how", "why", "not", "no", "yes",
        "just", "like", "really", "very", "much", "more", "most", "some",
        "every", "all", "about", "into", "over", "under", "with", "from",
        "for", "of", "on", "at", "by", "to", "in", "out", "up", "down",
        "next", "time", "one", "two", "thing", "things", "says", "said",
        "say", "tell", "tells", "told", "ask", "asks", "asking", "will",
        "would", "can", "could", "should", "was", "are", "is", "were",
        "been", "being", "have", "has", "had", "do", "does", "did", "now",
        "here", "there", "where", "who", "whom", "whose", "its", "get",
        "gets", "got", "make", "makes", "made", "see", "know", "think",
    ]
)


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    if not parts and text.strip():
        return [text.strip()]
    # merge tiny fragments into the previous sentence
    merged: list[str] = []
    for part in parts:
        if merged and len(part.split()) < 4:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged or ([text.strip()] if text.strip() else [])


def _tokens(text: str) -> set[str]:
    words = _TOKEN_STRIP.sub(" ", text.lower()).split()
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def _motif_tokens(motif: Motif) -> set[str]:
    tokens = _tokens(motif.name) | _tokens(motif.description)
    for term in motif.search_terms:
        tokens |= _tokens(term)
    return tokens


_MOTIF_TOKENS = {m.id: _motif_tokens(m) for m in MOTIFS}


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


def _pexels_fetch(query: str, fetch_dir: Path, key: str) -> Path | None:
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

def assign_sentence_assets(
    texts: list[str], *, used: set[str] | None = None, last_cat: str | None = None, fetch_dir: Path | None = None
) -> list[VisualAsset]:
    """One motif per sentence in ``texts``, timed by trigger/position."""

    if used is None:
        used = set()
    all_sents: list[tuple[int, int, str]] = []
    for ti, text in enumerate(texts):
        sents = _sentences(text)
        for si, sent in enumerate(sents):
            all_sents.append((ti, si, sent))
    if not all_sents:
        return []

    assets: list[VisualAsset] = []
    for idx, (_ti, _si, sent) in enumerate(all_sents):
        tokens = _tokens(sent)
        scored = []
        for motif in MOTIFS:
            if motif.id in used:
                continue
            overlap = len(tokens & _MOTIF_TOKENS[motif.id])
            penalty = 2 if motif.category == last_cat else 0
            scored.append((overlap - penalty, overlap, motif.id, motif))
        scored.sort(key=lambda item: (-item[0], item[2]))
        best_score = scored[0][1] if scored else 0
        best = scored[0][3] if scored else MOTIFS[idx % len(MOTIFS)]
        used.add(best.id)
        last_cat = best.category
        position = idx / max(len(all_sents) - 1, 1) if len(all_sents) > 1 else 0.0
        file_path = MOTIF_DIR / f"{best.id}.png"
        url = f"file://{file_path.resolve()}"
        page_url = f"motif:{best.id}"
        caption = best.description
        fname = file_path.name
        # fallback to Pexels (darkened) when no good match or file missing
        need_fetch = (best_score <= 0 and bool(tokens)) or not file_path.exists()
        if need_fetch and fetch_dir is not None:
            fetched = _pexels_fetch(sent[:120], fetch_dir, "")
            if fetched is not None:
                url = f"file://{fetched.resolve()}"
                page_url = f"pexels:{fetched.name}"
                caption = f"Dark visual for: {sent[:80]}"
                fname = fetched.name
        assets.append(
            VisualAsset(
                kind="image",
                file=fname,
                url=url,
                page_url=page_url,
                caption=caption,
                trigger=_trigger(sent),
                position=round(position, 3),
            )
        )
    return assets


def build_asset_pack(script: VideoScript, *, slug: str) -> AssetPack:
    """Assign one motif image per sentence of every scene (and hook)."""

    used: set[str] = set()
    last_cat: str | None = None
    scenes: list[SceneAssets] = []
    fetch_dir = Path("output") / slug / "fetched"

    # hook gets its own visual (scene_index -1, consumed by render)
    hook_assets = assign_sentence_assets([script.hook], used=used, last_cat=last_cat, fetch_dir=fetch_dir)
    if hook_assets:
        last_cat = hook_assets[-1].page_url.split(":")[-1].split("_")[0]
    scenes.append(
        SceneAssets(
            scene_index=-1,
            visual=script.hook,
            query="hook",
            assets=hook_assets,
        )
    )

    for index, scene in enumerate(script.scenes):
        # derive last_cat from last used motif
        if assets_prev := scenes[-1].assets:
            mid = assets_prev[-1].page_url.split(":")[-1]
            last_cat = mid.split("_")[0]
        sent_assets = assign_sentence_assets(
            [scene.narration], used=used, last_cat=last_cat, fetch_dir=fetch_dir
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
        article_title=f"motif library ({MOTIF_DIR})",
        article_url="",
        scenes=scenes,
    )


def choose_motifs(script: VideoScript) -> list[Motif]:
    """Compatibility: one motif per scene (first sentence's pick)."""
    used: set[str] = set()
    last_cat: str | None = None
    result: list[Motif] = []
    for scene in script.scenes:
        assets = assign_sentence_assets([scene.narration], used=used, last_cat=last_cat)
        if assets:
            mid = assets[0].page_url.split(":")[-1]
            motif = next((m for m in MOTIFS if m.id == mid), MOTIFS[0])
            result.append(motif)
            last_cat = motif.category
    return result
