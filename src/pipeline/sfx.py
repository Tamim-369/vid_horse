"""Horror sound-effects shelf: PD/CC audio pulled from Wikimedia Commons.

``sfx-pull`` grows a small, curated shelf of creaks, wind, thunder and
similar effects in ``assets/sfx``. A manifest keeps URL + license per file;
the sound-design step later picks from the shelf by keyword.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlsplit

from providers.sources.mediawiki import MediaWikiClient

logger = logging.getLogger(__name__)

SFX_TERMS: dict[str, str] = {
    "thunder": "thunder rumble filetype:audio",
    "wind": "wind howling filetype:audio",
    "creak": "door creaking filetype:audio",
    "steps": "footsteps concrete filetype:audio",
    "static": "radio static filetype:audio",
    "heart": "heartbeat filetype:audio",
    "drone": "dark ambient drone filetype:audio",
    "impact": "loud boom filetype:audio",
    "whisper": "whisper voice filetype:audio",
    "knock": "knocking door filetype:audio",
}

_KEEP = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _KEEP.sub("-", text.lower()).strip("-")[:40]


def _audio_seconds(path: Path) -> float | None:
    """Probe an audio file's duration with torchaudio metadata."""

    import torchaudio

    try:
        meta = torchaudio.info(str(path))
        return meta.num_frames / meta.sample_rate
    except Exception:  # noqa: BLE001 — unknown formats get rejected
        return None


def pull_sfx(settings, commons: MediaWikiClient, out_dir: Path) -> Path:
    """Fetch a few audio hits per term and store them with a manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sfx.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    have = {entry["url"] for entry in manifest}

    for term, query in SFX_TERMS.items():
        existing = sum(1 for entry in manifest if entry["term"] == term)
        if existing >= 2:
            continue
        try:
            hits = commons.search_media(query, limit=6, thumb_width=1280)
        except Exception as exc:  # noqa: BLE001 — one bad term must not stop the shelf
            logger.warning("SFX term '%s' failed: %s", term, exc)
            continue
        for hit in hits:
            if existing >= 2:
                break
            url = hit.get("url", "")
            if not url or url in have or not hit.get("mime", "").startswith("audio/"):
                continue
            if hit.get("size", 0) and hit.get("size", 0) > 8 * 1024 * 1024:
                continue
            clean_path = Path(urlsplit(url).path)
            suffix = clean_path.suffix if clean_path.suffix else ".ogg"
            path = out_dir / f"{term}-{existing+1}{suffix}"
            try:
                payload = commons.fetch_bytes(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SFX download failed for %s: %s", url, exc)
                continue
            path.write_bytes(payload)
            duration = _audio_seconds(path)
            if duration is None or duration > 15.0:
                path.unlink(missing_ok=True)
                continue
            manifest.append(
                {
                    "term": term,
                    "file": str(path),
                    "title": hit.get("title") or hit.get("file", ""),
                    "url": url,
                    "license": hit.get("license", ""),
                    "artist": hit.get("artist", ""),
                    "seconds": round(duration, 2),
                }
            )
            have.add(url)
            existing += 1
            logger.info("SFX shelf: %s -> %s (%s)", term, path.name, hit.get("license"))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("SFX shelf ready: %d files at %s", len(manifest), out_dir)
    return manifest_path
