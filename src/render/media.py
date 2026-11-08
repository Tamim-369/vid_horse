"""Shared render helpers: ffmpeg runner, downloads, adjacency rule."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import urllib.request
from pathlib import Path

import torchaudio

from core.models import VisualAsset

logger = logging.getLogger(__name__)

WIDTH = 1080
HEIGHT = 1920
FPS = 25
MUSIC_GAIN = 0.35
MUSIC = Path("assets/music/normal.mp3")
MUSIC_SHELF = Path("assets/music/music.json")
CARD_BG = "0x0d1117"
CARD_TEXT_LINES = 26

# Working canvas for motion: 4x the output frame, keeps every animated crop
# sub-pixel smooth and keeps a cropped wallpaper shot readable.
CANVAS_W = 4320
CANVAS_H = 7680

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
UA_HEADER = {"User-Agent": "videoai-pipeline/0.1"}


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {' '.join(args[:8])}...\n{result.stderr[-1500:]}"
        )


def _font() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _duration(wav_path: Path) -> float:
    info = torchaudio.info(str(wav_path))
    return info.num_frames / info.sample_rate


def _no_back_to_back(cards: list[list[VisualAsset]]) -> None:
    """Reorder every card's visuals so ADJACENT cards never lead with the
    same image. An image may return later in the video — it just may not
    play for one sentence and then immediately for the next sentence.
    """

    previous: str | None = None
    for visuals in cards:
        if not visuals:
            continue
        if previous is not None and visuals[0].url == previous:
            pick = next(
                (i for i in range(1, len(visuals)) if visuals[i].url != previous),
                None,
            )
            if pick is not None:
                visuals.insert(0, visuals.pop(pick))
        previous = visuals[0].url


def _fetch_media(url: str, work: Path, name: str) -> Path:
    """Download ``url`` into ``work`` under a content-addressed name.

    Slot-named caching (img-00-0.jpg) made every re-render reuse the
    previous run's download, so an updated asset pack still showed stale
    pictures — the return of the grandpa. The digest name keeps repeat
    runs free of re-downloads without ever confusing two images. Videos
    ride the same path; ``name`` carries the real extension.
    """

    digest = hashlib.sha1(url.encode()).hexdigest()[:10]
    stem = name.rsplit(".", 1)[0]
    path = work / f"{stem}-{digest}{Path(name).suffix or '.jpg'}"
    if not path.exists():
        source = Path(url.removeprefix("file://"))
        if source.is_file():
            # local asset (motif library): copy instead of download
            path.write_bytes(source.read_bytes())
        else:
            request = urllib.request.Request(url, headers=UA_HEADER)
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
    return path


def _asset_source(asset, work: Path, name: str) -> Path:
    """Fetch one asset's pixels: full file for video, thumbnail for stills."""

    suffix = Path(asset.file).suffix.lower()
    if not suffix:
        suffix = ".webm" if asset.kind == "video" else ".jpg"
    url = asset.url if asset.kind == "video" else asset.thumb_url or asset.url
    if not url:
        raise ValueError(f"asset {name} has no usable url (placeholder)")
    return _fetch_media(url, work, f"{name}{suffix}")


def _is_video_file(source: Path) -> bool:
    return source.suffix.lower() in {".webm", ".ogg", ".ogv", ".mp4", ".mov"}


