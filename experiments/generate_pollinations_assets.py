"""Generate images via the Pollinations API.

Three modes:

1. library (default when no --queries): generate the fixed 100-motif
   library into assets/motifs_pollinations/ (skips existing files).

2. queries: generate one image per query phrase (crowd, protest,
   classroom, man fearing, ...) into assets/motifs_sentences/ — the
   generated pool the sentence-asset pipeline indexes alongside the
   motif library. Filename = slug of the phrase; re-runs skip existing.
   Pass phrases as CLI args or with --file (one phrase per line).

Usage:
    uv run python experiments/generate_pollinations_assets.py \
        --queries "crowd protest marching" "classroom full of students" \
                  "man frozen in fear"
    uv run python experiments/generate_pollinations_assets.py \
        --file assets/motif_seed_queries.txt
    uv run python experiments/generate_pollinations_assets.py --library

Env: POLLINATIONS_API_KEY (optional, raises rate limits).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LIBRARY_DIR = Path("assets/motifs_pollinations")
SENTENCES_DIR = Path("assets/motifs_sentences")
KEY = os.getenv("POLLINATIONS_API_KEY", "")
BASE = "https://image.pollinations.ai/prompt"

STYLE = (
    "dark cinematic noir psychological, 9:16 vertical, near-black gradient, "
    "heavy vignette, film grain, centered subject, high contrast, shallow depth, "
    "colored accent rim light, photorealistic scene, no text, no watermark, "
    "original, not a person photo"
)


def build_prompt(subject: str, seed: int, *, subject_style: bool = False) -> str:
    if subject_style:
        return f"{subject}. {STYLE}. seed {seed}."
    return f"{subject}. {STYLE}. seed {seed}."


def slugify(phrase: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower().strip()).strip("_")[:60]
    return slug or hashlib.sha1(phrase.encode()).hexdigest()[:10]


def fetch(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=120, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            time.sleep(2**attempt)
    return None


def _duplicates(out_dir: Path) -> set[str]:
    seen: set[str] = set()
    for existing in out_dir.glob("*.png"):
        seen.add(hashlib.sha256(existing.read_bytes()).hexdigest())
    return seen


def generate_one(
    phrase: str,
    out_path: Path,
    seen_hashes: set[str],
    *,
    attempts: int = 4,
) -> bool:
    """Generate one image for ``phrase`` at ``out_path``; True on success."""

    for _attempt in range(attempts):
        seed = random.randint(1, 10_000_000)
        prompt = urllib.parse.quote(build_prompt(phrase, seed))
        params = f"width=1080&height=1920&model=flux&nologo=true&seed={seed}"
        if KEY:
            params += f"&key={KEY}"
        data = fetch(f"{BASE}/{prompt}?{params}")
        if not data:
            time.sleep(2)
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_hashes:
            time.sleep(1)
            continue
        seen_hashes.add(digest)
        out_path.write_bytes(data)
        print("ok", out_path.name, "(seed", seed, ")", flush=True)
        return True
    print("fail", phrase, flush=True)
    return False


def run_library() -> None:
    from render.motifs import MOTIFS

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    seen = _duplicates(LIBRARY_DIR)
    for m in MOTIFS:
        out = LIBRARY_DIR / f"{m.id}.png"
        if out.exists():
            seen.add(hashlib.sha256(out.read_bytes()).hexdigest())
            continue
        generate_one(m.description + ". " + m.cues, out, seen)
        time.sleep(2)


def run_queries(phrases: list[str]) -> None:
    SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
    seen = _duplicates(SENTENCES_DIR)
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase or phrase.startswith("#"):
            continue
        out = SENTENCES_DIR / f"{slugify(phrase)}.png"
        if out.exists() and out.stat().st_size > 0:
            print("skip", out.name, "(exists)", flush=True)
            continue
        generate_one(phrase, out, seen)
        time.sleep(2)


def read_phrase_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.split("=", 1)[-1].strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", action="store_true", help="Generate the 100-motif library.")
    parser.add_argument("queries", nargs="*", help="Query phrases (scene-topic images).")
    parser.add_argument("--file", help="File with one phrase per line (also motif=phrase).")
    args = parser.parse_args()

    if args.library:
        run_library()
        return
    phrases = list(args.queries)
    if args.file:
        phrases.extend(read_phrase_file(Path(args.file)))
    if not phrases:
        parser.error("nothing to generate: pass --queries phrases, --file, or --library")
    run_queries(phrases)


if __name__ == "__main__":
    main()
