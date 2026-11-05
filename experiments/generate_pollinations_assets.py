"""Generate motif assets via Pollinations API.

Usage:
    POLLINATIONS_API_KEY=... uv run python experiments/generate_pollinations_assets.py
Outputs:
    assets/motifs_pollinations/<id>.png
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from render.motifs import MOTIFS

OUT = Path("assets/motifs_pollinations")
OUT.mkdir(parents=True, exist_ok=True)
KEY = os.getenv("POLLINATIONS_API_KEY", "")
BASE = "https://image.pollinations.ai/prompt"

STYLE = (
    "dark cinematic noir psychological, 9:16 vertical, near-black gradient, "
    "heavy vignette, film grain, centered subject, high contrast, shallow depth, "
    "colored accent rim light, no text, no watermark, original, not a person photo"
)


def build_prompt(m, seed: int) -> str:
    return f"{m.description}. {m.cues}. {STYLE}. category {m.category}. seed {seed}."


def fetch(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=90, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def main() -> None:
    seen_hashes: set[str] = set()
    for existing in OUT.glob("*.png"):
        seen_hashes.add(hashlib.sha256(existing.read_bytes()).hexdigest())

    for m in MOTIFS:
        out = OUT / f"{m.id}.png"
        if out.exists():
            seen_hashes.add(hashlib.sha256(out.read_bytes()).hexdigest())
            continue

        success = False
        for attempt in range(5):
            seed = random.randint(1, 10_000_000)
            prompt = urllib.parse.quote(build_prompt(m, seed))
            params = f"width=1080&height=1920&model=flux&nologo=true&seed={seed}"
            if KEY:
                params += f"&key={KEY}"
            url = f"{BASE}/{prompt}?{params}"
            data = fetch(url)
            if not data:
                time.sleep(2)
                continue
            h = hashlib.sha256(data).hexdigest()
            if h in seen_hashes:
                time.sleep(1)
                continue
            seen_hashes.add(h)
            out.write_bytes(data)
            print("ok", m.id, "seed", seed, flush=True)
            success = True
            break
        if not success:
            print("fail", m.id, flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
