"""QC motif assets with qwen3-vl:2b for distortions/garbled text.

Usage:
    uv run python experiments/qc_assets.py
Outputs:
    output/bad_assets.txt  (list of ids flagged BAD)
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx

ASSETS = Path("assets/motifs_pollinations")
OUT = Path("output/bad_assets.txt")
HOST = "http://localhost:11434"
MODEL = "qwen3-vl:2b"

PROMPT = (
    "Inspect this image. Is it distorted, unnatural, has garbled or unreadable text, "
    "or disturbing artifacts (especially deformed human faces/hands)? "
    "Reply with exactly one word first: BAD or OK, then a short reason."
)


def judge(img_path: Path) -> str:
    from io import BytesIO
    from PIL import Image
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((512, 512))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT, "images": [b64]},
        ],
        "stream": False,
    }
    try:
        r = httpx.post(f"{HOST}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


PROGRESS = Path("output/qc_progress.jsonl")


def main() -> None:
    done = set()
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    bad = []
    if OUT.exists():
        bad = [x for x in OUT.read_text().split() if x]
    for img in sorted(ASSETS.glob("*.png")):
        if img.stem in done:
            continue
        verdict = judge(img)
        print(img.stem, "->", verdict, flush=True)
        PROGRESS.write_text(
            (PROGRESS.read_text() if PROGRESS.exists() else "")
            + json.dumps({"id": img.stem, "verdict": verdict}) + "\n"
        )
        if verdict.upper().startswith("BAD"):
            bad.append(img.stem)
            OUT.write_text("\n".join(sorted(set(bad))) + "\n")
        time.sleep(0.5)
    OUT.write_text("\n".join(sorted(set(bad))) + "\n")
    print("BAD count:", len(set(bad)))


if __name__ == "__main__":
    main()
