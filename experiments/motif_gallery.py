"""Build a reviewable HTML gallery of the 100 visual motifs.

Usage:
    uv run python experiments/motif_gallery.py
Outputs:
    output/motif_gallery/index.html
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from render.motifs import MOTIFS

OUT = Path("output/motif_gallery")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "urls.json"

HEADERS = {"User-Agent": "ash_speaks/1.0"}


def fetch_url(term: str) -> str | None:
    try:
        r = httpx.get(
            "https://api.openverse.org/v1/images/",
            params={"q": term, "page_size": 1},
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        return results[0].get("thumbnail") or results[0].get("url")
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    urls: dict[str, str | None] = {}
    if CACHE.exists():
        urls = json.loads(CACHE.read_text())
    for m in MOTIFS:
        local = Path("assets/motifs_pollinations") / f"{m.id}.png"
        if local.exists():
            urls[m.id] = "../../assets/motifs_pollinations/" + f"{m.id}.png"
    CACHE.write_text(json.dumps(urls, indent=2))

    cards = []
    for m in MOTIFS:
        img = urls.get(m.id)
        search = "https://openverse.org/search/image?q=" + m.search_terms[0].replace(" ", "+")
        img_tag = (
            f'<a href="{search}" target="_blank"><img src="{img}" loading="lazy" alt="{m.name}"></a>'
            if img
            else f'<a href="{search}" target="_blank" class="placeholder">search images</a>'
        )
        cards.append(
            f"""<div class="card">
  <div class="img">{img_tag}</div>
  <div class="meta">
    <span class="cat">{m.category}</span>
    <h3>{m.id} — {m.name}</h3>
    <p>{m.description}</p>
    <small>{m.cues}</small>
  </div>
</div>"""
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Motif Gallery</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0b0b0f;color:#e6e6e6;margin:0;padding:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
.card{{background:#16161d;border-radius:10px;overflow:hidden;border:1px solid #2a2a33}}
.img{{aspect-ratio:4/3;background:#000;display:flex;align-items:center;justify-content:center}}
.img img{{width:100%;height:100%;object-fit:cover}}
.placeholder{{color:#666}}
.meta{{padding:10px}}
.cat{{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#9aa;opacity:.8}}
h3{{margin:4px 0;font-size:14px}}
p{{margin:4px 0;font-size:12px;opacity:.85}}
small{{font-size:11px;opacity:.6}}
</style></head><body>
<h1>Visual Motif Gallery ({len(MOTIFS)})</h1>
<div class="grid">
{''.join(cards)}
</div></body></html>"""
    (OUT / "index.html").write_text(html)
    print(f"Wrote {OUT/'index.html'}")


if __name__ == "__main__":
    main()
