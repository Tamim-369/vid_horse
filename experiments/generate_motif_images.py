"""Generate 100 original dark-psychology motif images (1080x1920).

Usage:
    uv run python experiments/generate_motif_images.py
Outputs:
    assets/motifs/<id>.png
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from render.motifs import MOTIFS

OUT = Path("assets/motifs")
OUT.mkdir(parents=True, exist_ok=True)
FONT_BOLD = "/usr/share/fonts/truetype/JetBrainsMonoNerdFont-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/JetBrainsMonoNerdFont-Regular.ttf"

PALETTES = {
    "face": ((8, 8, 12), (180, 60, 80)),
    "body": ((10, 10, 14), (90, 120, 180)),
    "shadow": ((5, 5, 8), (120, 120, 140)),
    "control": ((8, 6, 10), (160, 120, 60)),
    "space": ((6, 8, 12), (80, 140, 160)),
    "text": ((12, 12, 12), (200, 200, 200)),
    "surveillance": ((4, 8, 6), (60, 180, 120)),
    "abstract": ((6, 6, 14), (120, 80, 200)),
    "weather": ((6, 8, 10), (100, 140, 180)),
    "motion": ((8, 6, 8), (200, 80, 100)),
}


def make_background(w, h, rng, base, accent):
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    # vertical gradient
    top = np.array(base, dtype=np.float32)
    bot = np.array([min(255, c + 30) for c in base], dtype=np.float32)
    ys = np.linspace(0, 1, h)[:, None]
    grad = top[None, None, :] * (1 - ys)[:, :, None] + bot[None, None, :] * ys[:, :, None]
    grad = np.broadcast_to(grad, (h, w, 3)).copy()

    # radial glow
    cx = rng.uniform(0.2, 0.8) * w
    cy = rng.uniform(0.15, 0.5) * h
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.exp(-dist / (rng.uniform(180, 320)))[:, :, None]
    accent_arr = np.array(accent, dtype=np.float32)
    grad = grad + glow * accent_arr * rng.uniform(0.6, 1.1)

    # film grain
    noise = np_rng.normal(0, rng.uniform(6, 14), (h, w, 3))
    grad = grad + noise

    # vignette
    yy2, xx2 = np.mgrid[0:h, 0:w]
    cx2, cy2 = w / 2, h / 2
    vign = 1 - np.clip(np.sqrt(((xx2 - cx2) / (w * 0.55)) ** 2 + ((yy2 - cy2) / (h * 0.55)) ** 2), 0, 1)
    vign = vign ** 1.6
    grad = grad * vign[:, :, None]

    grad = np.clip(grad, 0, 255).astype(np.uint8)
    return Image.fromarray(grad, mode="RGB")


def draw_shape(draw, cat, rng, w, h, accent):
    color = tuple(int(c) for c in accent) + (120,)
    if cat == "face":
        r = int(w * 0.22)
        draw.ellipse((w // 2 - r, h // 2 - r - 200, w // 2 + r, h // 2 + r - 200), outline=accent, width=3)
        draw.ellipse((w // 2 - 40, h // 2 - 240, w // 2 - 10, h // 2 - 210), fill=accent)
        draw.ellipse((w // 2 + 10, h // 2 - 240, w // 2 + 40, h // 2 - 210), fill=accent)
    elif cat == "body":
        x = w // 2
        draw.line((x, h // 2 - 300, x, h // 2 + 300), fill=accent, width=4)
        draw.line((x - 120, h // 2 - 100, x + 120, h // 2 - 100), fill=accent, width=4)
    elif cat == "shadow":
        for i in range(6):
            x = rng.randint(0, w)
            draw.line((x, 0, x + 400, h), fill=accent, width=2)
    elif cat == "control":
        # grid
        for i in range(0, w, 80):
            draw.line((i, 0, i, h), fill=(80, 80, 80), width=1)
        for i in range(0, h, 80):
            draw.line((0, i, w, i), fill=(80, 80, 80), width=1)
    elif cat == "space":
        # perspective lines
        vp = (w // 2, h // 3)
        for i in range(-6, 7):
            draw.line((vp[0] + i * 120, h, vp[0], vp[1]), fill=accent, width=1)
        draw.line((0, vp[1], w, vp[1]), fill=accent, width=2)
    elif cat == "text":
        draw.line((w // 2 - 200, h // 2, w // 2 + 200, h // 2), fill=accent, width=4)
    elif cat == "surveillance":
        for y in range(0, h, 6):
            draw.line((0, y, w, y), fill=(0, 40, 0), width=1)
    elif cat == "abstract":
        for _ in range(12):
            x, y = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(20, 120)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=accent, width=2)
    elif cat == "weather":
        for _ in range(60):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            draw.line((x, y, x + 4, y + 30), fill=accent, width=1)
    elif cat == "motion":
        for _ in range(8):
            y = rng.randint(h // 3, 2 * h // 3)
            draw.line((0, y, w, y), fill=accent, width=rng.randint(1, 4))


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = (cur + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def generate_one(motif):
    rng = random.Random(motif.id)
    base, accent = PALETTES[motif.category]
    w, h = 1080, 1920
    img = make_background(w, h, rng, base, accent)
    draw = ImageDraw.Draw(img, "RGBA")

    # shape overlay
    shape_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shape_layer)
    draw_shape(sd, motif.category, rng, w, h, accent)
    img = Image.alpha_composite(img.convert("RGBA"), shape_layer)

    draw = ImageDraw.Draw(img)
    # category label top
    font_cat = ImageFont.truetype(FONT_REG, 28)
    cat_text = motif.category.upper()
    draw.text((60, 80), cat_text, font=font_cat, fill=(200, 200, 200, 180))

    # motif id small
    font_id = ImageFont.truetype(FONT_REG, 22)
    draw.text((60, 120), motif.id, font=font_id, fill=(140, 140, 140, 160))

    # main title
    font_title = ImageFont.truetype(FONT_BOLD, 58)
    title_lines = wrap_text(motif.name, font_title, w - 120, draw)
    y = h // 2 - (len(title_lines) * 70) // 2 + 100
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=font_title)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        # shadow
        draw.text((x + 3, y + 3), line, font=font_title, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=font_title, fill=(240, 240, 245, 255))
        y += 72

    # description
    font_desc = ImageFont.truetype(FONT_REG, 26)
    desc_lines = wrap_text(motif.description, font_desc, w - 120, draw)
    y = h // 2 + 180
    for line in desc_lines[:4]:
        bbox = draw.textbbox((0, 0), line, font=font_desc)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        draw.text((x, y), line, font=font_desc, fill=(180, 180, 190, 200))
        y += 34

    # bottom cue
    font_cue = ImageFont.truetype(FONT_REG, 20)
    cue = motif.cues
    cue_lines = wrap_text(cue, font_cue, w - 120, draw)
    y = h - 120
    for line in cue_lines[:2]:
        bbox = draw.textbbox((0, 0), line, font=font_cue)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        draw.text((x, y), line, font=font_cue, fill=(120, 120, 130, 160))
        y += 26

    img = img.convert("RGB")
    img.save(OUT / f"{motif.id}.png", "PNG", optimize=True)


def main():
    for m in MOTIFS:
        out = OUT / f"{m.id}.png"
        if out.exists():
            continue
        generate_one(m)
        print("generated", m.id)
    print("done")


if __name__ == "__main__":
    main()
