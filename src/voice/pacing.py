"""Per-video pacing variation: every video breathes slightly differently.

Two videos from the same factory must not feel cloned. The safest lever is
pacing: voice speed, sentence gap and line gap each get a small deterministic
variation per output slug, always inside the ranges known to sound good
(calm, deliberate, never rushed).

The values are persisted to ``<output>/<slug>/pacing.json`` on first use so
later steps (voice re-render, subtitle alignment) read EXACTLY the values
the voice was rendered with — the subtitle aligner recomputes sentence
timing from these numbers and must never disagree with the wav cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from config.settings import get_settings

logger = logging.getLogger(__name__)

# variation ranges (settings defaults sit inside them)
_SPEED_MIN, _SPEED_STEP, _SPEED_STEPS = 0.90, 0.02, 4        # 0.90–0.96
_GAP_MIN, _GAP_STEP, _GAP_STEPS = 0.36, 0.02, 5              # 0.36–0.44
_LINE_GAP_MIN, _LINE_GAP_STEP, _LINE_GAP_STEPS = 0.60, 0.03, 4  # 0.60–0.69


def pacing_for(slug: str) -> dict[str, float]:
    """Deterministic pacing values derived from the output slug."""

    h = hashlib.sha256(slug.encode("utf-8")).digest()
    speed = round(_SPEED_MIN + (h[0] % _SPEED_STEPS) * _SPEED_STEP, 3)
    sentence_gap = round(_GAP_MIN + (h[1] % _GAP_STEPS) * _GAP_STEP, 3)
    line_gap = round(_LINE_GAP_MIN + (h[2] % _LINE_GAP_STEPS) * _LINE_GAP_STEP, 3)
    return {"speed": speed, "sentence_gap": sentence_gap, "line_gap": line_gap}


def _pacing_path(slug: str) -> Path:
    return Path(get_settings().output_dir) / slug / "pacing.json"


def ensure_pacing(slug: str) -> dict[str, float]:
    """First-use persistence: write once, then always read the same values."""

    path = _pacing_path(slug)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if {"speed", "sentence_gap", "line_gap"} <= set(data):
                return {k: float(data[k]) for k in ("speed", "sentence_gap", "line_gap")}
        except Exception:  # noqa: BLE001 - corrupt file: rewrite below
            pass
    values = pacing_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    logger.info("pacing for %s: %s", slug, values)
    return values


def load_pacing(slug: str | None) -> dict[str, float] | None:
    """The stored pacing for a slug, or None when absent/unknown."""

    if not slug:
        return None
    path = _pacing_path(slug)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if {"speed", "sentence_gap", "line_gap"} <= set(data):
            return {k: float(data[k]) for k in ("speed", "sentence_gap", "line_gap")}
    except Exception:  # noqa: BLE001
        return None
    return None
