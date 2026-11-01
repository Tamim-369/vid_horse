"""Quality gates: luminance floor, CLIP thresholds, vision judge."""

from __future__ import annotations

import logging

from .media import _CONFIDENT_CLIP

logger = logging.getLogger(__name__)




class JudgingMixin:
    """Mixin: quality gates, attached to AssetResearchBlock."""

    def _luminance(self, info: dict) -> float | None:
        """Mean brightness (0-255) of the thumbnail, for horror mood ranking."""

        if not self._fiction:
            return None
        url = info.get("thumburl") or info.get("url")
        if not url:
            return None
        try:
            from io import BytesIO

            from PIL import Image

            response = self._commons.fetch_bytes(url)
            image = Image.open(BytesIO(response)).convert("L").resize((32, 32))
            pixels = list(image.getdata())
            return sum(pixels) / len(pixels)
        except Exception:  # noqa: BLE001 — mood ranking is best-effort
            return None

    def _vision_policy(self, clip_score: float | dict | None) -> bool:
        """True when this pick still needs the vision model's judgement.

        CLIP is a trained semantic judge: a confident score needs no
        second opinion. A borderline one does, so the model looks at the
        actual pixels before the image ships. Accepts either a raw score
        or an info dict with ``_clip_score``.
        """

        if isinstance(clip_score, dict):
            clip_score = clip_score.get("_clip_score")
        return float(clip_score or 0.0) < _CONFIDENT_CLIP

    def _vision_vet(self, info: dict, scene: str, title: str = "") -> bool:
        """Ask the local vision model whether this image fits the moment.

        Semantics of the verdict:
        - model looked and said relevant (confident) -> ship
        - model looked and said irrelevant -> drop
        - image unreachable -> drop (a shot we cannot inspect sits on
          chance alone)
        - model error / inconclusive -> keep: CLIP already scored these
          pixels above the floor; absence of a second opinion is not
          evidence against them
        """

        if self._vision is None:
            return True
        try:
            check = self._vision.check_relevance(
                info.get("thumburl") or info.get("url"),
                info.get("description") or info.get("title") or "",
                scene,
                title,
            )
        except Exception as exc:
            logger.warning("Vision vet failed for %s: %s", info.get("url"), exc)
            info["_vision_reason"] = f"vet error: {exc}"
            return True  # inconclusive beats blind rejection
        info["_vision_relevant"] = None if not check.checked else check.relevant
        info["_vision_confidence"] = None if not check.checked else check.confidence
        info["_vision_reason"] = check.reason
        if not check.checked:
            serious = "unreachable:" in check.reason
            logger.info(
                "Vision verdict for %s: %s", info.get("url"), (
                    "unreachable - reject" if serious else "inconclusive - keep"
                )
            )
            return not serious
        return check.relevant and check.confidence >= 0.5
