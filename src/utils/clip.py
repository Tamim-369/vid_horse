"""CLIP scoring: judge thumbnails by what they DEPICT, not what they say.

A ViT-B/32 CLIP model embeds the subject text and each candidate's
thumbnail pixels; cosine similarity then ranks candidates by actual
visual match. This is what lexical caption-matching cannot do: a photo of
people sitting at a conference scores low against "interferometer" no
matter what its caption claims.
"""

from __future__ import annotations

import logging
from io import BytesIO

import httpx
import torch

logger = logging.getLogger(__name__)


class ClipScorer:
    """Lazy CLIP scorer over candidate thumbnails."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self._failed = False

    def _load(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        try:
            import open_clip

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._model.eval()
            self._tokenize = open_clip.get_tokenizer("ViT-B-32")
            return True
        except Exception as exc:  # noqa: BLE001 — scoring is best-effort
            logger.warning("CLIP unavailable, falling back to caption ranking: %s", exc)
            self._failed = True
            return False

    def score(
        self,
        subject_text: str,
        thumb_url: str | None,
        client: httpx.Client,
    ) -> float | None:
        """Cosine similarity between the subject text and the thumbnail."""

        if not self._load() or not thumb_url:
            return None
        try:
            response = client.get(thumb_url)
            response.raise_for_status()
            import PIL.Image

            image = self._preprocess(
                PIL.Image.open(BytesIO(response.content)).convert("RGB")
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Thumbnail fetch/prep failed for %s: %s", thumb_url, exc)
            return None
        with torch.no_grad():
            text = self._tokenize([subject_text[:180]])
            image_features = self._model.encode_image(image.unsqueeze(0))
            text_features = self._model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return float(
                (image_features @ text_features.T).squeeze().item()
            )
