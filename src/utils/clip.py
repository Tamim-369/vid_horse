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
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ClipScorer:
    """Lazy CLIP scorer over candidate thumbnails.

    ``threads`` pins torch intra-op threads to the burst size the
    scheduler agreed on: several scorer pools run side by side while
    other stages' encodes ebb and flow — every pool owning exactly its
    slice of the cores instead of each grabbing the whole machine.
    """

    def __init__(self, user_agent: str, threads: int | None = None) -> None:
        self._user_agent = user_agent
        self._threads = threads
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

            if self._threads:
                torch.set_num_threads(self._threads)
                torch.set_num_interop_threads(1)
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._model.eval()
            self._tokenize = open_clip.get_tokenizer("ViT-B-32")
            if self._threads:
                logger.info("CLIP scorer pinned to %d torch threads", self._threads)
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

        return self.analyze(subject_text, thumb_url, client)[0]

    def text_embed(self, subject_text: str) -> torch.Tensor | None:
        """Unit-normalized CLIP text embedding (shelf matching)."""

        if not self._load():
            return None
        with torch.no_grad():
            features = self._model.encode_text(self._tokenize([subject_text[:180]]))
            return features / features.norm(dim=-1, keepdim=True)

    def image_embed(
        self, thumb_url: str | None, client: httpx.Client
    ) -> np.ndarray | None:
        """Unit-normalized CLIP image embedding from one thumb download."""

        if not self._load() or not thumb_url:
            return None
        try:
            response = client.get(thumb_url)
            response.raise_for_status()
            import PIL.Image

            image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embed fetch failed for %s: %s", thumb_url, exc)
            return None
        return self._image_features(image)

    def _image_features(self, image) -> torch.Tensor | None:
        prepared = self._preprocess(image)
        with torch.no_grad():
            features = self._model.encode_image(prepared.unsqueeze(0))
            return features / features.norm(dim=-1, keepdim=True)

    def analyze(
        self,
        subject_text: str,
        thumb_url: str | None,
        client: httpx.Client,
    ) -> tuple[float | None, bool, bool]:
        """Score one thumbnail from a single download.

        Returns ``(similarity, collage_like, texty)``. ``collage_like``
        means the image is a multi-panel sheet — photo collages with
        white interior gutters — whose separate panels never match a
        single narration moment on the vertical feed. ``texty`` means
        OCR finds a wall of words on the pixels: a poster or diagram,
        not a photograph of the subject.
        """

        if not self._load() or not thumb_url:
            return None, False, False
        try:
            response = client.get(thumb_url)
            response.raise_for_status()
            import PIL.Image

            image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Thumbnail fetch/prep failed for %s: %s", thumb_url, exc)
            return None, False, False
        return (
            self._similarity(subject_text, image),
            self._collage_like(image),
            self._texty(image),
        )

    def _similarity(self, subject_text: str, image) -> float | None:
        prepared = self._preprocess(image)
        with torch.no_grad():
            text = self._tokenize([subject_text[:180]])
            image_features = self._model.encode_image(prepared.unsqueeze(0))
            text_features = self._model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return float(
                (image_features @ text_features.T).squeeze().item()
            )

    def _collage_like(self, image) -> bool:
        """Multi-panel sheets: 2+ interior seams between panels.

        A seam is a short near-flat column/row (any shade — white padding
        or the thin dark bars of NASA photo collages) whose shade sits at
        an extreme and contrasts with the panel beside it. Uniform sky or
        starfield regions fail on width (too wide), shade (not extreme),
        and contrast. Runs must sit inside the image so page margins and
        single photos never count; a 3-panel sheet has 2 seams, a 2x2
        grid has 1 of each.
        """

        import numpy as np

        scaled = image.copy()
        scaled.thumbnail((1024, 1024))
        grid = np.asarray(scaled.convert("L"), dtype=np.float64)

        total = 0
        for axis in (0, 1):
            strip = grid if axis == 0 else grid.T
            ptp = np.ptp(strip, axis=0)
            mean = strip.mean(axis=0)
            flat = ptp <= 60
            n = len(flat)
            index = 0
            while index < n - 4:
                if not flat[index]:
                    index += 1
                    continue
                end = index
                while end < n and flat[end]:
                    end += 1
                width = end - index
                shade = float(mean[index:end].mean())
                extreme = shade <= 45 or shade >= 215
                neighbours = []
                if index > 0 and not flat[index - 1]:
                    neighbours.append(float(mean[index - 1]))
                if end < n and not flat[end]:
                    neighbours.append(float(mean[end]))
                contrast = neighbours and max(
                    abs(shade - nb) for nb in neighbours
                ) >= 60
                if 1 <= width <= 8 and extreme and index > 2 and end < n - 3 and contrast:
                    total += 1
                index = max(end, index + 1)
        return total >= 2

    def _texty(self, image) -> bool:
        """Posters and diagrams: OCR reads many confident words.

        Artist concepts and photographs carry no readable text beyond an
        occasional one-word label; an infographic carries a title plus
        caption lines — dozens of characters at high confidence. Any
        tesseract failure counts as not-texty so detection never blocks
        a usable image.
        """

        import subprocess

        try:
            scaled = image.copy()
            scaled.thumbnail((1024, 1024))
            buffer = BytesIO()
            scaled.save(buffer, format="PNG")
            result = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
                input=buffer.getvalue(),
                capture_output=True,
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("OCR unavailable, skipping texty check: %s", exc)
            return False
        words = 0
        chars = 0
        strong = 0
        for line in result.stdout.decode("utf-8", "replace").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12 or parts[0] != "5":
                continue
            try:
                conf = float(parts[10])
            except ValueError:
                continue
            text = parts[11].strip()
            if conf < 60 or len(text) < 3 or not text.isalnum():
                continue
            words += 1
            chars += len(text)
            if conf >= 80:
                strong += 1
        return strong >= 3 or words >= 6 or chars >= 30
