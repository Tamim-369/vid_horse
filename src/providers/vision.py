"""Vision model provider for image relevance checking."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class ImageUnreachable(RuntimeError):
    """The image bytes could not be fetched from any source."""


@dataclass
class DecisionJudgeCLI:
    """Ultra-fast caption judge on ollama's ``--ask`` decision models.

    Sub-2s verdicts with logit confidence. Works on text alone: it reads
    the caption, the narration and the video subject and decides whether
    the two plausibly belong together. It cannot see pixels — that is
    the deep look's job; this stage just kills the obvious junk cheaply.
    """

    def __init__(self, model: str = "d3:2b") -> None:
        self._model = model

    def judge(self, text: str, ask: str) -> tuple[str, float]:
        """Returns ("yes"|"no"|"unknown", confidence 0..1)."""

        import subprocess

        try:
            proc = subprocess.run(
                ["ollama", "run", self._model, "--ask", ask],
                input=text,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except Exception:  # busy host, missing CLI, timeout: judge absent
            return "unknown", 0.0
        tail = proc.stdout.strip().splitlines()
        tail = [line for line in tail if line.strip()]
        if not tail:
            return "unknown", 0.0
        words = tail[-1].split()
        verdict = "yes" if words[0].lower().startswith("yes") else (
            "no" if words[0].lower().startswith("no") else "unknown"
        )
        if len(words) > 1 and words[1].endswith("%"):
            with contextlib.suppress(ValueError):
                conf = float(words[1].rstrip("%")) / 100
        else:
            # a bare yes/no states no confidence; pretending it is 0.9
            # makes the cascade auto-accept on the threshold. A verdict
            # without a stated confidence defers to the pixel judge
            return "unknown", 0.0
        return verdict, conf


@dataclass
class VisionCheckResult:
    relevant: bool
    confidence: float
    reason: str
    # False when the model could not actually judge (fetch or ollama failed)
    checked: bool = True


class VisionRelevanceProvider:
    """Uses a local vision model (qwen3-vl:2b) to check if an image is relevant to a topic."""

    def __init__(
        self,
        model: str = "qwen3-vl:2b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._cache: dict[str, VisionCheckResult] = {}
        self._decision: DecisionJudgeCLI | None = None
        self._cache_path = None
        self._cache_db = None
        self._cache_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._img_headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://yandex.com/",
        }
        self._plain_headers = {
            "User-Agent": "videoai-image-qc/1.0 (contact: qc@videoai.local)",
            "Accept": "image/*,*/*;q=0.8",
        }

    def d3_judge(self, caption: str, narration: str) -> tuple[bool, float]:
        """Fast text decision via d3:2b (caption vs narration). Returns (relevant, prob)."""

        try:
            import ollama

            resp = ollama.systemone(
                model="d3:2b",
                state=f"Narration moment: {narration[:300]}\nImage caption: {caption[:300]}",
                questions={
                    "relevant": {
                        "type": "noul",
                        "instructions": (
                            "Does this image give the right mood/vibe for this "
                            "narration moment in a serious documentary?"
                        ),
                        "criteria": {
                            "true": "Image evokes the right atmosphere and could plausibly "
                                    "illustrate this moment.",
                            "false": "Unrelated, wrong mood, wrong subject, or text-heavy.",
                        },
                    }
                },
            )
            prob = float(resp.answers["relevant"].noul)
            return prob >= 0.3, prob
        except Exception as exc:  # noqa: BLE001
            logger.debug("d3 judge failed: %s", exc)
            return False, 0.0

    def attach_cache(self, path) -> None:
        """Persist verdicts in a sqlite table so checks are paid once ever."""

        import sqlite3

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = str(path)
        self._cache_db = sqlite3.connect(str(path), check_same_thread=False)
        self._cache_lock = threading.Lock()
        self._cache_db.execute(
            "CREATE TABLE IF NOT EXISTS vision_cache ("
            "key TEXT PRIMARY KEY, relevant INTEGER, confidence REAL, reason TEXT)"
        )
        self._cache_db.commit()

    def _fetch_image(self, image_url: str) -> bytes:
        """Fetch the image bytes, degrading gracefully per host.

        Wikimedia 403s the hotlink-style browser headers, so every fetch
        walks a ladder of (url, headers) pairs until one answers.
        """

        import httpx as _httpx

        clean_url = image_url.split("?")[0]
        attempts = [
            (clean_url, self._plain_headers),
            (image_url, self._img_headers),
        ]
        seen: set[tuple[str, str]] = set()
        last_error: Exception | None = None
        for url, headers in attempts:
            if (url, headers["User-Agent"]) in seen:
                continue
            seen.add((url, headers["User-Agent"]))
            try:
                response = _httpx.get(
                    url, timeout=8.0, follow_redirects=True, headers=headers
                )
                response.raise_for_status()
                return response.content
            except Exception as exc:  # try the next rung
                last_error = exc
        raise ImageUnreachable(str(last_error or "no fetch attempt"))

    @staticmethod
    def _shrink(data: bytes, edge: int = 1024) -> bytes:
        """Encode the image small enough that the local model accepts it.

        Ollama refuses raw multi-thousand-pixel originals ("Failed to
        load image"); a 1024px JPEG keeps judgement quality without the
        memory spike.
        """

        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as img:
            img = img.convert("RGB")
            img.thumbnail((edge, edge))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()

    def _write_cache(self, key: str, result: VisionCheckResult) -> None:
        """Persist one verdict (thread-safe, no-op without disk cache)."""

        if self._cache_db is None:
            return
        try:
            with self._cache_lock:
                self._cache_db.execute(
                    "INSERT OR REPLACE INTO vision_cache VALUES (?,?,?,?)",
                    (key, int(result.relevant), result.confidence, result.reason),
                )
                self._cache_db.commit()
        except Exception:  # cache failure never blocks the judge
            pass

    def attach_decision(self, model: str = "d3:2b") -> None:
        """Turn on the fast caption-judge stage ahead of the pixel look."""

        self._decision = DecisionJudgeCLI(model=model)

    def check_relevance(
        self,
        image_url: str,
        topic: str,
        narration: str = "",
        title: str = "",
    ) -> VisionCheckResult:
        """Check if an image plausibly illustrates one narration moment.

        The judge is an EDITOR, not a fact-checker: no photograph exists
        of a unique historical event, so relevance means the image shows
        the kind of person, object or place the moment describes, in a
        plausible era/style — not that it depicts the literal scene.
        """
        # Check cache first (memory, then disk)
        import hashlib

        key = hashlib.sha1(
            f"{self._model}|{image_url}|{topic}|{narration}".encode()
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self._cache_db is not None:
            with self._cache_lock:
                row = self._cache_db.execute(
                    "SELECT relevant, confidence, reason FROM vision_cache WHERE key=?",
                    (key,),
                ).fetchone()
            if row is not None:
                result = VisionCheckResult(
                    relevant=bool(row[0]), confidence=row[1], reason=row[2]
                )
                self._cache[key] = result
                return result
        if self._decision is not None:
            # cascased stage 1: the fast caption judge decides what it
            # confidently can (sparse captions and wavy verdicts fall
            # through to the pixel look below)
            words = [
                w for w in (topic or "").split()
                if any(ch.isalpha() or ch.isdigit() for ch in w)
            ]
            dense = len(words) >= 3
            if dense:
                brief_text = (
                    f"Video: {title}\n\n"
                    f"Narration moment: {narration}\n\n"
                    f"Caption of the image: {topic}\n"
                )
                ask = (
                    "Would the captioned image plausibly appear on screen"
                    " during this narration moment in this video? Judge like"
                    " a film editor: too-generic stock, wrong-era, unrelated"
                    " subjects or anything displaying mostly text is a no."
                    " Answer yes or no."
                )
                verdict_text, conf = self._decision.judge(brief_text, ask)
                if verdict_text == "yes" and conf >= 0.85:
                    result = VisionCheckResult(True, conf, "caption judge accept")
                    self._cache[key] = result
                    self._write_cache(key, result)
                    return result
                if verdict_text == "no" and conf >= 0.60:
                    result = VisionCheckResult(False, conf, "caption judge reject")
                    self._cache[key] = result
                    self._write_cache(key, result)
                    return result

        try:
            img_data = self._shrink(self._fetch_image(image_url))
            img_b64 = base64.b64encode(img_data).decode()

            # Prepare the prompt — judge like a video editor
            prompt = (
                'Output only JSON: {"relevant":true,"confidence":0.9,"reason":""} '
                'or {"relevant":false,"confidence":0.1,"reason":""}.\n'
                f'Video is about: "{title}" (a speculative documentary).\n'
                f'Narration moment this image must illustrate: "{narration[:300]}"\n'
                'Judge like a film editor: relevant=true only if a viewer '
                "watching that narration with this picture on screen would "
                "accept the shot — i.e. it depicts the SAME KIND of thing the "
                "moment describes (right subject type, plausible era and "
                "style). Unique historical moments have no photographs: do "
                "NOT demand the literal event or the literal person. Good "
                "examples: for an early-1960s secret Soviet test-launch "
                "moment, any genuine 1950s-60s Soviet missile or Vostok "
                "rocket launch photo is relevant.\n"
                "Bad examples judged not relevant: a modern airliner for a "
                "1960s flight, a random dark portrait for a rocket launch, "
                "abstract shapes, unrelated logos, screens or text slides "
                "with title-card text, generic decorative stock.\n"
                f'Context of what the shot was searched for: "{topic}"'
            )

            payload = {
                "model": self._model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "format": "json",
                # the thinking channel must not swallow the answer token
                "think": False,
                "options": {"temperature": 0.1},
            }

            with self._model_lock:  # the CPU model serves ONE call at a time
                resp = self._client.post(
                    f"{self._host}/api/generate",
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                result = resp.json()

            content = result.get("response", "") or ""
            parsed = None
            if content.strip():
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
            thinking = result.get("thinking", "") or ""
            if parsed is None and thinking:
                probe = re.search(
                    r'\{[^{}\n]*"relevant"[^{}\n]*\}', thinking
                )
                if probe:
                    try:
                        parsed = json.loads(probe.group(0))
                    except json.JSONDecodeError:
                        parsed = None
            if parsed is None:
                raise ValueError("no JSON verdict in think or response channel")
            vision_result = VisionCheckResult(
                relevant=bool(parsed.get("relevant", False)),
                confidence=float(parsed.get("confidence", 0.0)),
                reason=parsed.get("reason", "No reason provided"),
                checked=True,
            )
            self._cache[key] = vision_result
            self._write_cache(key, vision_result)
            return vision_result

        except ImageUnreachable as exc:
            logger.warning("Vision fetch failed for %s: %s", image_url, exc)
            return VisionCheckResult(
                relevant=False,
                confidence=0.0,
                reason=f"Image unreachable: {exc}",
                checked=False,
            )
        except Exception as exc:
            logger.warning("Vision check failed for %s: %s", image_url, exc)
            return VisionCheckResult(
                relevant=False,
                confidence=0.0,
                reason=f"Check failed: {exc}",
                checked=False,
            )

    def close(self):
        self._client.close()


def build_vision_provider(
    model: str = "qwen3-vl:2b",
    host: str = "http://localhost:11434",
    timeout: float = 240.0,
) -> VisionRelevanceProvider:
    return VisionRelevanceProvider(model=model, host=host, timeout=timeout)
