"""OpenRouter-backed LLM provider with model rotation.

Uses OpenRouter's OpenAI-compatible chat-completions API. Holds a pool of
free models; if a model is rate-limited, unavailable, or fails, it rotates
to the next model so one bad model cannot stop a run.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import httpx

from core.errors import ConfigurationError, LLMError
from providers.llm.base import BaseLLMProvider
from utils.cache import DiskCache

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "recursive/flint-v1.0:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
]


class OpenRouterProvider(BaseLLMProvider):
    name = "openrouter"
    supports_native_schema = False

    def __init__(
        self,
        *,
        api_key: str,
        models: Sequence[str] | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        cache: DiskCache | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ConfigurationError("OpenRouterProvider requires an API key")
        super().__init__(
            model=(models[0] if models else DEFAULT_MODELS[0]),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache=cache,
        )
        self._api_key = api_key.strip()
        self._models = [m.strip() for m in (models or DEFAULT_MODELS) if m and m.strip()]
        if not self._models:
            self._models = list(DEFAULT_MODELS)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._lock = threading.Lock()
        self._cooldown: dict[int, float] = {}
        self._dead: set[int] = set()
        self._index = 0
        self._client = client or httpx.Client(timeout=timeout)

    def fingerprint(self) -> dict:
        return {**super().fingerprint(), "base_url": self._base_url, "models": self._models}

    def _model_order(self):
        total = len(self._models)
        now = __import__("time").monotonic()
        with self._lock:
            self._cooldown = {i: t for i, t in self._cooldown.items() if t > now}
            hot = set(self._cooldown)
        fresh, cooling = [], []
        for offset in range(total):
            idx = (self._index + offset) % total
            if idx in self._dead:
                continue
            (cooling if idx in hot else fresh).append(idx)
        yield from fresh
        yield from cooling

    def _generate(self, messages, *, options, schema):
        import time as _time
        payload = {
            "model": options.get("model", self.model),
            "messages": list(messages),
            "temperature": options.get("temperature", self.temperature),
            "max_tokens": options.get("num_predict") or self.max_tokens or 4096,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if "seed" in options:
            payload["seed"] = options["seed"]

        errors = []
        for idx in self._model_order():
            model = self._models[idx]
            payload["model"] = model
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://videoai.local",
                        "X-Title": "videoai",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model}: request error {exc}")
                logger.warning("OpenRouter model '%s' request error: %s", model, exc)
                with self._lock:
                    self._cooldown[idx] = _time.monotonic() + 10.0
                continue
            if response.status_code in (401, 403):
                raise LLMError(f"OpenRouter auth failure: HTTP {response.status_code}")
            if response.status_code == 429:
                errors.append(f"{model}: rate limit (429)")
                logger.warning("OpenRouter model '%s' rate limited; rotating", model)
                with self._lock:
                    self._cooldown[idx] = _time.monotonic() + 30.0
                continue
            if response.status_code == 404:
                errors.append(f"{model}: not found (404)")
                logger.warning("OpenRouter model '%s' not found; rotating", model)
                self._dead.add(idx)
                continue
            if response.status_code >= 400:
                text = response.text[:200]
                errors.append(f"{model}: HTTP {response.status_code}: {text}")
                logger.warning("OpenRouter model '%s' HTTP %s", model, response.status_code)
                self._dead.add(idx)
                continue
            try:
                data = response.json()
            except ValueError as exc:
                errors.append(f"{model}: invalid JSON: {exc}")
                self._dead.add(idx)
                continue
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                errors.append(f"{model}: bad response: {exc}")
                self._dead.add(idx)
                continue
            if not content:
                errors.append(f"{model}: empty content")
                self._dead.add(idx)
                continue
            with self._lock:
                self._index = idx
            return content

        detail = "; ".join(errors) if errors else "no usable model"
        raise LLMError(f"All OpenRouter models failed ({detail}).")
