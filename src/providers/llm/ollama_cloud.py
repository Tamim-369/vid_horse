"""Ollama Cloud LLM provider with fallback model rotation.

Uses the OpenAI-compatible chat-completions API at ``https://ollama.com/v1``.
Holds an ordered list of cloud models and falls back to a local Ollama provider
when all cloud models are exhausted or failing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from core.errors import ConfigurationError, LLMError
from core.interfaces import LLMMessage
from providers.llm.base import BaseLLMProvider
from utils.cache import DiskCache

logger = logging.getLogger(__name__)


class OllamaCloudProvider(BaseLLMProvider):
    """Ollama Cloud models with fallback model rotation.

    Tries the configured cloud models in order (default chain: gemma4:31b,
    gpt-oss:120b, gpt-oss:20b, nemotron-3-nano:30b, nemotron-3-super,
    nemotron-3-ultra). On failure (rate-limit, exhausted credits,
    server error) it rotates to the next model. After all cloud models fail,
    it delegates to a local ``OllamaProvider`` if ``local_fallback`` is set.
    """

    name = "ollama-cloud"
    supports_native_schema = False

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://ollama.com/v1",
        models: Sequence[str],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        cache: DiskCache | None = None,
        client: httpx.Client | None = None,
        local_fallback: BaseLLMProvider | None = None,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache=cache,
        )
        if not api_key:
            raise ConfigurationError("OllamaCloudProvider requires an API key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._models = list(models)
        self._timeout = timeout
        self._local_fallback = local_fallback
        self._dead: set[int] = set()
        self._cooldown: dict[int, float] = {}
        self._lock = threading.Lock()
        self._client = client or httpx.Client(timeout=timeout)
        self._cloud_index = 0

    def fingerprint(self) -> dict:
        fp = {
            **super().fingerprint(),
            "base_url": self._base_url,
            "models": self._models,
            "local_fallback": bool(self._local_fallback),
        }
        if self._local_fallback:
            fp["local_fingerprint"] = self._local_fallback.fingerprint()
        return fp

    def _generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        options: dict,
        schema: dict | None,
    ) -> str:
        payload = self._payload(messages, options, schema)
        errors: list[str] = []

        for idx in self._model_order():
            try:
                content = self._post(self._models[idx], payload)
            except _CloudFailure as exc:
                errors.append(f"model '{self._models[idx]}': {exc}")
                logger.warning(
                    "Ollama Cloud model '%s' failed (%s); rotating",
                    self._models[idx],
                    exc,
                )
                if "rate limit" not in str(exc).lower():
                    self._dead.add(idx)
                else:
                    with self._lock:
                        self._cooldown[idx] = time.monotonic() + 30.0
                continue
            # remembered start of the last successful rotation, so the next
            # call does not always slam whatever failed most recently
            with self._lock:
                self._cloud_index = idx
            return content

        if self._local_fallback is None:
            detail = "; ".join(errors) if errors else "no usable model"
            raise LLMError(
                f"All Ollama Cloud models failed ({detail}). "
                "No local fallback configured."
            )
        logger.info(
            "All Ollama Cloud models failed; delegating to local fallback"
        )
        local_options = {**options, "model": self._local_fallback.model}
        return self._local_fallback._generate(
            messages, options=local_options, schema=schema
        )

    def _model_order(self) -> Iterator[int]:
        total = len(self._models)
        with self._lock:
            now = time.monotonic()
            self._cooldown = {i: t for i, t in self._cooldown.items() if t > now}
            hot = set(self._cooldown)
        fresh, cooling = [], []
        for offset in range(total):
            index = (self._cloud_index + offset) % total
            if index in self._dead:
                continue
            (cooling if index in hot else fresh).append(index)
        yield from fresh
        yield from cooling

    def _payload(
        self,
        messages: Sequence[LLMMessage],
        options: dict,
        schema: dict | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": options.get("model", self.model),
            "messages": list(messages),
            "temperature": options.get("temperature", self.temperature),
            "max_tokens": options.get("num_predict") or self.max_tokens or 4096,
            "stream": False,
        }
        if "seed" in options:
            payload["seed"] = options["seed"]
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, model: str, payload: dict[str, Any]) -> str:
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise _CloudFailure(f"request error: {exc}") from exc

        if response.status_code in (401, 403):
            raise _CloudFailure(f"auth failure: HTTP {response.status_code}")
        if response.status_code == 429:
            raise _CloudFailure("rate limit (429)")
        if response.status_code == 402:
            raise _CloudFailure("exhausted credits (402)")
        if response.status_code >= 400:
            text = _error_text(response)[:200]
            raise _CloudFailure(f"HTTP {response.status_code}: {text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise _CloudFailure(f"invalid JSON response: {exc}") from exc

        content = _extract_content(data)
        if not content:
            raise _CloudFailure(f"empty response (finish_reason={_finish_reason(data)})")
        return content


class _CloudFailure(Exception):
    """Base class for Ollama Cloud failures."""


def _extract_content(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _finish_reason(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return "unknown"
    return str(choices[0].get("finish_reason") or "unknown")


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message", error))
        if error is not None:
            return str(error)
    return str(data)[:200]