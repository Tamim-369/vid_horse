"""Groq-backed LLM provider.

Talks to Groq's OpenAI-compatible chat-completions API. The provider holds a
*pool* of API keys: if a key is rejected, rate-limited, or a request otherwise
fails, it rotates to the next key so one bad or exhausted key cannot stop a run.
Only authentication failures (401/403) retire a key; rate limits and transient
errors just move on to the next key.
"""

from __future__ import annotations

import logging
import re
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


class GroqProvider(BaseLLMProvider):
    """Groq's hosted models, with rotating API keys."""

    name = "groq"
    supports_native_schema = False

    def __init__(
        self,
        *,
        model: str,
        api_keys: Sequence[str],
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
        cache: DiskCache | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        keys = [key.strip() for key in api_keys if key and key.strip()]
        if not keys:
            raise ConfigurationError("GroqProvider requires at least one API key")
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache=cache,
        )
        self._api_keys = keys
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._reasoning_effort = reasoning_effort
        self._dead: set[int] = set()
        self._index = 0
        # parallel corrective waves share one provider across threads:
        # rotation state mutates under a lock and a rate-limited key gets
        # a short cooldown window instead of being hammered again by the
        # next thread
        self._lock = threading.Lock()
        self._cooldown: dict[int, float] = {}
        self._client = client or httpx.Client(timeout=timeout)

    def fingerprint(self) -> dict:
        return {
            **super().fingerprint(),
            "base_url": self._base_url,
            "reasoning_effort": self._reasoning_effort,
        }

    def _generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        options: dict,
        schema: dict | None,
    ) -> str:
        payload = self._payload(messages, options, schema)
        errors: list[str] = []

        for index in self._key_order():
            attempt = payload
            shrank = False
            while True:
                try:
                    content = self._post(self._api_keys[index], attempt)
                except _TooLarge as exc:
                    if not shrank and attempt.get("max_tokens", 0) > exc.limit:
                        logger.warning(
                            "Groq output cap is %d tokens; retrying key #%d with "
                            "a smaller max_tokens",
                            exc.limit,
                            index + 1,
                        )
                        attempt = {**attempt, "max_tokens": exc.limit}
                        shrank = True
                        continue
                    errors.append(f"key #{index + 1}: {exc}")
                    break
                except _KeyRejected as exc:
                    self._dead.add(index)
                    errors.append(f"key #{index + 1}: {exc}")
                    logger.warning(
                        "Groq key #%d rejected (%s); rotating", index + 1, exc
                    )
                    break
                except _KeyFailure as exc:
                    if "HTTP 429" in str(exc):
                        # parallel waves share these keys; let a judged
                        # key rest briefly instead of walkers hammering it
                        with self._lock:
                            self._cooldown[index] = time.monotonic() + 6.0
                        logger.warning("Groq key #%d rate-limited; cooling 6s", index + 1)
                    errors.append(f"key #{index + 1}: {exc}")
                    logger.warning("Groq key #%d failed (%s); rotating", index + 1, exc)
                    break
                with self._lock:
                    self._index = index
                    self._cooldown.pop(index, None)
                return content

        if not errors:
            raise LLMError("No usable Groq API keys remain")
        raise LLMError("All Groq API keys failed: " + "; ".join(errors))

    # -- internals ----------------------------------------------------------
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
        effort = options.get("reasoning_effort", self._reasoning_effort)
        if effort:
            payload["reasoning_effort"] = effort
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _key_order(self) -> Iterator[int]:
        """Fresh keys first, rate-limited keys at the tail as last resort."""

        total = len(self._api_keys)
        with self._lock:
            now = time.monotonic()
            self._cooldown = {i: t for i, t in self._cooldown.items() if t > now}
            hot = set(self._cooldown)
        fresh, cooling = [], []
        for offset in range(total):
            index = (self._index + offset) % total
            if index in self._dead:
                continue
            (cooling if index in hot else fresh).append(index)
        yield from fresh
        yield from cooling

    def _post(self, key: str, payload: dict[str, Any]) -> str:
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise _KeyFailure(f"request error: {exc}") from exc

        if response.status_code in (401, 403):
            raise _KeyRejected(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            text = _error_text(response)
            limit = _output_limit(text)
            if limit is not None:
                raise _TooLarge(limit, f"HTTP {response.status_code}: {text}")
            raise _KeyFailure(f"HTTP {response.status_code}: {text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise _KeyFailure(f"invalid JSON response: {exc}") from exc

        content = _extract_content(data)
        if not content:
            raise _KeyFailure(f"empty response (finish_reason={_finish_reason(data)})")
        return content


class _KeyFailure(Exception):
    """A single key's request failed; try the next one."""


class _KeyRejected(_KeyFailure):
    """A key was rejected (bad credentials); retire it."""


class _TooLarge(_KeyFailure):
    """The requested ``max_tokens`` exceeded the account's output cap."""

    def __init__(self, limit: int, message: str) -> None:
        super().__init__(message)
        self.limit = limit


_OUTPUT_LIMIT = re.compile(r"on output tokens per minute.*?Limit (\d+)", re.IGNORECASE)


def _output_limit(text: str) -> int | None:
    """Extract the enforced output-token limit from a Groq error, if present."""

    match = _OUTPUT_LIMIT.search(text)
    return int(match.group(1)) if match else None


def _extract_content(data: Any) -> str:
    """Read ``choices[0].message.content`` from a chat-completions payload."""

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _finish_reason(data: Any) -> str:
    """Read ``choices[0].finish_reason`` for error messages."""

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
