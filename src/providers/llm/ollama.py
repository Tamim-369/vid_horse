"""Ollama-backed LLM provider.

Speaks to a local Ollama server. Uses Ollama's ``format`` parameter to pass a
JSON Schema for grammar-constrained decoding, which makes even small models
structurally reliable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ollama import Client

from core.errors import LLMError
from core.interfaces import LLMMessage
from providers.llm.base import BaseLLMProvider
from utils.cache import DiskCache


class OllamaProvider(BaseLLMProvider):
    """Local Ollama models."""

    name = "ollama"
    supports_native_schema = True

    def __init__(
        self,
        *,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 4096,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        repeat_penalty: float = 1.3,
        repeat_last_n: int = 256,
        keep_alive: str = "5m",
        timeout: float = 600.0,
        max_retries: int = 2,
        cache: DiskCache | None = None,
        client: Client | None = None,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache=cache,
        )
        self._host = host
        self._num_ctx = num_ctx
        self._repeat_penalty = repeat_penalty
        self._repeat_last_n = repeat_last_n
        self._keep_alive = keep_alive
        self._client = client or Client(host=host, timeout=timeout)

    def fingerprint(self) -> dict:
        return {
            **super().fingerprint(),
            "host": self._host,
            "num_ctx": self._num_ctx,
            "repeat_penalty": self._repeat_penalty,
            "repeat_last_n": self._repeat_last_n,
            "keep_alive": self._keep_alive,
        }

    def _generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        options: dict,
        schema: dict | None,
    ) -> str:
        chat_options = {key: value for key, value in options.items() if key != "model"}
        chat_options["num_ctx"] = self._num_ctx
        chat_options["repeat_penalty"] = self._repeat_penalty
        chat_options["repeat_last_n"] = self._repeat_last_n

        kwargs: dict[str, Any] = {
            "model": options["model"],
            "messages": list(messages),
            "options": chat_options,
            "keep_alive": self._keep_alive,
            "stream": False,
        }
        if schema is not None:
            kwargs["format"] = schema

        try:
            response = self._client.chat(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalise client errors
            raise LLMError(f"Ollama request failed: {exc}") from exc

        content = _extract_content(response)
        if not content:
            raise LLMError("Ollama returned an empty response")
        return content


def _extract_content(response: Any) -> str:
    """Read message content from a dict- or attribute-style response."""

    if isinstance(response, dict):
        message = response.get("message")
    else:
        message = getattr(response, "message", None)
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return getattr(message, "content", "") or ""
