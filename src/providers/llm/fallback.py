"""Composite provider: try primary, fall back to secondary on failure."""

from __future__ import annotations

from core.errors import LLMError
from providers.llm.base import BaseLLMProvider
from utils.cache import DiskCache


class FallbackLLMProvider(BaseLLMProvider):
    name = "fallback"
    supports_native_schema = False

    def __init__(
        self,
        *,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(
            model=primary.model,
            temperature=primary.temperature,
            max_tokens=primary.max_tokens,
            max_retries=primary.max_retries,
            cache=cache or primary._cache,
        )
        self._primary = primary
        self._fallback = fallback
        self.supports_native_schema = primary.supports_native_schema

    def fingerprint(self) -> dict:
        return {
            "provider": "fallback",
            "primary": self._primary.fingerprint(),
            "fallback": self._fallback.fingerprint(),
        }

    def _generate(self, messages, *, options, schema):
        try:
            return self._primary._generate(messages, options=options, schema=schema)
        except LLMError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Primary LLM failed (%s); falling back to %s", exc, self._fallback.name
            )
            fallback_options = {**options, "model": self._fallback.model}
            return self._fallback._generate(messages, options=fallback_options, schema=schema)
