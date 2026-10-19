"""Base class for LLM providers.

Concrete providers implement only :meth:`_generate`; this base adds the
engineering that makes small local models reliable:

- native (grammar-constrained) JSON schema support when available,
- transparent disk caching of every call,
- a validation + self-repair loop around structured output.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from core.errors import LLMError
from core.interfaces import LLMMessage, TModel
from utils.cache import DiskCache
from utils.json_parsing import (
    json_schema_instruction,
    parse_json_as,
    repair_instruction,
)

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Common behaviour shared by all LLM backends."""

    name: str = "base"
    supports_native_schema: bool = False

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 2,
        cache: DiskCache | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._cache = cache

    # -- subclass hook ------------------------------------------------------
    @abstractmethod
    def _generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        options: dict,
        schema: dict | None,
    ) -> str:
        """Perform one model call and return raw text."""

    def fingerprint(self) -> dict:
        """Fields that affect output and therefore belong in the cache key."""

        return {
            "provider": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    # -- public API ---------------------------------------------------------
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        """Return the assistant's text for ``messages``."""

        options = self._options(model, temperature, max_tokens, seed)
        return self._invoke(messages, options, None)

    def complete_structured(
        self,
        messages: Sequence[LLMMessage],
        schema: type[TModel],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        """Return raw text constrained to ``schema`` where supported."""

        options = self._options(model, temperature, max_tokens, seed)
        if self.supports_native_schema:
            return self._invoke(messages, options, schema.model_json_schema())
        return self._invoke([*messages, json_schema_instruction(schema)], options, None)

    def complete_json(
        self,
        messages: Sequence[LLMMessage],
        schema: type[TModel],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> TModel:
        """Ask the model for JSON and validate it, repairing on failure."""

        conversation = list(messages)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.complete_structured(
                    conversation,
                    schema,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            except LLMError as exc:
                last_error = exc
                logger.debug(
                    "Generation failed for %s (attempt %d/%d): %s",
                    schema.__name__,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                continue
            try:
                return parse_json_as(raw, schema)
            except LLMError as exc:
                last_error = exc
                logger.debug(
                    "Repairing %s output (attempt %d/%d)",
                    schema.__name__,
                    attempt + 1,
                    self.max_retries + 1,
                )
                conversation = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": repair_instruction(exc)},
                ]

        raise LLMError(
            f"Could not produce a valid {schema.__name__} after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )

    # -- internals ----------------------------------------------------------
    def _resolve(
        self,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, float, int | None]:
        return (
            model or self.model,
            self.temperature if temperature is None else temperature,
            self.max_tokens if max_tokens is None else max_tokens,
        )

    def _options(
        self,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None,
    ) -> dict:
        resolved_model, temp, tokens = self._resolve(model, temperature, max_tokens)
        options: dict = {"model": resolved_model, "temperature": temp}
        if tokens is not None:
            options["num_predict"] = tokens
        if seed is not None:
            options["seed"] = seed
        return options

    def _invoke(
        self,
        messages: Sequence[LLMMessage],
        options: dict,
        schema: dict | None,
    ) -> str:
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = DiskCache.make_key(
                "llm", self.fingerprint(), options, schema, list(messages)
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        text = self._generate(messages, options=options, schema=schema)

        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, text)
        return text
