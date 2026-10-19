"""Factory and registry for LLM providers.

Adding a backend is two lines: implement
:class:`~providers.llm.base.BaseLLMProvider` and call :func:`register_provider`.
"""

from __future__ import annotations

from collections.abc import Callable

from config.settings import Settings, get_settings
from core.errors import ConfigurationError
from providers.llm.base import BaseLLMProvider
from providers.llm.groq import GroqProvider
from providers.llm.ollama import OllamaProvider
from utils.cache import DiskCache

ProviderBuilder = Callable[[Settings], BaseLLMProvider]

_REGISTRY: dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Register ``builder`` under ``name`` (case-insensitive)."""

    _REGISTRY[name.lower()] = builder


def available_providers() -> list[str]:
    """Return the names of all registered providers."""

    return sorted(_REGISTRY)


def create_llm_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """Instantiate the provider configured in ``settings``."""

    settings = settings or get_settings()
    builder = _REGISTRY.get(settings.llm_provider.lower())
    if builder is None:
        raise ConfigurationError(
            f"Unknown LLM provider '{settings.llm_provider}'. "
            f"Available: {', '.join(available_providers())}."
        )
    return builder(settings)


def _build_ollama(settings: Settings) -> BaseLLMProvider:
    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)
    return OllamaProvider(
        model=settings.ollama_model,
        host=settings.ollama_host,
        num_ctx=settings.ollama_num_ctx,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_num_predict,
        repeat_penalty=settings.ollama_repeat_penalty,
        repeat_last_n=settings.ollama_repeat_last_n,
        keep_alive=settings.ollama_keep_alive,
        timeout=settings.ollama_timeout,
        max_retries=settings.max_retries,
        cache=cache,
    )


def _build_groq(settings: Settings) -> BaseLLMProvider:
    keys = settings.groq_api_keys
    if not keys:
        raise ConfigurationError(
            "No Groq API keys found. Set GROQ_API_KEY (single mode) or the "
            "numbered GROQ_API_KEY_FIRST..GROQ_API_KEY_FOURTH (pool mode)."
        )
    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)
    return GroqProvider(
        model=settings.groq_model,
        api_keys=keys,
        base_url=settings.groq_base_url,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        timeout=settings.groq_timeout,
        max_retries=settings.max_retries,
        cache=cache,
    )


register_provider("ollama", _build_ollama)
register_provider("groq", _build_groq)
