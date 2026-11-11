"""Factory and registry for LLM providers.

Local-only: the one backend is Ollama. Adding another backend is two
lines: implement :class:`~providers.llm.base.BaseLLMProvider` and call
:func:`register_provider`.
"""

from __future__ import annotations

from collections.abc import Callable

from config.settings import Settings, get_settings
from core.errors import ConfigurationError
from providers.llm.base import BaseLLMProvider
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


def create_llm_provider(settings: Settings | None = None, *, role: str = "general") -> BaseLLMProvider:
    """Instantiate the provider configured in ``settings``.

    ``role`` selects the local model for work that has a dedicated model:
    ``"reader"``/``"writer"``/``"reviewer"`` -> ``ollama_model`` (gemma3:4b),
    ``"decision"`` -> ``ollama_decision_model`` (d3:2b), ``"assets"`` ->
    ``ollama_asset_model`` (gemma3:1b).
    """

    settings = settings or get_settings()
    builder = _REGISTRY.get(settings.llm_provider.lower())
    if builder is None:
        raise ConfigurationError(
            f"Unknown LLM provider '{settings.llm_provider}'. "
            f"Available: {', '.join(available_providers())}. "
            "This pipeline runs on local models only (ollama)."
        )
    attr = _ROLE_MODEL_ATTR.get(role)
    model = getattr(settings, attr) if attr else None
    return builder(settings, model=model)


_ROLE_MODEL_ATTR = {
    "general": "ollama_model",
    "reader": "ollama_model",
    "writer": "ollama_model",
    "reviewer": "ollama_model",
    "decision": "ollama_decision_model",
    "assets": "ollama_asset_model",
}


def _build_ollama(settings: Settings, model: str | None = None) -> BaseLLMProvider:
    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)
    from utils import computing

    return OllamaProvider(
        model=model or settings.ollama_model,
        host=settings.ollama_host,
        num_ctx=settings.ollama_num_ctx,
        num_thread=settings.ollama_num_thread or max(2, computing.machine().cores // 2),
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_num_predict,
        repeat_penalty=settings.ollama_repeat_penalty,
        repeat_last_n=settings.ollama_repeat_last_n,
        keep_alive=settings.ollama_keep_alive,
        timeout=settings.ollama_timeout,
        max_retries=settings.max_retries,
        cache=cache,
    )


register_provider("ollama", _build_ollama)
