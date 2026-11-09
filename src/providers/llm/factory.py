"""Factory and registry for LLM providers.

Adding a backend is two lines: implement
:class:`~providers.llm.base.BaseLLMProvider` and call :func:`register_provider`.
"""

from __future__ import annotations

from collections.abc import Callable

from config.settings import Settings, get_settings
from core.errors import ConfigurationError
from providers.llm.base import BaseLLMProvider
from providers.llm.fallback import FallbackLLMProvider
from providers.llm.groq import GroqProvider
from providers.llm.ollama import OllamaProvider
from providers.llm.ollama_cloud import OllamaCloudProvider
from providers.llm.openrouter import OpenRouterProvider
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


def _build_groq_chain(settings: Settings) -> BaseLLMProvider:
    keys = settings.groq_api_keys
    if not keys:
        raise ConfigurationError(
            "No Groq API keys found. Set GROQ_API_KEY (single mode) or the "
            "numbered GROQ_API_KEY_FIRST..GROQ_API_KEY_FOURTH (pool mode)."
        )
    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)
    primary = GroqProvider(
        model=settings.groq_model,
        api_keys=keys,
        base_url=settings.groq_base_url,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        timeout=settings.groq_timeout,
        max_retries=settings.max_retries,
        cache=cache,
    )
    # Fall back to Ollama Cloud when all Groq keys are exhausted
    api_key = settings.ollama_cloud_api_key
    if api_key:
        cloud_models = [m.strip() for m in settings.ollama_cloud_models.split(",") if m.strip()]
        if cloud_models:
            # Cloud must NOT silently delegate to local; it should raise so
            # the outer fallback can move on to Groq, then local.
            fallback = OllamaCloudProvider(
                model=cloud_models[0],
                api_key=api_key,
                base_url=settings.ollama_cloud_host,
                models=cloud_models,
                temperature=settings.ollama_cloud_temperature,
                max_tokens=settings.ollama_cloud_max_tokens,
                timeout=settings.ollama_cloud_timeout,
                max_retries=settings.max_retries,
                cache=cache,
                local_fallback=None,
            )
            # Groq -> local as the second tier.
            groq_tier = primary
            if settings.ollama_cloud_fallback_to_local:
                local_provider = OllamaProvider(
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
                groq_tier = FallbackLLMProvider(
                    primary=primary, fallback=local_provider, cache=cache
                )
            rest = FallbackLLMProvider(
                primary=fallback,   # ollama cloud
                fallback=groq_tier,  # groq, then local
                cache=cache,
            )
            # OpenRouter goes first if a key is available
            if settings.openrouter_api_key:
                or_models = [m.strip() for m in settings.openrouter_models.split(",") if m.strip()]
                or_provider = OpenRouterProvider(
                    api_key=settings.openrouter_api_key,
                    models=or_models,
                    base_url=settings.openrouter_base_url,
                    temperature=settings.openrouter_temperature,
                    max_tokens=settings.openrouter_max_tokens,
                    timeout=settings.openrouter_timeout,
                    cache=cache,
                )
                return FallbackLLMProvider(primary=or_provider, fallback=rest, cache=cache)
            return rest
    # no cloud key: groq (+local) chain, optionally fronted by OpenRouter
    if settings.openrouter_api_key:
        or_models = [m.strip() for m in settings.openrouter_models.split(",") if m.strip()]
        or_provider = OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            models=or_models,
            base_url=settings.openrouter_base_url,
            temperature=settings.openrouter_temperature,
            max_tokens=settings.openrouter_max_tokens,
            timeout=settings.openrouter_timeout,
            cache=cache,
        )
        return FallbackLLMProvider(primary=or_provider, fallback=primary, cache=cache)
    return primary


def _build_ollama_cloud(settings: Settings) -> BaseLLMProvider:
    api_key = settings.ollama_cloud_api_key
    if not api_key:
        raise ConfigurationError(
            "Ollama Cloud API key not set. Set VIDEOAI_OLLAMA_CLOUD_API_KEY."
        )
    cloud_models = [m.strip() for m in settings.ollama_cloud_models.split(",") if m.strip()]
    if not cloud_models:
        raise ConfigurationError("No Ollama Cloud models configured.")

    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)

    # Build local fallback if enabled
    local_fallback: BaseLLMProvider | None = None
    if settings.ollama_cloud_fallback_to_local:
        local_fallback = OllamaProvider(
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

    return OllamaCloudProvider(
        model=cloud_models[0],
        api_key=api_key,
        base_url=settings.ollama_cloud_host,
        models=cloud_models,
        temperature=settings.ollama_cloud_temperature,
        max_tokens=settings.ollama_cloud_max_tokens,
        timeout=settings.ollama_cloud_timeout,
        max_retries=settings.max_retries,
        cache=cache,
        local_fallback=local_fallback,
    )


def _build_openrouter(settings: Settings) -> BaseLLMProvider:
    if not settings.openrouter_api_key:
        raise ConfigurationError("OpenRouter API key not set. Set VIDEOAI_OPENROUTER_API_KEY.")
    models = [m.strip() for m in settings.openrouter_models.split(",") if m.strip()]
    if not models:
        raise ConfigurationError("No OpenRouter models configured.")
    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)
    return OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        models=models,
        base_url=settings.openrouter_base_url,
        temperature=settings.openrouter_temperature,
        max_tokens=settings.openrouter_max_tokens,
        timeout=settings.openrouter_timeout,
        max_retries=settings.max_retries,
        cache=cache,
    )


register_provider("ollama", _build_ollama)
register_provider("groq", _build_groq_chain)
register_provider("openrouter", _build_openrouter)
register_provider("ollama-cloud", _build_ollama_cloud)
