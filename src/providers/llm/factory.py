"""Create LLM providers from settings."""

from providers.llm.base import LLMProvider


def create_llm_provider() -> LLMProvider:
    """Create the configured LLM provider."""
    ...
