"""Domain error hierarchy."""

from __future__ import annotations


class VideoAIError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(VideoAIError):
    """Raised when runtime configuration is missing or invalid."""


class ProviderError(VideoAIError):
    """Raised when an infrastructure provider fails."""


class LLMError(ProviderError):
    """Raised when a language-model call fails or returns an unusable result."""


class SearchError(ProviderError):
    """Raised when a search provider fails."""


class BlockError(VideoAIError):
    """Raised when a pipeline block cannot produce its output."""


class PipelineError(VideoAIError):
    """Raised when orchestrating blocks fails."""
