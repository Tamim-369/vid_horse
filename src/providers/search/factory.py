"""Create search providers from settings."""

from providers.search.base import SearchProvider


def create_search_provider() -> SearchProvider:
    """Create the configured search provider."""
    ...
