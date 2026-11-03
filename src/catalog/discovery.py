"""Discovery orchestration: seed the catalog from connectors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from catalog.store import TheoryCatalog
from config.settings import Settings, get_settings
from core.models import PopularityTier, SourceTheory, Theory, TheorySource
from providers.sources.base import SourceConnector
from utils.text import slugify

_META_PREFIXES = ("list of ", "outline of ", "index of ", "category: ", "bibliography of ")
_META_NAMES = {"conspiracy theory", "conspiracy theories"}

# Load niche config
_NICHES_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "niches.yaml"
with _NICHES_CONFIG_PATH.open() as f:
    _NICHES_CONFIG = yaml.safe_load(f)

NICHES = _NICHES_CONFIG["niches"]
DEFAULT_ACTIVE_NICHES = _NICHES_CONFIG.get("active_niches", list(NICHES.keys()))


@dataclass
class DiscoveryReport:
    """Summary of a discovery run."""

    discovered: int = 0
    skipped: int = 0
    per_source: dict[str, int] = field(default_factory=dict)
    per_niche: dict[str, int] = field(default_factory=dict)


def is_meta(name: str) -> bool:
    """Whether ``name`` is a list/index page rather than an actual theory."""

    lowered = name.strip().lower()
    return lowered in _META_NAMES or lowered.startswith(_META_PREFIXES)


def classify_popularity(views: float | None, settings: Settings) -> PopularityTier:
    """Bucket average monthly pageviews into a popularity tier."""

    if views is None:
        return "unknown"
    if views >= settings.popularity_mainstream:
        return "mainstream"
    if views >= settings.popularity_popular:
        return "popular"
    if views >= settings.popularity_moderate:
        return "moderate"
    return "underground"


def to_theory(hit: SourceTheory, now: datetime, settings: Settings) -> Theory:
    """Build a canonical :class:`Theory` from a raw discovery hit."""

    return Theory(
        slug=slugify(hit.name),
        name=hit.name,
        aliases=hit.aliases,
        wikidata_id=hit.wikidata_id,
        popularity_tier=classify_popularity(hit.metrics.avg_monthly_views, settings),
        metrics=hit.metrics,
        categories=hit.categories,
        sources=[
            TheorySource(
                source=hit.source,
                url=hit.url,
                title=hit.name,
                retrieved_at=now,
            )
        ],
        first_seen=now,
        last_updated=now,
    )


def _build_connectors_for_niche(niche_name: str, settings: Settings) -> list[SourceConnector]:
    """Build source connectors configured for a specific niche."""

    from providers.sources.factory import build_connectors

    niche = NICHES[niche_name]
    # Temporarily override settings for this niche
    original_categories = settings.discovery_categories
    original_seed_pages = settings.discovery_seed_pages
    original_fandom_wikis = settings.fandom_wikis

    settings.discovery_categories = list(niche["categories"])
    settings.discovery_seed_pages = list(niche["seed_pages"])
    settings.fandom_wikis = list(niche["fandom_wikis"])

    try:
        connectors = build_connectors(settings)
    finally:
        # Restore original settings
        settings.discovery_categories = original_categories
        settings.discovery_seed_pages = original_seed_pages
        settings.fandom_wikis = original_fandom_wikis

    return connectors


def discover(
    catalog: TheoryCatalog,
    connectors: Iterable[SourceConnector] | None = None,
    *,
    limit: int | None = None,
    with_metrics: bool = True,
    settings: Settings | None = None,
    niches: list[str] | None = None,
) -> DiscoveryReport:
    """Run discovery for specified niches and merge results into ``catalog``.

    If ``connectors`` is provided, uses those directly (backward compat).
    Otherwise, builds connectors per niche from the niche config.
    """

    settings = settings or get_settings()
    niches = niches or DEFAULT_ACTIVE_NICHES
    now = datetime.now(UTC)
    report = DiscoveryReport()

    if connectors is not None:
        # Legacy path: use provided connectors directly
        for connector in connectors:
            for hit in connector.discover(limit=limit, with_metrics=with_metrics):
                if is_meta(hit.name):
                    report.skipped += 1
                    continue
                catalog.upsert(to_theory(hit, now, settings))
                report.discovered += 1
                report.per_source[hit.source] = report.per_source.get(hit.source, 0) + 1
        return report

    # Multi-niche path
    for niche_name in niches:
        if niche_name not in NICHES:
            continue
        niche_connectors = _build_connectors_for_niche(niche_name, settings)
        niche_discovered = 0

        for connector in niche_connectors:
            for hit in connector.discover(limit=limit, with_metrics=with_metrics):
                if is_meta(hit.name):
                    report.skipped += 1
                    continue
                catalog.upsert(to_theory(hit, now, settings))
                report.discovered += 1
                niche_discovered += 1
                report.per_source[hit.source] = report.per_source.get(hit.source, 0) + 1

        report.per_niche[niche_name] = niche_discovered

    return report


def discover_niche(
    catalog: TheoryCatalog,
    niche_name: str,
    *,
    limit: int | None = None,
    with_metrics: bool = True,
    settings: Settings | None = None,
) -> DiscoveryReport:
    """Discover theories for a single niche."""

    return discover(
        catalog,
        None,
        limit=limit,
        with_metrics=with_metrics,
        settings=settings,
        niches=[niche_name],
    )


def list_niches() -> dict[str, dict]:
    """Return all configured niche definitions."""

    return NICHES


def list_active_niches() -> list[str]:
    """Return the list of niches active by default."""

    return DEFAULT_ACTIVE_NICHES