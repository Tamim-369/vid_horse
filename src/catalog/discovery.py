"""Discovery orchestration: seed the catalog from connectors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from catalog.store import TheoryCatalog
from config.settings import Settings, get_settings
from core.models import PopularityTier, SourceTheory, Theory, TheorySource
from providers.sources.base import SourceConnector
from utils.text import slugify

_META_PREFIXES = ("list of ", "outline of ", "index of ", "category:")
_META_NAMES = {"conspiracy theory", "conspiracy theories"}


@dataclass
class DiscoveryReport:
    """Summary of a discovery run."""

    discovered: int = 0
    skipped: int = 0
    per_source: dict[str, int] = field(default_factory=dict)


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


def discover(
    catalog: TheoryCatalog,
    connectors: Iterable[SourceConnector],
    *,
    limit: int | None = None,
    with_metrics: bool = True,
    settings: Settings | None = None,
) -> DiscoveryReport:
    """Run every connector and merge results into ``catalog``."""

    settings = settings or get_settings()
    now = datetime.now(UTC)
    report = DiscoveryReport()

    for connector in connectors:
        for hit in connector.discover(limit=limit, with_metrics=with_metrics):
            if is_meta(hit.name):
                report.skipped += 1
                continue
            catalog.upsert(to_theory(hit, now, settings))
            report.discovered += 1
            report.per_source[hit.source] = report.per_source.get(hit.source, 0) + 1

    return report
