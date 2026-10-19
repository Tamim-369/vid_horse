"""SQLite-backed conspiracy-theory catalog.

Stores only a topic, pointers to where its detail lives, and cheap metadata.
Upserts *merge* rather than overwrite, so repeated discovery runs accumulate
aliases, sources and metrics without creating duplicates.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.models import Theory, TheoryMetrics, TheorySource

_SCHEMA = """
CREATE TABLE IF NOT EXISTS theories (
    id             INTEGER PRIMARY KEY,
    slug           TEXT NOT NULL UNIQUE,
    name           TEXT NOT NULL,
    aliases        TEXT NOT NULL DEFAULT '[]',
    wikidata_id    TEXT,
    theory_type    TEXT,
    popularity_tier TEXT NOT NULL DEFAULT 'unknown',
    metrics        TEXT NOT NULL DEFAULT '{}',
    categories     TEXT NOT NULL DEFAULT '[]',
    first_seen     TEXT NOT NULL,
    last_updated   TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_theories_wikidata
    ON theories(wikidata_id) WHERE wikidata_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_theories_tier ON theories(popularity_tier);

CREATE TABLE IF NOT EXISTS theory_sources (
    id           INTEGER PRIMARY KEY,
    theory_id    INTEGER NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    UNIQUE(theory_id, source, url)
);

CREATE INDEX IF NOT EXISTS idx_sources_theory ON theory_sources(theory_id);
"""

_ORDER_BY = {
    "popularity": (
        "(json_extract(metrics, '$.avg_monthly_views') IS NULL), "
        "json_extract(metrics, '$.avg_monthly_views') DESC"
    ),
    "size": "(json_extract(metrics, '$.size_bytes') IS NULL), "
    "json_extract(metrics, '$.size_bytes') DESC",
    "name": "name COLLATE NOCASE ASC",
}


class TheoryCatalog:
    """A local, persistent catalog of conspiracy theories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TheoryCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- reads --------------------------------------------------------------
    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM theories").fetchone()
        return int(row["n"])

    def get(self, slug: str) -> Theory | None:
        row = self._conn.execute(
            "SELECT * FROM theories WHERE slug = ?", (slug,)
        ).fetchone()
        return self._load(row) if row else None

    def list(
        self,
        *,
        limit: int = 50,
        order: str = "popularity",
        tier: str | None = None,
        theory_type: str | None = None,
    ) -> list[Theory]:
        clauses: list[str] = []
        params: list[object] = []
        if tier:
            clauses.append("popularity_tier = ?")
            params.append(tier)
        if theory_type:
            clauses.append("theory_type = ?")
            params.append(theory_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = _ORDER_BY.get(order, _ORDER_BY["popularity"])
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM theories {where} ORDER BY {order_by} LIMIT ?",  # noqa: S608
            params,
        ).fetchall()
        return [self._load(row) for row in rows]

    def sources_for(self, theory_id: int) -> list[TheorySource]:
        rows = self._conn.execute(
            "SELECT source, url, title, retrieved_at FROM theory_sources "
            "WHERE theory_id = ? ORDER BY source",
            (theory_id,),
        ).fetchall()
        return [
            TheorySource(
                source=row["source"],
                url=row["url"],
                title=row["title"],
                retrieved_at=row["retrieved_at"],
            )
            for row in rows
        ]

    # -- writes -------------------------------------------------------------
    def upsert(self, theory: Theory) -> int:
        """Insert or merge ``theory`` and return its row id."""

        theory_id = self._find_id(theory)
        if theory_id is None:
            cursor = self._conn.execute(
                "INSERT INTO theories (slug, name, aliases, wikidata_id, theory_type, "
                "popularity_tier, metrics, categories, first_seen, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._row_values(theory),
            )
            theory_id = int(cursor.lastrowid)
        else:
            existing = self._get_by_id(theory_id)
            merged = _merge(existing, theory)
            self._conn.execute(
                "UPDATE theories SET slug = ?, name = ?, aliases = ?, wikidata_id = ?, "
                "theory_type = ?, popularity_tier = ?, metrics = ?, categories = ?, "
                "first_seen = ?, last_updated = ? WHERE id = ?",
                (*self._row_values(merged), theory_id),
            )

        for source in theory.sources:
            self._conn.execute(
                "INSERT OR IGNORE INTO theory_sources "
                "(theory_id, source, url, title, retrieved_at) VALUES (?, ?, ?, ?, ?)",
                (
                    theory_id,
                    source.source,
                    source.url,
                    source.title,
                    source.retrieved_at.isoformat(),
                ),
            )
        self._conn.commit()
        return theory_id

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _row_values(theory: Theory) -> tuple:
        return (
            theory.slug,
            theory.name,
            json.dumps(theory.aliases),
            theory.wikidata_id,
            theory.theory_type,
            theory.popularity_tier,
            theory.metrics.model_dump_json(),
            json.dumps(theory.categories),
            theory.first_seen.isoformat(),
            theory.last_updated.isoformat(),
        )

    def _find_id(self, theory: Theory) -> int | None:
        if theory.wikidata_id:
            row = self._conn.execute(
                "SELECT id FROM theories WHERE wikidata_id = ?", (theory.wikidata_id,)
            ).fetchone()
            if row:
                return int(row["id"])
        row = self._conn.execute(
            "SELECT id FROM theories WHERE slug = ?", (theory.slug,)
        ).fetchone()
        return int(row["id"]) if row else None

    def _get_by_id(self, theory_id: int) -> Theory:
        row = self._conn.execute(
            "SELECT * FROM theories WHERE id = ?", (theory_id,)
        ).fetchone()
        return self._load(row)

    def _load(self, row: sqlite3.Row) -> Theory:
        return Theory(
            id=row["id"],
            slug=row["slug"],
            name=row["name"],
            aliases=json.loads(row["aliases"]),
            wikidata_id=row["wikidata_id"],
            theory_type=row["theory_type"],
            popularity_tier=row["popularity_tier"],
            metrics=TheoryMetrics.model_validate_json(row["metrics"]),
            categories=json.loads(row["categories"]),
            sources=self.sources_for(row["id"]),
            first_seen=row["first_seen"],
            last_updated=row["last_updated"],
        )


def _merge(existing: Theory, incoming: Theory) -> Theory:
    """Combine two records without losing data."""

    aliases = _dedupe([*existing.aliases, *incoming.aliases])
    if incoming.name != existing.name:
        aliases = _dedupe([*aliases, incoming.name])

    metrics = existing.metrics.model_copy(
        update={
            key: value
            for key, value in incoming.metrics.model_dump().items()
            if value is not None
        }
    )
    return existing.model_copy(
        update={
            "aliases": aliases,
            "wikidata_id": existing.wikidata_id or incoming.wikidata_id,
            "theory_type": incoming.theory_type or existing.theory_type,
            "popularity_tier": (
                incoming.popularity_tier
                if incoming.popularity_tier != "unknown"
                else existing.popularity_tier
            ),
            "metrics": metrics,
            "categories": _dedupe([*existing.categories, *incoming.categories]),
            "last_updated": max(existing.last_updated, incoming.last_updated),
        }
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
