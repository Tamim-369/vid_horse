"""The asset shelf: a local, self-replenishing candidate store.

Every image that ever passed the full research gauntlet (junk veto,
CLIP floor, license context) lands in one SQLite table together with
its CLIP image embedding (fp16, 512-d from ViT-B/32). Later research
passes match a subject's TEXT embedding against the shelf locally —
zero network, zero politeness delays, zero rate limits — and re-verify
survivors through the same CLIP gates before any of them is used, so
the quality contract never changes. A use counter deprecates heavy
repeaters so variety across slugs survives the cache.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AssetShelf:
    """Searchable CLIP pool of every asset the pipeline ever approved."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS assets (
                file TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                thumburl TEXT,
                width INTEGER,
                height INTEGER,
                mime TEXT,
                license TEXT,
                artist TEXT,
                description TEXT,
                page_url TEXT,
                kind TEXT,
                embed BLOB NOT NULL,
                clip REAL NOT NULL,
                uses INTEGER NOT NULL DEFAULT 0,
                added REAL NOT NULL
            )"""
        )
        self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    # -- write -----------------------------------------------------------------
    def store(self, info: dict, embed: np.ndarray, clip_score: float) -> bool:
        """Record one approved candidate. Returns False when refused."""

        try:
            embed = np.asarray(embed, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return False
        if embed.size != 512 or not np.isfinite(embed).all():
            return False
        blob = embed.astype(np.float16).tobytes()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO assets (file, url, thumburl, width, height, mime,"
                    " license, artist, description, page_url, kind, embed, clip,"
                    " uses, added) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)"
                    " ON CONFLICT(file) DO UPDATE SET url=excluded.url,"
                    " thumburl=excluded.thumburl, width=excluded.width,"
                    " height=excluded.height, mime=excluded.mime,"
                    " license=excluded.license, artist=excluded.artist,"
                    " description=excluded.description,"
                    " page_url=excluded.page_url, kind=excluded.kind,"
                    " embed=excluded.embed, clip=excluded.clip",
                    (
                        info["file"],
                        info.get("url", ""),
                        info.get("thumburl", ""),
                        info.get("width"),
                        info.get("height"),
                        info.get("mime", ""),
                        info.get("license", ""),
                        info.get("artist", ""),
                        (info.get("description") or "")[:2000],
                        info.get("page_url", ""),
                        info.get("kind", "image"),
                        blob,
                        float(clip_score),
                        time.time(),
                    ),
                )
            return True
        except sqlite3.Error as exc:
            logger.debug("shelf store failed for %s: %s", info.get("file"), exc)
            return False

    def touch(self, file: str) -> None:
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE assets SET uses = uses + 1 WHERE file = ?", (file,)
                )
        except sqlite3.Error:
            pass

    def count(self) -> int:
        try:
            with self._lock:
                return int(
                    self._conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
                )
        except sqlite3.Error:
            return 0

    # -- read ------------------------------------------------------------------
    def match(
        self,
        text_embed: np.ndarray,
        *,
        limit: int = 12,
        max_uses: int = 5,
        min_clip: float = 0.0,
    ) -> list[dict]:
        """Top shelf candidates by cosine; [] when the shelf is empty.

        ``max_uses`` drops heavy repeaters so parallel slugs keep variety;
        ``min_clip`` pre-filters by the score recorded at take time (many
        gated takes carry 0.0 — their search path applied its own gate and
        the route re-verifies every hit through CLIP analyze again). The
        shelf only recalls; it never grants content approval itself.
        """

        query = np.asarray(text_embed, dtype=np.float32).reshape(-1)
        if query.size != 512 or not np.isfinite(query).all():
            return []
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT file, url, thumburl, width, height, mime, license,"
                    " artist, description, page_url, kind, embed, clip, uses"
                    " FROM assets"
                    " WHERE uses < ? AND clip >= ?",
                    (max_uses, float(min_clip)),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("shelf read failed: %s", exc)
            return []
        if not rows:
            return []
        embeds = np.stack(
            [np.frombuffer(row[11], dtype=np.float16).astype(np.float32) for row in rows]
        )
        scores = embeds @ (query / max(np.linalg.norm(query), 1e-8))
        order = np.argsort(-scores)[: limit * 3]
        hits: list[dict] = []
        for index in order:
            row = rows[int(index)]
            hits.append(
                {
                    "file": row[0],
                    "url": row[1],
                    "thumburl": row[2],
                    "width": row[3],
                    "height": row[4],
                    "mime": row[5],
                    "license": row[6],
                    "artist": row[7],
                    "description": row[8],
                    "page_url": row[9],
                    "kind": row[10],
                    "clip_score": float(row[12]),
                    "_shelf_similarity": float(scores[index]),
                }
            )
        return hits[:limit]
