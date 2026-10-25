"""A tiny disk cache.

Local, CPU-only inference is slow and prompt iteration is repetitive, so every
model call can be memoised. Keys are hashes of the full request; values are
stored as JSON. The cache is deliberately dumb and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    """File-backed key/value cache storing JSON blobs."""

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        self._root = Path(root)
        self._enabled = enabled
        if self._enabled:
            self._root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Build a stable key from arbitrary JSON-serialisable parts."""

        return json.dumps(parts, sort_keys=True, default=str)

    def get(self, key: str) -> Any | None:
        if not self._enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        if not self._enabled:
            return
        try:
            self._path(key).write_text(json.dumps(value), encoding="utf-8")
        except OSError:
            return

    def forget(self, key: str) -> None:
        """Remove one cached entry (a rejected sample must not persist)."""

        if not self._enabled:
            return
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError:
            return

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"
