"""Per-run claim sheet: one media file serves one scene only."""

from __future__ import annotations

import logging
import threading

from .media import _stem_of

logger = logging.getLogger(__name__)




class _Claims:
    """Which files are already claimed by any scene in this research pass.

    Parallel scene workers share one registry: a file may serve one scene
    only, and the claim is atomic — check and take happen inside the same
    lock, so two scenes racing for the same MediaWiki file can never both
    win. ``stem`` twins (same file, different extension) land together.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._files: set[str] = set()
        self._stems: set[str] = set()

    def blocked(self, file: str) -> bool:
        with self.lock:
            return file in self._files or _stem_of(file) in self._stems

    def claim(self, file: str) -> bool:
        with self.lock:
            stem = _stem_of(file)
            if file in self._files or stem in self._stems:
                return False
            self._files.add(file)
            self._stems.add(stem)
            return True
