"""Ledger tracking which book lesson chunks already produced a script.

Statuses:
- "done":   chunk produced a script/video -> never picked again.
- "failed": the run errored (e.g. transient LLM/rate-limit failure). The
            chunk is NOT consumed; it is picked again on the next run.
Failed entries carry a retry counter and the last error note.
"""

from __future__ import annotations

import json
from pathlib import Path

LEDGER_PATH = Path("content/psychology_books/taken.json")


class TakenLedger:
    """Persistent record of lesson chunks that already generated a script."""

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def _status(entry: dict) -> str:
        """Old entries (no status field) count as done."""
        return entry.get("status", "done")

    def is_taken(self, lesson_path: str | Path) -> bool:
        entry = self._data.get(str(lesson_path))
        if entry is None:
            return False
        return self._status(entry) == "done"

    def mark(self, lesson_path: str | Path, *, script_dir: str, note: str = "") -> None:
        self._data[str(lesson_path)] = {
            "status": "done",
            "script_dir": script_dir,
            "note": note,
        }
        self.save()

    def fail(
        self,
        lesson_path: str | Path,
        *,
        note: str = "",
        max_retries: int = 3,
    ) -> bool:
        """Record a failure WITHOUT consuming the chunk.

        Returns True only when the retry budget is exhausted and the chunk
        should be abandoned (marked done with a failure note) so the runner
        can make forward progress on a permanently broken chunk.
        """

        key = str(lesson_path)
        entry = self._data.get(key, {})
        retries = entry.get("retries", 0) + 1
        abandoned = retries >= max_retries
        self._data[key] = {
            "status": "done" if abandoned else "failed",
            "retries": retries,
            "script_dir": entry.get("script_dir", ""),
            "note": note or entry.get("note", ""),
        }
        self.save()
        return abandoned

    def unmark(self, lesson_path: str | Path) -> None:
        self._data.pop(str(lesson_path), None)
        self.save()

    def taken_paths(self) -> list[str]:
        return [
            path for path, entry in self._data.items()
            if self._status(entry) == "done"
        ]

    def failed_paths(self) -> list[str]:
        return [
            path for path, entry in self._data.items()
            if self._status(entry) == "failed"
        ]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
