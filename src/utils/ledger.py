"""Ledger tracking which book lesson chunks already produced a script."""

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

    def is_taken(self, lesson_path: str | Path) -> bool:
        return str(lesson_path) in self._data

    def mark(self, lesson_path: str | Path, *, script_dir: str, note: str = "") -> None:
        self._data[str(lesson_path)] = {
            "script_dir": script_dir,
            "note": note,
        }
        self.save()

    def unmark(self, lesson_path: str | Path) -> None:
        self._data.pop(str(lesson_path), None)
        self.save()

    def taken_paths(self) -> list[str]:
        return list(self._data)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
