"""Helpers for turning a book into chapter-sized topics."""

from __future__ import annotations

import re
from pathlib import Path

from utils.pdf import extract_text, extract_text_simple


_HEADING_RE = re.compile(
    r"^(?:(?:chapter|part|lesson|section|book|unit)\s+[\divxlc\d]+.*|[A-Z][A-Z0-9 ,:'\-]{6,}$|\d{1,3}\.\s+[A-Z].*)$",
    re.IGNORECASE | re.MULTILINE,
)


def read_book(path: Path) -> str:
    """Read a book file (PDF or text) into a single string."""
    if path.suffix.lower() == ".pdf":
        return extract_text_simple(path)
    return path.read_text(encoding="utf-8", errors="replace")


def split_chapters(text: str, *, min_chars: int = 200) -> list[tuple[str, str]]:
    """Split a book into (title, body) chapters using heading heuristics."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        # Fall back to splitting on blank-line-separated blocks of headings.
        return [("Full text", text.strip())]

    chapters: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) >= min_chars:
            chapters.append((title, body))
    if not chapters:
        chapters.append(("Full text", text.strip()))
    return chapters


def list_books(directory: Path = Path("assets/books")) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".pdf", ".txt", ".md"}
    )
