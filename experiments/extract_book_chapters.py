"""Extract chapter topics from every book in assets/books.

Usage:
    uv run python experiments/extract_book_chapters.py
Output:
    output/book_chapters.json
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.books import list_books, read_book, split_chapters


def main() -> None:
    out = Path("output/book_chapters.json")
    out.parent.mkdir(exist_ok=True)

    result = []
    for book in list_books():
        print(f"Reading {book.name}...", flush=True)
        text = read_book(book)
        chapters = split_chapters(text)
        result.append(
            {
                "book": book.name,
                "chapter_count": len(chapters),
                "chapters": [
                    {"title": title, "chars": len(body), "preview": body[:300]}
                    for title, body in chapters
                ],
            }
        )
        print(f"  -> {len(chapters)} chapters", flush=True)

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
