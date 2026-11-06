"""Extract psychology books into per-book / per-part / per-lesson text files.

Usage:
    uv run python experiments/build_psychology_book_content.py
Output:
    content/psychology_books/<book_slug>/...
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pymupdf as fitz

from utils.pdf import _has_text, _ocr_page

BOOKS_DIR = Path("assets/psycology_books")
OUT_ROOT = Path("content/psychology_books")


def slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return name or "book"


def page_texts(path: Path) -> list[str]:
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        if _has_text(page):
            pages.append(page.get_text())
        else:
            pages.append(_ocr_page(page))
    doc.close()
    return pages


def toc_segments(pages: list[str], toc: list) -> list[tuple[str, str, list[str]]]:
    """Return (part_title, lesson_title, page_texts) using TOC."""
    # Normalize toc to (level, title, page)
    entries = [(lvl, title.strip(), p) for lvl, title, p in toc]
    # Build page ranges
    segs = []
    for i, (lvl, title, p) in enumerate(entries):
        start = p - 1
        end = (entries[i + 1][2] - 1) if i + 1 < len(entries) else len(pages)
        start = max(0, min(start, len(pages)))
        end = max(start, min(end, len(pages)))
        body_pages = pages[start:end]
        # Determine part: nearest previous level-1 heading
        part = "Part 1"
        for j in range(i, -1, -1):
            if entries[j][0] == 1 and entries[j][1].lower().startswith(("part", "book")):
                part = entries[j][1]
                break
        segs.append((part, title, body_pages))
    # Filter junk
    skip = {"title page", "dedication", "copyright page", "contents", "index"}
    return [(p, t, bp) for p, t, bp in segs if t.lower() not in skip and any(x.strip() for x in bp)]


def heading_segments(pages: list[str]) -> list[tuple[str, str, list[str]]]:
    """Split into chapters by detecting heading-like lines."""
    heading_re = re.compile(
        r"^(?:chapter|part|book|section|lesson|unit)\b.*$|^[IVXLC]+\.\s+.+$|^[A-Z][A-Z0-9 ,:'\-]{8,}$",
        re.IGNORECASE,
    )
    chapters = []
    current_title = "Introduction"
    current_pages = []
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        head = None
        for ln in lines[:8]:
            if heading_re.match(ln):
                head = ln
                break
        if head and current_pages:
            chapters.append(("Part 1", current_title, current_pages))
            current_title = head
            current_pages = [page]
        else:
            current_pages.append(page)
            if head and not current_pages[:-1]:
                current_title = head
    if current_pages:
        chapters.append(("Part 1", current_title, current_pages))
    # Merge tiny trailing
    return [c for c in chapters if any(p.strip() for p in c[2])]


def write_segments(book_path: Path, segments: list[tuple[str, str, list[str]]]) -> None:
    slug = slugify(book_path.stem)
    book_dir = OUT_ROOT / slug
    book_dir.mkdir(parents=True, exist_ok=True)

    # Group by part
    parts: dict[str, list[tuple[str, list[str]]]] = {}
    for part, title, body_pages in segments:
        parts.setdefault(part, []).append((title, body_pages))

    manifest = {"book": book_path.name, "parts": []}
    for part_name, lessons in parts.items():
        part_slug = slugify(part_name)
        part_dir = book_dir / part_slug
        part_dir.mkdir(parents=True, exist_ok=True)
        part_info = {"part": part_name, "lessons": []}
        for idx, (title, body_pages) in enumerate(lessons, 1):
            lesson_slug = f"{idx:03d}_{slugify(title)[:60]}"
            out_file = part_dir / f"{lesson_slug}.txt"
            text = "\n\n".join(p.strip() for p in body_pages if p.strip())
            out_file.write_text(text, encoding="utf-8")
            part_info["lessons"].append({"title": title, "file": str(out_file.relative_to(book_dir))})
        manifest["parts"].append(part_info)

    (book_dir / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {len(segments)} lessons -> {book_dir}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for book_path in sorted(BOOKS_DIR.glob("*.pdf")):
        import pymupdf as _fitz
        _d=_fitz.open(str(book_path)); _pages=_d.page_count; _d.close()
        if _pages > 300:
            print(f"Skip large book for now: {book_path.name} ({_pages} pages)", flush=True)
            continue
        slug = slugify(book_path.stem)
        if (OUT_ROOT / slug / "manifest.json").exists():
            print(f"Skip (already built): {book_path.name}", flush=True)
            continue
        print(f"Processing {book_path.name}...", flush=True)
        pages = page_texts(book_path)
        doc = __import__("pymupdf").open(str(book_path))
        toc = doc.get_toc()
        doc.close()
        if toc:
            segments = toc_segments(pages, toc)
        else:
            segments = heading_segments(pages)
        if not segments:
            segments = [("Part 1", "Full text", pages)]
        write_segments(book_path, segments)


if __name__ == "__main__":
    main()
