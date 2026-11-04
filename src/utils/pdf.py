"""PDF text extraction: text-based via PyMuPDF, scanned via OCR."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


def _text_quality(text: str) -> float:
    """Ratio of alphanumeric characters to total (0.0-1.0)."""
    if not text:
        return 0.0
    alnum = sum(1 for c in text if c.isalnum())
    return alnum / len(text)


def _has_text(page: fitz.Page, min_chars: int = 50, min_quality: float = 0.73) -> bool:
    """True when the page carries enough readable embedded text."""
    text = page.get_text().strip()
    if len(text) < min_chars:
        return False
    return _text_quality(text) >= min_quality


def _ocr_page(page: fitz.Page, dpi: int = 150) -> str:
    """Render a page to an image and OCR it with tesseract."""
    pix = page.get_pixmap(dpi=dpi)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = tmp.name
    try:
        pix.save(img_path)
        result = subprocess.run(
            ["tesseract", img_path, "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip()
    finally:
        import contextlib
        with contextlib.suppress(OSError):
            os.unlink(img_path)


def extract_text(path: str | Path, *, ocr: bool = True, dpi: int = 150) -> dict:
    """Extract text from a PDF.

    Returns a dict with:
        - text: full extracted text
        - pages: number of pages
        - method: 'text' or 'ocr' or 'mixed'
        - scanned_pages: list of page numbers that were OCR'd
    """
    path = Path(path)
    doc = fitz.open(str(path))
    pages_text: list[str] = []
    scanned_pages: list[int] = []
    methods: set[str] = set()

    for i, page in enumerate(doc):
        if _has_text(page):
            pages_text.append(page.get_text())
            methods.add("text")
        elif ocr:
            ocr_text = _ocr_page(page, dpi=dpi)
            pages_text.append(ocr_text)
            scanned_pages.append(i + 1)
            methods.add("ocr")
        else:
            pages_text.append("")
            methods.add("empty")

    doc.close()

    if len(methods) == 1:
        method = methods.pop()
    elif "ocr" in methods and "text" in methods:
        method = "mixed"
    else:
        method = "text"

    return {
        "text": "\n\n".join(pages_text),
        "pages": len(pages_text),
        "method": method,
        "scanned_pages": scanned_pages,
        "source": str(path),
    }


def extract_text_simple(path: str | Path) -> str:
    """Convenience wrapper: returns just the text string."""
    return extract_text(path)["text"]
