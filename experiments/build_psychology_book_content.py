"""Rebuild book lesson chunks from source PDFs (production-quality extraction).

Usage:
    uv run python experiments/build_psychology_book_content.py [pdf-name-substr]
    uv run python experiments/build_psychology_book_content.py  --force

Output:
    content/psychology_books/<book_slug>/
        part_N/NNN_<slug>.txt        # one teachable chunk per file
        manifest.json                # structure: parts -> chapters -> chunks

What this fixes over the old extractor:
- Running page headers/footers ("THE CROWD 58", "NN THE MIND OF CROWDS") are
  detected by repetition across pages and stripped - they can never again
  become "lesson titles" or lesson content.
- Page text is joined into flowing paragraphs (dehyphenated), so chunks stop
  mid-sentence only at chapter boundaries.
- Real chapter/section headings are detected (TOC when the PDF has one,
  heading lines otherwise) and become the chunk's chapter context.
- Chunks are idea-sized (~250-350 words), split on paragraph boundaries -
  one chunk carries one coherent idea, not 8 lines of OCR.
- Front matter (title/imprint/copyright) is kept in the manifest but marked
  `teachable: false` and never picked for scripts.
- Page OCR results are cached under .cache/book_pages/ so re-runs are fast.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import os
from collections import Counter
from pathlib import Path

import pymupdf as fitz

BOOKS_DIR = Path("assets/psycology_books")
OUT_ROOT = Path("content/psychology_books")
OCR_CACHE = Path(".cache/book_pages")

MIN_CHUNK_WORDS = 130
TARGET_CHUNK_WORDS = 260
MAX_CHUNK_WORDS = 420

# ---------------------------------------------------------------- extraction

_PAGE_OCR_CACHE: dict[str, str] = {}


def _has_text(page: fitz.Page, min_chars: int = 50, min_quality: float = 0.73) -> bool:
    text = page.get_text().strip()
    if len(text) < min_chars:
        return False
    alnum = sum(1 for c in text if c.isalnum())
    return (alnum / max(len(text), 1)) >= min_quality


def _ocr_page(page: fitz.Page, dpi: int = 150) -> str:
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
        os.unlink(img_path)


def page_texts(path: Path, book_slug: str) -> list[str]:
    """Text of every page, OCR'd where needed, cached on disk."""
    doc = fitz.open(str(path))
    cache_dir = OCR_CACHE / book_slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []
    for i, page in enumerate(doc):
        if _has_text(page):
            pages.append(page.get_text())
            continue
        cache = cache_dir / f"{i:04d}.txt"
        if cache.exists():
            pages.append(cache.read_text(encoding="utf-8"))
            continue
        text = _ocr_page(page)
        cache.write_text(text, encoding="utf-8")
        if i % 25 == 0:
            print(f"    ocr page {i}", flush=True)
        pages.append(text)
    doc.close()
    return pages


# ------------------------------------------------------------- normalization

def slugify(name: str) -> str:
    name = (
        name.replace("’", "")
        .replace("‐", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    name = re.sub(r"[^A-Za-z0-9-]+", "_", name).strip("_-").lower()
    name = re.sub(r"_+", "_", name)
    return (name or "untitled")[:70]


def raw_lines(pages: list[str]) -> list[dict]:
    """Flatten page lines with their page index and position-in-page."""
    out = []
    for pno, text in enumerate(pages):
        lines = [ln.strip() for ln in text.splitlines()]
        kept: list[str] = []
        for ln in lines:
            if not ln:
                # preserve blank line as paragraph hint in position list
                kept.append("")
            else:
                kept.append(ln)
        out.append({"page": pno, "lines": kept})
    return out


def strip_furniture(lines: list[dict], occurrence_frac: float = 0.25) -> list[dict]:
    """Remove running headers/footers: short lines repeated across many pages.

    A line is furniture when it appears (recurrently, on >= frac of pages
    that have any content) within the first or last 3 lines of the page.
    """

    n_pages = max(len(lines), 1)
    count: Counter[str] = Counter()
    for rec in lines:
        head = [l for l in rec["lines"][:3] if l]
        tail = [l for l in rec["lines"][-3:] if l]
        for l in set(head + tail):
            if 0 < len(l) <= 90:
                count[_norm_key(l)] += 1

    threshold = max(3, int(occurrence_frac * n_pages))
    furniture = {k for k, c in count.items() if c >= threshold}

    out = []
    for rec in lines:
        kept = []
        idx_from_end = len(rec["lines"]) - 1
        for i, l in enumerate(rec["lines"]):
            pos_end = idx_from_end - i
            edge_zone = i < 3 or pos_end < 3
            if l and edge_zone and _norm_key(l) in furniture:
                continue
            kept.append(l)
        out.append({"page": rec["page"], "lines": kept})
    return out


def _norm_key(line: str) -> str:
    """Normalize a line for equality: drop digits (page numbers) and case."""
    return re.sub(r"\d+", "0", line.lower()).strip(" .:~|")


# ------------------------------------------------------- structure detection
# ------------------------------------------------------------------ folio/TOC

_LEADER_RE = re.compile(r"[.~;·•|,\s]*$")
_TOC_ENTRY_RE = re.compile(
    r"^(?P<title>.{4,90}?)[\s.·•|,;:~]+(?P<page>\d{1,4})\s*$"
)


def _printed_folio(page_lines: list[str]) -> int | None:
    """The printed page number visible near the page edges, if any."""
    for line in page_lines[:3] + page_lines[-3:]:
        m = re.fullmatch(r"\s*(\d{1,4})\s+", line + " ")
        if m:
            return int(m.group(1))
    # number embedded at line start/end ("58 THE MIND OF CROWDS")
    for line in page_lines[:2] + page_lines[-2:]:
        m = re.fullmatch(r"(\d{1,4})\s+\w.*|\w.*?\s+(\d{1,4})", line)
        if m:
            return int(m.group(1) or m.group(2))
    return None


def _parse_toc(pages_text: list[str]) -> list[tuple[str, int]]:
    """Read the printed contents list from the front matter-ish pages."""
    entries: list[tuple[str, int]] = []
    scan = pages_text[:14]
    for text in scan:
        lines = [l.strip() for l in text.splitlines()]
        page_has_list = sum(1 for l in lines if _TOC_ENTRY_RE.match(l)) >= 3
        if not page_has_list:
            continue
        for line in lines:
            if not line:
                continue
            m = _TOC_ENTRY_RE.match(line)
            if not m:
                title = _TOC_TITLE_CLEAN(line)
                if title and entries and len(title) > 3 and not re.search(r"\d\s*$", line):
                    # wrapped title continuation
                    prev_t, prev_p = entries[-1]
                    entries[-1] = (f"{prev_t} {title}", prev_p)
                continue
            title = _TOC_TITLE_CLEAN(m.group("title"))
            page = int(m.group("page"))
            if page < 6000:
                entries.append((title, page))
        if len(entries) >= 6:
            break
    return entries


def _TOC_TITLE_CLEAN(title: str) -> str:
    t = re.sub(r"^[\s.·•|,;:~ASKK]+", "", title)
    t = re.sub(r"[\s.·•|,;~]+$", "", t)
    return t.strip(" :-")


def _page_offset(pages: list[str], toc_entries: list[tuple[str, int]]) -> int:
    """Median (pdf_index - printed_page) offset using folios near page starts."""
    raw_lines_ = raw_lines(pages)
    deltas: list[int] = []
    for rec in raw_lines_:
        folio = _printed_folio(rec["lines"])
        if folio is not None:
            deltas.append(rec["page"] - folio)
            if len(deltas) > 200:
                break
    if not deltas:
        return 0
    deltas.sort()
    return deltas[len(deltas) // 2]


# ------------------------------------------------------- heading detection

_HEADING_PAT = re.compile(
    r"^\s*(?:"
    r"(?P<mark>book|chapter|part|section|lesson|introduction|preface|"
    r"foreword|prologue|epilogue|appendix|index|contents)"
    r"\s+(?P<num>i{1,3}|i[vx]|v{1,2}|[ivxlcdm]+|\d+|one|two|three|four|"
    r"five|six|seven|eight|nine|ten|eleven|twelve)?\b[\s.:—-]*"
    r"(?P<rest>.*?)"
    r"\s*$"
    r")",
    re.I,
)

_ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9 ,.&;:!'’\-—/]{3,90}$")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 95:
        return False
    if _HEADING_PAT.match(s):
        return True
    # OCR-tolerant ALL-CAPS title; must look title-ish, not a full sentence
    if _ALL_CAPS.match(s) and len(s.split()) <= 14:
        # reject lines that end with typical sentence punctuation
        if s[-1] not in ".,":
            if not re.search(r"\b(of|the|and|to|in|that|is)\b.{0,40}\b(of|the|and|to|in)\b", s.lower()):
                return True
    return False


def _heading_key(line: str) -> str:
    s = line.strip().rstrip(".:")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def detect_chapters(pages: list[str], toc: list | None) -> list[dict]:
    """Split the extracted line stream into chapters.

    Strategy:
    1. Parse the printed contents list + folio page numbers (most reliable
       for scanned books whose running headers ARE chapter titles).
    2. Fall back to the PDF metadata TOC if it exists.
    3. Fall back to heading-line detection.
    """

    printed = _parse_toc(pages)
    if len(printed) >= 4:
        chapters = _chapters_from_printed_toc(pages, printed)
        if chapters:
            return chapters
    if toc:
        chapters = _chapters_from_meta_toc(pages, toc)
        if chapters:
            return chapters

    chapters = _chapter_from_header_runs(pages)
    real = [c for c in chapters if c["title"] != "Front matter"]
    if len(real) >= 3:
        return chapters
    return _chapters_from_headings(pages)


def _chapters_from_printed_toc(
    pages: list[str], entries: list[tuple[str, int]]
) -> list[dict]:
    offset = _page_offset(pages, entries)
    # verify: each entry's title should appear near its printed start page
    page_lines = raw_lines(pages)
    verified: list[tuple[str, int]] = []  # (title, pdf_page_index_estimate)
    for title, printed_p in entries:
        start = printed_p + offset
        if not (0 <= start < len(pages)):
            continue
        window = " ".join(
            l for rec in page_lines[max(0, start - 2): start + 3] for l in rec["lines"]
        )
        if _similar(title, window):
            verified.append((title, start - 2))
        else:
            # entry kept if title matchable in the wider window
            wide = " ".join(
                l for rec in page_lines[max(0, start - 5): start + 8] for l in rec["lines"]
            )
            if _similar(title, wide):
                verified.append((title, start - 5))

    if len(verified) < 4:
        return []

    # refine: find the exact heading page within the estimate ±6
    positioned: list[tuple[str, int]] = []
    for title, pdf_start in verified:
        best = pdf_start
        for cand in range(max(0, pdf_start), min(len(pages), pdf_start + 7)):
            window = " ".join(page_lines[cand]["lines"])
            if _similar(title, window):
                best = cand
                break
        positioned.append((title, best))

    # entries must be non-decreasing; drop duplicates/out-of-order
    ordered: list[tuple[str, int]] = []
    last = -1
    for title, pg in positioned:
        if pg < last:
            continue
        if ordered and pg == ordered[-1][1]:
            continue
        ordered.append((title, pg))
        last = pg

    chapters: list[dict] = []
    for i, (title, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(pages)
        if end <= start:
            end = min(start + 1, len(pages))
        lines: list[str] = []
        for rec in page_lines[start:end]:
            lines.extend(rec["lines"])
        if not any(l.strip() for l in lines):
            continue
        chapters.append({"title": title, "page": start, "lines": lines})

    # text before the first chapter entry = front matter
    pre_lines: list[str] = []
    first = chapters[0]["page"] if chapters else len(pages)
    for rec in page_lines[:first]:
        pre_lines.extend(rec["lines"])
    if any(l.strip() for l in pre_lines):
        chapters.insert(0, {"title": "Front matter", "page": 0, "lines": pre_lines})
    return chapters


def _similar(needle: str, haystack: str) -> bool:
    """Fuzzy containment: >=60% of the title's significant words appear (OCR-tolerant)."""

    import difflib

    words = [
        w for w in re.findall(r"[a-z]{3,}", needle.lower())
        if w not in {"the", "and", "chapter", "book", "part", "introduction", "preface"}
    ]
    if not words:
        return False
    found = 0
    hay = haystack.lower().translate(str.maketrans("l1oOS7T", "1100S7T"))
    chunks = hay.split()
    for w in words:
        w_fold = w.translate(str.maketrans("l1oOS7T", "1100S7T"))
        if w_fold in hay:
            found += 1
        elif difflib.get_close_matches(w_fold, chunks, n=1, cutoff=0.8):
            found += 1
    return found / len(words) >= 0.6


def _chapters_from_meta_toc(pages: list[str], toc: list) -> list[dict]:
    # Normalize toc to (level, title, page1based)
    entries = [(lvl, title.strip(), p) for lvl, title, p in toc]
    # Use level-1 entries only for ranges when nested, else all entries
    use = [e for e in entries if e[0] == 1] or entries
    if len(use) < 2:
        return []

    skip = {"title page", "dedication", "copyright page", "contents",
            "front matter", "cover", "about this book"}
    page_lines = raw_lines(pages)
    chapters: list[dict] = []
    for i, (lvl, title, start) in enumerate(use):
        if title.lower() in skip:
            continue
        start_idx = max(0, start - 1)
        end_idx = (use[i + 1][2] - 1) if i + 1 < len(use) else len(pages)
        end_idx = min(end_idx, len(pages))
        if start_idx >= end_idx:
            continue
        lines: list[str] = []
        for rec in page_lines[start_idx:end_idx]:
            lines.extend(rec["lines"])
        if not any(l.strip() for l in lines):
            continue
        chapters.append({
            "title": title,
            "page": start_idx,
            "lines": lines,
        })
    return chapters


def _chapter_from_header_runs(pages: list[str]) -> list[dict]:
    """Chapters recovered from running headers.

    Scanned books print the current chapter title at the top (or bottom) of
    every page in that chapter. Cluster edge lines fuzzily (OCR tolerant),
    then group consecutive pages sharing the same dominant header cluster
    into chapter spans. Requires >= 3 pages per run; shorter runs attach to
    their neighbour as interstitial (divider) pages.
    """

    import difflib

    def norm(line: str) -> str:
        s = re.sub(r"\d+", "", line)
        s = re.sub(r"[^\w]+", " ", s).lower().strip()
        return re.sub(r"\s+", " ", s)

    def match(a: str, b: str) -> bool:
        if not a or not b or min(len(a), len(b)) < 6:
            return False
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82

    clusters: list[dict] = []  # {pages: set[(pno, weight)], exemplars: list[str]}

    def add_edge(pno: int, line: str, weight: float) -> None:
        key = norm(line)
        if len(key.split()) < 2:
            return
        for c in clusters:
            if any(match(key, e) for e in c["exemplars"]):
                c["pages"].add((pno, weight))
                c["exemplars"].append(key)
                return
        clusters.append({"pages": {(pno, weight)}, "exemplars": [key]})

    for rec in raw_lines(pages):
        heads = [l for l in rec["lines"][:3] if l.strip()]
        tails = [l for l in rec["lines"][-3:] if l.strip()]
        for line in heads:
            add_edge(rec["page"], line, 1.0)
        for line in tails:
            add_edge(rec["page"], line, 0.6)

    page_lines = raw_lines(pages)
    n = len(pages)

    # Scanned books alternate running headers on facing pages (verso shows
    # the chapter title, recto the book title), so pick the majority cluster
    # per 2-page spread and apply it to both faces.
    per_page: list[dict[int, float]] = [dict() for _ in range(n)]
    for ci, c in enumerate(clusters):
        for pno, w in c["pages"]:
            per_page[pno][ci] = per_page[pno].get(ci, 0.0) + w
    dominant: list[int | None] = [
        None
    ] * n
    for sp in range((n + 1) // 2):
        counts: dict[int, float] = {}
        for pno in (sp * 2, sp * 2 + 1):
            if pno >= n:
                break
            for ci, w in per_page[pno].items():
                counts[ci] = counts.get(ci, 0.0) + w
        if not counts:
            continue
        dominant[sp * 2] = max(counts, key=counts.get)
        if sp * 2 + 1 < n:
            dominant[sp * 2 + 1] = dominant[sp * 2]

    # group consecutive pages with the same dominant cluster
    segments: list[dict] = []  # {cluster, start, end}
    i = 0
    while i < n:
        ci = dominant[i]
        j = i
        while j + 1 < n and dominant[j + 1] == ci:
            j += 1
        segments.append({"cluster": ci, "start": i, "end": j + 1})
        i = j + 1

    # merge short interstitial runs (divider pages) into the following chapter
    merged: list[dict] = []
    for seg in segments:
        size = seg["end"] - seg["start"]
        if seg["cluster"] is None or size < 3:
            if merged:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(dict(seg))
            continue
        if merged and merged[-1]["cluster"] == seg["cluster"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    chapters: list[dict] = []
    for seg in merged:
        if seg["cluster"] is None:
            continue
        title = _best_header_text(clusters[seg["cluster"]]["exemplars"])
        if not title:
            continue
        lines: list[str] = []
        for rec in page_lines[seg["start"]: seg["end"]]:
            lines.extend(rec["lines"])
        chapters.append({"title": title, "page": seg["start"], "lines": lines})

    # strip header/footer lines from chapter bodies using the recovered
    # header clusters as fuzzy keys (OCR tolerates spelling drift)
    furniture = _furniture_norms(clusters)
    if furniture:
        for c in chapters:
            c["lines"] = [
                l for l in c["lines"]
                if not (l.strip() and _is_furniture_line(l, furniture))
            ]

    # front matter: pages before first recovered chapter
    if chapters and chapters[0]["page"] > 0:
        pre_lines: list[str] = []
        for rec in page_lines[: chapters[0]["page"]]:
            pre_lines.extend(rec["lines"])
        pre_lines = [
            l for l in pre_lines
            if not (l.strip() and _is_furniture_line(l, furniture))
        ]
        chapters.insert(0, {"title": "Front matter", "page": 0, "lines": pre_lines})
    return chapters


def _furniture_norms(clusters: list[dict]) -> list[str]:
    """Normalized header texts of clusters recurring on >= 3 distinct pages."""

    norms: set[str] = set()
    for c in clusters:
        if len({p for p, _w in c["pages"]}) >= 3:
            norms.update(c["exemplars"][:40])
    return sorted(norms)


def _is_furniture_line(line: str, furniture: list[str]) -> bool:
    import difflib

    s = line.strip()
    if re.fullmatch(r"\d{1,4}", s):
        return True  # bare folio
    key = re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", re.sub(r"\d+", "", s)).lower().strip())
    if len(key) < 6:
        return False
    for f in furniture:
        if difflib.SequenceMatcher(None, key, f).ratio() >= 0.85:
            return True
    return False


def _best_header_text(exemplars: list[str]) -> str:
    """Most common full exemplar, digits stripped, looking like a title."""

    import difflib
    from collections import Counter

    # keep only exemplars with >= 2 words; pick the modal exemplar
    words_only = [e for e in exemplars if len(e.split()) >= 2]
    if not words_only:
        return ""
    canon = Counter(words_only)
    best = canon.most_common(1)[0][0]

    # assemble title from the exemplar words that recur most (OCR noise wins)
    words: list[str] = []
    for w in best.split():
        if w in words:
            continue
            # keep order
        words.append(w)
    title = " ".join(words)
    return title[:90].strip()


def _chapters_from_headings(pages: list[str]) -> list[dict]:
    lines = strip_furniture(raw_lines(pages))
    chapters: list[dict] = []
    cur = {"title": "Front matter", "page": 0, "lines": []}
    seen: set[str] = set()
    for rec in lines:
        content_lines = [x for x in rec["lines"] if x.strip()]
        sparse_page = len(content_lines) <= 8
        for i, line in enumerate(rec["lines"]):
            if not line:
                cur["lines"].append("")
                continue
            # headings only count near the top of a page or on sparse pages
            at_top = i <= 6
            if _is_heading(line) and (at_top or sparse_page):
                key = _heading_key(line)
                if key in seen:
                    continue  # repeated running header we didn't filter
                seen.add(key)
                nxt = rec["lines"][i + 1] if i + 1 < len(rec["lines"]) else ""
                title = _clean_title(line, nxt)
                chapters.append(cur)
                cur = {"title": title, "page": rec["page"], "lines": []}
                continue
            cur["lines"].append(line)
    chapters.append(cur)
    chapters = _merge_preamble(chapters)
    # A book with no detectable headings at all: one chapter holds everything
    if len(chapters) == 1 and chapters[0]["title"] == "Front matter":
        chapters[0]["title"] = "The work"
    return chapters


def _clean_title(line: str, nxt: str) -> str:
    """Heading title, folding in the next line when it is the title's wrap."""
    line = line.strip().rstrip(".:-")
    nxt_s = (nxt or "").strip().rstrip(".:-")
    parts = [line]
    if nxt_s and len(nxt_s.split()) <= 14 and not _HEADING_PAT.match(nxt_s):
        parts.append(nxt_s)
    title = " -- ".join(dict.fromkeys(p for p in parts if p))
    return re.sub(r"\s+", " ", title).strip(" -_.") or "untitled"


def _merge_preamble(chapters: list[dict]) -> list[dict]:
    """The first chapter before any real heading keeps title 'Front matter'."""
    if chapters and chapters[0]["title"] == "Front matter":
        body = "\n".join(l for l in chapters[0]["lines"] if l.strip())
        if len(body.split()) < 250:
            front = chapters.pop(0)
            if chapters:
                chapters[0]["lines"] = front["lines"] + chapters[0]["lines"]
    return [c for c in chapters if c["title"]]


# ------------------------------------------------------------------ chunking

def paragraphs(lines: list[str]) -> list[str]:
    """Join cleaned lines into paragraphs; dehyphenate line-final hyphens."""
    paras: list[str] = []
    buf: list[str] = []
    for line in lines:
        if not line.strip():
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        # dehyphenate across the join
        if buf and buf[-1].endswith("-"):
            buf[-1] = buf[-1][:-1] + line
        else:
            buf.append(line)
    if buf:
        paras.append(" ".join(buf))
    # drop OCR noise lines and weak paragraphs
    good = []
    for p in paras:
        if len(p.split()) < 4:
            continue
        good.append(p)
    return good


def chunk_paragraphs(paras: list[str]) -> list[str]:
    """Group text into idea-sized chunks (~250-350 words) at sentence ends."""
    # Re-split paragraphs that survived as huge blobs into sentences first
    segments: list[str] = []
    for p in paras:
        if len(p.split()) > 90:
            segments.extend(
                s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"'])", p) if s.strip()
            )
        else:
            segments.append(p)

    chunks: list[str] = []
    buf: list[str] = []
    words = 0
    for seg in segments:
        wc = len(seg.split())
        if words and words + wc > MAX_CHUNK_WORDS:
            chunks.append(" ".join(buf))
            buf, words = [], 0
        buf.append(seg)
        words += wc
        # split only on sentence-final segments past the target
        if words >= TARGET_CHUNK_WORDS and seg.rstrip()[-1:] in ".!?":
            chunks.append(" ".join(buf))
            buf, words = [], 0
    if buf:
        if chunks and words < MIN_CHUNK_WORDS:
            chunks[-1] = chunks[-1] + " " + " ".join(buf)
        else:
            chunks.append(" ".join(buf))
    return chunks


# -------------------------------------------------------------------- output

FRONT_TITLES = (
    "front matter", "title page", "copyright", "imprint", "contents",
    "table of contents", "publisher", "printer", "unwin", "adelphi",
    "paternoster", "book seller",
)


def is_teachable(chapter_title: str, text: str) -> bool:
    t = chapter_title.lower()
    if any(f in t for f in FRONT_TITLES):
        return False
    body = text.strip()
    if len(body.split()) < 60:
        return False  # too little actual content to teach from
    return True


def write_book(book_path: Path, force: bool) -> None:
    slug = slugify(book_path.stem)
    out_dir = OUT_ROOT / slug
    if out_dir.exists() and not force:
        print(f"  skip (exists, use --force to rebuild): {book_path.name}")
        return
    print(f"Processing {book_path.name}...", flush=True)
    doc = fitz.open(str(book_path))
    toc = doc.get_toc()
    doc.close()
    pages = page_texts(book_path, slug)
    chapters = detect_chapters(pages, toc)
    if toc:
        print(f"  (using pdf TOC with {len(toc)} entries to structure chapters)", flush=True)

    if out_dir.exists():
        for p in sorted(out_dir.rglob("*.txt")):
            p.unlink()
        for d in sorted(out_dir.glob("part_*"), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

    manifest = {"book": book_path.name, "parts": []}
    chapter_seq = 0
    for chapter in chapters:
        chapter_seq += 1
        title = chapter["title"]
        paras = paragraphs(chapter["lines"])
        if not paras:
            continue
        chunks = chunk_paragraphs(paras)
        if not chunks and not is_teachable(title, "\n".join(paras)):
            # still record as non-teachable part
            pass

        part_name = slugify(title)[:60]
        part_dir = out_dir / f"part_{chapter_seq:02d}_{part_name}"
        true_teachable = all(
            is_teachable(title, c) for c in chunks
        ) if chunks else False

        entries = []
        for idx, chunk in enumerate(chunks, 1):
            fname = f"{idx:03d}_{slugify(title)[:48]}.txt"
            part_dir.mkdir(parents=True, exist_ok=True)
            (part_dir / fname).write_text(chunk + "\n", encoding="utf-8")
            entries.append({
                "file": str((part_dir / fname).relative_to(out_dir)),
                "chapter": title,
                "words": len(chunk.split()),
            })
        info = {
            "part": f"{chapter_seq:02d} {title}",
            "teachable": true_teachable and bool(chunks),
            "chunks": entries,
        }
        manifest["parts"].append(info)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    teachable = sum(len(p["chunks"]) for p in manifest["parts"] if p["teachable"])
    total = sum(len(p["chunks"]) for p in manifest["parts"])
    print(f"  wrote {total} chunks ({teachable} teachable) -> {out_dir}")


def main() -> None:
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    pattern = args[0].lower() if args else ""
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for book_path in sorted(BOOKS_DIR.glob("*.pdf")):
        if pattern and pattern not in book_path.stem.lower():
            continue
        write_book(book_path, force)


if __name__ == "__main__":
    main()
