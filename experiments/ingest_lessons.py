"""Ingest extracted psychology-book lessons into the RAG knowledge base.

Usage:
    uv run python experiments/ingest_lessons.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.ingest import _connect, _embed_batch
from utils.chunking import chunk_paragraphs

BOOKS_ROOT = Path("content/psychology_books")
DB_PATH = Path("assets/rag.sqlite3")


def main() -> None:
    conn = _connect(DB_PATH)
    files = sorted(BOOKS_ROOT.rglob("*.txt"))
    print(f"Ingesting {len(files)} lesson files from {BOOKS_ROOT}")
    total = 0
    for path in files:
        rel = str(path.relative_to(BOOKS_ROOT))
        existing = conn.execute(
            "SELECT id FROM sources WHERE filename = ?", (rel,)
        ).fetchone()
        if existing:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 200:
            continue
        chunks = chunk_paragraphs(text, max_tokens=300, source=path.stem)
        if not chunks:
            continue
        book = path.relative_to(BOOKS_ROOT).parts[0]
        cur = conn.execute(
            "INSERT INTO sources (filename, title, type, pages, chunks, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rel, f"{book} / {path.stem}", "txt", 1, len(chunks), time.time()),
        )
        source_id = cur.lastrowid
        chunk_ids = []
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (source_id, chunk_index, text, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?)",
                (source_id, c.index, c.text, c.char_start, c.char_end),
            )
            chunk_ids.append(cur.lastrowid)
        embeddings = _embed_batch([c.text for c in chunks])
        for cid, emb in zip(chunk_ids, embeddings, strict=True):
            conn.execute(
                "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)",
                (cid, json.dumps(emb)),
            )
        conn.commit()
        total += len(chunks)
        print(f"  {rel}: {len(chunks)} chunks")
    print(f"done: {total} new chunks")
    conn.close()


if __name__ == "__main__":
    main()
