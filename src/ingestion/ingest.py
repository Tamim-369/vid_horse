"""Ingestion pipeline: PDFs and text files -> chunks -> embeddings -> sqlite-vec."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import ollama
import sqlite_vec

from utils.chunking import chunk_paragraphs
from utils.pdf import extract_text

EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768
DB_PATH = Path("assets/rag.sqlite3")
BOOKS_DIR = Path("assets/books")


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the RAG database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            title TEXT,
            type TEXT,
            pages INTEGER,
            chunks INTEGER,
            ingested_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source_id INTEGER REFERENCES sources(id),
            chunk_index INTEGER,
            text TEXT,
            char_start INTEGER,
            char_end INTEGER,
            FOREIGN KEY (source_id) REFERENCES sources(id)
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vecs USING vec0(
            embedding float[{EMBED_DIMS}]
        )
    """)
    conn.commit()


def _extract_source(path: Path) -> dict:
    """Extract text from a file (PDF or text)."""
    if path.suffix.lower() == ".pdf":
        return extract_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "text": text,
            "pages": 1,
            "method": "text",
            "scanned_pages": [],
            "source": str(path),
        }


def _embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts using ollama."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(resp["embeddings"])
    return all_embeddings


def ingest_file(path: Path, conn: sqlite3.Connection) -> int:
    """Ingest a single file. Returns number of chunks created."""
    # Check if already ingested
    existing = conn.execute(
        "SELECT id FROM sources WHERE filename = ?", (path.name,)
    ).fetchone()
    if existing:
        print(f"  Skip (already ingested): {path.name}")
        return 0

    print(f"  Extracting: {path.name}...", end=" ", flush=True)
    t0 = time.time()
    result = _extract_source(path)
    text = result["text"]
    if not text.strip():
        print("NO TEXT")
        return 0

    # Chunk
    chunks = chunk_paragraphs(text, max_tokens=300, source=path.stem)
    if not chunks:
        print("NO CHUNKS")
        return 0

    # Insert source
    cur = conn.execute(
        ("INSERT INTO sources (filename, title, type, pages, chunks, ingested_at) "
         "VALUES (?, ?, ?, ?, ?, ?)"),
        (
            path.name, path.stem, path.suffix.lower().lstrip("."),
            result["pages"], len(chunks), time.time(),
        ),
    )
    source_id = cur.lastrowid

    # Insert chunks and get their IDs
    chunk_ids = []
    for c in chunks:
        cur = conn.execute(
            ("INSERT INTO chunks (source_id, chunk_index, text, char_start, char_end) "
             "VALUES (?, ?, ?, ?, ?)"),
            (source_id, c.index, c.text, c.char_start, c.char_end),
        )
        chunk_ids.append(cur.lastrowid)

    # Embed and store vectors with matching rowids
    texts = [c.text for c in chunks]
    embeddings = _embed_batch(texts)
    for cid, emb in zip(chunk_ids, embeddings, strict=True):
        conn.execute("INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", (cid, json.dumps(emb)))

    # Map chunk IDs to vector rowids (they should be sequential)
    conn.commit()

    elapsed = time.time() - t0
    print(f"{len(chunks)} chunks, {result['pages']} pages, {result['method']}, {elapsed:.1f}s")
    return len(chunks)


def ingest_directory(directory: Path = BOOKS_DIR, db_path: Path = DB_PATH) -> dict:
    """Ingest all files in a directory."""
    conn = _connect(db_path)
    files = sorted(directory.iterdir())
    files = [f for f in files if f.suffix.lower() in (".pdf", ".txt", ".md")]

    print(f"Ingesting {len(files)} files from {directory}")
    total_chunks = 0
    for f in files:
        n = ingest_file(f, conn)
        total_chunks += n

    conn.commit()
    print(f"\nDone: {len(files)} files, {total_chunks} chunks")
    conn.close()
    return {"files": len(files), "chunks": total_chunks}


def search(
    query: str,
    top_k: int = 5,
    db_path: Path = DB_PATH,
    model: str = EMBED_MODEL,
) -> list[dict]:
    """Search the RAG database for relevant chunks."""
    conn = _connect(db_path)

    # Embed query
    resp = ollama.embed(model=model, input=[query])
    q_emb = resp["embeddings"][0]

    # Dense retrieval
    rows = conn.execute(
        "SELECT rowid, distance FROM vecs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (json.dumps(q_emb), top_k * 2),
    ).fetchall()

    # Map rowids to chunks
    results = []
    for rowid, dist in rows:
        chunk_row = conn.execute(
            ("SELECT c.text, c.source_id, s.filename FROM chunks c "
             "JOIN sources s ON c.source_id = s.id WHERE c.id = ?"),
            (rowid,),
        ).fetchone()
        if chunk_row:
            results.append({
                "text": chunk_row[0],
                "source": chunk_row[2],
                "distance": dist,
                "score": 1 - dist,  # higher is better
            })

    conn.close()
    return results[:top_k]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        results = search(query)
        for r in results:
            print(f"\n[{r['source']}] (score={r['score']:.3f})")
            print(r["text"][:200])
    else:
        ingest_directory()
