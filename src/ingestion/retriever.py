"""RAG retriever: hybrid search (dense + BM25) over the knowledge base."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

import ollama
import sqlite_vec

from ingestion.ingest import DB_PATH, EMBED_MODEL

EMBED_DIMS = 768


class RAGRetriever:
    """Hybrid retriever over the ingested knowledge base."""

    def __init__(self, db_path: Path = DB_PATH, model: str = EMBED_MODEL):
        self.db_path = db_path
        self.model = model
        self._conn = None
        self._corpus: list[str] | None = None
        self._chunk_ids: list[int] | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        return self._conn

    def _load_corpus(self) -> None:
        """Load all chunks for BM25."""
        if self._corpus is not None:
            return
        conn = self._connect()
        rows = conn.execute("SELECT id, text FROM chunks").fetchall()
        self._chunk_ids = [r[0] for r in rows]
        self._corpus = [r[1] for r in rows]
        # Build BM25 index
        self._tokenized = [self._tokenize(t) for t in self._corpus]
        self._N = len(self._corpus)
        self._df: dict[str, int] = {}
        for doc in self._tokenized:
            for t in set(doc):
                self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = sum(len(d) for d in self._tokenized) / max(self._N, 1)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _bm25_scores(self, query: str) -> dict[int, float]:
        """Compute BM25 scores for all chunks."""
        self._load_corpus()
        q_tokens = self._tokenize(query)
        k1, b = 1.5, 0.75
        scores = {}
        for i, doc in enumerate(self._tokenized):
            dl = len(doc)
            tf_map = {}
            for t in doc:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for t in q_tokens:
                if t in tf_map:
                    tf = tf_map[t]
                    idf = math.log(
                        (self._N - self._df.get(t, 0) + 0.5)
                        / (self._df.get(t, 0) + 0.5)
                        + 1
                    )
                    score += idf * (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * dl / self._avgdl)
                    )
            if score > 0:
                scores[self._chunk_ids[i]] = score
        return scores

    def _dense_scores(self, query: str, top_k: int) -> dict[int, float]:
        """Compute dense (embedding) scores for top-k chunks."""
        conn = self._connect()
        resp = ollama.embed(model=self.model, input=[query])
        q_emb = resp["embeddings"][0]
        rows = conn.execute(
            "SELECT rowid, distance FROM vecs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (json.dumps(q_emb), top_k * 3),
        ).fetchall()
        # Convert distance to similarity (lower distance = higher similarity)
        scores = {}
        for rowid, dist in rows:
            scores[rowid] = 1.0 / (1.0 + dist)
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Hybrid retrieval: BM25 + dense via Reciprocal Rank Fusion.

        Returns list of dicts with keys: text, source, score, chunk_id
        """
        conn = self._connect()

        # Keyword relevance: chunk must contain at least one content word from query
        query_words = set(re.findall(r"\w+", query.lower()))
        query_words = {w for w in query_words if len(w) >= 3}

        # Dense scores
        dense = self._dense_scores(query, top_k)

        # BM25 scores
        bm25 = self._bm25_scores(query)

        # RRF fusion
        rrf_k = 60
        rrf_scores: dict[int, float] = {}
        dense_ranked = sorted(dense, key=dense.get, reverse=True)
        bm25_ranked = sorted(bm25, key=bm25.get, reverse=True)

        for rank, cid in enumerate(dense_ranked):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
        for rank, cid in enumerate(bm25_ranked):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        # Fetch chunk text
        results = []
        for cid in sorted_ids[:top_k * 2]:
            row = conn.execute(
                "SELECT c.text, c.source_id, s.filename FROM chunks c "
                "JOIN sources s ON c.source_id = s.id WHERE c.id = ?",
                (cid,),
            ).fetchone()
            if not row:
                continue
            if source_filter and source_filter not in row[2]:
                continue
            # Relevance threshold: only return chunks with meaningful score
            if rrf_scores[cid] < 0.028:
                continue
            # Keyword filter: chunk must share a RARE content word with the query
            if query_words:
                common = {"time", "cube", "plot", "story", "book", "people", "world"}
                rare_words = query_words - common
                if rare_words:
                    chunk_lower = row[0].lower()
                    if not any(w in chunk_lower for w in rare_words):
                        continue
            results.append({
                "chunk_id": cid,
                "text": row[0],
                "source": row[2],
                "score": rrf_scores[cid],
            })
            if len(results) >= min(top_k, 3):
                break

        return results

    def retrieve_for_topic(
        self,
        topic: str,
        sub_queries: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve chunks for a topic, with optional sub-queries.

        Uses query decomposition: breaks the topic into sub-queries and
        retrieves for each, then deduplicates.
        """
        queries = [topic]
        if sub_queries:
            queries.extend(sub_queries)

        seen_ids = set()
        all_results = []
        for q in queries:
            results = self.retrieve(q, top_k=top_k)
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    all_results.append(r)

        # Sort by score and return top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k * 2]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
