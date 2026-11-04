"""Text chunking: paragraph, semantic, and hierarchical strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int
    source: str = ""
    page: int | None = None
    parent_index: int | None = None
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Normalize whitespace, fix hyphenation, remove control chars."""
    # Fix hyphenated line breaks: "word-\nword" -> "wordword"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace single newlines with space (within paragraph)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove control characters (keep newlines and tabs)
    text = "".join(c for c in text if c == "\n" or c == "\t" or c.isprintable())
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines."""
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def chunk_fixed(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    source: str = "",
) -> list[Chunk]:
    """Fixed-size chunking with overlap (by characters)."""
    text = clean_text(text)
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    source=source,
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1
        start = end - overlap if end < len(text) else len(text)
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


def chunk_paragraphs(
    text: str,
    max_tokens: int = 500,
    source: str = "",
    page: int | None = None,
) -> list[Chunk]:
    """Chunk by paragraphs, merging small ones up to max_tokens."""
    paragraphs = split_paragraphs(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    char_pos = 0
    idx = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            # Flush current chunk
            chunk_text = "\n\n".join(current)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    source=source,
                    page=page,
                    char_start=char_pos - len(chunk_text),
                    char_end=char_pos,
                )
            )
            idx += 1
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens
        char_pos += len(para) + 2

    # Flush remaining
    if current:
        chunk_text = "\n\n".join(current)
        chunks.append(
            Chunk(
                text=chunk_text,
                index=idx,
                source=source,
                page=page,
                char_start=char_pos - len(chunk_text),
                char_end=char_pos,
            )
        )
    return chunks


def chunk_semantic(
    text: str,
    max_tokens: int = 500,
    similarity_threshold: float = 0.5,
    embed_fn=None,
    source: str = "",
    page: int | None = None,
) -> list[Chunk]:
    """Semantic chunking: group sentences by embedding similarity.

    Falls back to paragraph chunking if no embed_fn provided.
    """
    if embed_fn is None:
        return chunk_paragraphs(text, max_tokens=max_tokens, source=source, page=page)

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # Embed all sentences
    embeddings = [embed_fn(s) for s in sentences]

    # Group by similarity
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]
    current_tokens = _estimate_tokens(sentences[0])

    for i in range(1, len(sentences)):
        sim = cosine(embeddings[i - 1], embeddings[i])
        sent_tokens = _estimate_tokens(sentences[i])
        if sim < similarity_threshold or current_tokens + sent_tokens > max_tokens:
            # Flush
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    source=source,
                    page=page,
                )
            )
            current = [sentences[i]]
            current_tokens = sent_tokens
        else:
            current.append(sentences[i])
            current_tokens += sent_tokens

    if current:
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                text=chunk_text,
                index=len(chunks),
                source=source,
                page=page,
            )
        )
    return chunks


@dataclass
class HierarchicalChunks:
    """Parent (large) and child (small) chunks with references."""

    parents: list[Chunk]
    children: list[Chunk]


def chunk_hierarchical(
    text: str,
    parent_tokens: int = 2000,
    child_tokens: int = 400,
    source: str = "",
    page: int | None = None,
) -> HierarchicalChunks:
    """Hierarchical chunking: large parents contain small children.

    Retrieve children for precision, return parent for context.
    """
    # Parent chunks: large sections
    parents = chunk_paragraphs(
        text, max_tokens=parent_tokens, source=source, page=page
    )
    # Child chunks: small paragraphs within each parent
    children: list[Chunk] = []
    child_idx = 0
    for parent_idx, parent in enumerate(parents):
        child_chunks = chunk_paragraphs(
            parent.text, max_tokens=child_tokens, source=source, page=page
        )
        for cc in child_chunks:
            cc.parent_index = parent_idx
            cc.index = child_idx
            children.append(cc)
            child_idx += 1
    return HierarchicalChunks(parents=parents, children=children)
