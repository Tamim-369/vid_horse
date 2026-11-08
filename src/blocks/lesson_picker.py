"""Lesson picker: read one book chunk and extract ONE teachable lesson idea.

The agent may also read neighbouring chunks as CONTEXT (for continuity), but
the script idea must come from the target chunk itself. Context chunks stay
un-taken in the ledger; only the target chunk gets marked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import Field

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import StrictModel
from utils.ledger import TakenLedger

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000
MAX_LESSON_CHARS = 12000

SYSTEM = (
    "You are a teacher preparing one short video mini-lesson from a book "
    "excerpt. Read the TARGET LESSON text and pick the single most teachable "
    "idea in it: something a viewer aged 10 to 40 can understand and use. "
    "Neighbouring excerpts are CONTEXT ONLY — use them to understand "
    "continuity, but the idea MUST come from the target text. Give the idea a "
    "short spoken-friendly topic name and quote 2-3 short passages from the "
    "target text that support it. Output JSON only."
)


class LessonIdea(StrictModel):
    """One teachable idea extracted from a book chunk."""

    topic: str = Field(description="Short spoken-friendly topic name, 2-5 words.")
    lesson: str = Field(description="One sentence stating what the viewer will learn.")
    source_chunks: list[str] = Field(
        default_factory=list,
        description="Short verbatim passages from the target text that support the lesson.",
    )
    context_notes: str = Field(
        default="", description="One sentence on how neighbouring context was used, if at all."
    )


def _read(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def pick_lesson(
    llm: LLMProvider,
    lesson_path: str | Path,
    *,
    context_paths: list[str | Path] | None = None,
    ledger: TakenLedger | None = None,
) -> LessonIdea:
    """Read one lesson chunk (plus optional context) and return one idea."""

    lesson_path = Path(lesson_path)
    if not lesson_path.exists():
        raise BlockError(f"lesson file not found: {lesson_path}")
    if ledger is not None and ledger.is_taken(lesson_path):
        raise BlockError(f"lesson already taken: {lesson_path}")

    target = _read(lesson_path, MAX_LESSON_CHARS)
    context = ""
    for path in context_paths or []:
        p = Path(path)
        if p.exists():
            context += f"\n--- context: {p.name} ---\n{_read(p, MAX_CONTEXT_CHARS)}\n"

    user = (
        f"TARGET LESSON ({lesson_path.name}):\n\"\"\"\n{target}\n\"\"\"\n"
        f"NEIGHBOURING CONTEXT (do NOT take the idea from here):{context or ' (none)'}\n\n"
        "Pick ONE teachable idea from the target lesson. Output JSON with keys: "
        "topic (str), lesson (str), source_chunks (list[str], short verbatim "
        "passages from the TARGET text), context_notes (str)."
    )
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(role="user", content=user),
    ]
    try:
        idea = llm.complete_json(msgs, LessonIdea, temperature=0.4)
    except (LLMError, BlockError) as exc:
        raise BlockError(f"lesson picking failed: {exc}") from exc
    logger.info("picked lesson idea '%s' from %s", idea.topic, lesson_path.name)
    return idea


def lesson_files(book_dir: str | Path) -> list[Path]:
    """Teachable lesson .txt files of one book, in book order.

    Prefers manifest.json: only parts marked ``teachable`` contribute files,
    so front matter / imprint chunks are never picked. Falls back to a plain
    recursive scan when no manifest exists.
    """

    root = Path(book_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = []
            for part in manifest.get("parts", []):
                if not part.get("teachable"):
                    continue
                for chunk in part.get("chunks", []):
                    p = root / chunk.get("file", "")
                    if p.is_file():
                        files.append(p)
            if files:
                return files
        except (json.JSONDecodeError, OSError):
            logger.warning("bad manifest in %s; falling back to file scan", root)
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def next_untaken(
    book_dir: str | Path,
    ledger: TakenLedger,
    skip: set[str] | None = None,
) -> Path | None:
    """First not-yet-taken lesson file of a book, honoring a per-run skip set."""

    banned = skip or set()
    for path in lesson_files(book_dir):
        if str(path) in banned:
            continue
        if not ledger.is_taken(path):
            return path
    return None


def neighbours(path: Path, count: int = 1) -> list[Path]:
    """The previous N lesson files of the same book part, for context."""
    siblings = sorted(p for p in path.parent.glob("*.txt") if p.is_file())
    index = siblings.index(path)
    return siblings[max(0, index - count) : index]


def load_manifest(book_dir: str | Path) -> dict:
    manifest = Path(book_dir) / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return {}
