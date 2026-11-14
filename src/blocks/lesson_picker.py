"""Lesson picker: read one book chunk and extract ONE teachable lesson idea.

The agent reads as much surrounding material as it needs for continuity —
previous and following parts, including parts that ALREADY produced a video
(those are context-only: they are labeled as used so the agent knows not to
mine them for ideas). The script idea must come from the target chunk, which
is always the next not-yet-used part in book order. Only the target chunk
gets marked in the ledger.
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

# Give the model enough of the book to understand the lesson, while
# staying within a safe context/memory budget (PC can crash if too large).
MAX_CONTEXT_CHARS = 2000
MAX_LESSON_CHARS = 8000
MAX_TOTAL_CONTEXT_CHARS = 6000

SYSTEM = (
    "You read a real BOOK excerpt and find the one hidden truth about how people act that "
    "would hit a viewer in everyday life (for a short dark-psychology video). Read the "
    "TARGET LESSON and pick that single idea. Read the surrounding parts for understanding "
    "and continuity, but the idea MUST come from the TARGET text (the next not-yet-used "
    "part). Parts marked ALREADY USED are context only — never take the idea from them.\n"
    "Use the SIMPLEST everyday words (not the book's style). Be direct and concrete, not "
    "academic. Stay true to what the target text means; do not invent numbers, studies, or "
    "facts it does not support.\n"
    "Return a grounded brief as JSON with: topic (2-5 plain words), lesson (one plain "
    "spoken sentence stating a SPECIFIC mechanism or observation about what you do or why "
    "— never a generic topic label like 'crowd psychology'; not a textbook definition), "
    "hook_moment (a concrete everyday moment/situation that shows it — work, "
    "friends, family, online, or a crowd), why_plain (why you do it, in simple words), "
    "takeaway (one simple thing to notice or do), source_chunks (1-3 short verbatim "
    "passages from the TARGET text), context_notes (one sentence, or empty).\n"
    "If the target has NO real idea to teach, return topic and lesson as empty strings. "
    "Judge the TARGET text AS A WHOLE: if most of it is publisher, copyright, title, "
    "edition or dedication noise (proper nouns, dates, ALL-CAPS lines, garbled OCR) with "
    "only a stray quote or salutation, it is front matter — NOT teachable, even if one "
    "sentence looks quotable. Also return empty for: imprint, table of contents, letters "
    "addressed to a person, author bios, lists of illustrations, and any text under ~40 "
    "real words. Do not invent a generic lesson to fill the gap — empty is the correct "
    "answer. Output JSON only."
)


class LessonIdea(StrictModel):
    """One grounded idea/brief extracted from a book chunk for the writer."""

    topic: str = Field(description="Short spoken-friendly topic name, 2-5 plain words.")
    lesson: str = Field(description="One plain spoken sentence stating the hidden truth about what you do or why (not a textbook definition).")
    hook_moment: str = Field(
        default="",
        description="A concrete everyday moment/situation that shows this (work, friends, family, online, crowd) — for the opening 'why' question.",
    )
    why_plain: str = Field(
        default="",
        description="Why you do it, in simple everyday words, grounded in the target text.",
    )
    takeaway: str = Field(
        default="",
        description="One simple thing to notice or do, in plain words.",
    )
    source_chunks: list[str] = Field(
        default_factory=list,
        description="1-3 short verbatim passages from the TARGET text that support the idea.",
    )
    context_notes: str = Field(
        default="", description="One sentence on how surrounding context helped understanding, if at all."
    )
    # Filled by picker (not by the LLM): grounded text for the script writer.
    target_text: str = Field(default="", description="Target lesson text.")
    context_text: str = Field(default="", description="Brief surrounding section context.")


def _read(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


_PAIR_SYSTEM = (
    "You compare two short lesson summaries from psychology videos. SAME point means "
    "a viewer who learned lesson A would learn nothing new from lesson B: the same "
    "mechanism, the same cause-effect, merely reworded. DIFFERENT means same broad "
    "theme, but a different mechanism, setting or consequence.\n\n"
    "Examples:\n"
    "A: \"People follow what seems popular because groups erase individual identity.\" "
    "B: \"People react on instinct instead of reasoning; they follow whatever feels "
    "right in the group.\" -> SAME (both: instinct/emotion replaces reasoning inside a group).\n"
    "A: \"People follow trends blindly because emotion beats reason.\" "
    "B: \"In an argument, staying calm while the opponent grows irritated makes them "
    "defeat themselves.\" -> DIFFERENT (B is one-on-one argument tactics, not group behavior).\n"
    "A: \"People follow the crowd because emotion beats reason.\" "
    "B: \"Crowds think in images, never in reasoning; one striking picture outweighs a "
    "hundred facts.\" -> DIFFERENT (B is about HOW ideas enter the mind — images vs "
    "arguments — not why people follow).\n"
    "A: \"Crowds adopt ideas as images, never through reasoning.\" "
    "B: \"Crowds demand illusions and never thank anyone who tells them the truth.\" "
    "-> DIFFERENT (both about crowds, but two distinct mechanisms).\n\n"
    "Judge the concrete cause-and-effect story, never the general theme. "
    'Answer the JSON only: {"same_point": true or false}.'
)


class PairVerdict(StrictModel):
    same_point: bool = Field(default=False)


def screen_similar_lessons(
    new_lesson: str,
    all_notes: list[str],
    *,
    top_k: int = 4,
    always_include_last: int = 2,
) -> list[str]:
    """Pick which past lessons the LLM judge should compare against.

    Embeddings cannot DECIDE whether two lessons teach the same point
    (their similarity distributions overlap), but they RANK well: the
    same-family pair scores 0.68-0.75 where cross-family sits lower.
    So the screener ranks the ENTIRE channel history and hands the judge
    only the top candidates plus the most recent lessons — full-history
    duplicate coverage at a small, constant number of judge calls.

    On any failure (embed server down, etc.) it degrades to the recent
    window so production never blocks.
    """

    recent = all_notes[-always_include_last:] if all_notes else []
    if not new_lesson.strip() or len(all_notes) <= top_k:
        return list(all_notes)
    try:
        import ollama as _ollama

        from ingestion.ingest import EMBED_MODEL

        texts = [new_lesson.strip(), *all_notes]
        embs = _ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
        new_emb = embs[0]
        sims = []
        for i, emb in enumerate(embs[1:]):
            dot = sum(a * b for a, b in zip(new_emb, emb))
            na = sum(a * a for a in new_emb) ** 0.5
            nb = sum(b * b for b in emb) ** 0.5
            sims.append((dot / (na * nb), i))
        sims.sort(reverse=True)
        ranked = [all_notes[i] for _, i in sims[:top_k]]
        # merge: most recent lessons always judged, top-ranked added after
        merged = list(dict.fromkeys(recent + ranked))
        logger.info("duplicate screen: %d candidates from %d notes", len(merged), len(all_notes))
        return merged
    except Exception as exc:  # noqa: BLE001 - screening must never block
        logger.warning("lesson similarity screening failed (%s); using recent window", exc)
        return list(recent) or list(all_notes)


def judge_lesson_duplicate(
    llm: LLMProvider,
    new_lesson: str,
    recent_notes: list[str],
    *,
    model: str | None = None,
) -> bool:
    """One small local LLM call per recent lesson: same point, reworded?

    Word/embedding overlap cannot detect paraphrase ("don't think for
    themselves" vs "react instinctively on impression" share nothing but
    mean the same), so a model judges. Judged as direct A-vs-B pairs
    (not against a numbered list) because a 4B model handles one binary
    comparison reliably and a list with a "closest" index poorly. Cheap
    and local; short-circuits on the first duplicate found.
    """

    new_text = new_lesson.strip()
    if not new_text or not recent_notes:
        return False
    for note in recent_notes:
        msgs = [
            LLMMessage(role="system", content=_PAIR_SYSTEM),
            LLMMessage(
                role="user",
                content=f"LESSON A:\n{note.strip()}\n\nLESSON B:\n{new_text}\n\nDoes B teach the SAME point as A?",
            ),
        ]
        try:
            verdict = llm.complete_json(msgs, PairVerdict, temperature=0.0, model=model)
        except (LLMError, BlockError) as exc:
            # the judge must never block production: treat this pair as
            # different and keep going
            logger.warning("lesson-duplicate pair judge failed (%s); assuming different", exc)
            continue
        if verdict.same_point:
            return True
    return False


def pick_lesson(
    llm: LLMProvider,
    lesson_path: str | Path,
    *,
    context_paths: list[str | Path] | None = None,
    ledger: TakenLedger | None = None,
    recent_lessons: list[str] | None = None,
) -> LessonIdea:
    """Read one lesson chunk (plus surrounding section context) and return one idea."""

    lesson_path = Path(lesson_path)
    if not lesson_path.exists():
        raise BlockError(f"lesson file not found: {lesson_path}")
    if ledger is not None and ledger.is_taken(lesson_path):
        raise BlockError(f"lesson already taken: {lesson_path}")

    target = _read(lesson_path, MAX_LESSON_CHARS)

    # Include surrounding parts so the model has enough memory to understand
    # the lesson fully (the whole section, in book order), within a budget.
    paths = [Path(p) for p in (context_paths or []) if Path(p).exists()]
    per_part = (
        min(MAX_CONTEXT_CHARS, max(1200, MAX_TOTAL_CONTEXT_CHARS // len(paths)))
        if paths else 0
    )
    context = ""
    for p in paths:
        used = ledger.is_taken(p) if ledger is not None else False
        label = "ALREADY USED in an earlier video - context only" if used else "not used yet"
        context += f"\n--- context: {p.name} [{label}] ---\n{_read(p, per_part)}\n"

    user = (
        f"TARGET LESSON ({lesson_path.name}) — NEXT UNUSED part; the idea must come from here:\n"
        f"\"\"\"\n{target}\n\"\"\"\n"
        f"SURROUNDING SECTION PARTS (for understanding/continuity only; "
        f"[ALREADY USED] parts must not be the idea source):{context or ' (none)'}\n\n"
        + (
            "RECENT VIDEOS ALREADY TAUGHT these ideas — do NOT repeat their point. "
            "Pick a clearly DIFFERENT mechanism, cause or consequence, or return empty "
            "topic/lesson if the target text only supports one of them:\n"
            + "\n".join(f"- {x}" for x in recent_lessons)
            + "\n\n"
            if recent_lessons else ""
        )
        + "Pick ONE teachable idea from the TARGET lesson and return the lesson brief JSON."
    )
    msgs = [
        LLMMessage(role="system", content=SYSTEM),
        LLMMessage(role="user", content=user),
    ]
    try:
        idea = llm.complete_json(msgs, LessonIdea, temperature=0.4)
    except (LLMError, BlockError) as exc:
        raise BlockError(f"lesson picking failed: {exc}") from exc
    if not idea.topic.strip() or not idea.lesson.strip():
        raise BlockError(f"no teachable idea in: {lesson_path.name}")

    # Build grounded, target-first text for the script writer (safe budget).
    target_for_writer = target[:3500]
    ctx_parts: list[str] = []
    ctx_budget = 1200
    sibs = sorted(x for x in lesson_path.parent.glob("*.txt") if x.is_file())
    try:
        idx = sibs.index(lesson_path)
    except ValueError:
        idx = -1
    if idx >= 0:
        ordered_nb: list[Path] = []
        for d in range(1, max(idx, len(sibs) - idx) + 1):
            if idx - d >= 0:
                ordered_nb.append(sibs[idx - d])
            if idx + d < len(sibs):
                ordered_nb.append(sibs[idx + d])
    else:
        ordered_nb = paths
    used_ctx = 0
    for p in ordered_nb:
        if used_ctx >= ctx_budget:
            break
        room = ctx_budget - used_ctx
        if room < 250:
            break
        snippet = _read(p, min(600, room)).strip()
        if snippet:
            ctx_parts.append(f"[{p.name}]\n{snippet}")
            used_ctx += len(snippet)
    context_for_writer = "\n\n".join(ctx_parts)

    idea = idea.model_copy(update={
        "target_text": target_for_writer,
        "context_text": context_for_writer,
    })
    logger.info(
        "picked lesson brief '%s' from %s (target=%d chars, context=%d chars, %d section parts read)",
        idea.topic, lesson_path.name, len(target_for_writer), len(context_for_writer), len(paths),
    )
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


def is_junk_chunk(text: str) -> bool:
    """Cheap deterministic front-matter / OCR-noise detector.

    The 4B picker rationalizes a lesson out of ANY stray quotable line
    (a copyright page's epigraph became 'Belief & Influence'), so junk
    must be filtered before it ever reaches an LLM call. High precision:
    when unsure, return False — a missed junk chunk still gets caught by
    the variety judge, but a wrongly eaten real chunk is lost forever.
    """

    import re

    body = text.strip()
    words = body.split()
    if len(words) < 60:
        return True  # matches the builder's own minimum

    low = body.lower()
    markers = sum(
        1
        for m in (
            "all rights reserved", "copyright", "isbn", "published by",
            "printed by", "printed in", "first impression", "publisher in ordinary",
            "translated by", "new york:", "london:", "boston:",
        )
        if m in low
    )
    if markers >= 2:
        return True

    tokens = re.findall(r"[A-Za-z0-9']+", body)
    real = [t for t in tokens if len(t) >= 3 and any(c.isalpha() for c in t)]
    if not real:
        return len(tokens) >= 5  # pure noise
    caps = sum(1 for t in real if t.isupper() and len(t) >= 3)
    if caps / len(real) > 0.25:
        return True  # title-page shouting: "MEMOIRS OF ... BY CHARLES MACKAY"
    # OCR garbage: mid-token case switches ("acozroRareD") or vowel-less tokens
    garbage = sum(
        1
        for t in real
        if len(t) >= 5 and any(c.isupper() for c in t[1:-1])
        or (len(t) >= 5 and not any(c in "aeiouAEIOU" for c in t))
    )
    if garbage / len(real) > 0.12:
        return True

    # letters: a salutation plus a sign-off in the same chunk
    if re.search(r"\bdear (mrs|mr|miss|dr|sir)\b", low) and re.search(
        r"yours (most |truly |sincerely |faithfully )", low
    ):
        return True
    return False


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
        if ledger.is_taken(path):
            continue
        # front matter / OCR junk never reaches an LLM call: it would be
        # re-detected (for free) on every later run, so no ledger write
        try:
            if is_junk_chunk(path.read_text(encoding="utf-8", errors="replace")):
                continue
        except OSError:
            pass
        return path
    return None


def neighbours(
    path: Path,
    count: int = 8,
    after: int = 3,
    ledger: TakenLedger | None = None,
) -> list[Path]:
    """All other lesson files in the same book section, for full context.

    The model reads the whole section (all parts in the same folder) so it
    has enough material in memory to understand the lesson fully. Parts
    already used are included as continuity-only context and labeled by the
    picker; the lesson idea must still come from the target part.
    """
    siblings = sorted(p for p in path.parent.glob("*.txt") if p.is_file())
    return [p for p in siblings if p != path]


def load_manifest(book_dir: str | Path) -> dict:
    manifest = Path(book_dir) / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return {}


def chapter_of(lesson_path: str | Path) -> str:
    """Chapter/title of one chunk from the book manifest (empty if unknown)."""

    path = Path(lesson_path)
    manifest_path = path.parent.parent / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    try:
        rel = str(path.relative_to(manifest_path.parent))
    except ValueError:
        rel = path.name
    for part in manifest.get("parts", []):
        for chunk in part.get("chunks", []):
            if chunk.get("file") == rel:
                return str(chunk.get("chapter", ""))
    return ""
