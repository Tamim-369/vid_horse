"""Script engine: grounded single-call script writing from a lesson brief."""

from __future__ import annotations

import logging
import re

from blocks.concept_metaphors import extract_concepts_from_text
from blocks.script_engine.validators import grounding_score, repeats_previous
from core.interfaces import LLMProvider
from core.models import ScriptFormat, ScriptScene, VideoScript
from ingestion.retriever import RAGRetriever

logger = logging.getLogger(__name__)

_DANGLING_WORDS = frozenset(
    [
        "the", "a", "an", "of", "to", "and", "but", "or", "is", "are", "was",
        "were", "in", "on", "at", "for", "with", "that", "which", "when",
        "while", "means", "just", "so", "because", "their", "your", "my",
        "our", "its", "his", "her", "even", "still", "also", "then",
        "perhaps", "maybe", "almost", "mostly", "mainly", "about", "into",
        "over", "under",
    ]
)

_VISUAL_STOPWORDS = frozenset(
    [
        "a", "an", "the", "this", "that", "these", "those", "you", "your",
        "they", "their", "them", "it", "its", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "should", "just", "like", "when",
        "what", "how", "why", "not", "no", "yes", "and", "but", "or", "so",
        "if", "then", "than", "because", "about", "into", "over", "under",
        "with", "from", "for", "of", "on", "at", "by", "to", "in", "out",
        "up", "down", "next", "time", "one", "two", "thing", "things",
        "really", "very", "much", "more", "most", "some", "every", "all",
        "say", "says", "said", "ask", "asks", "asking", "tell", "tells",
        "told", "where", "there", "here", "now", "always", "never",
    ]
)

_META_LEAK_RE = re.compile(
    r"\b(as an ai|here is|here's|sure[,!]|i can help|let me|i will|i'll|"
    r"beat\s*\d|scene\s*\d|hook\s*:|outro\s*:|narration\s*:|visual\s*:|"
    r"lesson brief|target passage|json)\b",
    re.I,
)

_TRAILING_OPEN = re.compile(
    r"\b(when|if|because|before|after|while|that|which|who|to|and|but|or|so|than|as|what|how|where|why)\s*[.!?]?\s*$",
    re.I,
)

_TRAILING_DANGLING = re.compile(
    r"[,;:]?\s*\b(?:the|a|an|of|to|and|but|or|is|are|was|were|in|on|at|for|with|that|which|who|when|while|as|so|because|if|than|into|about|their|your|my|our|its|his|her|this|these|those|what|how|where|why)\b\s*$",
    re.I,
)


_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
)


def _scrub_text(text: str) -> str:
    """Remove unreadable/control characters, JSON/markdown artifacts, and normalize whitespace."""
    t = _CONTROL_CHARS_RE.sub("", str(text or ""))
    # braces/brackets/backslashes are never prose: the writer sometimes
    # leaks a JSON closing brace into a field ("...join its current.}.")
    t = re.sub(r"[{}\[\]\\`|]+", "", t)
    t = t.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()


def _normalize_punctuation(text: str) -> str:
    """Normalize punctuation/spacing for clear, readable spoken lines."""
    t = str(text or "")
    # normalize ellipsis to clear pauses/stops (avoid trailing-off ambiguity)
    def _ellipsis(m: re.Match[str]) -> str:
        rest = t[m.end() :].lstrip()
        if not rest or rest[0] in ".!?":
            return ". "
        return ", "

    t = re.sub(r"\.{2,}|…", _ellipsis, t)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)
    t = re.sub(r",{2,}", ",", t)
    t = re.sub(r"\.\s*,|,\s*\.", ".", t)
    t = re.sub(r"!\s*,|,\s*!", "!", t)
    t = re.sub(r"\?\s*,|,\s*\?", "?", t)
    t = re.sub(r",\s*([.!?])", r"\1", t)
    t = re.sub(r"([.!?])\s*,", r"\1", t)
    t = re.sub(r"\s+([,\.!?;:])", r"\1", t)
    t = re.sub(r"([,!?;:])(?=[A-Za-z0-9])", r"\1 ", t)
    t = re.sub(r"\.(?=[A-Za-z])", ". ", t)
    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"(?<=\s)[,;:](?=\s|$)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _clean_speech(text: str) -> str:
    """Strip labels, markdown, meta leakage, and unreadable/garbled artifacts."""
    t = _scrub_text(text)
    if not t:
        return ""
    # strip common field/line labels the model may echo
    t = re.sub(r"^\s*(hook|scene\s*\d+|outro|line\s*\d+|narration)\s*[:\-]\s*", "", t, flags=re.I)
    t = re.sub(r"^\s*\d+\s*[.)]\s*", "", t)
    t = re.sub(r"[*_`#<>]", "", t)
    t = t.replace("—", ", ").replace("–", ", ").replace("…", "...")
    t = re.sub(r'"|“|”', "", t)
    t = t.replace("‘", "'").replace("’", "'")
    t = re.sub(r"(?<!\w)'|'(?!\w)", "", t)
    # if meta leakage appears, keep only the part before it
    m = _META_LEAK_RE.search(t)
    if m and m.start() > 10:
        t = t[: m.start()].strip()
    t = _normalize_punctuation(t)
    t = re.sub(r"^[\s\-:;,]+|[\s\-:;,]+$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_garbled(text: str) -> bool:
    """Detect degenerate/corrupted output that should be replaced."""
    t = (text or "").strip()
    if not t:
        return True
    words = t.split()
    if not words:
        return True
    # broken punctuation clusters signal corruption/degeneration
    if re.search(r",\s*,|\.\s*,|,\s*\.", t):
        return True
    # too many hard sentence breaks for a single narration line
    if len(re.findall(r"[.!?]", t)) > 3:
        return True
    # excessive comma density or very long rambling output
    commas = t.count(",")
    if commas >= max(6, len(words) // 3 + 2):
        return True
    if len(words) > 90:
        return True
    # mostly non-letter content
    letters = sum(ch.isalpha() for ch in t)
    if letters < 6 or (len(words) >= 5 and letters < len(t) * 0.4):
        return True
    return False


def _is_bad(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t.split()) < 3:
        return True
    if _META_LEAK_RE.search(t):
        return True
    if _is_garbled(t):
        return True
    return False


def _strip_trailing_open(text: str) -> str:
    """Drop a trailing incomplete clause, keeping earlier complete sentences."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t or not _TRAILING_OPEN.search(t):
        return t
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    if len(parts) > 1 and _TRAILING_OPEN.search(parts[-1]):
        kept = " ".join(parts[:-1]).strip()
        if kept:
            return kept
    # single fragment ending open: remove dangling tail markers
    for _ in range(3):
        new = _TRAILING_OPEN.sub("", t).strip(" ,;:-.!?")
        new = _TRAILING_DANGLING.sub("", new).strip(" ,;:-.!?")
        new = re.sub(r"\s+", " ", new).strip()
        if new == t or not new:
            t = new
            break
        t = new
        if not _TRAILING_OPEN.search(t):
            break
    return re.sub(r"\s+", " ", t).strip()


def _ensure_sentence(text: str, end: str = ".") -> str:
    t = _normalize_punctuation(_scrub_text(text))
    if not t:
        return t
    t = _strip_trailing_open(t)
    if not t:
        return ""
    if t[-1] not in ".!?":
        t += end
    # avoid ending on a dangling connector
    t = re.sub(
        r"[,;:]?\s*\b(?:and|but|or|so|because|which|that|to|like|as|when|if|than)\b\s*[.!?]?\s*$",
        end,
        t,
        flags=re.I,
    )
    t = _normalize_punctuation(t)
    # if we stripped down to almost nothing, let caller decide fallback
    if len(t.split()) < 2:
        return ""
    return t.strip()


def _shape_fallback(text: str, *, end: str = ".", max_words: int = 26) -> str:
    """Turn fallback/brief text into a clear, complete spoken line."""
    t = _clean_speech(text)
    if not t:
        return ""
    # prefer the first clear sentence/fragment
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    if parts:
        t = parts[0]
    t = _strip_trailing_open(t)
    # if still long, take a sensible leading chunk at a clause boundary
    words = t.split()
    if len(words) > max_words:
        chunk = " ".join(words[:max_words]).rstrip(" ,;:.-!?")
        cut = max(chunk.rfind(","), chunk.rfind(";"), chunk.rfind("."))
        if cut >= max(12, len(chunk) // 2):
            chunk = chunk[:cut].rstrip()
        t = chunk
    # trim dangling tail words so the fallback ends as a complete thought
    for _ in range(4):
        stripped = _TRAILING_DANGLING.sub("", t).strip(" ,;:-.!?")
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped or stripped == t:
            t = stripped or t
            break
        t = stripped
    t = _ensure_sentence(t, end=end)
    return _normalize_punctuation(t)


def _cap_to_complete(text: str, max_words: int) -> str:
    """Shorten to whole sentences within ~max_words; never cut mid-thought."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t or len(t.split()) <= max_words:
        return t
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    if len(parts) > 1:
        kept: list[str] = []
        total = 0
        for p in parts:
            w = len(p.split())
            if kept and total + w > max_words + 6:
                break
            kept.append(p)
            total += w
            if total >= max_words:
                break
        if kept:
            return " ".join(kept)
    # Single long sentence: keep it intact rather than chopping into nonsense.
    return t


def _visual_keywords(text: str, limit: int = 4) -> str:
    words = re.findall(r"[a-z][a-z-]{2,}", (text or "").lower())
    picked: list[str] = []
    for word in words:
        if word in _VISUAL_STOPWORDS or word in picked:
            continue
        picked.append(word)
        if len(picked) >= limit:
            break
    return " ".join(picked)


class ScriptEngineBlock:
    name = "script_engine"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        target_seconds: int = 45,
        max_scene_retries: int = 2,  # kept for API compatibility
        retriever: RAGRetriever | None = None,
    ) -> None:
        from config import get_settings

        self._llm = llm
        self._target_seconds = target_seconds
        self._retriever = retriever
        settings = get_settings()
        self._writer_model = settings.ollama_writer_model or settings.ollama_model

    def run(
        self,
        topic: str,
        *,
        format: ScriptFormat = "short",
        lesson: str = "",
        hook_moment: str = "",
        why_plain: str = "",
        takeaway: str = "",
        key_quotes: list[str] | None = None,
        target_text: str = "",
        context_text: str = "",
        source_chunks: list[str] | None = None,
        context_note: str = "",
        shape_seed: str = "",
    ) -> VideoScript:
        # Optional RAG (non-lesson flows)
        if not target_text and source_chunks is None and self._retriever is not None:
            try:
                results = self._retriever.retrieve_for_topic(topic, top_k=6)
                source_chunks = [r["text"] for r in results]
                logger.info("RAG: retrieved %d chunks for '%s'", len(source_chunks), topic)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG retrieval failed: %s", exc)
                source_chunks = None

        if not target_text and source_chunks:
            # backward compat: first chunk is primary text
            target_text = source_chunks[0]
            if not context_text and len(source_chunks) > 1:
                context_text = "\n\n".join(source_chunks[1:3])

        key_quotes = list(key_quotes or [])
        if not key_quotes and source_chunks and len(source_chunks) > 1 and not context_text:
            # legacy picker put quotes after combined blob; if target_text came from elsewhere, treat extras as quotes
            key_quotes = [c for c in source_chunks[1:4] if c and c.strip()]

        # Budget for pacing: use only as many words as needed (cap, not a quota)
        wps = 2.5
        total_words = int(self._target_seconds * wps)
        cap_mode = self._target_seconds >= 120
        per_scene_ceiling = max(12, total_words // 6)
        target = min(per_scene_ceiling, 30) if cap_mode else max(10, min(per_scene_ceiling, 22))

        from blocks.script_engine.writer import write_full_script, pick_shape

        shape_name, shape_text = pick_shape(shape_seed or topic)
        logger.info("script shape: %s", shape_name)
        result = write_full_script(
            self._llm,
            topic=topic,
            lesson=lesson or context_note or topic,
            hook_moment=hook_moment,
            why_plain=why_plain,
            takeaway=takeaway,
            key_quotes=key_quotes,
            target_text=target_text,
            context_text=context_text,
            target_words=target,
            model=self._writer_model,
            flexible=cap_mode,
            shape=shape_text,
        )

        raw = [
            (result.hook or "").strip(),
            (result.scene1 or "").strip(),
            (result.scene2 or "").strip(),
            (result.scene3 or "").strip(),
            (result.scene4 or "").strip(),
            (result.outro or "").strip(),
        ]
        lines = [_clean_speech(x) for x in raw]

        # Dedupe: drop later duplicates (keep first occurrence)
        seen: set[str] = set()
        for i, ln in enumerate(lines):
            key = re.sub(r"\W+", " ", ln.lower()).strip()
            if not key:
                continue
            if key in seen:
                lines[i] = ""
            else:
                seen.add(key)

        # Deterministic, brief-grounded fallbacks (no invented "beats")
        fallbacks = [
            hook_moment or lesson or topic,
            lesson or why_plain or topic,
            why_plain or lesson or topic,
            takeaway or why_plain or lesson or topic,
            takeaway or lesson or "Notice this the next time it happens.",
            takeaway or lesson or "Now you can see it more clearly.",
        ]

        fixed: list[str] = []
        for i, ln in enumerate(lines):
            end = "?" if i == 0 else "."
            # resolve trailing-open/incomplete lines before acceptance
            ln = _strip_trailing_open(ln)
            if _is_bad(ln) or repeats_previous(ln, fixed):
                replacement = ""
                for fb_src in [fallbacks[i] if i < len(fallbacks) else "", fallbacks[-1]]:
                    fb = _shape_fallback(fb_src, end=end, max_words=target)
                    if fb and not _is_bad(fb) and not repeats_previous(fb, fixed):
                        replacement = fb
                        break
                ln = replacement or _shape_fallback(
                    fallbacks[i] if i < len(fallbacks) else fallbacks[-1],
                    end=end,
                    max_words=target,
                )
            ln = _ensure_sentence(ln, end=end)
            if not ln or len(ln.split()) < 3 or _is_garbled(ln):
                fb = _shape_fallback(
                    fallbacks[i] if i < len(fallbacks) else "",
                    end=end,
                    max_words=target,
                ) or _ensure_sentence(
                    "Why does this happen to you?" if i == 0 else "You feel it before you see it.",
                    end=end,
                )
                ln = fb
            ln = _normalize_punctuation(_scrub_text(ln))
            fixed.append(ln)

        # Enforce shape: hook asks "why" about a moment; line 2 may be question or statement;
        # lines 3-6 are statements.
        hook = fixed[0]
        # prefer an actual question sentence if one exists
        q = re.search(r"[^.!?]*\?", hook)
        if q:
            hook = q.group(0).strip()
        else:
            hook = re.sub(r"[.!]+\s*$", "?", hook).strip() or hook
            if "?" not in hook:
                hook = hook.rstrip() + "?"
        hook = _cap_to_complete(hook, 16)
        if "?" not in hook:
            hook = hook.rstrip(".! ") + "?"
        hook = _normalize_punctuation(_scrub_text(hook))
        fixed[0] = hook

        fixed[1] = _ensure_sentence(fixed[1]) or _shape_fallback(fallbacks[1], end=".", max_words=target)
        for i in range(2, 6):  # scene2, scene3, scene4, outro -> statements
            fixed[i] = fixed[i].replace("?", ".")
            fixed[i] = _ensure_sentence(fixed[i]) or _shape_fallback(
                fallbacks[i] if i < len(fallbacks) else fallbacks[-1], end=".", max_words=target
            )

        # Only shorten very overlong lines, and never by cutting mid-thought
        soft_cap = per_scene_ceiling + (12 if cap_mode else 8)
        for i in range(1, 6):
            if len(fixed[i].split()) > soft_cap:
                capped = _ensure_sentence(_cap_to_complete(fixed[i], per_scene_ceiling + 4))
                if capped:
                    fixed[i] = capped

        scene_lines = [_normalize_punctuation(_scrub_text(x)) for x in fixed[1:5]]
        outro = _normalize_punctuation(_scrub_text(fixed[5]))
        hook = _normalize_punctuation(_scrub_text(fixed[0]))

        scenes: list[ScriptScene] = []
        for narr in scene_lines:
            found = extract_concepts_from_text(narr)
            visual = found[0] if found else _visual_keywords(narr) or "people thinking"
            if len(visual.split()) > 5:
                visual = _visual_keywords(visual)
            secs = max(3, int(len(narr.split()) / 2.5))
            scenes.append(ScriptScene(narration=_scrub_text(narr), visual=_scrub_text(visual), seconds=secs))

        script = VideoScript(
            title=_scrub_text(topic),
            hook=hook,
            format=format,
            target_seconds=self._target_seconds,
            scenes=scenes,
            outro=outro,
        )

        # Grounding signal against the target passage (if available)
        ref = (target_text or " ".join(source_chunks or [])).strip()
        if ref:
            full_text = " ".join([hook] + scene_lines + [outro])
            try:
                score = grounding_score(full_text, ref)
                logger.info("grounding score: %.2f (content words traced to source)", score)
                if score < 0.25:
                    logger.warning("weak grounding (%.2f): script may drift off the target text", score)
            except Exception as exc:  # noqa: BLE001
                logger.debug("grounding score failed: %s", exc)

        logger.info("script generated (brief-driven single write, writer=%s)", self._writer_model)
        return script

