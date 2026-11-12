"""Multi-stage script engine: facts -> beats -> scenes -> validate -> revise."""

from __future__ import annotations

import logging
import re

from blocks.concept_metaphors import CONCEPT_VISUALS, extract_concepts_from_text
from blocks.script_engine.beats import generate_beats
from blocks.script_engine.facts import generate_fact_sheet
from blocks.script_engine.polish import polish_script_lines
from blocks.script_engine.validators import (
    check_claims,
    grounding_score,
    repeats_previous,
    validate_pov,
    validate_scene,
)
from blocks.script_engine.writer import condense_line, write_middle, write_scene
from core.interfaces import LLMProvider
from core.models import ScriptFormat, ScriptScene, VideoScript
from ingestion.retriever import RAGRetriever

logger = logging.getLogger(__name__)

BEAT_NAMES = ["hook", "situation", "mechanism", "turn", "action", "payoff"]

# Asset-aware writing: the writer must pick VISUAL hints from the same
# concept vocabulary the asset agent uses, and weaving a concept's plain
# phrase into the narration lets the asset agent lock onto that concept.
_CONCEPT_KEYS: list[str] = sorted(CONCEPT_VISUALS)
_CONCEPT_KEY_BY_NORM = {k.replace("_", " "): k for k in _CONCEPT_KEYS}


_REASONING_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s|\*{1,2}[A-Za-z]|<[a-z/]|Constraints?:|Analyze|Role:|Tone:|Format:|Length"
    r"|The user|Let me|Let's break|I'll|Here is|Here's|Okay,|Sure,|Step \d)",
    re.I,
)


_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:text|query|narration|visual|scene)\b[\s/.*]*",
    re.I,
)


def _is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(text)) or len(text.split()) < 3


def _same_text(a: str, b: str) -> bool:
    """True when two lines are the same idea (punctuation/case insensitive)."""

    def norm(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    wa, wb = norm(a), norm(b)
    if not wa or not wb:
        return False
    if wa == wb:
        return True
    overlap = len(set(wa) & set(wb)) / max(len(set(wa) | set(wb)), 1)
    return overlap >= 0.8


def _clean_speech(text: str) -> str:
    """Strip markdown/prompt artifacts; keep apostrophes inside words."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"[*_`<>{}]", "", text)
    # slash-markup the model leaks ("/br", "./br", "\br")
    text = re.sub(r"[./\\]+\s*br\b\.?", " ", text, flags=re.I)
    # drop everything from the first reasoning marker on
    cut = _REASONING_MARKERS.search(text)
    if cut:
        text = text[: cut.start()]
    # model appends its own search-word tail to narration — strip it
    text = re.sub(r"\s*(?:search\s*words?|keywords?)\s*:.*$", "", text, flags=re.I)
    # straight/curly double quotes off; apostrophes between letters stay
    text = re.sub(r'"|“|”', "", text)
    text = text.replace("‘", "'").replace("’", "'")
    # keep ' only between two letters (contractions); all other quotes go
    text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)
    # stray digit the model left dangling after the final punctuation
    text = re.sub(r"([.!?])\s*\d{1,3}\s*$", r"\1", text.strip())
    # unlabeled keyword tail the model tacked on after the last sentence
    # (lowercase start — a real next sentence would begin with a capital)
    tail = re.search(
        r"([.!?])\s+([a-z][A-Za-z']*(?:\s*[;,]?\s*[A-Za-z'][A-Za-z']*){1,3})"
        r"\s*[.;,]?\s*$",
        text,
    )
    if tail:
        tail_words = re.findall(r"[A-Za-z']+", tail.group(2))
        if all(w.lower() not in _VISUAL_STOPWORDS for w in tail_words):
            text = text[: tail.start()] + tail.group(1)
    # trailing dangling conjunction the model left before the final stop
    text = re.sub(
        r"[,;:]?\s*(?:and|but|or|so|because|which|that|to)\s*[.\s]*$",
        ".",
        text.strip(),
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def _parse_scene(raw: str) -> tuple[str, str]:
    import re

    narr = ""
    vis = ""
    # typo-tolerant labels: 4B models emit NARRATIOAN/NARROTAION/VISION etc.
    m = re.search(r"NARR\w*\s*:\s*(.*?)(?=VIS\w*\s*:|$)", raw, re.I | re.S)
    if m:
        narr = m.group(1).strip()
    m2 = re.search(r"VIS\w*\s*:\s*(.*)", raw, re.I | re.S)
    if m2:
        # keep only the first line: free models echo instructions after it
        vis_lines = m2.group(1).strip().splitlines()
        vis = vis_lines[0].strip() if vis_lines else ""
        # strip markdown emphasis/code marks; keep '_' (concept-key delimiter)
        vis = re.sub(r"[*<>`]", "", vis).strip()
    # remove any leaked VISUAL/VISION/FOMO labels from narration
    narr = re.sub(r"\s*VIS\w*\s*:.*", "", narr, flags=re.I | re.S).strip()
    narr = re.sub(r"\s*FOMO\s*:.*", "", narr, flags=re.I | re.S).strip()
    # scrub stray mislabelled tokens the model leaks mid-line
    # ("...stay quiet.NARROTION." -> "...stay quiet.")
    narr = re.sub(r"\bNARR\w*[.:]?\s*", "", narr, flags=re.I).strip()
    narr = re.sub(r"\bVIS\w*[.:]?\s*", "", narr, flags=re.I).strip()
    narr = narr.strip(" .,;:-")
    if not narr:
        narr = raw.strip()
    narr = _clean_speech(narr)
    narr = _ensure_complete(narr)
    # a concept key (with or without underscores) is valid at any length and
    # must be checked BEFORE _clean_speech (which strips underscores);
    # snap it to the canonical key the asset agent knows
    vis_key = _CONCEPT_KEY_BY_NORM.get(vis.lower().replace("_", " ").strip())
    if vis_key is not None:
        return narr, vis_key
    vis = _clean_speech(vis)
    # a visual is a 2-4 word image query; anything longer is leaked narration
    if len(vis.split()) > 5:
        vis = _visual_keywords(vis)
    if _is_placeholder(vis):
        vis = ""
    # scene failure mode: model puts the same line in both fields —
    # drop the visual so the block falls back to the beat text
    if narr and vis and _same_text(narr, vis):
        vis = ""
    return narr, vis


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



_INCOMPLETE_TAIL = re.compile(
    r"\b(when|if|as|while|before|after|because|since|though|although|unless|until|"
    r"that|which|who|whom|where|and|or|but|so|than|to|for|with|about|into|"
    r"like|from)\s+(everyone|everybody|someone|somebody|anyone|anybody|people|"
    r"they|you|we|he|she|it|things|something|anything|nothing|a\s+\w+|the\s+\w+|your\s+\w+|their\s+\w+)\s*[.!?]?\s*$",
    re.I,
)
_TRAILING_OPEN = re.compile(
    r"\b(when|if|because|before|after|while|that|which|who|to|and|but|or|so|than|as)\s*[.!?]?\s*$",
    re.I,
)


def _is_incomplete(text: str) -> bool:
    """True when narration looks cut off / not a complete thought."""
    t = (text or "").strip()
    if not t or len(t.split()) < 4:
        return True
    if _INCOMPLETE_TAIL.search(t):
        return True
    if _TRAILING_OPEN.search(t):
        return True
    # ends without sentence-final punctuation and smells fragmentary
    if t[-1] not in ".!?":
        if len(t.split()) < 6:
            return True
    return False


def _ensure_complete(text: str) -> str:
    """If clearly cut off, trim the dangling tail to the last clean boundary."""
    t = (text or "").strip()
    if not _is_incomplete(t):
        return t
    # try trimming the dangling tail (drop last 1-3 words) and re-close
    words = re.findall(r"\S+", t.rstrip(".!?"))
    for drop in (2, 3, 1):
        if len(words) <= drop + 3:
            continue
        cand = " ".join(words[:-drop]).rstrip(",;: ")
        # don't end on a dangling word
        cw = cand.split()
        while cw and cw[-1].lower().strip(",;:") in _DANGLING_WORDS:
            cw.pop()
        if len(cw) >= 4 and not _TRAILING_OPEN.search(" ".join(cw)):
            return " ".join(cw).rstrip(",;: ") + "."
    # fallback: return as-is but ensure terminal punct
    if t and t[-1] not in ".!?":
        t += "."
    return t

def _first_sentence(text: str, max_words: int = 12, min_words: int = 0) -> str:
    """First sentence of ``text``, hard-capped at ``max_words`` words.

    With ``min_words`` set, short lead sentences pull in the next sentence
    so the line does not end as a weak fragment. Truncation never ends on a
    dangling function word ("...changed the rules of the.").
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sent = parts[0] if parts else text.strip()
    if min_words and len(sent.split()) < min_words and len(parts) > 1:
        sent = parts[0] + " " + parts[1]
    words = sent.split()
    if len(words) > max_words:
        words = words[:max_words]
    while len(words) > 3 and words[-1].lower().strip(",;:.!?") in _DANGLING_WORDS:
        words.pop()
    if len(words) < len(sent.split()):
        sent = " ".join(words).rstrip(",;:.!?") + "."
    return _ensure_complete(sent)


def _parse_batch(
    raw: str,
    hook_line: str = "",
    payoff_line: str = "",
) -> dict[int, tuple[str, str]]:
    """Split a batch reply into {beat_index: (narration, visual)}.

    Preferred format: numbered lines ("2. <text>") where the number is
    the line's position in the full script (2 = first middle line), so
    misalignment is impossible even when the model echoes the hook.
    Falls back to NARRATION/VISUAL blocks, then blank-line paragraphs
    (assigned sequentially). Lines that near-duplicate the hook or
    payoff are dropped.
    """
    known = [line for line in (hook_line, payoff_line) if line]
    out: dict[int, tuple[str, str]] = {}
    for m in re.finditer(r"^\s*(\d+)\s*[.)]\s*(.+)$", raw, re.M):
        text = m.group(2).strip()
        if not text or _is_placeholder(text) or repeats_previous(text, known):
            continue
        out[int(m.group(1)) - 2] = (text, "")
    if out:
        return out
    pairs: list[tuple[str, str]] = []
    chunks = re.split(r"(?=NARR\w*\s*:)", raw, flags=re.I)
    for chunk in chunks:
        if not re.match(r"NARR\w*\s*:", chunk, re.I):
            continue
        narr, vis = _parse_scene(chunk)
        if narr:
            pairs.append((narr, vis))
    if not pairs:
        for chunk in re.split(r"\n\s*\n", raw):
            chunk = chunk.strip()
            if chunk and not _is_placeholder(chunk):
                pairs.append((chunk, ""))
    idx = 0
    for narr, vis in pairs:
        if repeats_previous(narr, known):
            continue
        out[idx] = (narr, vis)
        idx += 1
    return out


def _trim_fresh(text: str, target: int, previous: list[str]) -> str:
    """Leading complete sentences of ``text`` within the word budget,
    skipping sentences that near-duplicate lines already in the script.

    A 4B model that rambles tends to open every attempt with the same
    image; taking the first sentence blindly then makes every scene
    sound identical. Skipping repeated sentences keeps each line fresh
    without an extra LLM call. Returns "" when nothing fresh fits.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    count = 0
    for part in parts:
        if not part.strip() or repeats_previous(part, previous):
            continue
        w = len(part.split())
        if count + w > target and kept:
            break
        kept.append(part)
        count += w
        if count >= max(4, target - 4):
            break
    out = " ".join(kept).strip()
    if not out:
        return ""
    if len(out.split()) > target:
        out = _first_sentence(out, max_words=target)
    return out


_POV_SWAPS = [
    (r"\bI\b", "you"),
    (r"\bmy\b", "your"),
    (r"\bmine\b", "yours"),
    (r"\bme\b", "you"),
    (r"\bwe\b", "you"),
    (r"\bus\b", "you"),
    (r"\bour\b", "your"),
    (r"\bourselves\b", "yourselves"),
]

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


def _visual_keywords(text: str, limit: int = 4) -> str:
    """Derive a short image-query from the most significant words."""
    words = re.findall(r"[a-z][a-z-]{2,}", text.lower())
    picked: list[str] = []
    for word in words:
        if word in _VISUAL_STOPWORDS or word in picked:
            continue
        picked.append(word)
        if len(picked) >= limit:
            break
    return " ".join(picked)


def _fix_pov(text: str) -> str:
    """Mechanically rewrite first-person words to second person."""
    for pattern, replacement in _POV_SWAPS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    if text:
        text = text[0].upper() + text[1:]
    return text


class ScriptEngineBlock:
    name = "script_engine"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        target_seconds: int = 45,
        max_scene_retries: int = 2,
        retriever: RAGRetriever | None = None,
    ) -> None:
        from config import get_settings

        self._llm = llm
        self._target_seconds = target_seconds
        self._max_scene_retries = max_scene_retries
        self._retriever = retriever
        settings = get_settings()
        self._writer_model = settings.ollama_writer_model or settings.ollama_model

    def run(
        self,
        topic: str,
        *,
        format: ScriptFormat = "short",
        source_chunks: list[str] | None = None,
        context_note: str = "",
    ) -> VideoScript:
        if source_chunks is None and self._retriever is not None:
            try:
                results = self._retriever.retrieve_for_topic(topic, top_k=6)
                source_chunks = [r["text"] for r in results]
                logger.info("RAG: retrieved %d chunks for '%s'", len(source_chunks), topic)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG retrieval failed: %s", exc)
                source_chunks = None

        fs = generate_fact_sheet(
            self._llm, topic, source_chunks=source_chunks, context_note=context_note
        )
        beats = generate_beats(self._llm, fs)
        source_text = " ".join(source_chunks or [])

        # --- simplified: plan, then write the whole script in one call ---
        # (facts+beats already done above; fewer calls = more coherence)
        wps = 2.5
        total_words = int(self._target_seconds * wps)
        cap_mode = self._target_seconds >= 120
        per_scene_ceiling = max(12, total_words // 6)
        # writer hint: comfortable length, not something to pad to
        if cap_mode:
            target = min(per_scene_ceiling, 30)
        else:
            target = max(10, min(per_scene_ceiling, 22))

        wm = self._writer_model
        from blocks.script_engine.writer import write_full_script

        raw = write_full_script(
            self._llm, fs, beats, target,
            source_text=source_text, model=wm, flexible=cap_mode,
        )

        # parse numbered lines 1-6
        lines: list[str] = []
        for m in re.finditer(r"^\s*(\d+)\s*[.)]\s*(.+)$", raw, re.M):
            text_line = _clean_speech(m.group(2).strip())
            if text_line and not _is_placeholder(text_line):
                lines.append(text_line)
        # fallback: split by non-empty lines if numbering missing
        if len(lines) < 6:
            lines = []
            for ln in raw.splitlines():
                ln = re.sub(r"^\s*\d+\s*[.)]\s*", "", ln).strip()
                ln = _clean_speech(ln)
                if ln and not _is_placeholder(ln):
                    lines.append(ln)
        # dedupe while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for ln in lines:
            key = re.sub(r"\W+", " ", ln.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(ln)
        lines = uniq

        # ensure we have 6 lines (hook + 4 scenes + outro)
        while len(lines) < 6:
            idx = len(lines)
            fb = _clean_speech(beats[idx] if idx < len(beats) else beats[-1])
            lines.append(_ensure_complete(fb) if fb else "Think about this in your own life.")

        def _clean_hook(text: str) -> str:
            if not text:
                return text
            text = text.strip()
            text = re.sub(r'^\{[^}]+\}\s*', '', text)
            text = re.sub(r'^\[[^\]]+\]\s*', '', text)
            text = re.sub(r'^["\']\s*', '', text)
            text = re.sub(r'\s*["\']$', '', text)
            text = text.replace("*", "").replace("…", "...")
            text = re.sub(r'\.{3,}', '', text)
            return re.sub(r'\s+', ' ', text).strip()

        hook_narr = _clean_hook(lines[0])
        if "?" not in hook_narr:
            hook_narr = hook_narr.rstrip(".! ") + "?"
        hook_narr = _ensure_complete(_first_sentence(hook_narr, max_words=14))
        if not hook_narr.endswith("?"):
            hook_narr = hook_narr.rstrip(".! ") + "?"
        if _is_placeholder(hook_narr) or len(hook_narr.split()) < 4:
            hook_narr = _ensure_complete(_first_sentence(_clean_speech(beats[0]), max_words=12))
            if "?" not in hook_narr:
                hook_narr = hook_narr.rstrip(".! ") + "?"

        scene_lines = list(lines[1:5])
        outro_narr = lines[5]

        fixed_scenes: list[str] = []
        prevs: list[str] = [hook_narr]
        for i, ln in enumerate(scene_lines):
            # line 2 (i==0) may ask one question; later must be statements
            if i >= 1 and "?" in ln:
                ln = ln.replace("?", ".")
            if len(ln.split()) > per_scene_ceiling + 4:
                trimmed = _trim_fresh(ln, per_scene_ceiling, prevs)
                if not trimmed:
                    trimmed = _first_sentence(ln, max_words=per_scene_ceiling)
                if trimmed and not _is_placeholder(trimmed):
                    ln = trimmed
            if _is_placeholder(ln) or len(ln.split()) < 5 or repeats_previous(ln, prevs):
                beat_fb = _clean_speech(beats[i + 1] if i + 1 < len(beats) else beats[-1])
                if beat_fb and not repeats_previous(beat_fb, prevs):
                    ln = _ensure_complete(beat_fb)
            fixed_scenes.append(ln)
            prevs.append(ln)

        if "?" in outro_narr:
            outro_narr = outro_narr.replace("?", ".")
        if len(outro_narr.split()) > per_scene_ceiling + 4:
            trimmed = _first_sentence(outro_narr, max_words=per_scene_ceiling)
            if trimmed and not _is_placeholder(trimmed):
                outro_narr = trimmed
        if _is_placeholder(outro_narr) or len(outro_narr.split()) < 5:
            outro_narr = _ensure_complete(_clean_speech(beats[5] if len(beats) > 5 else beats[-1]))

        # POV repair
        all_narrs = [hook_narr] + fixed_scenes + [outro_narr]
        pov_res = validate_pov(all_narrs)
        if not pov_res.ok:
            logger.info("POV drift: %s — repairing", pov_res.issues)
            hook_narr = _fix_pov(hook_narr)
            outro_narr = _fix_pov(outro_narr)
            fixed_scenes = [_fix_pov(s) for s in fixed_scenes]

        scenes: list[ScriptScene] = []
        for i, narr in enumerate(fixed_scenes):
            beat = beats[i + 1] if i + 1 < len(beats) else narr
            found = extract_concepts_from_text(narr)
            visual = found[0] if found else _visual_keywords(narr) or _visual_keywords(beat)
            if len(visual.split()) > 5:
                visual = _visual_keywords(visual)
            if not visual or _is_placeholder(visual):
                visual = _visual_keywords(beat)
            secs = max(3, int(len(narr.split()) / 2.5))
            scenes.append(ScriptScene(narration=narr, visual=visual, seconds=secs))

        script = VideoScript(
            title=fs.concept,
            hook=hook_narr,
            format=format,
            target_seconds=self._target_seconds,
            scenes=scenes,
            outro=outro_narr,
        )

        if source_chunks:
            full_text = " ".join([hook_narr] + fixed_scenes + [outro_narr])
            score = grounding_score(full_text, " ".join(source_chunks))
            logger.info("grounding score: %.2f (content words traced to source)", score)
            if score < 0.3:
                logger.warning("weak grounding (%.2f): script may drift off the book text", score)

        logger.info("script generated in simplified single-write mode")
        return script
