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
    validate_pov,
    validate_scene,
)
from blocks.script_engine.writer import write_scene
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
    text = re.sub(r"[*_`<>]", "", text)
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
    m = re.search(r"NARRATION:\s*(.*?)(?=VIS(?:UAL|ION):|$)", raw, re.I | re.S)
    if m:
        narr = m.group(1).strip()
    m2 = re.search(r"VIS(?:UAL|ION):\s*(.*)", raw, re.I | re.S)
    if m2:
        # keep only the first line: free models echo instructions after it
        vis = m2.group(1).strip().splitlines()[0].strip()
        # strip markdown emphasis/code marks; keep '_' (concept-key delimiter)
        vis = re.sub(r"[*<>`]", "", vis).strip()
    # remove any leaked VISUAL/VISION/FOMO labels from narration
    narr = re.sub(r"\s*VIS(?:UAL|ION):.*", "", narr, flags=re.I | re.S).strip()
    narr = re.sub(r"\s*FOMO\s*:.*", "", narr, flags=re.I | re.S).strip()
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
        self._llm = llm
        self._target_seconds = target_seconds
        self._max_scene_retries = max_scene_retries
        self._retriever = retriever

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

        # words per scene: distribute target seconds
        wps = 2.5
        total_words = int(self._target_seconds * wps)
        per_scene = max(6, total_words // len(beats))

        # Generate payoff first, then hook referencing it.
        payoff_beat = beats[5] if len(beats) >= 6 else beats[-1]
        payoff_raw = write_scene(self._llm, payoff_beat, fs, [], 12, source_text=source_text, concepts=_CONCEPT_KEYS)
        payoff_narr, payoff_vis = _parse_scene(payoff_raw)
        if not payoff_narr or _is_placeholder(payoff_narr):
            payoff_narr = _clean_speech(payoff_beat)
        payoff_narr = _ensure_complete(_first_sentence(payoff_narr, max_words=16, min_words=8))

        def _clean_hook(text: str) -> str:
            """Strip JSON-like prefixes/labels like {Question} from hook text."""
            if not text:
                return text
            text = text.strip()
            text = re.sub(r'^\{[^}]+\}\s*', '', text)
            text = re.sub(r'^\[[^\]]+\]\s*', '', text)
            text = re.sub(r'^["\']\s*', '', text)
            text = re.sub(r'\s*["\']$', '', text)
            # markdown/emphasis noise the small model leaks into hooks
            text = text.replace("*", "").replace("…", "...")
            text = re.sub(r'\.{3,}', '', text)  # spoken pauses read as noise
            text = re.sub(r'\s+', ' ', text).strip()
            return text.strip()

        def _hook_ok(text: str) -> bool:
            return (
                bool(text)
                and not _is_placeholder(text)
                and 4 <= len(text.split()) <= 14
                and not _is_incomplete(text)
            )

        # The hook is the writer's first regular call. There is no dedicated
        # hook writer or hook mode: the same writer writes every line, with
        # the attention craft built into its system prompt. It sees the
        # payoff first so the opening can promise it without giving it away.
        hook_beat = beats[0]
        hook_narr, _hook_vis = _parse_scene(
            write_scene(self._llm, hook_beat, fs, [payoff_narr], 10,
                        source_text=source_text, concepts=_CONCEPT_KEYS)
        )
        hook_narr = _clean_hook(hook_narr)
        # mechanical clamps only (they never invent content)
        hook_narr = _ensure_complete(_first_sentence(hook_narr, max_words=12))
        if not _hook_ok(hook_narr):
            # single retry on hard constraints only
            hook_narr, _hook_vis = _parse_scene(
                write_scene(self._llm, hook_beat, fs, [payoff_narr], 10,
                            source_text=source_text, concepts=_CONCEPT_KEYS)
            )
            hook_narr = _clean_hook(hook_narr)
            hook_narr = _ensure_complete(_first_sentence(hook_narr, max_words=12))
        if not _hook_ok(hook_narr):
            # last resort: the beat itself, spoken plainly (never a generic
            # template — that reads as canned)
            hook_narr = _ensure_complete(_first_sentence(
                _clean_speech(hook_beat), max_words=12))
            logger.warning("hook fallback used beat text: %s", hook_narr)

        # Generate middle scenes.
        middle_beats = beats[1:5]
        scenes: list[ScriptScene] = []
        scene_beats: list[str] = []
        narrations: list[str] = [hook_narr]
        for i, beat in enumerate(middle_beats):
            target = per_scene
            attempt = 0
            narr = ""
            vis = ""
            # the line right after the hook is still prime attention time:
            # no warm-up, no recap, straight into the proof
            write_beat = (
                beat + " (This follows the hook — go straight at the viewer, "
                "no warm-up, no recapping the hook, keep the tension.)"
                if i == 0 else beat
            )
            while attempt < self._max_scene_retries:
                try:
                    raw = write_scene(self._llm, write_beat, fs, narrations, target, source_text=source_text, concepts=_CONCEPT_KEYS)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("scene %d write failed: %s", i, exc)
                    attempt += 1
                    continue
                narr, vis = _parse_scene(raw)
                if not narr or _is_placeholder(narr):
                    attempt += 1
                    logger.info("scene %d retry: placeholder/empty output", i)
                    continue
                res = validate_scene(narr, target)
                if _is_incomplete(narr):
                    res.ok = False
                    res.issues.append("incomplete/cut-off narration")
                if not res.ok:
                    attempt += 1
                    logger.info("scene %d retry: %s", i, res.issues)
                    continue
                pov = validate_pov([narr])
                if not pov.ok:
                    attempt += 1
                    logger.info("scene %d retry (POV): %s", i, pov.issues)
                    continue
                break
            if not narr or _is_placeholder(narr):
                narr = _clean_speech(beat)
                vis = ""
            visual = vis or ""
            if not visual or _is_placeholder(visual):
                # small models often skip the VISUAL line; bind the visual to
                # the concept vocabulary the asset agent matches on
                found = extract_concepts_from_text(narr)
                visual = found[0] if found else beat
            if len(visual.split()) > 5:
                visual = _visual_keywords(visual)
            if not visual or _is_placeholder(visual) or _same_text(narr, visual):
                visual = _visual_keywords(narr)
            secs = max(3, int(len(narr.split()) / 2.5))
            scenes.append(ScriptScene(narration=narr, visual=visual, seconds=secs))
            scene_beats.append(beat)
            narrations.append(narr)

        # POV check — repair first-person slips mechanically
        all_narrs = [hook_narr] + [s.narration for s in scenes] + [payoff_narr]
        pov_res = validate_pov(all_narrs)
        if not pov_res.ok:
            logger.info("POV drift: %s — repairing", pov_res.issues)
            hook_narr = _fix_pov(hook_narr)
            payoff_narr = _fix_pov(payoff_narr)
            for scene in scenes:
                scene.narration = _fix_pov(scene.narration)

        # coherence polish: one LLM pass rewriting all lines as natural
        # spoken prose (fixes telegraph fragments and scene-to-scene seams;
        # falls back to the original lines on any validation failure)
        raw_lines = [hook_narr] + [s.narration for s in scenes] + [payoff_narr]
        hook_before_polish = hook_narr
        polished = polish_script_lines(self._llm, raw_lines, context_note=context_note)
        if polished != raw_lines:
            logger.info(
                "polish pass rewrote %d/%d lines",
                sum(1 for a, b in zip(raw_lines, polished) if a != b),
                len(raw_lines),
            )
        # the hook is the attention asset: if polish turned it into a
        # fragment/placeholder, restore the pre-polish line
        if not _hook_ok(polished[0]) and _hook_ok(hook_before_polish):
            logger.info("polish mangled the hook; restoring original")
            polished[0] = hook_before_polish
        hook_narr, *polished_scenes, payoff_narr = polished
        for scene, beat, narr in zip(scenes, scene_beats, polished_scenes):
            # a line that is STILL a fragment or placeholder after polish
            # gets replaced by its beat text (a complete planned thought)
            # rather than shipped as cut-off narration
            if _is_placeholder(narr) or _is_incomplete(narr) or len(narr.split()) < 5:
                fixed = _clean_speech(beat)
                if fixed and not _is_placeholder(fixed):
                    logger.info("polish left garbage %r -> beat text", narr)
                    narr = fixed
            scene.narration = narr
        # same guard for hook and outro
        if _is_placeholder(hook_narr) or len(hook_narr.split()) < 5:
            fixed = _clean_speech(hook_beat)
            if fixed and not _is_placeholder(fixed):
                logger.info("polish left hook garbage -> beat text")
                hook_narr = fixed
        if _is_placeholder(payoff_narr) or len(payoff_narr.split()) < 5:
            fixed = _clean_speech(payoff_beat)
            if fixed and not _is_placeholder(fixed):
                logger.info("polish left outro garbage -> beat text")
                payoff_narr = fixed

        # claims check
        full_text = " ".join([hook_narr] + [s.narration for s in scenes] + [payoff_narr])
        claim_issues = check_claims(full_text, "\n".join(fs.facts) + " " + fs.source)
        if claim_issues:
            logger.warning("claims: %s", claim_issues)
        if source_chunks:
            score = grounding_score(full_text, " ".join(source_chunks))
            logger.info("grounding score: %.2f (content words traced to source)", score)
            if score < 0.3:
                logger.warning(
                    "weak grounding (%.2f): script may drift off the book text", score
                )

        return VideoScript(
            title=fs.concept,
            hook=hook_narr,
            format=format,
            target_seconds=self._target_seconds,
            scenes=scenes,
            outro=payoff_narr,
        )
