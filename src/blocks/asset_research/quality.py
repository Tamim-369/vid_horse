"""Mechanical quality validation for the visual director's output."""

from __future__ import annotations

import logging
import re

from core.models import SceneIntents, SceneQueries, SceneQuery, Subject, VideoScript

logger = logging.getLogger(__name__)


_TEMPERATURE = 0.4
_JSON_JUNK = re.compile("[\\{\\}\\[\\]\"“”`]")


def _wordish(value: str | None) -> bool:
    return bool(value) and not _JSON_JUNK.search(value)


def _scrub_debris(value: str) -> str:
    return re.sub("\\s+", " ", _JSON_JUNK.sub(" ", value)).strip()


# only function words are filtered generically; semantic judgement of a
# subject (filler vs depictable thing) is the agent's job, enforced by the
# system prompt, not by banned-word lists
_STOPWORD_WORDS = (
    "a an the of in on at to for from by with his her its their this that "
    "these those is are was were be been it it's as and or but not no yet so "
    "then there here"
)
_STOPWORDS_GATE = frozenset(_STOPWORD_WORDS.split())


def _content_words(text: str, *, min_len: int = 4) -> set[str]:
    return {
        w for w in re.findall("[a-z0-9']+", text.lower())
        if len(w) >= min_len and w not in _STOPWORDS_GATE
    }


def _grounded(fragment: str, scene_words: set[str]) -> bool:
    """True when ``fragment`` carries content words, not pure filler."""

    return bool(_content_words(fragment) & scene_words)


def _share_word(candidate: str, scene_words: set[str]) -> bool:
    """True when ``candidate`` reuses at least one scene word verbatim."""

    return bool(_content_words(candidate) & scene_words)


def _words_of(text: str) -> list[str]:
    return re.findall("[a-z0-9']+", text.lower())


def _repair_triggers(
    scenes: list[SceneIntents], script: VideoScript
) -> list[SceneIntents]:
    """Deterministic last resort after the race exhausts.

    Subjects lacking a usable trigger get the first narration words
    verbatim, so a pushed-through answer never re-creates the empty
    ``trigger=""`` fields that make assets air off-time; intents with
    leftover debris are scrubbed of the offending characters only.
    """

    lines = {-1: script.hook or ""}
    for position, scene in enumerate(script.scenes):
        lines[position] = scene.narration
    repaired: list[SceneIntents] = []
    for scene in scenes:
        line = lines.get(scene.scene_index, "")
        words = line.split()
        fallback = " ".join(words[:3])
        words_of_line = set(re.findall("[a-z0-9']+", line.lower()))
        subjects: list[Subject] = []
        for subject in scene.subjects:
            trigger = subject.trigger
            if not _wordish(trigger) or (
                trigger.lower().split() and not set(_words_of(trigger)) <= words_of_line
            ):
                # not verbatim from this scene: air-timing anchors must
                # exist inside the narration, or the image lands off-beat
                trigger = fallback
            subjects.append(
                subject.model_copy(
                    update={
                        "trigger": trigger,
                        "intent": _scrub_debris(subject.intent),
                    }
                )
            )
        repaired.append(scene.model_copy(update={"subjects": subjects}))
    return repaired


def _repair_queries(scenes: list[SceneQueries]) -> list[SceneQueries]:
    """Deterministic last resort: scrub punctuation/JSON debris from queries."""

    repaired: list[SceneQueries] = []
    for scene in scenes:
        queries: list[SceneQuery] = []
        for item in scene.queries:
            queries.append(
                item.model_copy(
                    update={
                        "query": re.sub("\\s+", " ", _JSON_JUNK.sub(" ", item.query)).strip(),
                    }
                )
            )
        repaired.append(scene.model_copy(update={"queries": queries}))
    return repaired


def _intent_problems(
    scenes: list[SceneIntents], narrations: dict[int, str] | None = None
) -> list[str]:
    """Content-quality rules the director must hold; [] means clean.

    ``narrations`` adds grounding beyond word hygiene: every subject and
    trigger must belong to the narration it is assigned to — a subject with
    no scene word, or a trigger quoting a neighbouring scene, produces
    searches that manufacture off-topic images.
    """

    global_words: set[str] = set()
    if narrations:
        global_words = _content_words(" ".join(narrations.values()))
    problems: list[str] = []
    for scene in scenes:
        tag = f"scene {scene.scene_index}"
        scene_text = ""
        if narrations is not None:
            scene_text = narrations.get(scene.scene_index, "")
        scene_words = _content_words(scene_text)
        if not scene.subjects:
            problems.append(f"{tag}: no subjects were given")
            continue
        for subject in scene.subjects:
            name = f"{tag} subject {subject.subject[:30]!r}"
            if len(subject.subject.split()) > 6:
                problems.append(f"{name}: subject name is over 6 words")
            if narrations is not None and not _grounded(
                subject.subject, scene_words
            ) and not _grounded(subject.subject, global_words):
                problems.append(
                    f"{name}: narrative filler that depicts nothing; name a"
                    " concrete thing this video talks about (proper noun,"
                    " object, place)"
                )
                problems.append(
                    f"{name}: subject is narrative filler or not grounded in"
                    " the scene narration; name a concrete thing the scene"
                    " literally talks about (proper noun, object, place)"
                )
            if not _wordish(subject.intent):
                problems.append(
                    f"{name}: intent has JSON debris/forbidden words; rewrite it as"
                    " a clean 3-9-word image description"
                )
            elif len(subject.intent.split()) > 12:
                problems.append(f"{name}: intent is over 12 words")
            if subject.trigger and (
                _JSON_JUNK.search(subject.trigger)
                or len(subject.trigger.split()) > 6
            ):
                problems.append(
                    f"{name}: trigger must be <= 6 words copied from the"
                    " narration, plain text"
                )
            elif (
                subject.trigger
                and scene_text
                and not set(_words_of(subject.trigger))
                <= set(_words_of(scene_text))
            ):
                problems.append(
                    f"{name}: trigger {subject.trigger[:30]!r} is not part of"
                    " this scene's narration; copy the exact words this scene"
                    " speaks when the image should air"
                )
    return problems


def _scene_count_problems(got: int, expected: int) -> list[str]:
    """Whole-document shape check; [] means the right number of scenes."""

    if got == expected:
        return []
    return [f"expected {expected} scenes in the answer, got {got}"]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _scene_names(text: str) -> list[str]:
    """Multi-word capitalized names spoken in this narration+visual text.

    "Vladimir Ilyushin pilot" -> {"Vladimir Ilyushin"}; sentence-initial
    capitalization alone never counts, so "According to the" yields no
    name. Two consecutive capitalized words in the same sentence.
    """

    names: list[str] = []
    for sentence in _SENT_SPLIT.split(text or ""):
        tokens = sentence.split()
        count = len(tokens)
        for i in range(count - 1):
            a, b = tokens[i], tokens[i + 1]
            clean_a, clean_b = a.strip(".,;:!?"), b.strip(".,;:!?")
            if (
                clean_a[:1].isupper() and clean_a.isalpha()
                and clean_b[:1].isupper() and clean_b.isalpha()
            ):
                name = f"{clean_a} {clean_b}"
                if not _STOPWORDS_GATE & {clean_a.lower(), clean_b.lower()}:
                    names.append(name)
    return names


def _query_problems(
    scenes: list[SceneQueries], narrations: dict[int, str] | None = None
) -> list[str]:
    """Content-quality rules the router must hold; [] means clean.

    With ``narrations`` present every query must reuse a content word of
    the scene it routes — keyword fragments like "the theory claims" or
    hallucinated subjects must never drive a search.
    """

    global_words: set[str] = set()
    if narrations:
        global_words = _content_words(" ".join(narrations.values()))
    problems: list[str] = []
    for scene in scenes:
        tag = f"scene {scene.scene_index}"
        scene_text = ""
        if narrations is not None:
            scene_text = narrations.get(scene.scene_index, "")
        scene_words = _content_words(scene_text)
        queries_text = " ".join(item.query for item in scene.queries).lower()
        for name in _scene_names(scene_text):
            hit = next(
                (part for part in name.lower().split() if part in queries_text),
                None,
            )
            if narrations is not None and hit is None:
                problems.append(
                    f"{tag}: the narration names the proper noun {name!r}"
                    " but no query searches for it; include a query built"
                    " around that name (a bare name is the strongest search)"
                )
        for item in scene.queries:
            words = item.query.split()
            tag2 = f"{tag} query {item.query[:30]!r}"
            if item.source not in ("commons", "nasa", "openverse"):
                problems.append(f"{tag2}: unknown source {item.source}")
            if not words:
                problems.append(f"{tag2}: empty query")
                continue
            if len(words) > 6:
                problems.append(f"{tag2}: over 6 keywords")
            if _JSON_JUNK.search(item.query):
                problems.append(f"{tag2}: query has punctuation or JSON debris")
            if narrations is not None and not _share_word(
                item.query, scene_words
            ) and not _share_word(item.query, global_words):
                problems.append(
                    f"{tag2}: query shares no content word with this scene's"
                    " narration; search for what the scene literally"
                    " describes (proper nouns first: names, places, ships)"
                )
    return problems
