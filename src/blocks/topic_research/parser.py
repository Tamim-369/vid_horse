"""Parse the model's labelled plain-text research brief into the domain model.

Like the script parser, this is intentionally forgiving: small models drift in
small ways, so we accept reasonable label variation, clean artifacts, and raise
:class:`BriefParseError` only when the result would be unusable — which lets the
block ask the model to retry.
"""

from __future__ import annotations

from pydantic import ValidationError

from blocks.textparse import clean, lines, split_label, split_timeline
from core.errors import BlockError
from core.models import ResearchBrief, TimelineEvent

_LABELS = {
    "TITLE",
    "ONE-LINE",
    "ONELINE",
    "LOGLINE",
    "ONE-LINER",
    "ORIGIN",
    "BACKGROUND",
    "CONTEXT",
    "HISTORY",
    "CLAIM",
    "CLAIMS",
    "COMPELLING",
    "WHY-COMPELLING",
    "APPEAL",
    "PERSON",
    "PEOPLE",
    "KEY-PERSON",
    "KEY-PEOPLE",
    "TERM",
    "TERMS",
    "KEY-TERM",
    "KEY-TERMS",
    "TIMELINE",
    "EVENT",
    "QUESTION",
    "QUESTIONS",
    "OPEN-QUESTION",
    "OPEN-QUESTIONS",
    "OUTRO",
}


class BriefParseError(BlockError):
    """Raised when model output cannot be parsed into a research brief."""


def parse_brief(text: str, *, fallback_title: str) -> ResearchBrief:
    """Parse labelled plain text into a :class:`ResearchBrief`."""

    title = fallback_title
    one_line = origin = background = ""
    claims: list[str] = []
    compelling: list[str] = []
    people: list[str] = []
    terms: list[str] = []
    questions: list[str] = []
    timeline: list[TimelineEvent] = []
    section = ""

    for raw in lines(text):
        line = raw.strip()
        if not line:
            continue

        key, rest = split_label(line)
        if key is not None and key in _LABELS:
            if key == "TITLE":
                title = clean(rest) or title
                section = ""
            elif key in ("ONE-LINE", "ONELINE", "LOGLINE", "ONE-LINER"):
                one_line = clean(rest)
                section = "one_line"
            elif key == "ORIGIN":
                origin = clean(rest)
                section = "origin"
            elif key in ("BACKGROUND", "CONTEXT", "HISTORY"):
                background = clean(rest)
                section = "background"
            elif key in ("CLAIM", "CLAIMS"):
                claims.append(clean(rest))
                section = "claims"
            elif key in ("COMPELLING", "WHY-COMPELLING", "APPEAL"):
                compelling.append(clean(rest))
                section = "compelling"
            elif key in ("PERSON", "PEOPLE", "KEY-PERSON", "KEY-PEOPLE"):
                people.append(clean(rest))
                section = "people"
            elif key in ("TERM", "TERMS", "KEY-TERM", "KEY-TERMS"):
                terms.append(clean(rest))
                section = "terms"
            elif key in ("TIMELINE", "EVENT"):
                timeline.append(_timeline(rest))
                section = "timeline"
            elif key in ("QUESTION", "QUESTIONS", "OPEN-QUESTION", "OPEN-QUESTIONS"):
                questions.append(clean(rest))
                section = "questions"
            else:  # OUTRO and other ignorable labels
                section = ""
            continue

        continuation = clean(line)
        if not continuation:
            continue
        if section == "one_line":
            one_line = clean(f"{one_line} {continuation}")
        elif section == "origin":
            origin = clean(f"{origin} {continuation}")
        elif section == "background":
            background = clean(f"{background} {continuation}")
        elif section == "claims" and claims:
            claims[-1] = clean(f"{claims[-1]} {continuation}")
        elif section == "compelling" and compelling:
            compelling[-1] = clean(f"{compelling[-1]} {continuation}")
        elif section == "people" and people:
            people[-1] = clean(f"{people[-1]} {continuation}")
        elif section == "terms" and terms:
            terms[-1] = clean(f"{terms[-1]} {continuation}")
        elif section == "questions" and questions:
            questions[-1] = clean(f"{questions[-1]} {continuation}")
        elif section == "timeline" and timeline:
            last = timeline[-1]
            timeline[-1] = last.model_copy(
                update={"event": clean(f"{last.event} {continuation}")}
            )

    if not one_line:
        raise BriefParseError("brief is missing ONE-LINE text")
    if not claims:
        raise BriefParseError("brief has no CLAIM lines")
    if not compelling:
        raise BriefParseError("brief has no COMPELLING lines")

    try:
        return ResearchBrief(
            title=title or fallback_title,
            one_line=one_line,
            origin=origin or background or one_line,
            background=background or origin or one_line,
            claims=claims,
            why_compelling=compelling,
            timeline=timeline,
            key_people=people,
            key_terms=terms,
            open_questions=questions,
        )
    except ValidationError as exc:
        raise BriefParseError(str(exc)) from exc


def _timeline(text: str) -> TimelineEvent:
    when, event = split_timeline(clean(text))
    return TimelineEvent(when=when or "undated", event=event or clean(text))
