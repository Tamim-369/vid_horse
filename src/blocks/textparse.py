"""Shared helpers for parsing labelled, model-generated plain text.

Both the research-brief and script parsers consume the same shape of output:
colon-labelled lines plus continuation lines. These helpers keep label
detection, artifact cleanup and whitespace handling in one place.
"""

from __future__ import annotations

import re

_BRACKET = re.compile(r"^\[\s*(.*?)\s*\]\s*$", re.DOTALL)
_LABEL = re.compile(r"^\[\s*scene\s*\d*\s*[:\-]?\s*\]\s*", re.IGNORECASE)
_INNER_LABEL = re.compile(r"^\s*scene\s*\d*\s*[:\-]\s*", re.IGNORECASE)
_TIMELINE_SPLIT = re.compile(r"\s+[—–-]\s+")
_LABEL_HEAD = re.compile(
    r"^[\s#>•*`\-]*(?P<label>[A-Za-z][A-Za-z0-9]*(?:[\s\-][A-Za-z0-9]+)?)[\s*_`]*:?\s*"
)
_MARKDOWN = "*_`# "
_WRAPPING = "\"'“”‘’"
_DOUBLED = re.compile(r"\b([A-Za-z][A-Za-z'’]*)(?:\s*-\s*|\s+)\1\b", re.IGNORECASE)


def lines(text: str) -> list[str]:
    """Split ``text`` into lines, normalizing line endings."""

    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def label_of(line: str) -> str:
    """Return the normalized uppercase label at the start of ``line``, if any.

    Leading markdown (``#``, ``**``, bullets) and trailing emphasis before the
    colon are tolerated, so ``**ONE-LINE:** x`` and ``### ONE LINE: x`` both
    yield ``"ONE-LINE"``.
    """

    match = _LABEL_HEAD.match(line)
    if not match:
        return ""
    return match.group("label").upper().replace(" ", "-")


def split_label(line: str) -> tuple[str | None, str]:
    """Return ``(label, value)`` for a labelled line, or ``(None, line)``."""

    key = label_of(line)
    if not key:
        return None, line
    position = line.find(":")
    rest = line[position + 1 :] if position != -1 else ""
    return key, rest.strip()


def value(line: str) -> str:
    """Return the text after the first colon, or the whole line."""

    position = line.find(":")
    return (line[position + 1 :] if position != -1 else line).strip()


def clean(text: str) -> str:
    """Strip scene labels, markdown, collapsing whitespace and wrapping quotes."""

    text = _strip_label(text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    text = _DOUBLED.sub(r"\1", text)
    text = text.strip(_WRAPPING).strip()
    return text.strip(_MARKDOWN).strip()


def first_sentence(text: str) -> str:
    """Return the first sentence of ``text``."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text


def split_timeline(text: str) -> tuple[str, str]:
    """Split ``"1947 — debris recovered"`` into ``("1947", "debris recovered")``."""

    parts = _TIMELINE_SPLIT.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    if ":" in text:
        when, _, event = text.partition(":")
        return when.strip(), event.strip()
    return "", text.strip()


def _strip_label(text: str) -> str:
    bracket = _BRACKET.match(text)
    if bracket:
        return _INNER_LABEL.sub("", bracket.group(1).strip())
    return _LABEL.sub("", text)
