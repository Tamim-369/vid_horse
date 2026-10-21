"""Spoken-text normalization for TTS.

Chatterbox reads raw digits poorly ("52" -> garble), so text destined for the
voice model is pre-expanded: numbers become words, years read as years
("1973" -> "nineteen seventy-three"), and common symbols are spoken.
"""

from __future__ import annotations

import re

from num2words import num2words

_CARDINAL = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _year_words(year: int) -> str:
    """Read a 4-digit year the way people say it aloud."""

    century, rest = divmod(year, 100)
    if rest == 0:
        return f"{num2words(century)} hundred"
    if rest < 10:
        return f"{num2words(century)} oh {num2words(rest)}"
    return f"{num2words(century)} {num2words(rest)}"


def _expand(match: re.Match) -> str:
    text = match.group(0).replace(",", "")
    number = int(text)
    if 1000 <= number <= 2099 and _YEAR.match(text):
        return _year_words(number)
    return num2words(number)


_SYMBOLS = (
    ("%", " percent"),
    ("&", " and "),
    ("+", " plus "),
    ("$", " dollars "),
)


def speakable(text: str) -> str:
    """Expand numbers, years and symbols into fully spoken English words."""

    text = text.replace("‑", "-")
    for symbol, spoken in _SYMBOLS:
        text = text.replace(symbol, spoken)
    text = _CARDINAL.sub(_expand, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def callsign(text: str) -> str:
    """Read digit groups individually, e.g. "Baron 52" -> "Baron five two"."""

    def read_digits(match: re.Match) -> str:
        return " ".join(num2words(int(d)) for d in match.group(0))

    return re.sub(r"\b\d\b|\b\d{2}\b", read_digits, text)
