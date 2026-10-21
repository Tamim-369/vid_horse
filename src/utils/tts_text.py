"""Spoken-text normalization for TTS.

Chatterbox reads raw digits poorly ("52" -> garble), so text destined for the
voice model is pre-expanded: numbers become words, years read as years
("1973" -> "nineteen seventy-three"), and common symbols are spoken.
"""

from __future__ import annotations

import re

from num2words import num2words

_CARDINAL = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b")
_DECIMAL = re.compile(r"\b\d+\.\d+\b")
_EXPONENT = re.compile(r"(\d+)\s*?([⁰¹²³⁴⁵⁶⁷⁷⁸⁹⁰]+)")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_SUPER = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}


def _exponent_words(match: re.Match) -> str:
    """Read superscript exponents aloud: 10⁴⁶ -> ten to the forty-six."""

    base = int(match.group(1))
    digits = "".join(_SUPER.get(char, "") for char in match.group(2))
    if not digits:
        return match.group(0)
    return f"{num2words(base)} to the {num2words(int(digits))}"


def _year_words(year: int) -> str:
    """Read a 4-digit year the way people say it aloud.

    Last-century years read as century pairs ("nineteen seventy-three");
    2000-2009 read flat ("two thousand five") — "twenty oh five" is not
    how anyone says it.
    """

    if 2000 <= year <= 2009:
        return f"two thousand {num2words(year - 2000)}" if year > 2000 else "two thousand"
    century, rest = divmod(year, 100)
    if rest == 0:
        return f"{num2words(century)} hundred"
    if rest < 10:
        return f"{num2words(century)} oh {num2words(rest)}"
    return f"{num2words(century)} {num2words(rest)}"


def _decimal_words(match: re.Match) -> str:
    whole, fraction = match.group(0).split(".")
    digits = " ".join(num2words(int(d)) for d in fraction if d.isdigit())
    return f"{num2words(int(whole))} point {digits}"


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
    text = _EXPONENT.sub(_exponent_words, text)
    text = _DECIMAL.sub(_decimal_words, text)
    text = _CARDINAL.sub(_expand, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def callsign(text: str) -> str:
    """Read digit groups individually, e.g. "Baron 52" -> "Baron five two"."""

    def read_digits(match: re.Match) -> str:
        return " ".join(num2words(int(d)) for d in match.group(0))

    return re.sub(r"\b\d\b|\b\d{2}\b", read_digits, text)
