"""Spoken-text normalization for TTS.

Chatterbox reads raw digits poorly ("52" -> garble), so text destined for the
voice model is pre-expanded: numbers become words, years read flat
("2022" -> "two thousand twenty two"), designators speak correctly
("F-35" -> "eff thirty five", "MiG-29" -> "mick twenty nine"), molecules get
their common names ("CO2" -> "carbon dioxide"), and common symbols are
spoken.
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
_SUB = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}

# Single-letter airframe designators (F-35 "eff thirty five"). The letter is
# a series name spoken as its letter sound; the number reads as plain words,
# with three-digit tails read as tens ("C-130" -> "see one thirty", never
# "one hundred thirty"). The (?=s?\b) tail lets plurals like "F-35s" match.
_JET_LETTER = {
    "A": "ay", "B": "bee", "C": "see", "E": "ee", "F": "eff",
    "N": "en", "P": "pee", "R": "arr", "S": "ess", "T": "tee",
    "U": "you", "V": "vee", "X": "ex",
}
_JET = re.compile(r"\b(F(?:/A)?|A|B|C|E|N|P|R|S|T|U|V|X)-(\d{1,3})(?=s?\b)")

# Word designators with their own spoken forms (MiG-29 -> "mick twenty nine").
_JET_WORD = re.compile(r"\b(MiG|Mig|MIG|SU|Su|TU|Tu)\s?-?(\d{1,3})(?=s?\b)")
_JET_SPOKEN = {"mig": "mick", "su": "soo", "tu": "too"}

# Named launch vehicles hyphenated to their flight number ("Falcon-9").
_LV_DASH = re.compile(r"\b(Falcon)-(\d+)\b")

# Molecules with spoken names; matching happens only on digit-bearing
# formulas so unrelated words like "COBE" are never touched.
_FORMULA = {
    "H2O2": "hydrogen peroxide",
    "H2O": "aitch two oh",
    "CO2": "carbon dioxide",
    "CO": "carbon monoxide",
    "O2": "oxygen",
    "O3": "ozone",
    "N2": "nitrogen",
    "NH3": "ammonia",
    "CH4": "methane",
    "SO2": "sulfur dioxide",
    "NO2": "nitrogen dioxide",
}
_FORMULA_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _FORMULA) + r")\b"
)

# Other digit formulas read letter by letter ("H2 -> aitch two",
# "O2" sits in _FORMULA above, "H2O" reads "aitch two oh").
_LETTER_SOUND = {"H": "aitch", "O": "oh", "N": "en", "S": "ess", "U": "you"}
_RAW_FORMULA_RE = re.compile(r"\b([HONSU][a-z]?)(\d+)\b")

_SYMBOLS = (
    ("%", " percent"),
    ("&", " and "),
    ("+", " plus "),
    ("$", " dollars "),
    ("km", " kilometers"),
    ("AU", " astronomical units"),
)


def _exponent_words(match: re.Match) -> str:
    """Read superscript exponents aloud: 10⁴⁶ -> ten to the forty-six."""

    base = int(match.group(1))
    digits = "".join(_SUPER.get(char, "") for char in match.group(2))
    if not digits:
        return match.group(0)
    return f"{num2words(base)} to the {num2words(int(digits))}"


def _jet_words(match: re.Match) -> str:
    """Read an airframe designator aloud: F-35 -> eff thirty five."""

    letter, number = match.group(1), match.group(2)
    if "/" in letter:
        head, tail = letter.split("/")
        return (
            f"{_JET_LETTER[head]} and {_JET_LETTER.get(tail, tail.lower())}"
            f" {num2words(int(number))}"
        )
    if len(number) == 3:
        num = f"{num2words(int(number[0]))} {num2words(int(number[1:]))}"
    else:
        num = num2words(int(number))
    return f"{_JET_LETTER[letter]} {num}"


def _jet_word(match: re.Match) -> str:
    word = _JET_SPOKEN.get(match.group(1).lower(), match.group(1).lower())
    return f"{word} {num2words(int(match.group(2)))}"


def _year_words(year: int) -> str:
    """Read a 4-digit year the way people say it aloud.

    Last-century years read as century pairs ("nineteen seventy three");
    everything from 2000 up reads flat ("two thousand twenty two") — always
    plain space-separated words, the form chatterbox pronounces reliably.
    """

    if year >= 2000:
        if year == 2000:
            return "two thousand"
        return f"two thousand {num2words(year - 2000)}"
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


def _raw_formula(match: re.Match) -> str:
    """Letter-and-digit formulas not in _FORMULA read out letter sounds."""

    head, digits = match.group(1), match.group(2)
    head = head.upper()
    head_sound = _LETTER_SOUND.get(head)
    if head_sound is None:
        return match.group(0)
    speech = " ".join(num2words(int(d)) for d in digits)
    return f"{head_sound} {speech}"


def speakable(text: str) -> str:
    """Expand numbers, years, symbols and designators into spoken words."""

    text = text.replace("‑", "-")
    text = re.sub(r"[₀-₉]", lambda m: _SUB.get(m.group(0), m.group(0)), text)
    for symbol, spoken in _SYMBOLS:
        text = text.replace(symbol, spoken)
    text = _EXPONENT.sub(_exponent_words, text)
    text = _DECIMAL.sub(_decimal_words, text)
    text = _FORMULA_RE.sub(
        lambda m: _FORMULA[m.group(0)], text
    )
    text = _JET_WORD.sub(_jet_word, text)
    text = _JET.sub(_jet_words, text)
    text = _LV_DASH.sub(r"\1 \2", text)
    text = _RAW_FORMULA_RE.sub(_raw_formula, text)
    text = _CARDINAL.sub(_expand, text)
    text = re.sub(r"\s+", " ", text).strip()
    # Years read as fully separate words: "two thousand twenty two", never
    # hyphenated, since chained hyphen tokens garble the voice model.
    text = text.replace("twenty-two", "twenty two").replace("thirty-two", "thirty two")
    return text
