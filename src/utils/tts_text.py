"""Spoken-text normalization for TTS.

Chatterbox reads raw digits poorly ("52" -> garble), so text destined for the
voice model is pre-expanded: numbers become words, years read flat
("2022" -> "two thousand twenty two"), designators speak correctly
("F-35" -> "eff thirty five", "MiG-29" -> "mick twenty nine"), molecules get
their common names ("CO2" -> "carbon dioxide"), and common symbols are
spoken.

Before expansion, text is mechanically sanitized for speech: spacing and
punctuation artifacts that confuse prosody are normalized without changing
the wording or meaning.
"""

from __future__ import annotations

import re

from num2words import num2words


def sanitize_for_speech(text: str) -> str:
    """Clean script punctuation/spacing for clear, unambiguous TTS.

    Meaning-preserving and mechanical only: fixes symbol/spacing artifacts
    that can make the voice pause or group words in confusing ways.
    """

    if not text:
        return ""

    t = str(text)
    # drop unreadable/control characters (keep normal whitespace/newlines)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", t)
    # normalize common unicode punctuation
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u2026", "...")
    t = t.replace("\u00a0", " ")

    # dash separators: a LONE dash between clauses is a sentence break the
    # writer used instead of a period; paired dashes are an aside and stay
    # a pause. Decide on the whole line BEFORE replacing anything.
    t = re.sub(r"\s*(?:--|\u2014|\u2013)\s*|\s-\s", " <DASH> ", t)
    dash_joiner = " . " if t.count(" <DASH> ") == 1 else " , "
    t = t.replace(" <DASH> ", dash_joiner)

    # semicolon joins two complete thoughts: a real sentence break, not a
    # comma pause (a comma splice erases the break the writer intended)
    t = re.sub(r"\s*;\s*", ". ", t)

    # protect decimals so spacing rules don't split them (e.g. 3.14)
    t = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", t)

    # ellipsis / multi-dots: avoid trailing-off prosody
    def _ellipsis(match: re.Match[str]) -> str:
        rest = t[match.end() :].lstrip()
        # terminal or before a sentence-ending mark -> clear stop; else a pause
        if not rest or rest[0] in ".!?":
            return ". "
        return ", "

    t = re.sub(r"\.{2,}", _ellipsis, t)

    # parentheses/brackets: keep aside text but set it off (don't fuse words)
    t = re.sub(r"\(([^)]*)\)", r", \1, ", t)
    t = re.sub(r"\[([^\]]*)\]", r", \1, ", t)
    t = re.sub(r"\{([^}]*)\}", r", \1, ", t)

    # slashes and dash artifacts
    t = re.sub(r"\band/or\b", "and or", t, flags=re.I)
    t = re.sub(r"(?<=\w)/(?=\w)", " ", t)
    t = re.sub(r"\s+--\s+|\s+-\s+", ", ", t)
    t = re.sub(r"--+", ", ", t)

    # collapse repeated/broken punctuation (preserve ? and ! intent)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)
    t = re.sub(r",{2,}", ",", t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r",\s*([.!?])", r"\1", t)
    t = re.sub(r"([.!?])\s*,", r"\1", t)

    # spacing: no space before punctuation, single space after where needed
    t = re.sub(r"\s+([,\.!?;:])", r"\1", t)
    # space after , ! ? ; : when followed by a letter/digit (start of next token)
    t = re.sub(r"([,!?;:])(?=[A-Za-z0-9])", r"\1 ", t)
    # space after period when followed by a letter (decimals already protected)
    t = re.sub(r"\.(?=[A-Za-z])", ". ", t)

    # restore protected decimals
    t = t.replace("<DOT>", ".")

    # tidy artifacts introduced by replacements
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"\s+([,\.!?;:])", r"\1", t)
    # fix broken punctuation combos that may remain after control-char removal
    t = re.sub(r"\.\s*,|,\s*\.", ".", t)
    t = re.sub(r"!\s*,|,\s*!", "!", t)
    t = re.sub(r"\?\s*,|,\s*\?", "?", t)
    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"(?<=\s)[,;:](?=\s|$)", "", t)  # lone separator tokens
    t = re.sub(r"^[\s,;:\-]+", "", t)
    t = re.sub(r"[\s,;:\-]+$", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    # a trailing tag question with no comma ("that's how it goes isn't it?")
    # is rushed straight into the sentence: set the tag off with a comma so
    # the question split below can give it its own breath
    t = re.sub(
        r"(?<!,)\s+(isn't it|isn't that|doesn't it|doesn't that|don't you|"
        r"aren't they|can't you|won't you)\?",
        lambda m: ", " + m.group(1) + "?",
        t,
        flags=re.IGNORECASE,
    )

    # a question tacked onto a statement with a comma ("..., what's pulling
    # you in?") is synthesized as ONE blob: the voice anchors on the opening
    # clause's declarative shape and never gives the question its own rise,
    # so it sounds welded to the sentence. Give the question its own
    # sentence: a breath before it and a full contour to make the rise.
    t = _QUESTION_TAIL.sub(
        lambda m: ". " + m.group(1)[:1].upper() + m.group(1)[1:] + "?", t
    )

    # the same weld when the question is a short "because ...?" punch glued
    # to a long statement ("...without complaint because someone just told
    # you?"): give the punch its own sentence too. Skip "just because" so
    # a bare "just." is never left dangling.
    t = _BECAUSE_TAIL.sub(lambda m: ". Because " + m.group(1) + "?", t)

    # a sentence break starts a sentence: capitalize the word after it so
    # the voice treats it as a new sentence and the subtitle shows it as one
    t = re.sub(r"([.!?])\s+([a-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    return t


_CARDINAL = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b")

# tail of a comma-glued question: the clause after the comma must start
# with a question word/auxiliary and have at least one more word, so list
# commas and asides never trigger a false split
_QUESTION_TAIL = re.compile(
    r",\s+((?:what(?:'s|s)?|why(?:'s|s)?|how|who(?:'s|s)?|when|where|which|"
    r"do(?:es|n't)?|did|is|are|was|were|can|could|would|will|should|has|have|"
    r"isn't|aren't|doesn't|don't|didn't)\b\s+[^.!?]+)\?",
    re.IGNORECASE,
)

# short "because ...?" question punch at the end of a statement (tail kept
# to a handful of words so long subordinate clauses are never split)
_BECAUSE_TAIL = re.compile(r"(?<!just)\s+because\s+([^,.!?]{4,45}?)\s*\?", re.IGNORECASE)
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

    text = sanitize_for_speech(text)
    text = text.replace("‑", "-")
    text = text.replace(":", ",").replace("—", ",")
    # asides are already set off by sanitize_for_speech; drop any leftover parens
    text = text.replace("(", ", ").replace(")", ", ")
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
    # ALL-CAPS shouting (e.g. DEMAND) makes neural TTS over-articulate;
    # normalize to natural case for speech (visual shout is handled in subs).
    text = re.sub(
        r"\b([A-Z][A-Z']{2,})\b",
        lambda m: m.group(1).capitalize() if m.group(1).isalpha() else m.group(1),
        text,
    )
    # word-word hyphen chains garble the voice model ("thirty‑seven" reads
    # as a single broken token); every remaining letter-hyphen-letter bond
    # becomes the space Chatterbox pronounces reliably
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Years read as fully separate words: "two thousand twenty two", never
    # hyphenated, since chained hyphen tokens garble the voice model.
    text = text.replace("twenty-two", "twenty two").replace("thirty-two", "thirty two")
    # final tidy in case normalization introduced double separators
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"\s+([,\.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text
