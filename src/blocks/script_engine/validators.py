"""Code-level validators for generated scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BANNED_END_WORDS = {
    "watch", "notice", "beware", "stay", "alert", "protect", "silent",
    "steady", "trap", "now", "today",
}

BANNED_PHRASES = [
    "studies show", "research shows", "it's not", "it is not",
]

FILLER_TAGS = re.compile(r"\b(watch|notice|beware|stay alert|protect yourself)\b", re.I)


@dataclass
class ValidationResult:
    ok: bool = True
    issues: list[str] = field(default_factory=list)


def word_count(text: str) -> int:
    return len(text.split())


def validate_scene(
    narration: str,
    target_words: int,
    tolerance: int = 2,
    *,
    allow_question: bool = False,
) -> ValidationResult:
    r = ValidationResult()
    wc = word_count(narration)
    if wc < target_words - tolerance or wc > target_words + tolerance:
        r.ok = False
        r.issues.append(f"word count {wc}, need {target_words}±{tolerance}")
    # question-mark policy: the hook may ask, the line after it may ask
    # once; everything later must be statements (no question bank)
    if not allow_question and "?" in narration:
        r.ok = False
        r.issues.append("statement line contains a question mark")
    elif allow_question and narration.count("?") > 1:
        r.ok = False
        r.issues.append("more than one question mark in this line")
    # banned end word
    last = narration.strip().rstrip(".!?").split()[-1].lower() if narration.strip() else ""
    if last in BANNED_END_WORDS:
        r.ok = False
        r.issues.append(f"ends with banned tag word '{last}'")
    # banned phrases
    low = narration.lower()
    for bp in BANNED_PHRASES:
        if bp in low:
            r.ok = False
            r.issues.append(f"contains banned phrase '{bp}'")
    # em-dash splice check
    if "—" in narration and narration.count("—") > 1:
        r.ok = False
        r.issues.append("multiple em-dash splices")
    return r


def validate_pov(texts: list[str]) -> ValidationResult:
    r = ValidationResult()
    joined = " ".join(texts)
    has_second = bool(re.search(r"\byou(?:r)?\b", joined))
    has_first = bool(re.search(r"\b(?:I|my|me)\b", joined))
    if has_second and has_first:
        r.ok = False
        r.issues.append("POV drift: mixes first and second person")
    return r


def validate_hook_payoff(hook: str, outro: str) -> ValidationResult:
    r = ValidationResult()
    hook_words = set(hook.lower().split())
    outro_words = set(outro.lower().split())
    overlap = hook_words & outro_words
    if len(overlap) < 2:
        r.ok = False
        r.issues.append("payoff does not reference hook")
    return r


def repeats_previous(narration: str, previous: list[str], threshold: float = 0.6) -> bool:
    """True when a line's content words heavily overlap an earlier line.

    The small model loves to ask the same question four times with tiny
    wording changes; this catches that mechanically.
    """

    def content_words(t: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in _STOPWORDS}

    cur = content_words(narration)
    if not cur:
        return False
    for prev in previous:
        pw = content_words(prev)
        if not pw:
            continue
        overlap = len(cur & pw) / min(len(cur), len(pw))
        if overlap >= threshold:
            return True
    return False


def check_claims(script_text: str, fact_sheet: str) -> list[str]:
    """Return list of suspicious claims (numbers/names) not in fact sheet."""
    issues = []
    # find numbers
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", script_text)
    for n in nums:
        if n not in fact_sheet:
            issues.append(f"unsourced number: {n}")
    return issues


_STOPWORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
        "for", "with", "is", "are", "was", "were", "be", "been", "will",
        "that", "this", "it", "its", "as", "by", "so", "not", "no", "from",
        "you", "your", "they", "their", "we", "i", "my", "me", "he", "she",
        "his", "her", "them", "then", "than", "there", "here", "when",
        "what", "how", "why", "who", "which", "would", "can", "could",
        "should", "do", "does", "did", "have", "has", "had", "into", "if",
        "one", "all", "more", "most", "very", "just", "now", "even", "out",
    ]
)


def grounding_score(script_text: str, source_text: str) -> float:
    """Fraction of the script's content words that trace back to the source.

    Rough lexical grounding: 0.0 means the script shares nothing with the
    book chunk (invented wallpaper), ~0.5+ means the script is strongly
    rooted in the source text. Stopwords excluded; both texts lowercased.
    """

    sw = re.findall(r"[a-z']+", (script_text or "").lower())
    src = re.findall(r"[a-z']+", (source_text or "").lower())
    src_set = {w for w in src if w not in _STOPWORDS}
    content = [w for w in sw if w not in _STOPWORDS]
    if not content:
        return 0.0
    hit = sum(1 for w in content if w in src_set)
    return hit / len(content)
