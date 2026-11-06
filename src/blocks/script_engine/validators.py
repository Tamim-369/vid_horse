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


def validate_scene(narration: str, target_words: int, tolerance: int = 2) -> ValidationResult:
    r = ValidationResult()
    wc = word_count(narration)
    if wc < target_words - tolerance or wc > target_words + tolerance:
        r.ok = False
        r.issues.append(f"word count {wc}, need {target_words}±{tolerance}")
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


def check_claims(script_text: str, fact_sheet: str) -> list[str]:
    """Return list of suspicious claims (numbers/names) not in fact sheet."""
    issues = []
    # find numbers
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", script_text)
    for n in nums:
        if n not in fact_sheet:
            issues.append(f"unsourced number: {n}")
    return issues
