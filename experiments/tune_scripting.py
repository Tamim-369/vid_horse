#!/usr/bin/env python3
"""Compare script-drafting strategies for small local models.

The problem: a single call asking gemma3:4b for a nested ``list[Scene]`` object
lets it leak JSON fragments into narration. This experiment drafts the *same*
short script several ways and reports quality signals so we can pick a design.

Usage:
    uv run python experiments/tune_scripting.py
    uv run python experiments/tune_scripting.py --candidate delimited
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from pydantic import Field

from config.settings import get_settings
from core.models import ResearchBrief, StrictModel
from providers.llm import create_llm_provider
from providers.llm.base import BaseLLMProvider

BRIEF_PATH = Path("output/baron-52/brief.json")

SYSTEM = """You write tight, engaging narration for short YouTube videos that \
EXPLAIN conspiracy theories.

Rules:
- Explain the theory and why people find it compelling. Never debunk it and \
never present its claims as true.
- Use distancing language: "the theory claims", "believers say".
- Every field is plain prose. Never put JSON, field names, quotes or braces \
inside a value.
- Narration is spoken aloud: short natural sentences, no headings."""

_LEAK = re.compile(r'"(narration|visual|seconds|scenes|hook|outro)"|\{|\}')


class DelimitedDraft(StrictModel):
    """Strategy B: one string per concern, scenes split by a delimiter line."""

    title: str
    hook: str
    narration: str = Field(
        description="All scenes; separate scenes with a line containing only ###."
    )
    visuals: str = Field(description='One visual per scene, same ### separators, same order.')
    outro: str


class BeatPlan(StrictModel):
    """Strategy C, step 1: a plain list of short beats."""

    title: str
    hook: str
    beats: list[str] = Field(min_length=3, max_length=8)
    outro: str


class SceneDraft(StrictModel):
    """Strategy C, step 2: narrate one beat."""

    narration: str
    visual: str


def _brief() -> ResearchBrief:
    return ResearchBrief.model_validate_json(BRIEF_PATH.read_text())


def _summary(brief: ResearchBrief) -> str:
    claims = "\n".join(f"- {c}" for c in brief.claims)
    return (
        f"Title: {brief.title}\nOne-line: {brief.one_line}\n"
        f"Origin: {brief.origin}\nBackground: {brief.background}\nClaims:\n{claims}"
    )


def run_delimited(llm: BaseLLMProvider, brief: ResearchBrief) -> str:
    user = f"""Write a ~45 second script (~110-130 words) in 4-6 scenes.

{_summary(brief)}

Put ALL narration in the "narration" field, one scene per section, sections
separated by a line containing only ###. Put the matching visuals in "visuals"
with the same ### separators in the same order."""
    draft = llm.complete_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        DelimitedDraft,
        temperature=0.6,
    )
    narration = [s.strip() for s in draft.narration.split("###") if s.strip()]
    visuals = [s.strip() for s in draft.visuals.split("###") if s.strip()]
    out = [f"# {draft.title}", f"HOOK: {draft.hook}"]
    for i, line in enumerate(narration):
        visual = visuals[i] if i < len(visuals) else "(none)"
        out.append(f"\n[{i + 1}] VISUAL: {visual}\n{line}")
    out.append(f"\nOUTRO: {draft.outro}")
    return "\n".join(out)


def run_plaintext(llm: BaseLLMProvider, brief: ResearchBrief) -> str:
    user = f"""Write a ~45 second script (~110-130 words) in 4-6 scenes.

{_summary(brief)}

Reply in EXACTLY this plain-text format, nothing else:

TITLE: <title>
HOOK: <one sentence>
SCENE
VISUAL: <what is on screen>
NARRATION: <spoken words for this scene>
SCENE
VISUAL: ...
NARRATION: ...
OUTRO: <one sentence>"""
    return llm.complete(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.6,
    )


def run_per_scene(llm: BaseLLMProvider, brief: ResearchBrief) -> str:
    plan = llm.complete_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "Plan a ~45 second script in 4-5 scenes. Return a title, a "
                    "hook, one-sentence beats, and an outro.\n\n" + _summary(brief)
                ),
            },
        ],
        BeatPlan,
        temperature=0.6,
    )
    out = [f"# {plan.title}", f"HOOK: {plan.hook}"]
    for i, beat in enumerate(plan.beats, start=1):
        scene = llm.complete_json(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Narrate this single beat in 1-2 spoken sentences "
                        f"(15-25 words): {beat}\nGive one matching visual."
                    ),
                },
            ],
            SceneDraft,
            temperature=0.6,
        )
        out.append(f"\n[{i}] VISUAL: {scene.visual}\n{scene.narration}")
    out.append(f"\nOUTRO: {plan.outro}")
    return "\n".join(out)


CANDIDATES = {
    "delimited": run_delimited,
    "plaintext": run_plaintext,
    "per_scene": run_per_scene,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=[*CANDIDATES, "all"])
    args = parser.parse_args()

    wanted = args.candidate or ["all"]
    names = list(CANDIDATES) if "all" in wanted else wanted

    settings = get_settings()
    llm = create_llm_provider(settings)
    brief = _brief()

    for name in names:
        started = time.monotonic()
        try:
            text = CANDIDATES[name](llm, brief)
        except Exception as exc:  # noqa: BLE001 - experiment harness
            print(f"\n===== {name} FAILED: {exc}\n")
            continue
        elapsed = time.monotonic() - started
        leaks = _LEAK.findall(text)
        print(f"\n===== {name}  ({elapsed:.0f}s, {len(text)} chars, leaks={len(leaks)})")
        print(text)


if __name__ == "__main__":
    main()
