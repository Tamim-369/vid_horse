"""Script review: independent critic pass + targeted rewrite.

Gives the script a separate, focused quality check before assets/images
are assigned. The critic looks for concrete hooks, simple words, real-life
examples, connected flow, and grounding — and a rewrite pass fixes issues
when the critic rejects the draft.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import ScriptScene, VideoScript

logger = logging.getLogger(__name__)


class ScriptReview(BaseModel):
    ok: bool = Field(description="True only if the script meets the quality bar.")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems to fix (empty when ok).",
    )
    summary: str = Field(default="", description="One short sentence of feedback.")


CRITIC_SYSTEM = (
    "You are a strict script editor for a short dark-psychology video "
    "(one narrator talking to 'you'). Judge the draft against this bar:\n"
    "- HOOK (line 1): a question about a concrete real-life moment that "
    "shows the pattern and asks WHY it happens. It must NOT name the "
    "pattern or ask in the abstract.\n"
    "- SIMPLE WORDS: everyday common words only. No fancy/bookish words, "
    "no copying the book's style.\n"
    "- CONNECTED FLOW: one argument from start to finish. Each line "
    "follows the previous line and adds something new (no repeats, no "
    "random jumps).\n"
    "- REAL-LIFE EXAMPLES: clear, concrete moments from daily life (work, "
    "friends, family, online, crowds) with a calm, quietly dark vibe.\n"
    "- GROUNDED & HONEST: drawn from the source idea, no invented "
    "numbers/statistics, never says 'studies show', no vague 'they "
    "found' without saying who/what in plain words.\n"
    "- FORM: line 1 (and maybe line 2) may be questions; later lines are "
    "statements. Every line is a complete, clear sentence.\n"
    "Be fair but strict. Return ok=true only when the script clearly "
    "meets the bar. If not, list the specific issues to fix. "
    "Output JSON only."
)


REWRITE_SYSTEM = (
    "You rewrite a short dark-psychology video script to fix specific "
    "issues, while keeping it as ONE connected argument. Voice: calm, "
    "confident, quietly dark — like a friend sharing a secret about how "
    "people really act. Use the SIMPLEST everyday words only; never copy "
    "the book's style. Use concrete real-life examples. "
    "Line 1 must be a question about a concrete real-life moment that "
    "shows the pattern and asks WHY it happens (not the pattern's name). "
    "Line 2 may be a question or a statement. Every line after that is a "
    "declarative statement. No invented numbers/statistics, never "
    "'studies show'. Keep the meaning grounded in the source idea. "
    "Output JSON: {\"hook\": \"...\", \"scenes\": [\"...\"], "
    "\"outro\": \"...\"} with the same number of scene lines as given."
)


class _Rewritten(BaseModel):
    hook: str
    scenes: list[str]
    outro: str


def _script_text(script: VideoScript) -> str:
    lines = [f"1 (HOOK): {script.hook}"]
    for i, sc in enumerate(script.scenes, start=2):
        lines.append(f"{i}: {sc.narration}")
    lines.append(f"{len(lines)+1} (OUTRO): {script.outro}")
    return "\n".join(lines)


def review_script(
    llm: LLMProvider,
    script: VideoScript,
    *,
    source_text: str = "",
    model: str | None = None,
) -> ScriptReview:
    """Critique the script; returns a pass/fail review with issues."""
    src = ""
    if source_text:
        src = f"\nSource idea/excerpt:\n'''\n{source_text[:2000]}\n'''\n"
    msgs = [
        LLMMessage(role="system", content=CRITIC_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Draft script:\n{_script_text(script)}\n"
                f"{src}"
                "Review this draft. Output JSON."
            ),
        ),
    ]
    try:
        return llm.complete_json(msgs, ScriptReview, temperature=0.2, model=model)
    except (LLMError, BlockError, Exception) as exc:  # noqa: BLE001
        logger.warning("script review skipped (error): %s", exc)
        return ScriptReview(ok=True, issues=[], summary="review unavailable")


def rewrite_script(
    llm: LLMProvider,
    script: VideoScript,
    issues: list[str],
    *,
    source_text: str = "",
    model: str | None = None,
) -> VideoScript | None:
    """Return a rewritten script addressing ``issues``, or None on failure."""
    src = ""
    if source_text:
        src = f"\nSource idea/excerpt (stay grounded in this):\n'''\n{source_text[:2000]}\n'''\n"
    issue_text = "\n".join(f"- {it}" for it in issues) or "- improve clarity and flow"
    msgs = [
        LLMMessage(role="system", content=REWRITE_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Draft script:\n{_script_text(script)}\n"
                f"{src}"
                f"Issues to fix:\n{issue_text}\n"
                f"Rewrite the script fixing these issues. Keep exactly "
                f"{len(script.scenes)} scene lines in the same order/roles. "
                "Output JSON."
            ),
        ),
    ]
    try:
        out = llm.complete_json(msgs, _Rewritten, temperature=0.4, model=model)
    except (LLMError, BlockError, Exception) as exc:  # noqa: BLE001
        logger.warning("script rewrite skipped (error): %s", exc)
        return None

    if len(out.scenes) != len(script.scenes):
        logger.warning(
            "script rewrite rejected: got %d scenes, expected %d",
            len(out.scenes), len(script.scenes),
        )
        return None

    hook = (out.hook or "").strip()
    outro = (out.outro or "").strip()
    scene_lines = [(s or "").strip() for s in out.scenes]
    if not hook or not outro or any(not s for s in scene_lines):
        logger.warning("script rewrite rejected: empty line(s)")
        return None

    new_scenes = []
    for old, text in zip(script.scenes, scene_lines):
        new_scenes.append(
            ScriptScene(narration=text, visual=old.visual, seconds=old.seconds)
        )
    return VideoScript(
        title=script.title,
        hook=hook,
        format=script.format,
        target_seconds=script.target_seconds,
        scenes=new_scenes,
        outro=outro,
    )


def review_and_improve(
    llm: LLMProvider,
    script: VideoScript,
    *,
    source_text: str = "",
    model: str | None = None,
    max_rounds: int = 1,
) -> VideoScript:
    """Review the script and, if needed, rewrite once (or up to max_rounds).

    Fail-open: on reviewer/rewriter errors the original script is kept so
    the pipeline never breaks.
    """
    current = script
    for rnd in range(max_rounds):
        rev = review_script(llm, current, source_text=source_text, model=model)
        if rev.ok:
            logger.info("script review passed%s", f": {rev.summary}" if rev.summary else "")
            return current
        logger.info(
            "script review round %d: issues=%s", rnd + 1, "; ".join(rev.issues) or rev.summary
        )
        improved = rewrite_script(
            llm, current, rev.issues or [rev.summary or "make it clearer and more connected"],
            source_text=source_text, model=model,
        )
        if improved is None:
            logger.info("script rewrite unavailable; keeping current draft")
            return current
        current = improved
    # final check (informational)
    final = review_script(llm, current, source_text=source_text, model=model)
    if final.ok:
        logger.info("script review passed after rewrite")
    else:
        logger.info("script review still has notes after rewrite: %s", "; ".join(final.issues))
    return current
