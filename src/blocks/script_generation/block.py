"""Script generation: turn a research brief into a video script.

Scripts are drafted as labelled plain text (see :mod:`prompts`) and parsed
into the domain model. Plain text avoids the JSON-grammar corruption that
small local models exhibit inside long string fields.

Short scripts are produced in one call. Long scripts are produced in chunks:
small models under-deliver on a global word budget, so we plan the beats first
and then narrate a few beats per call, which reliably reaches the target length.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator

from blocks.base import Block, generate_parsed
from blocks.script_generation.parser import (
    hook_problems,
    parse_beats,
    parse_hook_scores,
    parse_hooks,
    parse_plan,
    parse_scenes,
    parse_script,
)
from blocks.script_generation.prompts import (
    build_chunk_messages,
    build_extension_messages,
    build_hook_messages,
    build_hook_score_messages,
    build_messages,
    build_plan_messages,
    corrective_instruction,
)
from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from core.models import ResearchBrief, ScriptFormat, ScriptPlan, ScriptScene, VideoScript

logger = logging.getLogger(__name__)


class ScriptGenerationBlock(Block):
    """Produce a scene-by-scene :class:`VideoScript` from a brief."""

    name = "script_generation"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        temperature: float = 0.6,
        max_attempts: int = 2,
        race: int = 0,
        words_per_second: float = 2.5,
        chunk_size: int = 3,
        scene_count: int = 10,
        hook_candidates: int = 1,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._max_attempts = max(1, max_attempts)
        self._race = max(0, race)
        self._words_per_second = words_per_second
        self._chunk_size = max(1, chunk_size)
        self._scene_count = max(3, scene_count)
        self._hook_candidates = max(1, hook_candidates)

    def run(
        self,
        brief: ResearchBrief,
        *,
        format: ScriptFormat = "short",
        target_seconds: int | None = None,
    ) -> VideoScript:
        """Write a script for ``brief`` in the requested ``format``."""

        seconds = target_seconds or _default_seconds(format)
        if format == "short":
            return self._run_short(brief, seconds)
        return self._run_long(brief, seconds)

    # -- short --------------------------------------------------------------
    def _run_short(self, brief: ResearchBrief, seconds: int) -> VideoScript:
        hook = self._choose_hook(brief)

        def parse(raw: str) -> VideoScript:
            return parse_script(
                raw,
                format="short",
                target_seconds=seconds,
                fallback_title=brief.title,
                words_per_second=self._words_per_second,
            )

        script = self._generate(build_messages(brief, "short", seconds, hook=hook), parse)
        return script.model_copy(update={"hook": hook}) if hook else script

    # -- long ---------------------------------------------------------------
    def _run_long(self, brief: ResearchBrief, seconds: int) -> VideoScript:
        plan = self._plan(brief, self._choose_hook(brief))
        beats = list(plan.beats)

        missing = self._scene_count - len(beats)
        if missing > 0:
            logger.info("Outline short by %d beat(s); extending", missing)
            beats.extend(
                self._generate(
                    build_extension_messages(plan, missing),
                    lambda raw: parse_beats(raw),
                )
            )

        scenes: list[ScriptScene] = []
        for start, chunk in _chunks(beats, self._chunk_size):
            messages = build_chunk_messages(plan, chunk, start)
            scenes.extend(
                self._generate(
                    messages,
                    lambda raw: parse_scenes(
                        raw, words_per_second=self._words_per_second
                    ),
                )
            )

        return VideoScript(
            title=plan.title,
            hook=plan.hook,
            format="long",
            target_seconds=seconds,
            scenes=scenes,
            outro=plan.outro,
        )

    def _plan(self, brief: ResearchBrief, hook: str | None = None) -> ScriptPlan:
        plan = self._generate(
            build_plan_messages(brief, self._scene_count, hook=hook),
            lambda raw: parse_plan(raw, fallback_title=brief.title),
        )
        return plan.model_copy(update={"hook": hook}) if hook else plan

    # -- hook selection -----------------------------------------------------
    def _choose_hook(self, brief: ResearchBrief) -> str | None:
        """Draft several hooks and keep the strongest clean one.

        Returns ``None`` when only one candidate is configured or selection
        fails, in which case the model writes the hook inline as before.
        """

        if self._hook_candidates <= 1:
            return None

        started = time.monotonic()
        try:
            raw = self._llm.complete(
                build_hook_messages(brief, self._hook_candidates),
                temperature=self._temperature,
            )
            candidates = parse_hooks(raw)
            logger.info(
                "hook candidates: %.1fs (%d)",
                time.monotonic() - started,
                len(candidates),
            )
        except (LLMError, BlockError) as exc:
            logger.warning("Hook candidates failed (%s); writing hook inline", exc)
            return None

        cleaned = [hook for hook in candidates if not hook_problems(hook)]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]

        started = time.monotonic()
        try:
            raw = self._llm.complete(
                build_hook_score_messages(cleaned), temperature=0.0
            )
            scored = parse_hook_scores(raw)
            logger.info("hook scoring: %.1fs", time.monotonic() - started)
        except (LLMError, BlockError) as exc:
            logger.warning("Hook scoring failed (%s); using first candidate", exc)
            return cleaned[0]

        return _pick_best(scored, cleaned) or cleaned[0]

    # -- shared retry loop --------------------------------------------------
    def _generate[T](self, messages: list[LLMMessage], parse: Callable[[str], T]) -> T:
        return generate_parsed(
            self._llm,
            messages,
            parse,
            attempts=self._max_attempts,
            temperature=self._temperature,
            corrective=corrective_instruction,
            label="script",
            race=self._race,
        )


def _pick_best(scored: list[tuple[int, str]], candidates: list[str]) -> str | None:
    """Return the highest-scored hook that matches a known candidate."""

    allowed = {_normalize(hook): hook for hook in candidates}
    for _, hook in scored:
        match = allowed.get(_normalize(hook))
        if match is not None:
            return match
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _chunks(beats: list[str], size: int) -> Iterator[tuple[int, list[str]]]:
    for offset in range(0, len(beats), size):
        yield offset + 1, beats[offset : offset + size]


def _default_seconds(fmt: ScriptFormat) -> int:
    return 45 if fmt == "short" else 600
