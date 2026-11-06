"""Multi-stage script engine: facts -> beats -> scenes -> validate -> revise."""

from __future__ import annotations

import logging

from blocks.script_engine.beats import generate_beats
from blocks.script_engine.facts import generate_fact_sheet
from blocks.script_engine.validators import (
    check_claims,
    validate_pov,
    validate_scene,
)
from blocks.script_engine.writer import write_scene
from core.interfaces import LLMProvider
from core.models import ScriptFormat, ScriptScene, VideoScript

logger = logging.getLogger(__name__)

BEAT_NAMES = ["hook", "situation", "mechanism", "turn", "action", "payoff"]


def _parse_scene(raw: str) -> tuple[str, str]:
    narr = ""
    vis = ""
    # handle inline "NARRATION: ... VISUAL: ..."
    import re
    m = re.search(r"NARRATION:\s*(.*?)(?:VISUAL:|$)", raw, re.I | re.S)
    if m:
        narr = m.group(1).strip()
    m2 = re.search(r"VIS(?:UAL|ION):\s*(.*)", raw, re.I | re.S)
    if m2:
        vis = m2.group(1).strip()
    # remove any leaked VISUAL/VISION from narration
    narr = re.sub(r"\s*VIS(?:UAL|ION):.*", "", narr, flags=re.I | re.S).strip()
    if not narr:
        narr = raw.strip()
    # clean trailing VISUAL if leaked
    narr = re.sub(r"\s*VISUAL:.*", "", narr, flags=re.I | re.S).strip()
    # strip quote characters for TTS safety
    for q in ('"', "'", "“", "”", "‘", "’"):
        narr = narr.replace(q, "")
    return narr, vis


class ScriptEngineBlock:
    name = "script_engine"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        target_seconds: int = 45,
        max_scene_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._target_seconds = target_seconds
        self._max_scene_retries = max_scene_retries

    def run(self, topic: str, *, format: ScriptFormat = "short") -> VideoScript:
        fs = generate_fact_sheet(self._llm, topic)
        beats = generate_beats(self._llm, fs)

        # words per scene: distribute target seconds
        wps = 2.5
        total_words = int(self._target_seconds * wps)
        per_scene = max(6, total_words // len(beats))

        # Generate payoff first, then hook referencing it.
        payoff_beat = beats[5] if len(beats) >= 6 else beats[-1]
        payoff_raw = write_scene(self._llm, payoff_beat, fs, [], 12)
        payoff_narr, payoff_vis = _parse_scene(payoff_raw)
        if not payoff_narr:
            payoff_narr = payoff_beat

        hook_beat = beats[0]
        hook_raw = write_scene(self._llm, hook_beat, fs, [payoff_narr], 10)
        hook_narr, hook_vis = _parse_scene(hook_raw)
        if not hook_narr:
            hook_narr = hook_beat

        # Generate middle scenes.
        middle_beats = beats[1:5]
        scenes: list[ScriptScene] = []
        narrations: list[str] = [hook_narr]
        for i, beat in enumerate(middle_beats):
            target = per_scene
            attempt = 0
            narr = ""
            vis = ""
            while attempt < self._max_scene_retries:
                try:
                    raw = write_scene(self._llm, beat, fs, narrations, target)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("scene %d write failed: %s", i, exc)
                    attempt += 1
                    continue
                narr, vis = _parse_scene(raw)
                res = validate_scene(narr, target)
                if res.ok:
                    break
                attempt += 1
                logger.info("scene %d retry: %s", i, res.issues)
            if not narr:
                narr = beat
            secs = max(3, int(len(narr.split()) / 2.5))
            scenes.append(ScriptScene(narration=narr, visual=vis or beat, seconds=secs))
            narrations.append(narr)

        # POV check
        pov_res = validate_pov([hook_narr] + [s.narration for s in scenes] + [payoff_narr])
        if not pov_res.ok:
            logger.warning("POV issue: %s", pov_res.issues)

        # claims check
        full_text = " ".join([hook_narr] + [s.narration for s in scenes] + [payoff_narr])
        claim_issues = check_claims(full_text, "\n".join(fs.facts) + " " + fs.source)
        if claim_issues:
            logger.warning("claims: %s", claim_issues)

        return VideoScript(
            title=fs.concept,
            hook=hook_narr,
            format=format,
            target_seconds=self._target_seconds,
            scenes=scenes,
            outro=payoff_narr,
        )
