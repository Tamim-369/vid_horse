"""Sound design for horror shorts: pick a few effects from the shelf.

Horror reels live on restraint — one deep hit under the hook, an ambience
creak around a build-up beat, a sting as the twist lands. This block reads
the script and the available effect keywords and returns at most ~4 cues.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.interfaces import LLMMessage, LLMProvider
from core.models import SfxPlan, VideoScript

logger = logging.getLogger(__name__)

_SYSTEM = """You are a horror sound designer for narration shorts.

Rules:
- Fear comes from LACK of sound: at most 4 effects for a 40-60s video.
- The FIRST cue lands on the hook line: a low impact or drone starting at
  0.0-0.5s into it, gain 0.4-0.6.
- One or two mid-story cues on the most tense beats: creak, knock, whisper.
  `at` = seconds into THAT line's audio, 0-1.5s typical.
- The LAST cue lands on the twist line: a sting — impact, drone swell, or
  whisper. Gain 0.5-0.7.
- scene_index counts NARRATION LINES: 0 = the hook, 1..N = the story scenes,
  N+1 = the twist. Point cues at the line that is being spoken when they
  should sound.
- Choose only terms from the AVAILABLE list. Gain 0.25-0.7.
"""


class SfxDesignerBlock:
    """Choose the few sound effects a horror short actually needs."""

    name = "sfx_design"

    def __init__(self, llm: LLMProvider, manifest_path: Path) -> None:
        self._llm = llm
        self._manifest_path = manifest_path

    def available_terms(self) -> dict[str, float]:
        """Effect keyword -> max duration we may use, from the shelf manifest."""

        terms: dict[str, float] = {}
        if not self._manifest_path.exists():
            return terms
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            term = entry.get("term", "")
            seconds = float(entry.get("seconds", 0) or 0)
            terms[term] = max(terms.get(term, 0.0), seconds)
        return terms

    def run(self, script: VideoScript, *, scene_count: int) -> SfxPlan:
        """Place cues on the hook, one tense beat, and the twist."""

        terms = self.available_terms()
        if not terms:
            logger.warning("No SFX shelf found; skipping sound design")
            return SfxPlan(cues=[])
        payload = [
            {
                "scene_index": index + 1,
                "narration": scene.narration,
            }
            for index, scene in enumerate(script.scenes)
        ]
        messages: list[LLMMessage] = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Hook (line index 0): {script.hook}\n"
                    f"Scenes: {json.dumps(payload, ensure_ascii=False)}\n"
                    f"Twist (line index {scene_count + 1}): {script.outro}\n"
                    "Place at most 4 cues on the full timeline as JSON."
                ),
            },
        ]
        plan = self._llm.complete_json(messages, SfxPlan)
        plan = SfxPlan(cues=[cue for cue in plan.cues[:8]])
        logger.info("SFX plan: %s", [(c.scene_index, c.effect) for c in plan.cues])
        return plan
