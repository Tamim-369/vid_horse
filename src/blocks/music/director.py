"""The music director: pick each video's bed roles from its content.

Roles (the table in ``assets/music/music.json``) describe what is
HAPPENING, not the channel lane — threat, big-event, mystery, discovery,
vast-reveal, dread-mission. One base bed covers the video; a scene that
demands a different track names its own override.
"""

from __future__ import annotations

import json
import logging

from core.models import MusicPlan, VideoScript

logger = logging.getLogger(__name__)

_TEMPERATURE = 0.25

ROLES = (
    "threat",
    "big-event",
    "mystery",
    "discovery",
    "vast-reveal",
    "dread-mission",
    "default",
)

MOOD_SYSTEM = f"""You are the music director of a documentary video channel.

You get a video script (hook, scenes, outro). Choose its BACKGROUND MUSIC
by labelling roles. A role describes WHAT IS HAPPENING, never the channel:

- threat: scary stuff — something attacking Earth or people, a monster
  lurking, an eerie presence hunting.
- big-event: cosmic-scale processes, giant events, huge things forming or
  being made — the sheer SCALE is the point.
- mystery: deep unexplained things — light behaving as wave AND particle,
  patterns nobody can explain; the video explains the phenomenon but the
  WHY stays unanswered.
- discovery: explanation paced as investigating — half the answer, then
  "but why is it like this?", then the other half; the viewer feels they
  are finding out as it plays.
- vast-reveal: the discovery rhythm aimed at something huge, vast, beyond
  human scale; also standalone awe at something enormous or beyond.
- dread-mission: deep-sea monsters, a new dangerous mission, a deep
  dangerous place that needs long explanation.
- default: none of the above — the neutral pulse.

Valid roles: {", ".join(ROLES)}.

RULES
1. `base_role` labels the video's DOMINANT content — what plays under most
   of the runtime.
2. Labelling hints:
   - A video whose spine is one catastrophic or enormous event, or the
     sheer scale of an immense thing, is big-event — NOT discovery.
   - A video about venturing into or landing on a dangerous, deep or
     barely-explored place (a descent, a landing, a probe mission) is
     dread-mission even when it also explains findings.
   - lone wandering objects far from home (probes streaking out of the
     solar system) are vast-reveal — distance and scale, not discovery.
   - discovery needs a teaching rhythm: question, half-answer, question,
     half-answer, throughout. Without that rhythm use a scale role.
3. `scene_roles` is RARE: only scenes whose content is a different KIND of
   moment than the base (the reveal scene of a discovery video becomes
   vast-reveal; an attack mention inside a calm explainer becomes threat;
   a lurking-monster scene becomes dread-mission). Most videos need zero
   or one override — never override the hook scene.
4. Scene keys are scene indexes starting at 0, as strings.

Respond with JSON only: {{"base_role": "...", "scene_roles": {{"0": "..."}}}}
"""


def annotate_music(llm, script: VideoScript) -> VideoScript:
    """Attach a :class:`MusicPlan` to ``script``; never blocks the pipeline."""

    from blocks.music.schema import MusicAnnotation
    from core.errors import LLMError

    payload = {
        "hook": script.hook,
        "scenes": [
            {"index": index, "narration": scene.narration, "visual": scene.visual}
            for index, scene in enumerate(script.scenes)
        ],
        "outro": script.outro,
    }
    try:
        answer = llm.complete_json(
            [
                {"role": "system", "content": MOOD_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            MusicAnnotation,
            temperature=_TEMPERATURE,
        )
    except LLMError as error:
        logger.warning("Music direction failed (%s); default bed stays.", error)
        return script
    roles = {
        key: role
        for key, role in answer.scene_roles.items()
        if role in ROLES and key.isdigit() and int(key) < len(script.scenes)
    }
    base = answer.base_role if answer.base_role in ROLES else "default"
    if base == "default" and not roles:
        return script
    return script.model_copy(update={"music": MusicPlan(base_role=base, scene_roles=roles)})
