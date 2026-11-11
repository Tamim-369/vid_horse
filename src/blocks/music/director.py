"""The music director: pick each video's bed roles from its content.

Roles (the table in ``assets/music/music.json``) describe what is
HAPPENING, not the channel lane — threat, big-event, mystery, discovery,
vast-reveal, dread-mission. One base bed covers the video; a scene that
demands a different track names its own override.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from core.models import MusicPlan, VideoScript

logger = logging.getLogger(__name__)

# ─── Music Catalog ────────────────────────────────────────────────────

def _load_music_catalog() -> dict:
    """Load music catalog with track metadata."""
    catalog_path = Path("assets/music/music.json")
    if catalog_path.exists():
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    return {"roles": {}, "tracks": []}


# ─── Role Definitions ──────────────────────────────────────────────────

ROLES = (
    "threat",
    "big-event",
    "mystery",
    "discovery",
    "vast-reveal",
    "dread-mission",
    "default",
)

# Detailed role descriptions for the LLM
ROLE_DESCRIPTIONS = {
    "threat": "scary stuff — something attacking Earth or people, a monster lurking, an eerie presence hunting, imminent danger, manipulation pressure, control tactics",
    "big-event": "cosmic-scale processes, giant events, huge things forming or being made — the sheer SCALE is the point",
    "mystery": "deep unexplained things — light behaving as wave AND particle, patterns nobody can explain; the video explains the phenomenon but the WHY stays unanswered",
    "discovery": "explanation paced as investigating — half the answer, then 'but why is it like this?', then the other half; the viewer feels they are finding out as it plays",
    "vast-reveal": "the discovery rhythm aimed at something huge, vast, beyond human scale; also standalone awe at something enormous or beyond",
    "dread-mission": "deep-sea monsters, a new dangerous mission, a deep dangerous place that needs long explanation, venturing into the unknown",
    "default": "none of the above — the neutral pulse, calm warning, protective advice",
}

# Build role descriptions string for system prompt
ROLE_DESC_STRING = "\n".join(f"- {role}: {ROLE_DESCRIPTIONS[role]}" for role in ROLES)
ROLES_STRING = ", ".join(ROLES)

# Role-to-track mapping from music.json
ROLE_TRACK_MAP = {
    "threat": "ancient-airy-mist.mp3",
    "big-event": "decrease_volume.mp3",
    "mystery": "forest-deep-drone.mp3",
    "discovery": "investigation-calm.mp3",
    "vast-reveal": "swell-tension.mp3",
    "dread-mission": "unease-suspense.mp3",
    "default": "normal.mp3",
}

# ─── Music Genre/Vibe Keywords for Automatic Fallback ──────────────────

ROLE_KEYWORDS = {
    "threat": [
        "threat", "danger", "attack", "monster", "lurking", "hunting", "dangerous",
        "manipulation", "control", "pressure", "trap", "coercion", "force",
        "predator", "victim", "abuse", "toxic", "harm", "violence", "fear",
        "intimidation", "dominance", "power over", "exploit", "weaponize",
        "gaslight", "manipulation tactic", "control tactic", "psychological abuse",
    ],
    "big-event": [
        "scale", "massive", "enormous", "giant", "colossal", "cosmic", "universe",
        "formation", "creation", "formation", "cataclysm", "extinction",
        "collapse", "civilization", "empire", "war", "revolution", "upheaval",
    ],
    "mystery": [
        "mystery", "unexplained", "unknown", "paradox", "phenomenon", "why",
        "how", "mechanism", "hidden", "secret", "concealed", "obscure",
        "enigma", "puzzle", "riddle", "cryptic", "baffling", "defies",
        "hidden mechanism", "trick revealed", "why this works",
    ],
    "discovery": [
        "discover", "reveal", "find", "learn", "understand", "realize",
        "insight", "breakthrough", "aha", "moment", "uncover", "expose",
        "investigate", "research", "study", "explain", "teach", "lesson",
        "half", "answer", "question", "why", "turn", "twist", "reveal",
        "spot the pattern", "naming the tactic", "learning to see",
    ],
    "vast-reveal": [
        "vast", "enormous", "immense", "scale", "scope", "beyond", "horizon",
        "infinite", "endless", "boundless", "limitless", "cosmic", "galaxy",
        "universe", "infinity", "eternal", "timeless", "awe", "wonder",
        "scale of", "immensity", "magnitude",
    ],
    "dread-mission": [
        "mission", "venture", "journey", "descent", "deep", "abyss", "unknown",
        "explore", "dangerous", "perilous", "treacherous", "hazardous",
        "expedition", "probe", "landing", "descent", "into the unknown",
        "dangerous mission", "deep dangerous", "venturing",
    ],
    "default": [
        "calm", "neutral", "advice", "warning", "protect", "careful",
        "aware", "notice", "observe", "reflect", "consider", "balanced",
        "trust your perception", "not crazy",
    ],
}

# Build valid roles string for validation
VALID_ROLES_STRING = ", ".join(f'"{role}"' for role in ROLES)

# ─── Enhanced System Prompt ─────────────────────────────────────────────

MOOD_SYSTEM = f"""You are the music director of a short-form video channel (documentary and dark-psychology teaching).

You get a video script (hook, scenes, outro). Choose its BACKGROUND MUSIC
by labelling roles. A role describes WHAT IS HAPPENING, never the channel:

{ROLE_DESC_STRING}

For DARK PSYCHOLOGY / MANIPULATION TEACHING lessons, map by feel:
- manipulation tactic, control, pressure, being trapped → threat or dread-mission
- hidden mechanism, "why this works on you", the trick revealed → mystery
- the "aha" turn, naming the tactic, learning to see it → discovery
- calm warning / protective advice → default or investigation-calm pacing (use discovery)
- the sheer scale of a phenomenon → big-event
- the vastness of the phenomenon → vast-reveal

VALID ROLES (must use EXACTLY one of these): {VALID_ROLES_STRING}.

RULES
1. `base_role` labels the video's DOMINANT content — what plays under most
   of the runtime. MUST be one of the valid roles above.
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

CRITICAL: Your response MUST use EXACTLY one of these roles for base_role: {VALID_ROLES_STRING}.
Do NOT use synonyms like "manipulation tactic" — use "threat" instead.

Respond with JSON only: {{"base_role": "...", "scene_roles": {{"0": "..."}}}}
"""

# ─── Keyword-Based Fallback Classifier ──────────────────────────────────

def _classify_by_keywords(text: str) -> str:
    """Fallback: classify text by keyword matching."""
    text_lower = text.lower()
    scores = {role: 0 for role in ROLES}

    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[role] += 1

    # Default gets a small boost
    scores["default"] += 1

    return max(scores, key=scores.get)


def _analyze_script_content(script: VideoScript) -> dict:
    """Analyze script content for automatic role detection."""
    full_text = " ".join([
        script.hook,
        *[s.narration for s in script.scenes],
        script.outro,
    ])

    # Count keyword matches per role
    scores = {role: 0 for role in ROLES}
    text_lower = full_text.lower()

    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[role] += 1

    return scores


# ─── Music Director ─────────────────────────────────────────────────────

_TEMPERATURE = 0.25


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
        logger.warning("Music direction failed (%s); using keyword fallback.", error)
        # Fallback to keyword-based classification
        scores = _analyze_script_content(script)
        base = max(scores, key=scores.get)
        return script.model_copy(update={"music": MusicPlan(base_role=base, scene_roles={})})

    roles = {
        key: role
        for key, role in answer.scene_roles.items()
        if role in ROLES and key.isdigit() and int(key) < len(script.scenes)
    }
    base = answer.base_role if answer.base_role in ROLES else "default"

    # Always attach the plan — even a plain "default" must resolve to the
    # default role's track. Leaving music unset makes the renderer fall
    # back to its legacy random bed, which is NOT an intelligent choice.
    return script.model_copy(update={"music": MusicPlan(base_role=base, scene_roles=roles)})


def get_track_for_role(role: str) -> str:
    """Get the music track filename for a given role."""
    return ROLE_TRACK_MAP.get(role, ROLE_TRACK_MAP["default"])


def get_role_keywords(role: str) -> list[str]:
    """Get keywords associated with a role."""
    return ROLE_KEYWORDS.get(role, [])


def classify_text_to_role(text: str) -> str:
    """Classify arbitrary text to a music role using keywords."""
    return _classify_by_keywords(text)