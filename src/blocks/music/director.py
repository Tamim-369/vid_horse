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

# ─── Track Vibe Catalog ────────────────────────────────────────────────

def _catalog_tracks() -> list[dict]:
    """The selectable tracks with their vibe descriptions, from music.json."""

    return _load_music_catalog().get("tracks", [])


def _track_files() -> list[str]:
    return [t["file"] for t in _catalog_tracks() if t.get("file")]


def _recent_beds(n: int = 3) -> list[str]:
    """Track file names used as base beds in the most recent videos.

    Read from the output dirs' short.json so the director can AVOID
    repeating beds and every track in the rack gets airtime. Old plans
    with only roles resolve through the role table.
    """

    from config.settings import get_settings

    beds: list[str] = []
    try:
        out_dir = Path(get_settings().output_dir)
        scripts = sorted(out_dir.glob("*/short.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in scripts:
            if len(beds) >= n:
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            plan = data.get("music") or {}
            track = str(plan.get("base_track") or "").strip()
            if not track:
                role = str(plan.get("base_role") or "default")
                track = _role_track_name(role)
            if track and track not in beds:
                beds.append(track)
    except OSError:
        return []
    return beds


# ─── Vibe System Prompt ────────────────────────────────────────────────

def _build_system(recent: list[str]) -> str:
    # HARD rotation: recently used beds are removed from the catalog the
    # director even sees — a small model cannot be trusted to politely
    # avoid them when one track vibes strongly. Whatever remains is still
    # a curated dark bed, so the vibe match stays meaningful.
    allowed = [t for t in _catalog_tracks() if t.get("file") and t["file"] not in recent]
    if not allowed:
        allowed = _catalog_tracks()
    vibe_lines = []
    for t in allowed:
        if not t.get("file"):
            continue
        mood = t.get("mood", "")
        genres = ", ".join(t.get("genres", []))
        vibe_lines.append(f"- {t['file']}: {mood}" + (f" [{genres}]" if genres else ""))
    vibe_block = "\n".join(vibe_lines)

    return f"""You are the music director of a short-form dark-psychology video channel.

You get the video's narration (hook, scenes, outro). Choose its BACKGROUND MUSIC
by VIBE: pick the track whose FEELING matches what the video does to the viewer.
You MUST pick from these tracks — exactly these file names:

{vibe_block}

How to feel your way (guides, not rules — the vibe decides):
- steady, calm, neutral warning or protective advice -> the neutral pulse (normal.mp3)
- something pressing on you, being watched, quietly controlled -> a closing-in unease (unease-suspense.mp3) or a menacing calm (forest-deep-drone.mp3)
- hidden machinery, an unexplained pattern, the trick under the surface -> a deep still drone (forest-deep-drone.mp3) or vast quiet space (ancient-airy-mist.mp3)
- finding out: the lesson unfolding, "look closer", the reveal paced -> an investigating level drone (investigation-calm.mp3)
- awe, scale, something bigger than you -> airy vast space (ancient-airy-mist.mp3) or a rising cinematic swell (swell-tension.mp3)
- a turn: the moment the truth lands and the ground shifts -> the swells (swell-tension.mp3)
- raw, heavier, edgier energy -> the distinct heavy top-end piece (decrease_volume.mp3)
- the longest videos (3min+) may use the extended pulse (full.mp3)

RULES
1. `base_track` labels the video's DOMINANT vibe — what plays under most
   of the runtime. MUST be one of the exact file names above.
2. `scene_tracks` is RARE: only scenes whose feeling is genuinely a
   different KIND of moment than the base (the reveal scene rises to a
   swell; a moment of being cornered gets the closing-in track). Most
   videos need zero or one override — never override the hook scene.
3. Scene keys are scene indexes starting at 0, as strings.

Respond with JSON only: {{"base_track": "...", "scene_tracks": {{"0": "..."}}}}
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
    """Attach a vibe-based :class:`MusicPlan` to ``script``; never blocks.

    The director chooses TRACKS by the feeling each one carries (the mood
    lines in ``assets/music/music.json``), and avoids the beds the last
    videos used so the whole rack gets airtime. Every failure path
    resolves to SOME valid track — a plan-less script would drop the
    renderer into its legacy keyword path, which is not an intelligent
    choice either.
    """

    from blocks.music.schema import MusicAnnotation
    from core.errors import LLMError

    files = _track_files()
    if not files:
        return script

    recent = _recent_beds()
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
                {"role": "system", "content": _build_system(recent)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            MusicAnnotation,
            temperature=_TEMPERATURE,
        )
    except LLMError as error:
        logger.warning("Music direction failed (%s); using keyword fallback.", error)
        scores = _analyze_script_content(script)
        base = max(scores, key=scores.get)
        track = _role_track_name(base)
        if track in set(recent):
            allowed = [t for t in files if t not in set(recent)] or files
            import hashlib

            seed_text = " ".join([script.hook, *(s.narration for s in script.scenes), script.outro])
            digest = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "big")
            track = allowed[digest % len(allowed)]
        return script.model_copy(update={"music": MusicPlan(base_role=base, base_track=track, scene_roles={}, scene_tracks={})})

    # validate: unknown track OR a recently used track -> the model ignored
    # the restricted catalog; fall back deterministically among the allowed
    base_track = (answer.base_track or "").strip()
    if base_track not in files and base_track + ".mp3" in files:
        base_track += ".mp3"  # tolerate a missing extension
    recent_set = set(recent)
    if base_track not in files or base_track in recent_set:
        logger.warning(
            "music director picked %r (unknown or recent); deterministic fallback", base_track
        )
        allowed = [t for t in files if t not in recent_set] or files
        import hashlib

        seed_text = " ".join([script.hook, *(s.narration for s in script.scenes), script.outro])
        digest = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "big")
        base_track = allowed[digest % len(allowed)]
        return script.model_copy(update={"music": MusicPlan(base_track=base_track, scene_tracks={})})

    scene_tracks = {
        key: (name.strip() if name.strip() in files else name.strip() + ".mp3")
        for key, name in (answer.scene_tracks or {}).items()
        if (name.strip() in files or name.strip() + ".mp3" in files)
        and key.isdigit() and int(key) < len(script.scenes)
    }
    logger.info("music bed: %s (recent: %s)", base_track, recent or "none")
    return script.model_copy(update={"music": MusicPlan(base_track=base_track, scene_tracks=scene_tracks)})


def _role_track_name(role: str) -> str:
    """Role -> track file (the music.json role table)."""

    catalog = _load_music_catalog()
    entry = (catalog.get("roles") or {}).get(role) or (catalog.get("roles") or {}).get("default") or {}
    return entry.get("track", "normal.mp3")


def get_track_for_role(role: str) -> str:
    """Get the music track filename for a given role."""
    return ROLE_TRACK_MAP.get(role, ROLE_TRACK_MAP["default"])


def get_role_keywords(role: str) -> list[str]:
    """Get keywords associated with a role."""
    return ROLE_KEYWORDS.get(role, [])


def classify_text_to_role(text: str) -> str:
    """Classify arbitrary text to a music role using keywords."""
    return _classify_by_keywords(text)