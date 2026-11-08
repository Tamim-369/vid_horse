"""Concept → Visual Metaphor mapping for dark psychology lessons.

Each psychological concept maps to concrete, searchable visual metaphors
that *embody* the mechanism rather than illustrate it literally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConceptVisuals:
    """Visual representation of a psychological concept."""
    concept: str
    # Search queries for stock footage (Pexels, Openverse, etc.)
    primary_queries: list[str]
    fallback_queries: list[str] = field(default_factory=list)
    # Motion style suited to this concept
    motion_style: Literal[
        "slow_creep",      # gradual influence, creeping normalization
        "sudden_snap",     # compliance, sudden realization
        "narrowing_focus", # tunneling, scarcity, tunneling
        "alignment_cascade", # social proof, herd behavior
        "distortion",      # manipulation, gaslighting, reframing
        "pressure_build",  # pressure, urgency, deadline
        "reflection",      # mirroring, projection, self-deception
        "lever",           # small input → large output, leverage
    ] = "slow_creep"
    # Color grade emphasis
    grade_emphasis: Literal[
        "cold", "noir", "raw", "academic", "tech", "neutral"
    ] = "neutral"
    # Crossfade timing (seconds)
    subject_crossfade: float = 0.45
    card_crossfade: float = 0.30
    # Music bed mood
    music_mood: str = "tension"


# Core concept → visual metaphor mapping
CONCEPT_VISUALS: dict[str, ConceptVisuals] = {
    # === CROWD / SOCIAL INFLUENCE ===
    "crowd_mind": ConceptVisuals(
        concept="crowd_mind",
        primary_queries=[
            "starling murmuration aerial",
            "synchronized swimming underwater",
            "iron filings magnetic field",
            "bird flock formation sky",
            "fish school bait ball",
            "fireflies synchronization night",
        ],
        fallback_queries=["crowd aerial pattern", "mass movement coordinated"],
        motion_style="alignment_cascade",
        grade_emphasis="noir",
        subject_crossfade=0.6,
        card_crossfade=0.5,
        music_mood="ominous_drone",
    ),

    "social_proof": ConceptVisuals(
        concept="social_proof",
        primary_queries=[
            "domino chain reaction close up",
            "pendulum synchronization multiple",
            "metronomes syncing table",
            "fireflies flashing unison",
            "clapping audience wave stadium",
        ],
        fallback_queries=["chain reaction physics", "synchronization phenomenon"],
        motion_style="alignment_cascade",
        grade_emphasis="neutral",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="building_tension",
    ),

    "herd_behavior": ConceptVisuals(
        concept="herd_behavior",
        primary_queries=[
            "wildebeest migration river crossing",
            "sheep flock moving pasture aerial",
            "lemming migration cliff",
            "buffalo herd stampede dust",
            "penguin colony marching ice",
        ],
        fallback_queries=["animal migration aerial", "herd movement pattern"],
        motion_style="alignment_cascade",
        grade_emphasis="raw",
        subject_crossfade=0.4,
        card_crossfade=0.3,
        music_mood="urgent_rhythm",
    ),

    "groupthink": ConceptVisuals(
        concept="groupthink",
        primary_queries=[
            "mirror maze infinite reflections",
            "echo chamber sound waves visualization",
            "feedback loop microphone speaker",
            "identical mannequins row factory",
            "copy paste duplicate pattern digital",
        ],
        fallback_queries=["echo chamber concept", "conformity pressure visual"],
        motion_style="distortion",
        grade_emphasis="noir",
        subject_crossfade=0.7,
        card_crossfade=0.6,
        music_mood="dissonant_hum",
    ),

    # === COMPLIANCE / OBEDIENCE ===
    "compliance": ConceptVisuals(
        concept="compliance",
        primary_queries=[
            "lever fulcrum mechanical advantage close",
            "hydraulic press crushing object",
            "key turning lock mechanism macro",
            "domino first piece falling chain",
            "water finding crack rock slow motion",
            "path least resistance river delta",
        ],
        fallback_queries=["mechanical leverage", "compliance mechanism"],
        motion_style="sudden_snap",
        grade_emphasis="cold",
        subject_crossfade=0.3,
        card_crossfade=0.25,
        music_mood="sharp_click",
    ),

    "foot_in_door": ConceptVisuals(
        concept="foot_in_door",
        primary_queries=[
            "door slightly ajar foot wedge",
            "small crack dam water seeping",
            "thin wedge splitting log",
            "camel nose under tent fabric",
            "first domino tipping line",
        ],
        fallback_queries=["small request compliance", "incremental escalation"],
        motion_style="slow_creep",
        grade_emphasis="clinical",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="creeping_tension",
    ),

    "door_in_face": ConceptVisuals(
        concept="door_in_face",
        primary_queries=[
            "heavy door slamming shut slow motion",
            "large request rejected hand gesture",
            "retreat compromise handshake",
            "large small comparison scale",
            "concession negotiation table",
        ],
        fallback_queries=["rejection then concession", "reciprocal compromise"],
        motion_style="sudden_snap",
        grade_emphasis="raw",
        subject_crossfade=0.35,
        card_crossfade=0.3,
        music_mood="rejection_then_relief",
    ),

    "lowball": ConceptVisuals(
        concept="lowball",
        primary_queries=[
            "price tag changing hands switch",
            "contract terms changing fine print",
            "bait switch product shelf",
            "promise broken chain link",
            "goalpost moving field markers",
        ],
        fallback_queries=["changing terms deal", "commitment trap"],
        motion_style="distortion",
        grade_emphasis="noir",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="betrayal_chord",
    ),

    # === AUTHORITY / POWER ===
    "authority": ConceptVisuals(
        concept="authority",
        primary_queries=[
            "low angle building looking up architecture",
            "judge bench gavel courtroom scale",
            "uniform insignia close up texture",
            "podium microphone single light",
            "crown throne empty room palace",
            "architect blueprint ruler compass",
        ],
        fallback_queries=["power symbol architecture", "authority figure silhouette"],
        motion_style="narrowing_focus",
        grade_emphasis="tech",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="solemn_chord",
    ),

    "milgram_obedience": ConceptVisuals(
        concept="milgram_obedience",
        primary_queries=[
            "switch lever panel labeled danger",
            "shock generator dial voltage increasing",
            "white coat clipboard observing",
            "learner behind glass partition",
            "authority figure pointing direction",
        ],
        fallback_queries=["obedience experiment", "authority compliance visual"],
        motion_style="pressure_build",
        grade_emphasis="clinical",
        subject_crossfade=0.4,
        card_crossfade=0.35,
        music_mood="rising_tension",
    ),

    # === MANIPULATION / GASLIGHTING ===
    "gaslighting": ConceptVisuals(
        concept="gaslighting",
        primary_queries=[
            "mirror distorted reflection face",
            "gas lamp flickering dimming brightening",
            "reality bending grid lines warping",
            "two reflections different actions mirror",
            "memory photograph fading changing",
            "hallucination visual distortion art",
        ],
        fallback_queries=["reality distortion", "memory manipulation visual"],
        motion_style="distortion",
        grade_emphasis="noir",
        subject_crossfade=0.8,
        card_crossfade=0.7,
        music_mood="unsettling_ambience",
    ),

    "love_bombing": ConceptVisuals(
        concept="love_bombing",
        primary_queries=[
            "rose petals overwhelming pile",
            "messages notifications overwhelming phone",
            "gifts stacked excessive wrapping",
            "attention spotlight single person crowd",
            "honeymoon phase sunset embrace",
        ],
        fallback_queries=["overwhelming affection", "manipulative affection"],
        motion_style="sudden_snap",
        grade_emphasis="raw",
        subject_crossfade=0.4,
        card_crossfade=0.3,
        music_mood="overwhelming_sweetness",
    ),

    "triangulation": ConceptVisuals(
        concept="triangulation",
        primary_queries=[
            "three points triangle tension cables",
            "puppeteer strings two puppets one hand",
            "messenger relay message distortion",
            "divide conquer map lines dividing",
            "jealousy third party shadow looming",
        ],
        fallback_queries=["three person manipulation", "divide and conquer"],
        motion_style="lever",
        grade_emphasis="noir",
        subject_crossfade=0.6,
        card_crossfade=0.5,
        music_mood="tense_strings",
    ),

    # === SCARCITY / URGENCY ===
    "scarcity": ConceptVisuals(
        concept="scarcity",
        primary_queries=[
            "hourglass sand running low macro",
            "evaporating puddle sun timelapse",
            "closing aperture camera lens",
            "last item shelf empty store",
            "countdown timer red numbers",
            "resource depletion graph declining",
        ],
        fallback_queries=["limited supply visual", "urgency time running out"],
        motion_style="narrowing_focus",
        grade_emphasis="raw",
        subject_crossfade=0.3,
        card_crossfade=0.2,
        music_mood="countdown_rhythm",
    ),

    "fomo": ConceptVisuals(
        concept="fomo",
        primary_queries=[
            "party through window looking in",
            "notification badge red increasing",
            "crowd rushing entrance doors",
            "exclusive velvet rope line",
            "limited edition sold out sign",
        ],
        fallback_queries=["fear missing out", "exclusion anxiety visual"],
        motion_style="pressure_build",
        grade_emphasis="raw",
        subject_crossfade=0.3,
        card_crossfade=0.25,
        music_mood="anxious_pulse",
    ),

    # === RECIPROCITY ===
    "reciprocity": ConceptVisuals(
        concept="reciprocity",
        primary_queries=[
            "boomerang returning thrower slow motion",
            "ripple pond returning center",
            "echo canyon sound wave returning",
            "pendulum swing back equal force",
            "gift exchanged hands simultaneous",
            "favor returned unexpected moment",
        ],
        fallback_queries=["reciprocal exchange", "obligation return favor"],
        motion_style="lever",
        grade_emphasis="neutral",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="resolution_chord",
    ),

    # === CONSISTENCY / COMMITMENT ===
    "consistency_trap": ConceptVisuals(
        concept="consistency_trap",
        primary_queries=[
            "signature line contract binding",
            "footprint path cement drying",
            "written statement pinned board",
            "public pledge raised hand crowd",
            "identity locked box key turning",
        ],
        fallback_queries=["commitment consistency", "self-image trap"],
        motion_style="narrowing_focus",
        grade_emphasis="clinical",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="binding_chord",
    ),

    "sunk_cost": ConceptVisuals(
        concept="sunk_cost",
        primary_queries=[
            "hole digging deeper shovel person",
            "investment graph red declining hold",
            "repair broken object cost exceeding value",
            "gambler chips pushing forward losing",
            "project half finished abandoned materials",
        ],
        fallback_queries=["throwing good after bad", "escalation commitment"],
        motion_style="pressure_build",
        grade_emphasis="raw",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="deepening_drone",
    ),

    # === LIKING / CHARISMA ===
    "halo_effect": ConceptVisuals(
        concept="halo_effect",
        primary_queries=[
            "halo light behind head portrait",
            "attractive person glowing aura lighting",
            "first impression lens flare bright",
            "beauty bias magazine cover glow",
            "charisma light radiating outward",
        ],
        fallback_queries=["attractiveness bias", "first impression power"],
        motion_style="narrowing_focus",
        grade_emphasis="tech",
        subject_crossfade=0.4,
        card_crossfade=0.3,
        music_mood="ethereal_glow",
    ),

    "mirroring": ConceptVisuals(
        concept="mirroring",
        primary_queries=[
            "two people posture matching perfectly",
            "mirror reflection mimicking movements",
            "chameleon color changing background",
            "synchronized swimmers perfect sync",
            "copy cat shadow following",
        ],
        fallback_queries=["rapport building mimicry", "behavioral matching"],
        motion_style="alignment_cascade",
        grade_emphasis="tech",
        subject_crossfade=0.4,
        card_crossfade=0.35,
        music_mood="sync_rhythm",
    ),

    # === FRAMING / REFRAMING ===
    "framing_effect": ConceptVisuals(
        concept="framing_effect",
        primary_queries=[
            "picture frame changing perspective painting",
            "glass half full half empty split",
            "camera lens zoom crop reframe",
            "window view different angles same scene",
            "words rearranged meaning changes",
        ],
        fallback_queries=["perspective shift", "reframing reality"],
        motion_style="distortion",
        grade_emphasis="noir",
        subject_crossfade=0.6,
        card_crossfade=0.5,
        music_mood="perspective_shift",
    ),

    "anchoring": ConceptVisuals(
        concept="anchoring",
        primary_queries=[
            "anchor dropping ocean heavy chain",
            "first price tag large bold numbers",
            "reference point map pin stuck",
            "heavy weight scale tipping balance",
            "first impression cement setting",
        ],
        fallback_queries=["first impression anchor", "reference point bias"],
        motion_style="lever",
        grade_emphasis="cold",
        subject_crossfade=0.4,
        card_crossfade=0.35,
        music_mood="heavy_impact",
    ),

    # === DECEPTION ===
    "deception": ConceptVisuals(
        concept="deception",
        primary_queries=[
            "magician sleight hand cards close",
            "mask two faces janus roman",
            "fog obscuring path ahead",
            "mirror maze dead end reflection",
            "liedetector needle spiking paper",
        ],
        fallback_queries=["lying deception visual", "hidden truth visual"],
        motion_style="distortion",
        grade_emphasis="noir",
        subject_crossfade=0.7,
        card_crossfade=0.6,
        music_mood="deceptive_melody",
    ),

    # === COGNITIVE PROCESSING ===
    "simplification": ConceptVisuals(
        concept="simplification",
        primary_queries=[
            "simple shape circle square triangle clean",
            "minimalist design geometric forms",
            "essence extracted complexity removed",
            "core form extracted from noise",
            "occam's razor blade cutting through",
        ],
        fallback_queries=["simplicity essence", "reducing complexity"],
        motion_style="slow_creep",
        grade_emphasis="tech",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="clarity_emerging",
    ),

    "cognitive_load": ConceptVisuals(
        concept="cognitive_load",
        primary_queries=[
            "brain overwhelmed too many inputs",
            "cluttered desk papers overwhelming",
            "traffic jam mental highway",
            "buffer overflow digital visualization",
            "working memory capacity limit diagram",
        ],
        fallback_queries=["mental overload", "too much information"],
        motion_style="pressure_build",
        grade_emphasis="raw",
        subject_crossfade=0.4,
        card_crossfade=0.3,
        music_mood="overwhelming_pulse",
    ),

    "perception": ConceptVisuals(
        concept="perception",
        primary_queries=[
            "optical illusion ambiguous figure",
            "brain interpreting visual input",
            "seeing vs believing brain processing",
            "visual cortex neural pathways lighting",
            "gestalt principles closure figure ground",
        ],
        fallback_queries=["visual perception", "how we see"],
        motion_style="narrowing_focus",
        grade_emphasis="tech",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="revelation",
    ),

    "heuristics": ConceptVisuals(
        concept="heuristics",
        primary_queries=[
            "mental shortcut path brain efficiency",
            "rule of thumb decision making",
            "pattern recognition brain autopilot",
            "cognitive shortcut maze solution",
            "intuition vs analysis brain split",
        ],
        fallback_queries=["mental shortcuts", "cognitive shortcuts"],
        motion_style="sudden_snap",
        grade_emphasis="cold",
        subject_crossfade=0.35,
        card_crossfade=0.3,
        music_mood="quick_insight",
    ),

    "gestalt": ConceptVisuals(
        concept="gestalt",
        primary_queries=[
            "gestalt principles visual grouping",
            "whole greater than sum parts",
            "figure ground reversal vase faces",
            "closure principle incomplete shapes",
            "similarity proximity continuity principles",
        ],
        fallback_queries=["gestalt psychology", "visual organization"],
        motion_style="alignment_cascade",
        grade_emphasis="tech",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="pattern_emerging",
    ),

    "essentialism": ConceptVisuals(
        concept="essentialism",
        primary_queries=[
            "essence extracted core revealed",
            "stripping away non-essential layers",
            "core truth beneath surface noise",
            "platonic form ideal shape glowing",
            "archetype essence purified",
        ],
        fallback_queries=["essential nature", "core truth"],
        motion_style="slow_creep",
        grade_emphasis="academic",
        subject_crossfade=0.6,
        card_crossfade=0.5,
        music_mood="deepening_clarity",
    ),

    "abstraction": ConceptVisuals(
        concept="abstraction",
        primary_queries=[
            "abstract art geometric reduction",
            "removing details keeping essence",
            "zoom out seeing pattern emerge",
            "from photo to line drawing evolution",
            "concrete to abstract spectrum",
        ],
        fallback_queries=["abstract thinking", "removing detail"],
        motion_style="narrowing_focus",
        grade_emphasis="tech",
        subject_crossfade=0.5,
        card_crossfade=0.4,
        music_mood="zooming_out",
    ),
    "default": ConceptVisuals(
        concept="default",
        primary_queries=[
            "psychology concept abstract visualization",
            "mind brain neural network connections",
            "decision making crossroads paths",
            "human behavior pattern abstract",
            "influence invisible forces visual",
        ],
        fallback_queries=["psychology concept", "mind manipulation"],
        motion_style="slow_creep",
        grade_emphasis="neutral",
        subject_crossfade=0.45,
        card_crossfade=0.3,
        music_mood="tension",
    ),
}


def get_concept_visuals(concept: str) -> ConceptVisuals:
    """Get visual config for a concept, with fuzzy matching."""
    concept = concept.lower().replace(" ", "_").replace("-", "_")
    if concept in CONCEPT_VISUALS:
        return CONCEPT_VISUALS[concept]
    # Fuzzy match
    for key, cv in CONCEPT_VISUALS.items():
        if key in concept or concept in key:
            return cv
    return CONCEPT_VISUALS["default"]


def extract_concepts_from_text(text: str) -> list[str]:
    """Extract known psychology concepts from narration text."""
    text_lower = text.lower()
    found = []
    # Keywords that indicate each concept
    keywords = {
        "crowd_mind": ["crowd mind", "crowd psychology", "mob mentality", "group mind", "collective mind"],
        "social_proof": ["social proof", "everyone else", "others do", "following the crowd", "proof that others"],
        "herd_behavior": ["herd", "flock", "follow the herd", "herd mentality", "mob behavior"],
        "groupthink": ["groupthink", "group think", "consensus", "unanimous", "dissent suppressed"],
        "compliance": ["comply", "compliance", "do as told", "obey", "follow orders", "do what"],
        "foot_in_door": ["foot in the door", "small request", "start small", "gradual escalation"],
        "door_in_face": ["door in the face", "large request", "reject then", "concession"],
        "lowball": ["lowball", "change the terms", "bait and switch", "promise broken"],
        "authority": ["authority", "expert says", "doctor says", "professor", "official", "uniform"],
        "milgram_obedience": ["milgram", "obedience experiment", "shock", "obey authority"],
        "gaslighting": ["gaslight", "crazy", "imagining things", "memory wrong", "reality"],
        "love_bombing": ["love bomb", "overwhelming affection", "too good to be true", "soulmate"],
        "triangulation": ["triangulation", "third party", "jealousy", "divide and conquer", "messenger"],
        "scarcity": ["scarcity", "limited", "running out", "only a few", "last chance", "exclusive"],
        "fomo": ["fear of missing", "missing out", "everyone getting", "don't miss", "last chance"],
        "reciprocity": ["reciprocity", "return the favor", "owe you", "pay back", "obligation"],
        "consistency_trap": ["consistency", "commitment", "said you would", "promised", "identity"],
        "sunk_cost": ["sunk cost", "already invested", "too far to quit", "throwing good"],
        "halo_effect": ["halo effect", "attractive", "beautiful", "good looking", "charming"],
        "mirroring": ["mirroring", "copying", "mimicking", "matching", "rapport"],
        "framing_effect": ["framing", "reframe", "perspective", "glass half", "how you see"],
        "anchoring": ["anchor", "first price", "reference point", "first number", "initial"],
        "crowd_mind": ["crowd mind", "crowd psychology", "mob mentality", "group mind", "collective mind", "unconscious crowd", "crowd at work", "crowd behavior", "mass mind", "hive mind"],
        "social_proof": ["social proof", "everyone else", "others do", "following the crowd", "proof that others", "others contribute", "see what others do", "follow others"],
        "herd_behavior": ["herd", "flock", "follow the herd", "herd mentality", "mob behavior", "join naturally", "follow the leader", "group movement"],
        "groupthink": ["groupthink", "group think", "consensus", "unanimous", "dissent suppressed"],
        "compliance": ["comply", "compliance", "do as told", "obey", "follow orders", "do what"],
        "foot_in_door": ["foot in the door", "small request", "start small", "gradual escalation"],
        "door_in_face": ["door in the face", "large request", "reject then", "concession"],
        "lowball": ["lowball", "change the terms", "bait and switch", "promise broken"],
        "authority": ["authority", "expert says", "doctor says", "professor", "official", "uniform"],
        "milgram_obedience": ["milgram", "obedience experiment", "shock", "obey authority"],
        "gaslighting": ["gaslight", "crazy", "imagining things", "memory wrong", "reality"],
        "love_bombing": ["love bomb", "overwhelming affection", "too good to be true", "soulmate"],
        "triangulation": ["triangulation", "third party", "jealousy", "divide and conquer", "messenger"],
        "scarcity": ["scarcity", "limited", "running out", "only a few", "last chance", "exclusive"],
        "fomo": ["fear of missing", "missing out", "everyone getting", "don't miss", "last chance"],
        "reciprocity": ["reciprocity", "return the favor", "owe you", "pay back", "obligation"],
        "consistency_trap": ["consistency", "commitment", "said you would", "promised", "identity"],
        "sunk_cost": ["sunk cost", "already invested", "too far to quit", "throwing good"],
        "halo_effect": ["halo effect", "attractive", "beautiful", "good looking", "charming"],
        "mirroring": ["mirroring", "copying", "mimicking", "matching", "rapport"],
        "framing_effect": ["framing", "reframe", "perspective", "glass half", "how you see"],
        "anchoring": ["anchor", "first price", "reference point", "first number", "initial"],
        "crowd_mind": ["crowd mind", "crowd psychology", "mob mentality", "group mind", "collective mind", "unconscious crowd", "crowd at work", "crowd behavior", "mass mind", "hive mind"],
        "social_proof": ["social proof", "everyone else", "others do", "following the crowd", "proof that others", "others contribute", "see what others do", "follow others"],
        "herd_behavior": ["herd", "flock", "follow the herd", "herd mentality", "mob behavior", "join naturally", "follow the leader", "group movement"],
        "groupthink": ["groupthink", "group think", "consensus", "unanimous", "dissent suppressed"],
        "compliance": ["comply", "compliance", "do as told", "obey", "follow orders", "do what"],
        "foot_in_door": ["foot in the door", "small request", "start small", "gradual escalation"],
        "door_in_face": ["door in the face", "large request", "reject then", "concession"],
        "lowball": ["lowball", "change the terms", "bait and switch", "promise broken"],
        "authority": ["authority", "expert says", "doctor says", "professor", "official", "uniform"],
        "milgram_obedience": ["milgram", "obedience experiment", "shock", "obey authority"],
        "gaslighting": ["gaslight", "crazy", "imagining things", "memory wrong", "reality"],
        "love_bombing": ["love bomb", "overwhelming affection", "too good to be true", "soulmate"],
        "triangulation": ["triangulation", "third party", "jealousy", "divide and conquer", "messenger"],
        "scarcity": ["scarcity", "limited", "running out", "only a few", "last chance", "exclusive"],
        "fomo": ["fear of missing", "missing out", "everyone getting", "don't miss", "last chance"],
        "reciprocity": ["reciprocity", "return the favor", "owe you", "pay back", "obligation"],
        "consistency_trap": ["consistency", "commitment", "said you would", "promised", "identity"],
        "sunk_cost": ["sunk cost", "already invested", "too far to quit", "throwing good"],
        "halo_effect": ["halo effect", "attractive", "beautiful", "good looking", "charming"],
        "mirroring": ["mirroring", "copying", "mimicking", "matching", "rapport"],
        "framing_effect": ["framing", "reframe", "perspective", "glass half", "how you see", "reframe", "perspective shift"],
        "anchoring": ["anchor", "first price", "reference point", "first number", "initial", "first impression"],
        "deception": ["deception", "lying", "manipulate", "trick", "fool", "mislead", "trap", "trick"],
        "simplification": ["simplify", "simplifies", "simplifying", "simplification", "simple form", "core form", "basic form", "essence", "strip away", "reduce to"],
        "cognitive_load": ["cognitive load", "mental load", "overwhelm", "too much", "clutter", "confusion", "processing", "capacity"],
        "perception": ["perception", "perceive", "see", "seeing", "visual", "brain sees", "interpret", "gestalt", "observe", "observation", "observing", "notice", "noticeable"],
        "heuristics": ["heuristic", "shortcut", "mental shortcut", "rule of thumb", "autopilot", "pattern recognition", "intuition", "jump to conclusion", "snap judgment"],
        "gestalt": ["gestalt", "whole greater", "figure ground", "closure", "proximity", "similarity"],
        "essentialism": ["essential", "essence", "core truth", "true nature", "stripped", "non-essential"],
        "abstraction": ["abstract", "abstraction", "geometric", "remove detail", "zoom out", "pattern emerge", "concrete to abstract"],
        "trap": ["trap", "pitfall", "snare", "easy trap", "fall into", "trap of"],
        "certainty": ["certainty", "certain", "sure", "confident", "confidence", "assumption", "assumed"],
        "hidden_layers": ["hidden", "unseen", "beneath", "surface", "layer", "depth", "underneath", "below surface"],
    }
    for concept, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            found.append(concept)
    return found if found else ["default"]


if __name__ == "__main__":
    # Quick test
    test_text = "The crowd mind takes over when everyone looks to others for proof"
    concepts = extract_concepts_from_text(test_text)
    print(f"Found: {concepts}")
    for c in concepts:
        cv = get_concept_visuals(c)
        print(f"  {c}: {cv.primary_queries[:2]} | motion={cv.motion_style} | grade={cv.grade_emphasis}")