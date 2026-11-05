"""Visual motif taxonomy for dark-psychology shorts.

Each motif is a self-contained visual idea that can carry a sentence on its
own, without narration. The render pipeline builds a "silent cut" by
assigning one motif per sentence, then layers voice and music on top.

Motifs are grouped into categories so the chooser can enforce variety
(no two adjacent scenes share a category, no global repeat within a video).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Motif:
    id: str
    category: str
    name: str
    description: str
    cues: str
    search_terms: tuple[str, ...]


CATEGORIES: tuple[str, ...] = (
    "face",
    "body",
    "shadow",
    "control",
    "space",
    "text",
    "surveillance",
    "abstract",
    "weather",
    "motion",
)


MOTIFS: tuple[Motif, ...] = (
    # ---- face ----
    Motif("face_01", "face", "Micro-expression freeze", "Extreme close-up of a face mid-micro-expression, frozen in time.", "tight framing, shallow depth, single catchlight", ("micro expression face", "close up eyes", "facial expression portrait")),
    Motif("face_02", "face", "Dead-eye stare", "Unblinking stare directly into lens, pupil dilated, no smile.", "frontal, flat light, clinical", ("stare portrait", "unblinking eyes", "intense gaze")),
    Motif("face_03", "face", "Half-shadow face", "Face split by a hard vertical shadow, one eye hidden.", "chiaroscuro, profile light", ("chiaroscuro portrait", "half face shadow", "dramatic portrait")),
    Motif("face_04", "face", "Smile that doesn't reach eyes", "Polite smile with cold, empty eyes.", "soft light, posed, uncanny", ("fake smile portrait", "insincere smile", "uncanny smile")),
    Motif("face_05", "face", "Tears without emotion", "Tears rolling while expression stays blank.", "macro, desaturated", ("tear close up", "crying blank face", "tears portrait")),
    Motif("face_06", "face", "Mouth covered", "Hand or tape covering mouth, eyes wide.", "centered, high contrast", ("mouth covered", "silenced face", "hand over mouth")),
    Motif("face_07", "face", "Mask removal", "Hands peeling a theatrical mask from a face.", "motion blur, dramatic", ("mask removal", "taking off mask", "masquerade face")),
    Motif("face_08", "face", "Reflection in eye", "Extreme macro of an eye reflecting a scene.", "iris detail, mirror", ("eye reflection", "iris macro", "reflection in eye")),
    Motif("face_09", "face", "Aged vs young split", "Split-screen same person young and old.", "diptych, neutral", ("age progression face", "young old portrait", "aging face")),
    Motif("face_10", "face", "Faceless silhouette", "Head and shoulders silhouette, no features.", "backlight, fog", ("silhouette head", "faceless figure", "dark silhouette")),
    # ---- body ----
    Motif("body_01", "body", "Power posture", "Person standing over seated subject, arms crossed, chin up.", "low angle, imposing", ("power posture", "dominant stance", "body language power")),
    Motif("body_02", "body", "Defensive crossing", "Arms crossed, shoulders hunched, turned away.", "side view, soft light", ("defensive body language", "crossed arms", "closed posture")),
    Motif("body_03", "body", "Leaning invasion", "One figure leaning into another's space, other recoiling.", "two-shot, tension", ("personal space invasion", "leaning close", "intimate distance")),
    Motif("body_04", "body", "Hand on shoulder control", "Hand gripping shoulder from behind, guiding.", "over-shoulder, subtle", ("hand on shoulder", "controlling touch", "guiding hand")),
    Motif("body_05", "body", "Pointing finger", "Aggressive finger point, other flinching.", "high contrast, freeze", ("pointing finger", "accusation gesture", "blame gesture")),
    Motif("body_06", "body", "Mirroring", "Two people in identical posture, one leading.", "symmetrical, study", ("mirroring body language", "rapport posture", "matched stance")),
    Motif("body_07", "body", "Hands steepled", "Fingertips touching, elbows out, thinking pose.", "desk, neutral", ("steepled hands", "thinking pose", "finger tent")),
    Motif("body_08", "body", "Pacing silhouette", "Figure pacing in profile, long shadow.", "side light, motion blur", ("pacing silhouette", "walking shadow", "restless figure")),
    Motif("body_09", "body", "Kneeling submission", "One figure kneeling, other standing over.", "low angle, stark", ("kneeling figure", "submission posture", "power dynamic")),
    Motif("body_10", "body", "Hands tied / restrained", "Wrists bound or held, tension.", "close-up, dramatic", ("restrained hands", "bound wrists", "captive hands")),
    # ---- shadow ----
    Motif("shadow_01", "shadow", "Long shadow pursuit", "Figure's shadow stretching far ahead, chasing.", "sunset light, ground", ("long shadow", "shadow pursuit", "elongated shadow")),
    Motif("shadow_02", "shadow", "Shadow without source", "Sharp shadow on wall, no visible caster.", "empty room, eerie", ("shadow without source", "mysterious shadow", "wall shadow")),
    Motif("shadow_03", "shadow", "Multiple shadows one body", "One person casting several shadows.", "hard light, surreal", ("multiple shadows", "split shadow", "surreal shadow")),
    Motif("shadow_04", "shadow", "Shadow hand reaching", "Hand-shaped shadow reaching across surface.", "projected, stark", ("shadow hand", "reaching shadow", "hand shadow")),
    Motif("shadow_05", "shadow", "Silhouette in doorway", "Dark figure framed in lit doorway.", "backlight, threshold", ("silhouette doorway", "figure in door", "backlit figure")),
    Motif("shadow_06", "shadow", "Window blinds stripes", "Striped light across face from blinds.", "venetian, noir", ("blinds shadow", "venetian blinds light", "striped face")),
    Motif("shadow_07", "shadow", "Eclipse shadow", "Circular shadow falling over face/scene.", "partial, ominous", ("eclipse shadow", "circular shadow", "darkening")),
    Motif("shadow_08", "shadow", "Shadow growing", "Shadow slowly enlarging across wall.", "time-lapse feel, dread", ("growing shadow", "expanding shadow", "creeping shadow")),
    Motif("shadow_09", "shadow", "Two shadows merging", "Two shadows overlap and become one.", "abstract, symbolic", ("merging shadows", "shadow overlap", "unified shadow")),
    Motif("shadow_10", "shadow", "Shadow detached", "Person and shadow moving independently.", "surreal, eerie", ("detached shadow", "shadow separate", "independent shadow")),
    # ---- control ----
    Motif("control_01", "control", "Puppet strings", "Marionette strings attached to a person's limbs.", "overhead, theatrical", ("puppet strings", "marionette", "control strings")),
    Motif("control_02", "control", "Chess pieces", "Chess board mid-game, king isolated.", "top-down, dramatic", ("chess pieces", "chess game", "king isolated")),
    Motif("control_03", "control", "Mask collection", "Row of theatrical masks on wall.", "symmetrical, eerie", ("mask collection", "theater masks", "wall of masks")),
    Motif("control_04", "control", "Mirror reflection lie", "Mirror shows different expression than face.", "uncanny, split", ("mirror reflection", "lying mirror", "false reflection")),
    Motif("control_05", "control", "Labyrinth / maze", "Top-down maze, single figure trapped.", "geometric, cold", ("maze aerial", "labyrinth", "trapped figure")),
    Motif("control_06", "control", "Cage / bars", "Figure behind bars or cage, reaching out.", "high contrast, stark", ("cage bars", "behind bars", "imprisoned figure")),
    Motif("control_07", "control", "Strings of code", "Text/strings wrapping around a person.", "abstract, digital", ("code strings", "digital control", "data tendrils")),
    Motif("control_08", "control", "Throne / empty chair", "Empty ornate chair, spotlight.", "symmetrical, ominous", ("empty throne", "empty chair", "spotlight chair")),
    Motif("control_09", "control", "Crown / halo", "Crown or halo over figure, ambiguous meaning.", "iconic, stark", ("crown silhouette", "halo figure", "power symbol")),
    Motif("control_10", "control", "Web of lines", "Person at center of radiating lines/connections.", "network, abstract", ("web lines", "network diagram", "radiating lines")),
    # ---- space ----
    Motif("space_01", "space", "Interrogation room", "Bare room, single table, harsh overhead light.", "institutional, cold", ("interrogation room", "bare room", "harsh light room")),
    Motif("space_02", "space", "Long empty corridor", "One-point perspective corridor, distant figure.", "vanishing point, dread", ("long corridor", "empty hallway", "vanishing point")),
    Motif("space_03", "space", "Staircase from below", "Looking up a staircase, figure descending.", "low angle, dramatic", ("staircase below", "looking up stairs", "descending figure")),
    Motif("space_04", "space", "Window at night", "Lit window in dark building, silhouette inside.", "noir, urban", ("lit window night", "window silhouette", "dark building")),
    Motif("space_05", "space", "Empty classroom", "Desks in rows, one chair occupied.", "institutional, eerie", ("empty classroom", "desks rows", "school room")),
    Motif("space_06", "space", "Therapy couch", "Couch and chair, empty, clinical.", "soft, sterile", ("therapy room", "couch chair", "clinical office")),
    Motif("space_07", "space", "Observation room", "One-way mirror, figure behind glass.", "surveillance, stark", ("one way mirror", "observation room", "behind glass")),
    Motif("space_08", "space", "Basement stairs", "Dark stairs descending, light at top only.", "chiaroscuro, dread", ("basement stairs", "dark staircase", "descending stairs")),
    Motif("space_09", "space", "Rooftop edge", "Figure at edge of rooftop, city below.", "height, vertigo", ("rooftop edge", "city below", "high angle")),
    Motif("space_10", "space", "Empty stage", "Spotlight on empty stage, curtain half-drawn.", "theatrical, ominous", ("empty stage", "spotlight stage", "theater curtain")),
    # ---- text ----
    Motif("text_01", "text", "Single word on black", "One word in large type on black background.", "minimal, stark", ("text on black", "single word", "typography black")),
    Motif("text_02", "text", "Redacted document", "Black bars over text, partial words visible.", "document, noir", ("redacted text", "censored document", "black bars")),
    Motif("text_03", "text", "Handwritten note", "Scrawled note, uneven handwriting.", "paper, intimate", ("handwritten note", "scrawled text", "note paper")),
    Motif("text_04", "text", "Typewriter page", "Close-up of typewriter keys / typed page.", "vintage, mechanical", ("typewriter", "typed page", "vintage text")),
    Motif("text_05", "text", "Neon sign", "Neon sign glowing in darkness, single phrase.", "noir, urban", ("neon sign", "glowing text", "night sign")),
    Motif("text_06", "text", "Subliminal flash", "Text flashing for a single frame, barely visible.", "strobe, unsettling", ("subliminal text", "flash text", "hidden message")),
    Motif("text_07", "text", "Graffiti wall", "Spray-painted phrase on concrete wall.", "urban, raw", ("graffiti text", "wall writing", "spray paint")),
    Motif("text_08", "text", "Page of a diary", "Open diary page, intimate handwriting.", "warm, personal", ("diary page", "journal entry", "handwritten diary")),
    Motif("text_09", "text", "Legal document", "Contract/clause close-up, signature line.", "formal, cold", ("contract document", "legal paper", "signature line")),
    Motif("text_10", "text", "Whiteboard mind map", "Whiteboard with connected concepts and arrows.", "analytical, busy", ("mind map", "whiteboard diagram", "concept map")),
    # ---- surveillance ----
    Motif("surv_01", "surveillance", "CCTV grain", "Low-res CCTV frame, timestamp overlay.", "grainy, greenish", ("cctv footage", "security camera", "surveillance frame")),
    Motif("surv_02", "surveillance", "Thermal imaging", "Thermal/heat-map figure in darkness.", "false color, eerie", ("thermal imaging", "heat map person", "infrared figure")),
    Motif("surv_03", "surveillance", "Night vision green", "Green night-vision footage, grain.", "monochrome green, eerie", ("night vision", "green footage", "nvg person")),
    Motif("surv_04", "surveillance", "Phone screen glow", "Face lit only by phone screen in dark room.", "blue glow, intimate", ("phone glow face", "screen light face", "dark room phone")),
    Motif("surv_05", "surveillance", "Screen reflection", "Face reflected in monitor/TV screen.", "tech, cold", ("screen reflection face", "monitor reflection", "display reflection")),
    Motif("surv_06", "surveillance", "Recording light", "Red recording LED in darkness.", "minimal, ominous", ("recording light", "red led", "camera recording")),
    Motif("surv_07", "surveillance", "Blurred identity", "Face blurred/pixelated, unrecognizable.", "anonymized, stark", ("blurred face", "pixelated face", "anonymized person")),
    Motif("surv_08", "surveillance", "Zoomed-in detail", "Extreme digital zoom, artifacts, subject.", "pixelated, invasive", ("digital zoom", "zoomed face", "surveillance zoom")),
    Motif("surv_09", "surveillance", "Multiple monitors", "Wall of monitors showing different scenes.", "control room, cold", ("monitor wall", "control room screens", "surveillance monitors")),
    Motif("surv_10", "surveillance", "Drone POV", "Aerial drone view of a person from above.", "top-down, detached", ("drone view person", "aerial surveillance", "overhead figure")),
    # ---- abstract ----
    Motif("abstract_01", "abstract", "Neural network", "Glowing nodes and connections, brain-like.", "dark, bioluminescent", ("neural network", "brain connections", "synapse glow")),
    Motif("abstract_02", "abstract", "Fractured glass", "Cracked glass with distorted reflection.", "macro, sharp", ("cracked glass", "shattered mirror", "fractured reflection")),
    Motif("abstract_03", "abstract", "Ripple in water", "Single drop ripple, dark water.", "macro, slow", ("water ripple", "drop ripple", "dark water")),
    Motif("abstract_04", "abstract", "Smoke tendrils", "Smoke curling in still air, backlit.", "atmospheric, slow", ("smoke tendrils", "curling smoke", "backlit smoke")),
    Motif("abstract_05", "abstract", "Ink in water", "Ink blooming in water, dark clouds.", "macro, organic", ("ink in water", "ink bloom", "dark cloud water")),
    Motif("abstract_06", "abstract", "Pulse / heartbeat", "ECG line pulsing on dark background.", "medical, rhythmic", ("heartbeat line", "ecg pulse", "heart monitor")),
    Motif("abstract_07", "abstract", "Maze pattern", "Geometric maze pattern, hypnotic.", "repetitive, cold", ("maze pattern", "geometric maze", "hypnotic pattern")),
    Motif("abstract_08", "abstract", "Fractal", "Fractal zoom, infinite recursion.", "psychedelic, dark", ("fractal", "infinite zoom", "recursive pattern")),
    Motif("abstract_09", "abstract", "Static noise", "TV static / white noise filling frame.", "analog, unsettling", ("tv static", "white noise", "analog noise")),
    Motif("abstract_10", "abstract", "Void / black hole", "Pure darkness with subtle gradient, nothingness.", "minimal, existential", ("black void", "darkness gradient", "empty space")),
    # ---- weather ----
    Motif("weather_01", "weather", "Rain on glass", "Rain streaks on window, blurred city beyond.", "noir, melancholy", ("rain on glass", "rain window", "wet window")),
    Motif("weather_02", "weather", "Fog / mist", "Thick fog, figure barely visible.", "atmospheric, eerie", ("fog mist", "foggy figure", "misty scene")),
    Motif("weather_03", "weather", "Storm clouds", "Dark storm clouds, lightning flash.", "dramatic, ominous", ("storm clouds", "dark clouds", "lightning sky")),
    Motif("weather_04", "weather", "Snow / ash falling", "Slow falling particles, ambiguous.", "quiet, surreal", ("falling ash", "snow fall", "particles falling")),
    Motif("weather_05", "weather", "Wind through trees", "Trees bending, leaves scattering.", "kinetic, restless", ("wind trees", "bending trees", "scattering leaves")),
    Motif("weather_06", "weather", "Moonlit night", "Full moon over dark landscape.", "nocturnal, vast", ("moonlit night", "full moon", "night landscape")),
    Motif("weather_07", "weather", "Dusk / twilight", "Blue hour, fading light, silhouettes.", "transitional, moody", ("twilight", "blue hour", "dusk silhouette")),
    Motif("weather_08", "weather", "Flooded street", "Water covering street, reflections.", "surreal, still", ("flooded street", "water reflection", "submerged street")),
    Motif("weather_09", "weather", "Dust / particles", "Dust motes in a single light beam.", "macro, quiet", ("dust motes", "light beam dust", "particles air")),
    Motif("weather_10", "weather", "Heat shimmer", "Mirage-like distortion over hot surface.", "surreal, disorienting", ("heat shimmer", "mirage", "air distortion")),
    # ---- motion ----
    Motif("motion_01", "motion", "Slow zoom in", "Gradual push-in on a still subject.", "creeping, hypnotic", ("slow zoom", "push in", "creeping camera")),
    Motif("motion_02", "motion", "Whip pan", "Fast blur transition between two scenes.", "kinetic, disorienting", ("whip pan", "motion blur transition", "fast pan")),
    Motif("motion_03", "motion", "Glitch / distortion", "Digital glitch, RGB split, tearing.", "digital, unsettling", ("glitch effect", "digital distortion", "rgb split")),
    Motif("motion_04", "motion", "Reverse playback", "Action played in reverse, uncanny.", "surreal, dreamlike", ("reverse playback", "rewind motion", "backward action")),
    Motif("motion_05", "motion", "Time-lapse clouds", "Fast-moving clouds over still subject.", "temporal, vast", ("time lapse clouds", "fast clouds", "sky time lapse")),
    Motif("motion_06", "motion", "Strobe / flicker", "Light flickering, unstable illumination.", "anxious, harsh", ("flickering light", "strobe", "unstable light")),
    Motif("motion_07", "motion", "Parallax depth", "Layered depth, slow parallax movement.", "cinematic, immersive", ("parallax", "depth layers", "3d parallax")),
    Motif("motion_08", "motion", "Split-screen sync", "Two panels, actions mirroring.", "symmetrical, tense", ("split screen", "dual panel", "mirrored action")),
    Motif("motion_09", "motion", "Fade to black", "Slow fade, lingering last frame.", "ominous, final", ("fade to black", "slow fade", "lingering frame")),
    Motif("motion_10", "motion", "Freeze frame", "Motion stops mid-action, held.", "dramatic, arrested", ("freeze frame", "frozen moment", "still action")),
)


def by_category(cat: str) -> list[Motif]:
    return [m for m in MOTIFS if m.category == cat]


def by_id(mid: str) -> Motif | None:
    for m in MOTIFS:
        if m.id == mid:
            return m
    return None


def pick_varied(count: int, *, seed: int = 0) -> list[Motif]:
    """Pick ``count`` motifs with no two adjacent sharing a category."""
    import random

    rng = random.Random(seed)
    pool = list(MOTIFS)
    rng.shuffle(pool)
    chosen: list[Motif] = []
    used_ids: set[str] = set()
    last_cat: str | None = None
    # greedy: prefer motifs whose category differs from last
    remaining = pool[:]
    while len(chosen) < count and remaining:
        # find best candidate
        cand = None
        for m in remaining:
            if m.id in used_ids:
                continue
            if m.category == last_cat:
                continue
            cand = m
            break
        if cand is None:
            # relax adjacency constraint
            for m in remaining:
                if m.id not in used_ids:
                    cand = m
                    break
        if cand is None:
            break
        chosen.append(cand)
        used_ids.add(cand.id)
        last_cat = cand.category
        remaining = [m for m in remaining if m.id != cand.id]
    return chosen


assert len(MOTIFS) == 100, len(MOTIFS)
assert len(CATEGORIES) == 10
