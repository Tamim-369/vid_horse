#!/usr/bin/env python3
"""
Generate ALL psychology video motifs/scenes via Pollinations (Flux model).
Dark cinematic noir theme. Handles rate limits with exponential backoff retry.

Covers 15 categories x 20 motifs = 300 unique dark psychology visuals.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings, setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)

# Pollinations Config
POLLINATIONS_URL = "https://image.pollinations.ai/prompt"
MODEL = "flux"
WIDTH = 1080
HEIGHT = 1920

# Dark cinematic noir style suffix applied to ALL prompts
STYLE_SUFFIX = (
    "dark cinematic noir psychological thriller, 9:16 vertical, near-black "
    "gradient, heavy vignette, film grain, centered subject, high "
    "contrast, shallow depth of field, colored accent rim light "
    "(cyan/amber/magenta), photorealistic 8k, no text, no watermark, "
    "no logos, original composition, unsettling atmosphere"
)

# Rate limit handling
BASE_DELAY = 2.0
MAX_DELAY = 60.0
MAX_RETRIES = 10
TIMEOUT = 180.0

# COMPREHENSIVE PSYCHOLOGY MOTIFS (15 categories x 20 = 300)
PSYCHOLOGY_MOTIFS = {
    # 1. FACES & MICRO-EXPRESSIONS (20)
    "faces": [
        ("face_01", "Micro-expression freeze", "Extreme close-up of a face mid-micro-expression frozen in time, one eyebrow raised, corner of mouth twitching, eyes wide with suppressed emotion"),
        ("face_02", "Dead-eye stare", "Unblinking stare directly into lens, pupils dilated, no smile, hollow eyes reflecting nothing, clinical flat lighting"),
        ("face_03", "Half-shadow face (chiaroscuro)", "Face split by hard vertical shadow, one eye hidden in darkness, other eye illuminated, dramatic Rembrandt lighting"),
        ("face_04", "Smile not reaching eyes", "Polite social smile with cold empty eyes, crow's feet absent, mouth curved but eyes dead, uncanny valley"),
        ("face_05", "Tears without emotion", "Tears rolling down cheeks while face remains completely blank, no muscle movement in forehead or jaw, macro detail"),
        ("face_06", "Mouth covered / silenced", "Hand pressed firmly over mouth, eyes wide with panic, fingers digging into skin, high contrast noir"),
        ("face_07", "Mask peeling off", "Fingers peeling theatrical porcelain mask from face, revealing different expression underneath, motion blur"),
        ("face_08", "Reflection in eye macro", "Extreme macro of iris reflecting a distorted scene - a figure, a room, a memory - impossible detail"),
        ("face_09", "Young/old split face", "Vertical split down center: left side youthful smooth skin, right side aged wrinkles, same eyes"),
        ("face_10", "Faceless silhouette", "Head and shoulders silhouette against bright backlight, zero facial features, fog swirling around edges"),
        ("face_11", "Gaslighting double reflection", "Mirror shows two different expressions - face calm, reflection screaming, uncanny horror"),
        ("face_12", "Dissociative blank stare", "Eyes open but nobody home, thousand-yard stare, pupils slightly unequal, derealization"),
        ("face_13", "Manipulative charm smile", "Charismatic smile, head tilted, eyes calculating, dimples, too perfect, predator disguise"),
        ("face_14", "Paranoid hypervigilance", "Eyes darting in all directions, tense jaw, sweat at temple, scanning for threats that aren't there"),
        ("face_15", "Narcissistic self-admiration", "Face close to mirror/camera, pupils dilated with self-love, faint smirk, perfectionist grooming"),
        ("face_16", "Borderline emotional storm", "Face contorted mid-emotion shift - anger to despair to fear in one frame, raw intensity"),
        ("face_17", "Psychopathic flat affect", "Completely relaxed face, zero tension, eyes cold like shark, mouth neutral, terrifyingly calm"),
        ("face_18", "Trauma freeze response", "Body rigid, eyes locked on invisible threat, breath held, time stopped, dissociation"),
        ("face_19", "Love bombing intensity", "Overwhelming eye contact, face too close, pupils blown, intensity burning, manufactured intimacy"),
        ("face_20", "Identity fragmentation", "Multiple exposure: same face layered 3-4 times with different expressions, splitting apart"),
    ],

    # 2. BODY LANGUAGE & POWER DYNAMICS (20)
    "body": [
        ("body_01", "Dominant power posture", "Figure looming over seated person, legs apart, hands on hips, chin raised, low angle shot"),
        ("body_02", "Defensive closed posture", "Arms crossed tight, shoulders hunched to ears, head tucked, turned away, protective"),
        ("body_03", "Space invasion", "Dominant figure leaning into personal space, subordinate recoiling backward, tension visible"),
        ("body_04", "Controlling hand on shoulder", "Hand gripping shoulder from behind, fingers pressing into trapezius, ownership gesture"),
        ("body_05", "Accusatory pointing", "Aggressive finger jab toward camera/viewer, other hand on hip, freeze frame intensity"),
        ("body_06", "Mirroring rapport", "Two figures in identical posture - legs crossed same, heads tilted same, one microsecond ahead"),
        ("body_07", "Steepled confidence", "Fingertips touching in pyramid, elbows wide, leaning back, evaluative judgment"),
        ("body_08", "Pacing anxiety", "Profile silhouette pacing, long shadow stretching, hands behind back or wringing"),
        ("body_09", "Submission kneeling", "Figure on knees looking up, hands visible palms up, other figure towering above"),
        ("body_10", "Restrained wrists", "Wrists bound with rope/zip ties/handcuffs, tension in arms, struggle marks on skin"),
        ("body_11", "Grooming manipulation", "One figure fixing other's collar/hair/tie, intimate invasion disguised as care"),
        ("body_12", "Territorial manspreading", "Seated figure taking maximum space, legs wide, arms on adjacent chairs, dominance"),
        ("body_13", "Self-soothing hug", "Arms wrapped around own torso, fingers digging into ribs, rocking slightly, comfort seeking"),
        ("body_14", "Aggressive finger gun", "Thumb up, index extended, 'shooting' at target, playful menace, power play"),
        ("body_15", "Submissive minimization", "Shoulders rounded, neck retracted, making self small, invisible, apology posture"),
        ("body_16", "Predatory stillness", "Zero micro-movements, breathing invisible, coiled spring, about to strike"),
        ("body_17", "Trauma protective curl", "Fetal position on side, knees to chest, arms around head, back to world"),
        ("body_18", "Charismatic open palms", "Hands wide, palms up, 'trust me' gesture, shoulders back, manufactured warmth"),
        ("body_19", "Gaslighting head tilt", "Head cocked 45 degrees, concerned expression, 'are you okay?' weaponized empathy"),
        ("body_20", "Dissociative floating", "Body present but weightless, feet not quite touching ground, disconnected from gravity"),
    ],

    # 3. SHADOWS & SILHOUETTES (20)
    "shadows": [
        ("shadow_01", "Long shadow pursuit", "Figure walking, shadow stretching impossibly long ahead, chasing the shadow self"),
        ("shadow_02", "Shadow without source", "Sharp human shadow on blank wall, no figure casting it, empty room, supernatural"),
        ("shadow_03", "Multiple shadows one body", "Single figure casting 3-4 shadows in different directions, fractured self"),
        ("shadow_04", "Shadow hand reaching", "Disembodied shadow hand stretching across floor/wall toward vulnerable figure"),
        ("shadow_05", "Doorway silhouette", "Dark figure filling doorway, backlit by harsh corridor light, threshold guardian"),
        ("shadow_06", "Venetian blind stripes", "Face/body sliced by horizontal light stripes through blinds, prison bars pattern"),
        ("shadow_07", "Eclipse shadow", "Circular shadow sweeping across face, progressive darkening, cosmic alignment"),
        ("shadow_08", "Growing shadow", "Time-lapse: shadow expanding across wall while figure stays same size, creeping dread"),
        ("shadow_09", "Merging shadows", "Two shadows overlapping until indistinguishable, enmeshment, loss of boundaries"),
        ("shadow_10", "Detached shadow", "Figure walking one way, shadow moving opposite, independence, betrayal"),
        ("shadow_11", "Shadow puppet mastery", "Hands forming animal shapes on wall, but shadows move on their own, sentient"),
        ("shadow_12", "Inner demon shadow", "Profile shadow shows horns/claws/wings while figure looks normal, true nature"),
        ("shadow_13", "Shadow in mirror wrong", "Reflection's shadow doesn't match body position, reality glitch"),
        ("shadow_14", "Cage shadow bars", "Shadow of window blinds falls like prison bars across body, trapped"),
        ("shadow_15", "Shadow consumption", "Darkness rising from feet upward, swallowing figure, depression visualization"),
        ("shadow_16", "Protective shadow", "Larger shadow looming over smaller figure, guardian or predator ambiguous"),
        ("shadow_17", "Shadow dissociation", "Shadow stepping out of feet, becoming 3D, walking away, abandonment"),
        ("shadow_18", "Projective identification", "Figure's shadow acting out emotions face suppresses - screaming, attacking"),
        ("shadow_19", "Shadow timeline", "Multiple shadows at different ages trailing behind walking figure, past selves"),
        ("shadow_20", "Void shadow", "Shadow is pure nothingness - light passes through, absence where presence should be"),
    ],

    # 4. CONTROL & MANIPULATION (20)
    "control": [
        ("control_01", "Puppet strings", "Marionette strings from above attached to limbs, controller hands invisible, helpless dance"),
        ("control_02", "Chess king isolated", "Ornate chessboard, king alone in center, all other pieces surrounding, checkmate inevitable"),
        ("control_03", "Mask collection wall", "Dozens of theatrical masks on wall - comedy, tragedy, animals, blank - identity wardrobe"),
        ("control_04", "Lying mirror", "Mirror reflection shows different expression/age/gender than person standing before it"),
        ("control_05", "Labyrinth trap", "Top-down maze, single tiny figure at center, walls closing in, no exit visible"),
        ("control_06", "Cage behind bars", "Figure gripping bars, face pressed to metal, reaching toward freedom just inches away"),
        ("control_07", "Digital code tendrils", "Green matrix code wrapping around body like vines, digital enslavement, modern control"),
        ("control_08", "Empty throne spotlight", "Ornate empty chair under single spotlight, dust motes dancing, power vacuum"),
        ("control_09", "Crown/halo ambiguity", "Golden crown or glowing halo hovering above head - blessing or burden unclear"),
        ("control_10", "Web of control", "Figure at center of glowing network lines connecting to dozens of puppets, spider"),
        ("control_11", "Gaslighting reality warp", "Room bending - walls melting, floor tilting, objects shifting - perception attack"),
        ("control_12", "Love bomb overwhelm", "Figure buried under avalanche of roses/gifts/letters/phones, drowning in 'affection'"),
        ("control_13", "Triangulation triangle", "Three figures, manipulator in center pulling strings to other two, divide conquer"),
        ("control_14", "Foot in door wedge", "Door barely open, foot blocking close, persistent pressure, incremental invasion"),
        ("control_15", "Door in face retreat", "Heavy door slamming, then smaller door opening, reciprocal concession psychology"),
        ("control_16", "Lowball changing terms", "Contract morphing - text rewriting itself, numbers shifting, fine print expanding"),
        ("control_17", "Authority uniform close", "Epaulettes, badges, medals, starched collar - symbol power not person"),
        ("control_18", "Milgram shock panel", "Voltage dials 15-450, labeled 'Slight Shock' to 'XXX', finger hovering"),
        ("control_19", "Hypnotic spiral", "Black-white spiral expanding from eyes outward, consciousness narrowing, trance"),
        ("control_20", "Invisible leash", "Figure walking freely but tension line visible from neck to unseen handler, subtle"),
    ],

    # 5. SPACES & ENVIRONMENTS (20)
    "spaces": [
        ("space_01", "Interrogation room", "Bare concrete, single metal table, two chairs, harsh overhead fluorescent, one-way mirror"),
        ("space_02", "Infinite corridor", "One-point perspective hallway stretching forever, doors on both sides, vanishing point"),
        ("space_03", "Staircase from below", "Looking up steep stairs, figure descending in silhouette, power position above"),
        ("space_04", "Window at night", "Single lit window in dark brutalist building, silhouette pacing inside, urban isolation"),
        ("space_05", "Empty classroom", "Rows of identical desks, one occupied, chalkboard with indecipherable writing, conformity"),
        ("space_06", "Therapy office", "Leather couch, empty chair, tissues, clock, soft light, clinical intimacy, safe container"),
        ("space_07", "Observation room", "Dark room, huge one-way mirror, figure on other side unaware, voyeuristic power"),
        ("space_08", "Basement stairs", "Steep narrow stairs descending into darkness, single bulb at top, childhood fear"),
        ("space_09", "Rooftop edge", "Concrete parapet, city lights far below, wind tearing clothes, vertigo invitation"),
        ("space_10", "Empty stage", "Velvet curtains half-drawn, single spotlight on center, waiting for performance"),
        ("space_11", "Childhood bedroom", "Twin bed, posters peeling, toys frozen in time, past preserved under dust"),
        ("space_12", "Hospital corridor", "Institutional green, fluorescent buzz, gurney wheels squeaking, sterile suffering"),
        ("space_13", "Confessional booth", "Dark wood screen, grate between, kneeler, absolute secrecy, absolute judgment"),
        ("space_14", "Panopticon center", "Circular prison, guard tower center, all cells visible, self-policing internalized"),
        ("space_15", "Abandoned asylum", "Peeling paint, rusted beds, restraint marks on walls, forgotten suffering"),
        ("space_16", "Luxury prison", "Gilded cage - marble, crystal, silk - but windows barred, golden handcuffs"),
        ("space_17", "Memory palace", "Impossible architecture - rooms within rooms, Escher stairs, mind's labyrinth"),
        ("space_18", "Safe room / panic room", "Steel walls, monitors, supplies, locked from inside, paranoia realized"),
        ("space_19", "Cult compound", "Identical white buildings, communal dining, watchtower, surrendered autonomy"),
        ("space_20", "Digital void", "Pure black, floating UI elements, notifications, algorithms watching, no escape"),
    ],

    # 6. TEXT & TYPOGRAPHY (20)
    "text": [
        ("text_01", "Single word on black", "One word - 'WHY' or 'LIAR' or 'WAKE UP' - huge white typography, stark minimal"),
        ("text_02", "Redacted document", "Classified pages, thick black marker bars, only 'THE' and 'AND' visible, truth hidden"),
        ("text_03", "Handwritten desperation", "Torn notebook page, frantic scrawl, pressed hard, ink bleeding, 'help me' repeated"),
        ("text_04", "Typewriter confession", "Vintage keys, paper with single sentence typed over and over, obsession loop"),
        ("text_05", "Neon truth sign", "Pink/blue neon in rain-slick alley: 'YOU ARE ENOUGH' flickering, ironic comfort"),
        ("text_06", "Subliminal flash", "Single frame text 'OBEY' 'CONSUME' 'SLEEP' - barely perceptible, programming"),
        ("text_07", "Graffiti confession", "Brick wall: 'I WAS NEVER REAL' in spray paint, raw, public secret"),
        ("text_08", "Diary locked", "Open leather journal, intimate cursive, last page torn out, key hanging broken"),
        ("text_09", "Legal contract fine print", "Magnified signature line, microscopic text selling soul, 'initial here' arrows"),
        ("text_10", "Mind map conspiracy", "Whiteboard: photos, red string, arrows, 'CONNECTED' centered, paranoid pattern"),
        ("text_11", "Affirmation mirror", "Lipstick on bathroom mirror: 'BEAUTIFUL' crossed out, 'WORTHLESS' written over"),
        ("text_12", "Anonymous threat", "Cut-out magazine letters on paper: 'WE KNOW WHAT YOU DID' ransom note"),
        ("text_13", "Gaslighting journal", "Two handwritings in same journal arguing: 'That never happened' / 'Yes it did'"),
        ("text_14", "Cult manifesto", "Photocopied pages, highlighted passages, 'THE LEADER IS NEVER WRONG' circled"),
        ("text_15", "Suicide note draft", "Crumpled paper, multiple starts, 'I'm sorry' 'I'm tired' 'Goodbye' - never sent"),
        ("text_16", "Autopilot script", "Teleprompter text reflecting in eyes: 'SMILE. AGREE. BUY. OBEY. REPEAT.'"),
        ("text_17", "Redacted memory", "Childhood photo with face scratched out, 'FORGOTTEN' written in red pen"),
        ("text_18", "Inner critic loop", "Typewriter paper scrolling endlessly: 'NOT ENOUGH NOT ENOUGH NOT ENOUGH'"),
        ("text_19", "Truth serum confession", "Handwritten on transparent overlay: 'I DON'T KNOW WHO I AM ANYMORE'"),
        ("text_20", "Freedom declaration", "Paper airplane flying from window, 'I CHOOSE ME' written on wing, liberation"),
    ],

    # 7. SURVEILLANCE & TECHNOLOGY (20)
    "surveillance": [
        ("surv_01", "CCTV grain", "Green-tinted low-res footage, timestamp 03:47:12, figure unaware, watched"),
        ("surv_02", "Thermal imaging", "False-color heat map: human shape glowing orange/red against cold blue background"),
        ("surv_03", "Night vision", "Monochrome green, grainy, eyes glowing white (IR reflection), predator vision"),
        ("surv_04", "Phone screen glow", "Face illuminated only by blue light in pitch dark, doom-scrolling, zombie"),
        ("surv_05", "Screen reflection", "Monitor reflection shows different content than face expresses, dual reality"),
        ("surv_06", "Recording LED", "Tiny red dot in darkness, blinking steadily, 'REC' - panopticon internalized"),
        ("surv_07", "Blurred identity", "Face pixelated/blurred by algorithm, anonymized but body language exposed"),
        ("surv_08", "Digital zoom enhance", "Extreme crop, JPEG artifacts, 'enhance' revealing nothing, false certainty"),
        ("surv_09", "Monitor wall", "Security room: 24 screens showing every angle, one guard, god complex"),
        ("surv_10", "Drone POV", "Top-down aerial, figure tiny in courtyard, crosshairs overlay, remote power"),
        ("surv_11", "Smart home eye", "Camera lens in smoke detector, wide angle, always recording, convenience trap"),
        ("surv_12", "Biometric scan", "Iris close-up, laser grid mapping, identity verified, soul catalogued"),
        ("surv_13", "Social credit display", "Floating AR numbers above heads: 420, 780, 150 - worth calculated"),
        ("surv_14", "Keystroke logger", "Fingers typing, ghost text appearing behind: 'password123' 'I hate him' 'help'"),
        ("surv_15", "Location history", "Map with glowing trail: home work lover's bar home - life tracked"),
        ("surv_16", "Predictive policing", "Holographic red zones on city map: 'CRIME PROBABILITY 94%' - prejudice coded"),
        ("surv_17", "Emotion recognition", "Face with algorithmic overlay: 'ANXIETY 87%' 'DECEPTION 12%' - feelings data"),
        ("surv_18", "Brainwave monitor", "EEG cap, squiggly lines, 'THOUGHT DETECTED: REBELLION' - mind read"),
        ("surv_19", "Digital twin", "Holographic clone standing beside original, perfect prediction, replacement ready"),
        ("surv_20", "Algorithm god", "Server room infinite, blinking lights, one cursor blinking: 'EXECUTE: TERMINATE'"),
    ],

    # 8. ABSTRACT & METAPHORICAL (20)
    "abstract": [
        ("abstract_01", "Neural network glow", "Dark brain silhouette, synapses firing gold/blue, consciousness emerging"),
        ("abstract_02", "Fractured mirror", "Shattered glass, each fragment showing different version of same face"),
        ("abstract_03", "Ripple in dark water", "Single drop impact, concentric rings expanding, distortion spreading"),
        ("abstract_04", "Smoke tendrils", "White smoke curling in black void, shaped by invisible forces, ephemeral"),
        ("abstract_05", "Ink bloom in water", "Black ink dropped, forming clouds/tendrils/creatures, Rorschach chaos"),
        ("abstract_06", "Heartbeat flatline", "ECG line on black: steady rhythm spike flatline new rhythm, transformation"),
        ("abstract_07", "Maze pattern", "Geometric labyrinth, no entrance/exit, hypnotic repetition, trapped logic"),
        ("abstract_08", "Fractal zoom", "Mandelbrot infinite recursion, self-similarity at every scale, pattern obsession"),
        ("abstract_09", "TV static", "Analog noise filling frame, hidden images in chaos, pareidolia, meaning from noise"),
        ("abstract_10", "Void gradient", "Pure black center fading to dark gray edges, existential nothingness, peaceful"),
        ("abstract_11", "Puppet strings geometry", "Invisible strings as golden geometry, sacred mathematics of control"),
        ("abstract_12", "Cognitive dissonance", "Two incompatible shapes forced together, tension lines, stress fractures"),
        ("abstract_13", "Repressed memory", "Locked box glowing, cracks forming, light leaking, inevitable emergence"),
        ("abstract_14", "Ego death", "Human silhouette dissolving into particles/light/birds, transcendence or annihilation"),
        ("abstract_15", "Shadow integration", "Dark figure embraced by light figure, merging, wholeness, Jungian gold"),
        ("abstract_16", "Gaslight fog", "Reality dissolving at edges, objects becoming suggestions, ground uncertain"),
        ("abstract_17", "Trauma vortex", "Spiral pulling everything inward - memories, self, time - event horizon"),
        ("abstract_18", "Narcissistic mirror", "Infinite reflection corridor, each self more perfect, none real"),
        ("abstract_19", "Borderline splitting", "Figure torn in half - angel/devil, saint/sinner - no middle ground"),
        ("abstract_20", "Healing spiral", "Upward spiral from darkness to light, not linear, cycling but ascending"),
    ],

    # 9. WEATHER & ATMOSPHERE (20)
    "weather": [
        ("weather_01", "Rain on window", "Streaks blurring city lights, face pressed to glass, tears or rain indistinguishable"),
        ("weather_02", "Thick fog", "Figure emerging/disappearing in whiteout, only eyes visible, isolation"),
        ("weather_03", "Storm clouds", "Supercell anvil, lightning illuminating from within, power uncontained"),
        ("weather_04", "Ash/snow falling", "Gray particles slow in dark air, ambiguous - volcanic winter or nuclear"),
        ("weather_05", "Wind through trees", "Branches thrashing, leaves stripped, resistance futile, nature's violence"),
        ("weather_06", "Moonlit landscape", "Silver light on dark water/fields, beauty in desolation, cold clarity"),
        ("weather_07", "Twilight blue hour", "Indigo sky, last warmth fading, liminal space, day dying night born"),
        ("weather_08", "Flooded reflection", "Submerged street, perfect mirror of buildings above, world inverted"),
        ("weather_09", "Dust motes in beam", "Single shaft light in dark room, dancing particles, microscopic universe"),
        ("weather_10", "Heat shimmer", "Horizon distorting, mirage rising, reality melting, perception unreliable"),
        ("weather_11", "Thunderhead anvil", "Massive cumulonimbus, mammatus pouches hanging, apocalypse aesthetic"),
        ("weather_12", "Freezing rain", "Everything encased in clear ice, beautiful deadly, touch shatters"),
        ("weather_13", "Sandstorm wall", "Orange-brown wall advancing, swallowing everything, erasure"),
        ("weather_14", "Aurora borealis", "Green/purple curtains dancing, magnetic field visible, cosmic radiation"),
        ("weather_15", "Acid rain", "Drops sizzling on contact, steam rising, invisible poison, corrupted nature"),
        ("weather_16", "Polar night", "Perpetual darkness, only stars/aurora, time meaningless, endurance test"),
        ("weather_17", "Microburst", "Sudden downdraft flattening everything, invisible violence from above"),
        ("weather_18", "Volcanic lightning", "Ash plume generating own lightning, earth's rage, creation/destruction"),
        ("weather_19", "Climate collapse", "Surreal weather - snow in desert, hurricane inland - new normal"),
        ("weather_20", "Calm eye", "Perfect stillness in hurricane center, blue sky above, temporary reprieve"),
    ],

    # 10. MOTION & CINEMATIC TECHNIQUES (20)
    "motion": [
        ("motion_01", "Slow push-in", "Gradual dolly zoom on frozen subject, creeping intimacy, inevitable revelation"),
        ("motion_02", "Whip pan", "Violent blur between scenes, disorientation, trauma jump cut"),
        ("motion_03", "Glitch distortion", "RGB split, datamoshing, tearing - reality corrupting, simulation breaking"),
        ("motion_04", "Reverse playback", "Action un-happening: glass un-breaking, words un-spoken, regret fantasy"),
        ("motion_05", "Time-lapse clouds", "Centuries in seconds, clouds boiling, shadows racing, human time irrelevant"),
        ("motion_06", "Strobe flicker", "Unstable light, frozen moments, seizure risk, interrogation technique"),
        ("motion_07", "Parallax depth", "Foreground/background moving at different speeds, 3D immersion, layers"),
        ("motion_08", "Split screen sync", "Two panels mirroring - same action different outcomes - choice visualization"),
        ("motion_09", "Fade to black", "Slow dissolve, last frame lingering, 'is it over?' uncertainty"),
        ("motion_10", "Freeze frame", "Mid-action halt - punch suspended, tear mid-fall - moment suspended forever"),
        ("motion_11", "Dolly zoom (vertigo)", "Background stretches while subject stays - disorientation, realization"),
        ("motion_12", "Long take unbroken", "Minutes without cut - raw reality, no escape from witnessing"),
        ("motion_13", "Jump cuts anxiety", "Same action repeated with micro-changes - OCD loop, perfectionism"),
        ("motion_14", "Slow motion impact", "Bullet/slap/word hitting - 1000fps - destruction in detail"),
        ("motion_15", "POV subjective", "Camera IS the eyes - breathing, blinking, shaking - embodied terror"),
        ("motion_16", "Zolly reverse", "Push-pull simultaneously - background rushes away - dissociation"),
        ("motion_17", "Frame within frame", "Doorways, windows, mirrors framing subject - trapped in layers"),
        ("motion_18", "Match cut metaphor", "Eye closes eye opens different person/place - identity fluid"),
        ("motion_19", "Infinite zoom", "Eye pupil retina neuron atom universe - scale meaningless"),
        ("motion_20", "Credits roll up", "Names scrolling over final shot - 'Gaslighter' 'Enabler' 'Victim' 'Survivor'"),
    ],

    # 11. ARCHETYPES & FIGURES (20)
    "archetypes": [
        ("archetype_01", "Beautiful woman mirror", "Stunning woman in vintage mirror, but reflection shows rotting corpse - vanitas"),
        ("archetype_02", "Machiavellian man staring", "Sharp suit, cold eyes, slight smirk, head tilted, calculating your value"),
        ("archetype_03", "Wounded healer", "Androgynous figure, hands glowing over another's wound, own chest bleeding"),
        ("archetype_04", "Trickster shapeshifter", "Face morphing between 4 expressions simultaneously, never settling"),
        ("archetype_05", "Tyrannical father", "Large silhouette in doorway, belt in hand, authority absolute, fear worshipped"),
        ("archetype_06", "Devouring mother", "Embrace becoming trap, arms multiplying, 'I only want what's best' suffocating"),
        ("archetype_07", "Golden child", "Perfect glowing figure on pedestal, cracks visible beneath gold leaf, fragile"),
        ("archetype_08", "Scapegoat", "Figure covered in projected shadows - 'lazy' 'crazy' 'ungrateful' - carrying all"),
        ("archetype_09", "Lost child", "Small figure in oversized adult clothes, invisible, hiding in plain sight"),
        ("archetype_10", "Mascot/clown", "Painted smile, tears underneath, performing for survival, 'everything's fine'"),
        ("archetype_11", "Wise old crone", "Ancient woman stirring pot, steam forming visions, knows your true name"),
        ("archetype_12", "Dark magician", "Hands weaving threads of fate, audience unaware they're the puppets"),
        ("archetype_13", "Innocent fool", "Walking off cliff eyes closed, smile trusting, universe catches or doesn't"),
        ("archetype_14", "Warrior shadow", "Battle-scarred, armor dented, fighting internal dragons, exhausted noble"),
        ("archetype_15", "Siren/mermaid", "Half in water, song visible as golden waves, love equals drowning"),
        ("archetype_16", "Hermit lantern", "Alone on mountain, light held high, not for others - for own path"),
        ("archetype_17", "Justice blindfolded", "Scales tipped by gold coins, sword rusted, peek through blindfold"),
        ("archetype_18", "Death as therapist", "Skeleton in chair taking notes, 'and how did that make you feel?'"),
        ("archetype_19", "Inner child locked", "Small figure behind barred window in adult's chest, knocking, muffled"),
        ("archetype_20", "Integrated self", "All archetypes in circle around center figure, council, wholeness achieved"),
    ],

    # 12. RELATIONSHIP DYNAMICS (20)
    "relationships": [
        ("rel_01", "Trauma bond chain", "Two figures chained together, walking into fire, 'we survive together'"),
        ("rel_02", "Push-pull dance", "One reaching, one retreating, then reversing, never meeting, dizzying"),
        ("rel_03", "Enmeshment vines", "Two bodies overgrown with same vines, roots tangled, where one ends unclear"),
        ("rel_04", "Pedestal fall", "Figure on pedestal crumbling, other catching or watching, idealization shatters"),
        ("rel_05", "Mirror neurons", "Two faces perfectly synced - micro-expressions identical - eerie resonance"),
        ("rel_06", "Power struggle", "Tug of war with invisible rope, both exhausted, neither releasing"),
        ("rel_07", "Caregiver exhaustion", "One figure carrying other (adult size), knees buckling, 'I can fix them'"),
        ("rel_08", "Narcissist supply", "Parasitic tendrils from hollow chest feeding on vibrant figure's light"),
        ("rel_09", "Codependent lock", "Two padlocks interlocked, no keys, each holds other's freedom"),
        ("rel_10", "Boundaries fortress", "One behind high walls, other throwing flowers over - never reaching"),
        ("rel_11", "Gaslighting fog", "One holding flashlight under other's chin, 'you're imagining the light'"),
        ("rel_12", "Love bombing shower", "One buried in rose petals/letters/phones, smiling predator above"),
        ("rel_13", "Triangulation triangle", "Three figures - manipulator center pulling strings - divide conquer"),
        ("rel_14", "Hoovering vacuum", "Figure walking away, massive suction pulling back - 'I changed'"),
        ("rel_15", "Discard phase", "Figure crumpled like used paper, new supply already being groomed"),
        ("rel_16", "Flying monkeys", "Swarm of masked figures attacking target on narcissist's command"),
        ("rel_17", "Gray rock defense", "Figure as featureless stone, attacker's tendrils sliding off, boredom"),
        ("rel_18", "No contact wall", "Concrete barrier dropping between figures, silence, peace, grief"),
        ("rel_19", "Reconciliation trap", "Arms open, 'I'm sorry' - but hands hidden behind back holding knife"),
        ("rel_20", "Healthy bond", "Two separate trees, roots touching gently, space between, choosing together"),
    ],

    # 13. HEALING & RECOVERY (20)
    "healing": [
        ("heal_01", "Kintsugi repair", "Broken gold-veined ceramic bowl - 'more beautiful for having been broken'"),
        ("heal_02", "Shadow integration", "Dark figure embraced by light figure, merging into wholeness, Jungian gold"),
        ("heal_03", "Inner child rescue", "Adult hands reaching through bars to small self, 'I'm here now, you're safe'"),
        ("heal_04", "Boundary fence", "White picket fence going up - 'this far, no further' - peaceful not hostile"),
        ("heal_05", "Voice returning", "Throat glowing, sound waves visible, first 'NO' spoken, power reclaimed"),
        ("heal_06", "Memory reconsolidation", "Painful memory burning, reforming gold - same event, new meaning"),
        ("heal_07", "Body homecoming", "Figure settling into own skin, shoulders dropping, breath deepening, arrival"),
        ("heal_08", "Compassion mirror", "Looking in mirror with kindness instead of criticism - first time"),
        ("heal_09", "Grief wave", "Figure on shore, wave crashes, stands ground, 'I survive this feeling'"),
        ("heal_10", "Parts integration", "Fragmented selves at round table - protector, critic, child - all heard"),
        ("heal_11", "Nervous system reset", "Vagus nerve glowing, heart rate slowing, safety signal received, ventral vagal"),
        ("heal_12", "Forgiveness release", "Heavy stone dropped from hands into water, ripples, lightness"),
        ("heal_13", "Self-trust anchor", "Hand on own heart, 'I believe you' - internal attachment formed"),
        ("heal_14", "New neural path", "Brain lighting new golden pathway beside old red trauma highway"),
        ("heal_15", "Post-traumatic growth", "Tree growing through boulder - 'impossible' made possible"),
        ("heal_16", "Community circle", "Diverse figures in circle, each holding piece of puzzle - belonging"),
        ("heal_17", "Creative expression", "Paint-stained hands, canvas showing chaos becoming art - sublimation"),
        ("heal_18", "Nature grounding", "Bare feet on earth, roots growing down, regulation through connection"),
        ("heal_19", "Nightmare mastery", "Dreamer lucid in nightmare, turning monster into guide - dream alchemy"),
        ("heal_20", "Wisdom earned", "Older self handing lantern to younger - 'you'll survive this too'"),
    ],

    # 14. DARK PSYCHOLOGY CONCEPTS (20)
    "concepts": [
        ("concept_01", "Cognitive dissonance", "Two gears jammed - 'I'm good' vs 'I did bad' - smoke from friction"),
        ("concept_02", "Confirmation bias", "Figure only seeing evidence supporting belief, blinders on, reality filtered"),
        ("concept_03", "Sunk cost fallacy", "Digging deeper in wrong hole, 'already invested so much' - shovel glowing"),
        ("concept_04", "Fundamental attribution", "Own mistake: 'circumstances' / Other's mistake: 'character' - double scale"),
        ("concept_05", "Halo effect", "Attractive figure glowing, 'good' words floating - intelligence/kindness assumed"),
        ("concept_06", "Anchoring bias", "First number (1000) glows, all subsequent judgments orbit it - gravity"),
        ("concept_07", "Availability heuristic", "Vivid memories (plane crash) loom large, statistics (car safety) invisible"),
        ("concept_08", "Dunning-Kruger peak", "Mount stupid: confidence high, competence low - valley of despair ahead"),
        ("concept_09", "Bystander effect", "Crowd watching figure fall, each thinking 'someone else will help' - diffusion"),
        ("concept_10", "Stockholm syndrome", "Hostage defending captor, chains seen as jewelry - survival bonding"),
        ("concept_11", "Learned helplessness", "Dog in shuttle box not jumping barrier - door open but not tried"),
        ("concept_12", "Reactance", "Forbidden fruit glowing, 'don't push' sign - finger hovering, freedom threatened"),
        ("concept_13", "Mere exposure", "Face repeating - 1x neutral, 5x familiar, 20x liked - repetition breeds comfort"),
        ("concept_14", "Foot-in-door", "Small yes (sign petition) large yes (donate kidney) - commitment ladder"),
        ("concept_15", "Door-in-face", "Huge request rejected small request accepted - reciprocal concession"),
        ("concept_16", "Lowball technique", "Agreed price 'mistake' price doubled still buy - commitment sunk"),
        ("concept_17", "Social proof", "Empty restaurant vs full - herd following herd - 'if others...'"),
        ("concept_18", "Authority obedience", "White coat says 'continue' - hand on shock lever - 65% comply"),
        ("concept_19", "Scarcity urgency", "Countdown timer 00:03, 'only 2 left!' - panic buying - FOMO triggered"),
        ("concept_20", "Reciprocity trap", "Free gift obligation compliance - 'I gave, you owe' - ancient wiring"),
    ],

    # 15. SURREAL & SYMBOLIC (20)
    "surreal": [
        ("surreal_01", "Brain as control room", "Tiny operators at consoles pulling levers labeled 'FEAR' 'DESIRE' 'ANGER'"),
        ("surreal_02", "Heart in cage", "Anatomical heart in birdcage, singing beautifully, door open but won't leave"),
        ("surreal_03", "Eyes on stalks", "Figure with eyes on long stalks watching own back - hypervigilance embodied"),
        ("surreal_04", "Mouth sewn shut", "Lips stitched with black thread, one stitch broken - 'I have things to say'"),
        ("surreal_05", "Head in cloud", "Literal cloud where head should be - thoughts weather, uncontrollable"),
        ("surreal_06", "Backpack of stones", "Figure carrying stones labeled '1998' '2003' '2011' - trauma weight"),
        ("surreal_07", "Mask eating face", "Porcelain mask fusing with skin, becoming new face - identity consumed"),
        ("surreal_08", "Puppeteer's hands", "Giant hands from clouds manipulating tiny world - but strings go UP further"),
        ("surreal_09", "Clockwork heart", "Open chest reveals gears - winding down - key missing - time borrowed"),
        ("surreal_10", "Mirror maze infinite", "Infinite reflections - each slightly different - which is real?"),
        ("surreal_11", "Thought police", "Small uniformed figures in brain arresting 'UNAPPROVED THOUGHT' - censorship"),
        ("surreal_12", "Emotion chemistry", "Test tubes: 'LOVE' (pink) 'RAGE' (red) 'SHAME' (black) - mixing dangerous"),
        ("surreal_13", "Child driving car", "Adult body, child hands on wheel - 'I don't know how!' - regression"),
        ("surreal_14", "Shadow therapist", "Patient on couch, shadow on wall doing therapy - unconscious healing"),
        ("surreal_15", "Memory library burning", "Bookshelves of memories aflame - 'forgetting is survival' - librarian weeping"),
        ("surreal_16", "Identity wardrobe", "Racks of person-suits - 'confident' 'victim' 'hero' - choosing daily costume"),
        ("surreal_17", "Gaslight architecture", "MC Escher house - stairs to ceiling, doors to walls - reality unstable"),
        ("surreal_18", "Trauma time frozen", "Clock melted (Dali) at 3:17 - 'it's always 3:17 in here' - stuck"),
        ("surreal_19", "Soul retrieval", "Shaman figure catching glowing fragment from wind - 'you lost this'"),
        ("surreal_20", "Phoenix ash", "Figure burning, ash forming new figure - not rebirth - TRANSFORMATION - different"),
    ],
}

# Generation Functions

def build_prompt(motif_name: str, description: str) -> str:
    """Build Pollinations prompt with consistent dark style."""
    return f"{description}, {motif_name}. {STYLE_SUFFIX}"


def generate_image(
    client: httpx.Client,
    api_key: str,
    category: str,
    motif_id: str,
    motif_name: str,
    description: str,
    out_dir: Path,
    seed: int | None = None,
) -> bool:
    """Generate single image with retry logic."""
    if seed is None:
        seed = random.randint(1, 10_000_000)

    prompt = build_prompt(motif_name, description)
    encoded = urllib.parse.quote(prompt)
    url = (
        f"{POLLINATIONS_URL}/{encoded}"
        f"?width={WIDTH}&height={HEIGHT}"
        f"&model={MODEL}&nologo=true&seed={seed}&key={api_key}"
    )

    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    out_path = category_dir / f"{motif_id}.png"

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("  Exists: %s/%s.png", category, motif_id)
        return True

    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, timeout=TIMEOUT, follow_redirects=True)
            if resp.status_code == 429:
                raise httpx.HTTPStatusError("Rate limited", request=resp.request, response=resp)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            logger.info("  Generated: %s/%s.png (seed=%d)", category, motif_id, seed)
            return True
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            if attempt == MAX_RETRIES:
                logger.error("  Failed %s/%s after %d attempts: %s", category, motif_id, MAX_RETRIES, e)
                return False
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                wait = min(delay * (2 ** (attempt - 1)) + random.uniform(0, 1), MAX_DELAY)
                logger.warning("  Rate limited %s/%s, waiting %.1fs (attempt %d/%d)",
                              category, motif_id, wait, attempt, MAX_RETRIES)
                time.sleep(wait)
            else:
                logger.warning("  Error %s/%s: %s, retrying in %.1fs",
                              category, motif_id, e, delay)
                time.sleep(delay)
    return False


def generate_category(
    client: httpx.Client,
    api_key: str,
    category: str,
    motifs: list[tuple],
    out_dir: Path,
    start_idx: int = 0,
) -> tuple[int, int]:
    """Generate all motifs in a category. Returns (success, total)."""
    logger.info("\n%s", "=" * 60)
    logger.info("CATEGORY: %s (%d motifs)", category.upper(), len(motifs))
    logger.info("%s", "=" * 60)

    success = 0
    for i, motif in enumerate(motifs[start_idx:], start=start_idx):
        motif_id, motif_name, description = motif[0], motif[1], motif[2]
        logger.info("[%d/%d] %s - %s", i + 1, len(motifs), motif_id, motif_name)
        if generate_image(client, api_key, category, motif_id, motif_name, description, out_dir):
            success += 1
        time.sleep(0.5)

    logger.info("Category %s: %d/%d successful", category, success, len(motifs))
    return success, len(motifs)


def main():
    parser = argparse.ArgumentParser(description="Generate ALL psychology motifs via Pollinations")
    parser.add_argument("--category", help="Generate only specific category")
    parser.add_argument("--start", type=int, default=0, help="Start index within category")
    parser.add_argument("--output", default="assets/motifs_pollinations", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List all categories and counts")
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.pollinations_api_key

    if not api_key:
        logger.error("No POLLINATIONS_API_KEY in settings/.env")
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("\nAVAILABLE CATEGORIES:")
        print("=" * 60)
        total = 0
        for cat, motifs in PSYCHOLOGY_MOTIFS.items():
            print(f"  {cat:20s} : {len(motifs):3d} motifs")
            total += len(motifs)
        print(f"  {'TOTAL':20s} : {total:3d} motifs")
        print("=" * 60)
        return 0

    client = httpx.Client(timeout=TIMEOUT, follow_redirects=True)

    try:
        if args.category:
            if args.category not in PSYCHOLOGY_MOTIFS:
                logger.error("Unknown category: %s", args.category)
                return 1
            categories = {args.category: PSYCHOLOGY_MOTIFS[args.category]}
        else:
            categories = PSYCHOLOGY_MOTIFS

        total_success = 0
        total_all = 0

        for cat, motifs in categories.items():
            s, t = generate_category(client, api_key, cat, motifs, out_dir, args.start if cat == args.category else 0)
            total_success += s
            total_all += t

        logger.info("\n%s", "=" * 60)
        logger.info("GENERATION COMPLETE: %d/%d images", total_success, total_all)
        logger.info("Output directory: %s", out_dir.absolute())
        logger.info("%s", "=" * 60)

        # Write manifest
        manifest = {
            "generated_at": time.time(),
            "total_requested": total_all,
            "total_successful": total_success,
            "categories": {cat: len(motifs) for cat, motifs in categories.items()},
            "style": STYLE_SUFFIX,
            "resolution": f"{WIDTH}x{HEIGHT}",
            "model": MODEL,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Manifest written to %s/manifest.json", out_dir)

    finally:
        client.close()

    return 0 if total_success == total_all else 1


if __name__ == "__main__":
    sys.exit(main())