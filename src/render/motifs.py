"""Visual motif taxonomy for dark-psychology shorts.

Each motif is a self-contained visual idea that can carry a sentence on its
own, without narration. The render pipeline builds a "silent cut" by
assigning one motif per sentence, then layers voice and music on top.

Motifs are grouped into categories so the chooser can enforce variety
(no two adjacent scenes share a category, no global repeat within a video).

Expanded to 15 categories x 20 motifs = 300 motifs for comprehensive coverage.
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
    "archetype",
    "relationship",
    "healing",
    "concept",
    "surreal",
)


MOTIFS: tuple[Motif, ...] = (
    # ---- face (20) ----
    Motif("face_01", "face", "Micro-expression freeze", "Extreme close-up of a face mid-micro-expression frozen in time, one eyebrow raised, corner of mouth twitching, eyes wide with suppressed emotion.", "tight framing, shallow depth, single catchlight", ("micro expression face", "close up eyes", "facial expression portrait")),
    Motif("face_02", "face", "Dead-eye stare", "Unblinking stare directly into lens, pupils dilated, no smile, hollow eyes reflecting nothing, clinical flat lighting.", "frontal, flat light, clinical", ("stare portrait", "unblinking eyes", "intense gaze")),
    Motif("face_03", "face", "Half-shadow face (chiaroscuro)", "Face split by hard vertical shadow, one eye hidden in darkness, other eye illuminated, dramatic Rembrandt lighting.", "chiaroscuro, profile light", ("chiaroscuro portrait", "half face shadow", "dramatic portrait")),
    Motif("face_04", "face", "Smile not reaching eyes", "Polite social smile with cold empty eyes, crow's feet absent, mouth curved but eyes dead, uncanny valley.", "soft light, posed, uncanny", ("fake smile portrait", "insincere smile", "uncanny smile")),
    Motif("face_05", "face", "Tears without emotion", "Tears rolling down cheeks while face remains completely blank, no muscle movement in forehead or jaw, macro detail.", "macro, desaturated", ("tear close up", "crying blank face", "tears portrait")),
    Motif("face_06", "face", "Mouth covered / silenced", "Hand pressed firmly over mouth, eyes wide with panic, fingers digging into skin, high contrast noir.", "centered, high contrast", ("mouth covered", "silenced face", "hand over mouth")),
    Motif("face_07", "face", "Mask peeling off", "Fingers peeling theatrical porcelain mask from face, revealing different expression underneath, motion blur.", "motion blur, dramatic", ("mask removal", "taking off mask", "masquerade face")),
    Motif("face_08", "face", "Reflection in eye macro", "Extreme macro of iris reflecting a distorted scene - a figure, a room, a memory - impossible detail.", "iris detail, mirror", ("eye reflection", "iris macro", "reflection in eye")),
    Motif("face_09", "face", "Young/old split face", "Vertical split down center: left side youthful smooth skin, right side aged wrinkles, same eyes.", "diptych, neutral", ("age progression face", "young old portrait", "aging face")),
    Motif("face_10", "face", "Faceless silhouette", "Head and shoulders silhouette against bright backlight, zero facial features, fog swirling around edges.", "backlight, fog", ("silhouette head", "faceless figure", "dark silhouette")),
    Motif("face_11", "face", "Gaslighting double reflection", "Mirror shows two different expressions - face calm, reflection screaming, uncanny horror.", "split mirror, horror", ("gaslighting mirror", "double reflection", "mirror horror")),
    Motif("face_12", "face", "Dissociative blank stare", "Eyes open but nobody home, thousand-yard stare, pupils slightly unequal, derealization.", "dissociated, empty", ("dissociative stare", "blank stare", "thousand yard stare")),
    Motif("face_13", "face", "Manipulative charm smile", "Charismatic smile, head tilted, eyes calculating, dimples, too perfect, predator disguise.", "charming, calculated", ("manipulative smile", "charm smile", "predator disguise")),
    Motif("face_14", "face", "Paranoid hypervigilance", "Eyes darting in all directions, tense jaw, sweat at temple, scanning for threats that aren't there.", "tense, scanning", ("paranoid eyes", "hypervigilance", "scanning eyes")),
    Motif("face_15", "face", "Narcissistic self-admiration", "Face close to mirror/camera, pupils dilated with self-love, faint smirk, perfectionist grooming.", "self-absorbed, perfect", ("narcissist mirror", "self admiration", "perfect grooming")),
    Motif("face_16", "face", "Borderline emotional storm", "Face contorted mid-emotion shift - anger to despair to fear in one frame, raw intensity.", "emotional chaos, raw", ("borderline emotion", "emotional storm", "rapid cycling")),
    Motif("face_17", "face", "Psychopathic flat affect", "Completely relaxed face, zero tension, eyes cold like shark, mouth neutral, terrifyingly calm.", "flat affect, predatory", ("psychopathic flat affect", "shark eyes", "zero tension")),
    Motif("face_18", "face", "Trauma freeze response", "Body rigid, eyes locked on invisible threat, breath held, time stopped, dissociation.", "frozen, dissociated", ("trauma freeze", "freeze response", "dissociation")),
    Motif("face_19", "face", "Love bombing intensity", "Overwhelming eye contact, face too close, pupils blown, intensity burning, manufactured intimacy.", "intense, manufactured", ("love bombing", "manufactured intimacy", "intense gaze")),
    Motif("face_20", "face", "Identity fragmentation", "Multiple exposure: same face layered 3-4 times with different expressions, splitting apart.", "fragmented, multiple exposure", ("identity fragmentation", "split personality", "fragmented face")),

    # ---- body (20) ----
    Motif("body_01", "body", "Dominant power posture", "Figure looming over seated person, legs apart, hands on hips, chin raised, low angle shot.", "low angle, imposing", ("power posture", "dominant stance", "body language power")),
    Motif("body_02", "body", "Defensive closed posture", "Arms crossed tight, shoulders hunched to ears, head tucked, turned away, protective.", "side view, soft light", ("defensive body language", "crossed arms", "closed posture")),
    Motif("body_03", "body", "Space invasion", "Dominant figure leaning into personal space, subordinate recoiling backward, tension visible.", "two-shot, tension", ("personal space invasion", "leaning close", "intimate distance")),
    Motif("body_04", "body", "Controlling hand on shoulder", "Hand gripping shoulder from behind, fingers pressing into trapezius, ownership gesture.", "over-shoulder, subtle", ("hand on shoulder", "controlling touch", "guiding hand")),
    Motif("body_05", "body", "Accusatory pointing", "Aggressive finger jab toward camera/viewer, other hand on hip, freeze frame intensity.", "high contrast, freeze", ("pointing finger", "accusation gesture", "blame gesture")),
    Motif("body_06", "body", "Mirroring rapport", "Two figures in identical posture - legs crossed same, heads tilted same, one microsecond ahead.", "symmetrical, study", ("mirroring body language", "rapport posture", "matched stance")),
    Motif("body_07", "body", "Steepled confidence", "Fingertips touching in pyramid, elbows wide, leaning back, evaluative judgment.", "desk, neutral", ("steepled hands", "thinking pose", "finger tent")),
    Motif("body_08", "body", "Pacing anxiety", "Profile silhouette pacing, long shadow stretching, hands behind back or wringing.", "side light, motion blur", ("pacing silhouette", "walking shadow", "restless figure")),
    Motif("body_09", "body", "Submission kneeling", "Figure on knees looking up, hands visible palms up, other figure towering above.", "low angle, stark", ("kneeling figure", "submission posture", "power dynamic")),
    Motif("body_10", "body", "Restrained wrists", "Wrists bound with rope/zip ties/handcuffs, tension in arms, struggle marks on skin.", "close-up, dramatic", ("restrained hands", "bound wrists", "captive hands")),
    Motif("body_11", "body", "Grooming manipulation", "One figure fixing other's collar/hair/tie, intimate invasion disguised as care.", "intimate, deceptive", ("grooming manipulation", "intimate invasion", "disguised care")),
    Motif("body_12", "body", "Territorial manspreading", "Seated figure taking maximum space, legs wide, arms on adjacent chairs, dominance.", "wide, territorial", ("manspreading", "territorial dominance", "space taking")),
    Motif("body_13", "body", "Self-soothing hug", "Arms wrapped around own torso, fingers digging into ribs, rocking slightly, comfort seeking.", "protective, vulnerable", ("self soothing", "comfort seeking", "protective hug")),
    Motif("body_14", "body", "Aggressive finger gun", "Thumb up, index extended, 'shooting' at target, playful menace, power play.", "playful menace", ("finger gun", "playful threat", "power play")),
    Motif("body_15", "body", "Submissive minimization", "Shoulders rounded, neck retracted, making self small, invisible, apology posture.", "small, invisible", ("submissive posture", "minimization", "apology posture")),
    Motif("body_16", "body", "Predatory stillness", "Zero micro-movements, breathing invisible, coiled spring, about to strike.", "still, coiled", ("predatory stillness", "coiled spring", "zero movement")),
    Motif("body_17", "body", "Trauma protective curl", "Fetal position on side, knees to chest, arms around head, back to world.", "fetal, protective", ("trauma curl", "fetal position", "protective ball")),
    Motif("body_18", "body", "Charismatic open palms", "Hands wide, palms up, 'trust me' gesture, shoulders back, manufactured warmth.", "open, manufactured", ("open palms", "trust gesture", "manufactured warmth")),
    Motif("body_19", "body", "Gaslighting head tilt", "Head cocked 45 degrees, concerned expression, 'are you okay?' weaponized empathy.", "head tilt, deceptive", ("gaslighting tilt", "weaponized empathy", "head cock")),
    Motif("body_20", "body", "Dissociative floating", "Body present but weightless, feet not quite touching ground, disconnected from gravity.", "weightless, disconnected", ("dissociative floating", "weightless body", "gravity disconnect")),

    # ---- shadow (20) ----
    Motif("shadow_01", "shadow", "Long shadow pursuit", "Figure walking, shadow stretching impossibly long ahead, chasing the shadow self.", "sunset light, ground", ("long shadow", "shadow pursuit", "elongated shadow")),
    Motif("shadow_02", "shadow", "Shadow without source", "Sharp human shadow on blank wall, no figure casting it, empty room, supernatural.", "empty room, eerie", ("shadow without source", "mysterious shadow", "wall shadow")),
    Motif("shadow_03", "shadow", "Multiple shadows one body", "Single figure casting 3-4 shadows in different directions, fractured self.", "hard light, surreal", ("multiple shadows", "split shadow", "surreal shadow")),
    Motif("shadow_04", "shadow", "Shadow hand reaching", "Disembodied shadow hand stretching across floor/wall toward vulnerable figure.", "projected, stark", ("shadow hand", "reaching shadow", "hand shadow")),
    Motif("shadow_05", "shadow", "Doorway silhouette", "Dark figure filling doorway, backlit by harsh corridor light, threshold guardian.", "backlight, threshold", ("silhouette doorway", "figure in door", "backlit figure")),
    Motif("shadow_06", "shadow", "Venetian blind stripes", "Face/body sliced by horizontal light stripes through blinds, prison bars pattern.", "venetian, noir", ("blinds shadow", "venetian blinds light", "striped face")),
    Motif("shadow_07", "shadow", "Eclipse shadow", "Circular shadow sweeping across face, progressive darkening, cosmic alignment.", "partial, ominous", ("eclipse shadow", "circular shadow", "darkening")),
    Motif("shadow_08", "shadow", "Growing shadow", "Time-lapse: shadow expanding across wall while figure stays same size, creeping dread.", "time-lapse feel, dread", ("growing shadow", "expanding shadow", "creeping shadow")),
    Motif("shadow_09", "shadow", "Merging shadows", "Two shadows overlapping until indistinguishable, enmeshment, loss of boundaries.", "abstract, symbolic", ("merging shadows", "shadow overlap", "unified shadow")),
    Motif("shadow_10", "shadow", "Detached shadow", "Figure walking one way, shadow moving opposite, independence, betrayal.", "surreal, eerie", ("detached shadow", "shadow separate", "independent shadow")),
    Motif("shadow_11", "shadow", "Shadow puppet mastery", "Hands forming animal shapes on wall, but shadows move on their own, sentient.", "sentient, puppet", ("shadow puppet", "sentient shadow", "puppet mastery")),
    Motif("shadow_12", "shadow", "Inner demon shadow", "Profile shadow shows horns/claws/wings while figure looks normal, true nature.", "true nature, hidden", ("inner demon", "shadow true nature", "hidden monster")),
    Motif("shadow_13", "shadow", "Shadow in mirror wrong", "Reflection's shadow doesn't match body position, reality glitch.", "reality glitch", ("wrong shadow", "mirror glitch", "shadow mismatch")),
    Motif("shadow_14", "shadow", "Cage shadow bars", "Shadow of window blinds falls like prison bars across body, trapped.", "prison bars, trapped", ("cage shadow", "prison bars shadow", "trapped shadow")),
    Motif("shadow_15", "shadow", "Shadow consumption", "Darkness rising from feet upward, swallowing figure, depression visualization.", "consuming, depression", ("shadow consumption", "depression shadow", "swallowing darkness")),
    Motif("shadow_16", "shadow", "Protective shadow", "Larger shadow looming over smaller figure, guardian or predator ambiguous.", "ambiguous, guardian", ("protective shadow", "shadow guardian", "ambiguous protection")),
    Motif("shadow_17", "shadow", "Shadow dissociation", "Shadow stepping out of feet, becoming 3D, walking away, abandonment.", "3D shadow, abandonment", ("shadow dissociation", "3d shadow", "shadow abandonment")),
    Motif("shadow_18", "shadow", "Projective identification", "Figure's shadow acting out emotions face suppresses - screaming, attacking.", "acting out, suppressed", ("projective identification", "shadow acting out", "suppressed emotions")),
    Motif("shadow_19", "shadow", "Shadow timeline", "Multiple shadows at different ages trailing behind walking figure, past selves.", "past selves, timeline", ("shadow timeline", "past selves", "age shadows")),
    Motif("shadow_20", "shadow", "Void shadow", "Shadow is pure nothingness - light passes through, absence where presence should be.", "absence, void", ("void shadow", "absence shadow", "nothingness shadow")),

    # ---- control (20) ----
    Motif("control_01", "control", "Puppet strings", "Marionette strings from above attached to limbs, controller hands invisible, helpless dance.", "overhead, theatrical", ("puppet strings", "marionette", "control strings")),
    Motif("control_02", "control", "Chess king isolated", "Ornate chessboard, king alone in center, all other pieces surrounding, checkmate inevitable.", "top-down, dramatic", ("chess pieces", "chess game", "king isolated")),
    Motif("control_03", "control", "Mask collection wall", "Dozens of theatrical masks on wall - comedy, tragedy, animals, blank - identity wardrobe.", "symmetrical, eerie", ("mask collection", "theater masks", "wall of masks")),
    Motif("control_04", "control", "Lying mirror", "Mirror reflection shows different expression/age/gender than person standing before it.", "uncanny, split", ("mirror reflection", "lying mirror", "false reflection")),
    Motif("control_05", "control", "Labyrinth trap", "Top-down maze, single tiny figure at center, walls closing in, no exit visible.", "geometric, cold", ("maze aerial", "labyrinth", "trapped figure")),
    Motif("control_06", "control", "Cage behind bars", "Figure gripping bars, face pressed to metal, reaching toward freedom just inches away.", "high contrast, stark", ("cage bars", "behind bars", "imprisoned figure")),
    Motif("control_07", "control", "Digital code tendrils", "Green matrix code wrapping around body like vines, digital enslavement, modern control.", "abstract, digital", ("code strings", "digital control", "data tendrils")),
    Motif("control_08", "control", "Empty throne spotlight", "Ornate empty chair under single spotlight, dust motes dancing, power vacuum.", "symmetrical, ominous", ("empty throne", "empty chair", "spotlight chair")),
    Motif("control_09", "control", "Crown/halo ambiguity", "Golden crown or glowing halo hovering above head - blessing or burden unclear.", "iconic, stark", ("crown silhouette", "halo figure", "power symbol")),
    Motif("control_10", "control", "Web of control", "Figure at center of glowing network lines connecting to dozens of puppets, spider.", "network, abstract", ("web lines", "network diagram", "radiating lines")),
    Motif("control_11", "control", "Gaslighting reality warp", "Room bending - walls melting, floor tilting, objects shifting - perception attack.", "reality warp", ("gaslighting warp", "reality bending", "perception attack")),
    Motif("control_12", "control", "Love bomb overwhelm", "Figure buried under avalanche of roses/gifts/letters/phones, drowning in 'affection'.", "overwhelming, drowning", ("love bombing", "affection overwhelm", "drowning gifts")),
    Motif("control_13", "control", "Triangulation triangle", "Three figures, manipulator in center pulling strings to other two, divide conquer.", "triangle, manipulation", ("triangulation", "divide conquer", "three person manipulation")),
    Motif("control_14", "control", "Foot in door wedge", "Door barely open, foot blocking close, persistent pressure, incremental invasion.", "incremental, persistent", ("foot in door", "incremental invasion", "door wedge")),
    Motif("control_15", "control", "Door in face retreat", "Heavy door slamming, then smaller door opening, reciprocal concession psychology.", "rejection concession", ("door in face", "reciprocal concession", "rejection retreat")),
    Motif("control_16", "control", "Lowball changing terms", "Contract morphing - text rewriting itself, numbers shifting, fine print expanding.", "morphing contract", ("lowball", "changing terms", "contract morph")),
    Motif("control_17", "control", "Authority uniform close", "Epaulettes, badges, medals, starched collar - symbol power not person.", "symbolic authority", ("authority uniform", "power symbol", "uniform close")),
    Motif("control_18", "control", "Milgram shock panel", "Voltage dials 15-450, labeled 'Slight Shock' to 'XXX', finger hovering.", "obedience experiment", ("milgram", "shock panel", "obedience experiment")),
    Motif("control_19", "control", "Hypnotic spiral", "Black-white spiral expanding from eyes outward, consciousness narrowing, trance.", "trance, narrowing", ("hypnotic spiral", "trance induction", "consciousness narrowing")),
    Motif("control_20", "control", "Invisible leash", "Figure walking freely but tension line visible from neck to unseen handler, subtle.", "subtle control", ("invisible leash", "subtle control", "hidden handler")),

    # ---- space (20) ----
    Motif("space_01", "space", "Interrogation room", "Bare concrete, single metal table, two chairs, harsh overhead fluorescent, one-way mirror.", "institutional, cold", ("interrogation room", "bare room", "harsh light room")),
    Motif("space_02", "space", "Infinite corridor", "One-point perspective hallway stretching forever, doors on both sides, vanishing point.", "vanishing point, dread", ("long corridor", "empty hallway", "vanishing point")),
    Motif("space_03", "space", "Staircase from below", "Looking up steep stairs, figure descending in silhouette, power position above.", "low angle, dramatic", ("staircase below", "looking up stairs", "descending figure")),
    Motif("space_04", "space", "Window at night", "Single lit window in dark brutalist building, silhouette pacing inside, urban isolation.", "noir, urban", ("lit window night", "window silhouette", "dark building")),
    Motif("space_05", "space", "Empty classroom", "Rows of identical desks, one occupied, chalkboard with indecipherable writing, conformity.", "institutional, eerie", ("empty classroom", "desks rows", "school room")),
    Motif("space_06", "space", "Therapy office", "Leather couch, empty chair, tissues, clock, soft light, clinical intimacy, safe container.", "soft, sterile", ("therapy room", "couch chair", "clinical office")),
    Motif("space_07", "space", "Observation room", "Dark room, huge one-way mirror, figure on other side unaware, voyeuristic power.", "surveillance, stark", ("one way mirror", "observation room", "behind glass")),
    Motif("space_08", "space", "Basement stairs", "Steep narrow stairs descending into darkness, single bulb at top, childhood fear.", "chiaroscuro, dread", ("basement stairs", "dark staircase", "descending stairs")),
    Motif("space_09", "space", "Rooftop edge", "Concrete parapet, city lights far below, wind tearing clothes, vertigo invitation.", "height, vertigo", ("rooftop edge", "city below", "high angle")),
    Motif("space_10", "space", "Empty stage", "Velvet curtains half-drawn, single spotlight on center, waiting for performance.", "theatrical, ominous", ("empty stage", "spotlight stage", "theater curtain")),
    Motif("space_11", "space", "Childhood bedroom", "Twin bed, posters peeling, toys frozen in time, past preserved under dust.", "nostalgic, frozen", ("childhood bedroom", "frozen past", "preserved toys")),
    Motif("space_12", "space", "Hospital corridor", "Institutional green, fluorescent buzz, gurney wheels squeaking, sterile suffering.", "institutional, suffering", ("hospital corridor", "sterile suffering", "institutional green")),
    Motif("space_13", "space", "Confessional booth", "Dark wood screen, grate between, kneeler, absolute secrecy, absolute judgment.", "secrecy, judgment", ("confessional booth", "absolute secrecy", "judgment booth")),
    Motif("space_14", "space", "Panopticon center", "Circular prison, guard tower center, all cells visible, self-policing internalized.", "panopticon, surveillance", ("panopticon", "self policing", "circular prison")),
    Motif("space_15", "space", "Abandoned asylum", "Peeling paint, rusted beds, restraint marks on walls, forgotten suffering.", "forgotten, suffering", ("abandoned asylum", "forgotten suffering", "rusted beds")),
    Motif("space_16", "space", "Luxury prison", "Gilded cage - marble, crystal, silk - but windows barred, golden handcuffs.", "gilded cage", ("luxury prison", "golden cage", "golden handcuffs")),
    Motif("space_17", "space", "Memory palace", "Impossible architecture - rooms within rooms, Escher stairs, mind's labyrinth.", "impossible, labyrinth", ("memory palace", "mind labyrinth", "impossible architecture")),
    Motif("space_18", "space", "Safe room / panic room", "Steel walls, monitors, supplies, locked from inside, paranoia realized.", "paranoia, safety", ("panic room", "safe room", "paranoia realized")),
    Motif("space_19", "space", "Cult compound", "Identical white buildings, communal dining, watchtower, surrendered autonomy.", "surrendered autonomy", ("cult compound", "surrendered autonomy", "communal living")),
    Motif("space_20", "space", "Digital void", "Pure black, floating UI elements, notifications, algorithms watching, no escape.", "digital, trapped", ("digital void", "algorithm surveillance", "no escape digital")),

    # ---- text (20) ----
    Motif("text_01", "text", "Single word on black", "One word - 'WHY' or 'LIAR' or 'WAKE UP' - huge white typography, stark minimal.", "minimal, stark", ("text on black", "single word", "typography black")),
    Motif("text_02", "text", "Redacted document", "Classified pages, thick black marker bars, only 'THE' and 'AND' visible, truth hidden.", "document, noir", ("redacted text", "censored document", "black bars")),
    Motif("text_03", "text", "Handwritten desperation", "Torn notebook page, frantic scrawl, pressed hard, ink bleeding, 'help me' repeated.", "paper, intimate", ("handwritten note", "scrawled text", "note paper")),
    Motif("text_04", "text", "Typewriter confession", "Vintage keys, paper with single sentence typed over and over, obsession loop.", "vintage, mechanical", ("typewriter", "typed page", "vintage text")),
    Motif("text_05", "text", "Neon truth sign", "Pink/blue neon in rain-slick alley: 'YOU ARE ENOUGH' flickering, ironic comfort.", "noir, urban", ("neon sign", "glowing text", "night sign")),
    Motif("text_06", "text", "Subliminal flash", "Single frame text 'OBEY' 'CONSUME' 'SLEEP' - barely perceptible, programming.", "strobe, unsettling", ("subliminal text", "flash text", "hidden message")),
    Motif("text_07", "text", "Graffiti confession", "Brick wall: 'I WAS NEVER REAL' in spray paint, raw, public secret.", "urban, raw", ("graffiti text", "wall writing", "spray paint")),
    Motif("text_08", "text", "Diary locked", "Open leather journal, intimate cursive, last page torn out, key hanging broken.", "warm, personal", ("diary page", "journal entry", "handwritten diary")),
    Motif("text_09", "text", "Legal contract fine print", "Magnified signature line, microscopic text selling soul, 'initial here' arrows.", "formal, cold", ("contract document", "legal paper", "signature line")),
    Motif("text_10", "text", "Mind map conspiracy", "Whiteboard: photos, red string, arrows, 'CONNECTED' centered, paranoid pattern.", "analytical, busy", ("mind map", "whiteboard diagram", "concept map")),
    Motif("text_11", "text", "Affirmation mirror", "Lipstick on bathroom mirror: 'BEAUTIFUL' crossed out, 'WORTHLESS' written over.", "mirror, twisted", ("affirmation mirror", "twisted affirmation", "mirror writing")),
    Motif("text_12", "text", "Anonymous threat", "Cut-out magazine letters on paper: 'WE KNOW WHAT YOU DID' ransom note.", "threatening, anonymous", ("ransom note", "anonymous threat", "cutout letters")),
    Motif("text_13", "text", "Gaslighting journal", "Two handwritings in same journal arguing: 'That never happened' / 'Yes it did'.", "conflicting, gaslight", ("gaslighting journal", "conflicting memories", "two handwritings")),
    Motif("text_14", "text", "Cult manifesto", "Photocopied pages, highlighted passages, 'THE LEADER IS NEVER WRONG' circled.", "indoctrination", ("cult manifesto", "indoctrination text", "leader worship")),
    Motif("text_15", "text", "Suicide note draft", "Crumpled paper, multiple starts, 'I'm sorry' 'I'm tired' 'Goodbye' - never sent.", "tragic, unsent", ("suicide note", "unsent letter", "crumpled paper")),
    Motif("text_16", "text", "Autopilot script", "Teleprompter text reflecting in eyes: 'SMILE. AGREE. BUY. OBEY. REPEAT.'", "programmed, automatic", ("autopilot script", "programmed behavior", "teleprompter eyes")),
    Motif("text_17", "text", "Redacted memory", "Childhood photo with face scratched out, 'FORGOTTEN' written in red pen.", "forgotten, scratched", ("redacted memory", "forgotten face", "childhood scratched")),
    Motif("text_18", "text", "Inner critic loop", "Typewriter paper scrolling endlessly: 'NOT ENOUGH NOT ENOUGH NOT ENOUGH'.", "endless, critical", ("inner critic", "not enough loop", "endless criticism")),
    Motif("text_19", "text", "Truth serum confession", "Handwritten on transparent overlay: 'I DON'T KNOW WHO I AM ANYMORE'.", "transparent, identity", ("truth confession", "identity loss", "transparent overlay")),
    Motif("text_20", "text", "Freedom declaration", "Paper airplane flying from window, 'I CHOOSE ME' written on wing, liberation.", "liberation, flight", ("freedom declaration", "paper airplane", "I choose me")),

    # ---- surveillance (20) ----
    Motif("surv_01", "surveillance", "CCTV grain", "Green-tinted low-res footage, timestamp 03:47:12, figure unaware, watched.", "grainy, greenish", ("cctv footage", "security camera", "surveillance frame")),
    Motif("surv_02", "surveillance", "Thermal imaging", "False-color heat map: human shape glowing orange/red against cold blue background.", "false color, eerie", ("thermal imaging", "heat map person", "infrared figure")),
    Motif("surv_03", "surveillance", "Night vision", "Monochrome green, grainy, eyes glowing white (IR reflection), predator vision.", "monochrome green, eerie", ("night vision", "green footage", "nvg person")),
    Motif("surv_04", "surveillance", "Phone screen glow", "Face illuminated only by blue light in pitch dark, doom-scrolling, zombie.", "blue glow, intimate", ("phone glow face", "screen light face", "dark room phone")),
    Motif("surv_05", "surveillance", "Screen reflection", "Monitor reflection shows different content than face expresses, dual reality.", "tech, cold", ("screen reflection face", "monitor reflection", "display reflection")),
    Motif("surv_06", "surveillance", "Recording LED", "Tiny red dot in darkness, blinking steadily, 'REC' - panopticon internalized.", "minimal, ominous", ("recording light", "red led", "camera recording")),
    Motif("surv_07", "surveillance", "Blurred identity", "Face pixelated/blurred by algorithm, anonymized but body language exposed.", "anonymized, stark", ("blurred face", "pixelated face", "anonymized person")),
    Motif("surv_08", "surveillance", "Digital zoom enhance", "Extreme crop, JPEG artifacts, 'enhance' revealing nothing, false certainty.", "pixelated, invasive", ("digital zoom", "zoomed face", "surveillance zoom")),
    Motif("surv_09", "surveillance", "Monitor wall", "Security room: 24 screens showing every angle, one guard, god complex.", "control room, cold", ("monitor wall", "control room screens", "surveillance monitors")),
    Motif("surv_10", "surveillance", "Drone POV", "Top-down aerial, figure tiny in courtyard, crosshairs overlay, remote power.", "top-down, detached", ("drone view person", "aerial surveillance", "overhead figure")),
    Motif("surv_11", "surveillance", "Smart home eye", "Camera lens in smoke detector, wide angle, always recording, convenience trap.", "hidden, always on", ("smart home camera", "hidden surveillance", "convenience trap")),
    Motif("surv_12", "surveillance", "Biometric scan", "Iris close-up, laser grid mapping, identity verified, soul catalogued.", "biometric, catalogued", ("biometric scan", "iris mapping", "identity verified")),
    Motif("surv_13", "surveillance", "Social credit display", "Floating AR numbers above heads: 420, 780, 150 - worth calculated.", "AR, calculated", ("social credit", "worth calculated", "AR numbers")),
    Motif("surv_14", "surveillance", "Keystroke logger", "Fingers typing, ghost text appearing behind: 'password123' 'I hate him' 'help'.", "ghost text, exposed", ("keystroke logger", "ghost typing", "exposed secrets")),
    Motif("surv_15", "surveillance", "Location history", "Map with glowing trail: home work lover's bar home - life tracked.", "tracked, mapped", ("location history", "life tracked", "map trail")),
    Motif("surv_16", "surveillance", "Predictive policing", "Holographic red zones on city map: 'CRIME PROBABILITY 94%' - prejudice coded.", "prejudice coded", ("predictive policing", "prejudice algorithm", "red zones")),
    Motif("surv_17", "surveillance", "Emotion recognition", "Face with algorithmic overlay: 'ANXIETY 87%' 'DECEPTION 12%' - feelings data.", "datafied feelings", ("emotion recognition", "feelings data", "algorithmic overlay")),
    Motif("surv_18", "surveillance", "Brainwave monitor", "EEG cap, squiggly lines, 'THOUGHT DETECTED: REBELLION' - mind read.", "mind reading", ("brainwave monitor", "thought detection", "mind reading")),
    Motif("surv_19", "surveillance", "Digital twin", "Holographic clone standing beside original, perfect prediction, replacement ready.", "replacement, predicted", ("digital twin", "holographic clone", "perfect prediction")),
    Motif("surv_20", "surveillance", "Algorithm god", "Server room infinite, blinking lights, one cursor blinking: 'EXECUTE: TERMINATE'.", "server room, execution", ("algorithm god", "server room", "execute terminate")),

    # ---- abstract (20) ----
    Motif("abstract_01", "abstract", "Neural network glow", "Dark brain silhouette, synapses firing gold/blue, consciousness emerging.", "dark, bioluminescent", ("neural network", "brain connections", "synapse glow")),
    Motif("abstract_02", "abstract", "Fractured mirror", "Shattered glass, each fragment showing different version of same face.", "macro, sharp", ("cracked glass", "shattered mirror", "fractured reflection")),
    Motif("abstract_03", "abstract", "Ripple in dark water", "Single drop impact, concentric rings expanding, distortion spreading.", "macro, slow", ("water ripple", "drop ripple", "dark water")),
    Motif("abstract_04", "abstract", "Smoke tendrils", "White smoke curling in black void, shaped by invisible forces, ephemeral.", "atmospheric, slow", ("smoke tendrils", "curling smoke", "backlit smoke")),
    Motif("abstract_05", "abstract", "Ink bloom in water", "Black ink dropped, forming clouds/tendrils/creatures, Rorschach chaos.", "macro, organic", ("ink in water", "ink bloom", "dark cloud water")),
    Motif("abstract_06", "abstract", "Heartbeat flatline", "ECG line on black: steady rhythm spike flatline new rhythm, transformation.", "medical, rhythmic", ("heartbeat line", "ecg pulse", "heart monitor")),
    Motif("abstract_07", "abstract", "Maze pattern", "Geometric labyrinth, no entrance/exit, hypnotic repetition, trapped logic.", "repetitive, cold", ("maze pattern", "geometric maze", "hypnotic pattern")),
    Motif("abstract_08", "abstract", "Fractal zoom", "Mandelbrot infinite recursion, self-similarity at every scale, pattern obsession.", "psychedelic, dark", ("fractal", "infinite zoom", "recursive pattern")),
    Motif("abstract_09", "abstract", "TV static", "Analog noise filling frame, hidden images in chaos, pareidolia, meaning from noise.", "analog, unsettling", ("tv static", "white noise", "analog noise")),
    Motif("abstract_10", "abstract", "Void gradient", "Pure black center fading to dark gray edges, existential nothingness, peaceful.", "minimal, existential", ("black void", "darkness gradient", "empty space")),
    Motif("abstract_11", "abstract", "Puppet strings geometry", "Invisible strings as golden geometry, sacred mathematics of control.", "golden, geometric", ("puppet geometry", "golden strings", "sacred mathematics")),
    Motif("abstract_12", "abstract", "Cognitive dissonance", "Two incompatible shapes forced together, tension lines, stress fractures.", "tension, fracture", ("cognitive dissonance", "incompatible shapes", "stress fractures")),
    Motif("abstract_13", "abstract", "Repressed memory", "Locked box glowing, cracks forming, light leaking, inevitable emergence.", "glowing, emerging", ("repressed memory", "locked box", "light leaking")),
    Motif("abstract_14", "abstract", "Ego death", "Human silhouette dissolving into particles/light/birds, transcendence or annihilation.", "dissolving, transcendent", ("ego death", "silhouette dissolving", "transcendence")),
    Motif("abstract_15", "abstract", "Shadow integration", "Dark figure embraced by light figure, merging, wholeness, Jungian gold.", "merging, gold", ("shadow integration", "Jungian gold", "wholeness")),
    Motif("abstract_16", "abstract", "Gaslight fog", "Reality dissolving at edges, objects becoming suggestions, ground uncertain.", "dissolving reality", ("gaslight fog", "reality dissolving", "uncertain ground")),
    Motif("abstract_17", "abstract", "Trauma vortex", "Spiral pulling everything inward - memories, self, time - event horizon.", "vortex, event horizon", ("trauma vortex", "memory spiral", "event horizon")),
    Motif("abstract_18", "abstract", "Narcissistic mirror", "Infinite reflection corridor, each self more perfect, none real.", "infinite, false", ("narcissistic mirror", "infinite reflection", "false selves")),
    Motif("abstract_19", "abstract", "Borderline splitting", "Figure torn in half - angel/devil, saint/sinner - no middle ground.", "torn, binary", ("borderline splitting", "angel devil", "no middle ground")),
    Motif("abstract_20", "abstract", "Healing spiral", "Upward spiral from darkness to light, not linear, cycling but ascending.", "ascending, cyclical", ("healing spiral", "upward spiral", "cyclical healing")),

    # ---- weather (20) ----
    Motif("weather_01", "weather", "Rain on window", "Streaks blurring city lights, face pressed to glass, tears or rain indistinguishable.", "noir, melancholy", ("rain on glass", "rain window", "wet window")),
    Motif("weather_02", "weather", "Thick fog", "Figure emerging/disappearing in whiteout, only eyes visible, isolation.", "atmospheric, eerie", ("fog mist", "foggy figure", "misty scene")),
    Motif("weather_03", "weather", "Storm clouds", "Supercell anvil, lightning illuminating from within, power uncontained.", "dramatic, ominous", ("storm clouds", "dark clouds", "lightning sky")),
    Motif("weather_04", "weather", "Ash/snow falling", "Gray particles slow in dark air, ambiguous - volcanic winter or nuclear.", "quiet, surreal", ("falling ash", "snow fall", "particles falling")),
    Motif("weather_05", "weather", "Wind through trees", "Branches thrashing, leaves stripped, resistance futile, nature's violence.", "kinetic, restless", ("wind trees", "bending trees", "scattering leaves")),
    Motif("weather_06", "weather", "Moonlit landscape", "Silver light on dark water/fields, beauty in desolation, cold clarity.", "nocturnal, vast", ("moonlit night", "full moon", "night landscape")),
    Motif("weather_07", "weather", "Twilight blue hour", "Indigo sky, last warmth fading, liminal space, day dying night born.", "transitional, moody", ("twilight", "blue hour", "dusk silhouette")),
    Motif("weather_08", "weather", "Flooded reflection", "Submerged street, perfect mirror of buildings above, world inverted.", "surreal, still", ("flooded street", "water reflection", "submerged street")),
    Motif("weather_09", "weather", "Dust motes in beam", "Single shaft light in dark room, dancing particles, microscopic universe.", "macro, quiet", ("dust motes", "light beam dust", "particles air")),
    Motif("weather_10", "weather", "Heat shimmer", "Horizon distorting, mirage rising, reality melting, perception unreliable.", "surreal, disorienting", ("heat shimmer", "mirage", "air distortion")),
    Motif("weather_11", "weather", "Thunderhead anvil", "Massive cumulonimbus, mammatus pouches hanging, apocalypse aesthetic.", "apocalyptic, massive", ("thunderhead", "mammatus", "apocalypse clouds")),
    Motif("weather_12", "weather", "Freezing rain", "Everything encased in clear ice, beautiful deadly, touch shatters.", "beautiful deadly", ("freezing rain", "ice encased", "touch shatters")),
    Motif("weather_13", "weather", "Sandstorm wall", "Orange-brown wall advancing, swallowing everything, erasure.", "advancing, erasure", ("sandstorm", "advancing wall", "erasure")),
    Motif("weather_14", "weather", "Aurora borealis", "Green/purple curtains dancing, magnetic field visible, cosmic radiation.", "cosmic, magnetic", ("aurora borealis", "magnetic field", "cosmic radiation")),
    Motif("weather_15", "weather", "Acid rain", "Drops sizzling on contact, steam rising, invisible poison, corrupted nature.", "corrupted nature", ("acid rain", "sizzling drops", "invisible poison")),
    Motif("weather_16", "weather", "Polar night", "Perpetual darkness, only stars/aurora, time meaningless, endurance test.", "endless night", ("polar night", "perpetual darkness", "endurance test")),
    Motif("weather_17", "weather", "Microburst", "Sudden downdraft flattening everything, invisible violence from above.", "invisible violence", ("microburst", "downdraft", "invisible violence")),
    Motif("weather_18", "weather", "Volcanic lightning", "Ash plume generating own lightning, earth's rage, creation/destruction.", "creation destruction", ("volcanic lightning", "ash plume", "earth rage")),
    Motif("weather_19", "weather", "Climate collapse", "Surreal weather - snow in desert, hurricane inland - new normal.", "surreal weather", ("climate collapse", "surreal weather", "new normal")),
    Motif("weather_20", "weather", "Calm eye", "Perfect stillness in hurricane center, blue sky above, temporary reprieve.", "temporary peace", ("hurricane eye", "calm center", "temporary reprieve")),

    # ---- motion (20) ----
    Motif("motion_01", "motion", "Slow push-in", "Gradual dolly zoom on frozen subject, creeping intimacy, inevitable revelation.", "creeping, hypnotic", ("slow zoom", "push in", "creeping camera")),
    Motif("motion_02", "motion", "Whip pan", "Violent blur between scenes, disorientation, trauma jump cut.", "kinetic, disorienting", ("whip pan", "motion blur transition", "fast pan")),
    Motif("motion_03", "motion", "Glitch distortion", "RGB split, datamoshing, tearing - reality corrupting, simulation breaking.", "digital, unsettling", ("glitch effect", "digital distortion", "rgb split")),
    Motif("motion_04", "motion", "Reverse playback", "Action un-happening: glass un-breaking, words un-spoken, regret fantasy.", "surreal, dreamlike", ("reverse playback", "rewind motion", "backward action")),
    Motif("motion_05", "motion", "Time-lapse clouds", "Centuries in seconds, clouds boiling, shadows racing, human time irrelevant.", "temporal, vast", ("time lapse clouds", "fast clouds", "sky time lapse")),
    Motif("motion_06", "motion", "Strobe flicker", "Unstable light, frozen moments, seizure risk, interrogation technique.", "anxious, harsh", ("flickering light", "strobe", "unstable light")),
    Motif("motion_07", "motion", "Parallax depth", "Foreground/background moving at different speeds, 3D immersion, layers.", "cinematic, immersive", ("parallax", "depth layers", "3d parallax")),
    Motif("motion_08", "motion", "Split screen sync", "Two panels mirroring - same action different outcomes - choice visualization.", "symmetrical, tense", ("split screen", "dual panel", "mirrored action")),
    Motif("motion_09", "motion", "Fade to black", "Slow dissolve, last frame lingering, 'is it over?' uncertainty.", "ominous, final", ("fade to black", "slow fade", "lingering frame")),
    Motif("motion_10", "motion", "Freeze frame", "Mid-action halt - punch suspended, tear mid-fall - moment suspended forever.", "dramatic, arrested", ("freeze frame", "frozen moment", "still action")),
    Motif("motion_11", "motion", "Dolly zoom (vertigo)", "Background stretches while subject stays - disorientation, realization.", "disorienting, realization", ("dolly zoom", "vertigo effect", "background stretch")),
    Motif("motion_12", "motion", "Long take unbroken", "Minutes without cut - raw reality, no escape from witnessing.", "raw, unbroken", ("long take", "unbroken shot", "no escape witnessing")),
    Motif("motion_13", "motion", "Jump cuts anxiety", "Same action repeated with micro-changes - OCD loop, perfectionism.", "OCD, perfectionism", ("jump cuts", "OCD loop", "perfectionism")),
    Motif("motion_14", "motion", "Slow motion impact", "Bullet/slap/word hitting - 1000fps - destruction in detail.", "destruction detail", ("slow motion impact", "1000fps", "destruction detail")),
    Motif("motion_15", "motion", "POV subjective", "Camera IS the eyes - breathing, blinking, shaking - embodied terror.", "embodied, subjective", ("POV subjective", "embodied camera", "breathing blinking")),
    Motif("motion_16", "motion", "Zolly reverse", "Push-pull simultaneously - background rushes away - dissociation.", "dissociation, rush", ("zolly reverse", "push pull", "background rushes")),
    Motif("motion_17", "motion", "Frame within frame", "Doorways, windows, mirrors framing subject - trapped in layers.", "trapped, layered", ("frame within frame", "doorway frame", "trapped layers")),
    Motif("motion_18", "motion", "Match cut metaphor", "Eye closes eye opens different person/place - identity fluid.", "identity fluid", ("match cut", "identity fluid", "eye transition")),
    Motif("motion_19", "motion", "Infinite zoom", "Eye pupil retina neuron atom universe - scale meaningless.", "scale meaningless", ("infinite zoom", "scale meaningless", "micro macro")),
    Motif("motion_20", "motion", "Credits roll up", "Names scrolling over final shot - 'Gaslighter' 'Enabler' 'Victim' 'Survivor'.", "credits, roles", ("credits roll", "role labels", "survivor credits")),

    # ---- archetype (20) ----
    Motif("archetype_01", "archetype", "Beautiful woman mirror", "Stunning woman in vintage mirror, but reflection shows rotting corpse - vanitas.", "vanitas, horror", ("beautiful woman mirror", "vanitas", "rotting reflection")),
    Motif("archetype_02", "archetype", "Machiavellian man staring", "Sharp suit, cold eyes, slight smirk, head tilted, calculating your value.", "calculating, predator", ("machiavellian man", "predator stare", "calculating value")),
    Motif("archetype_03", "archetype", "Wounded healer", "Androgynous figure, hands glowing over another's wound, own chest bleeding.", "wounded, healing", ("wounded healer", "bleeding healer", "self sacrifice")),
    Motif("archetype_04", "archetype", "Trickster shapeshifter", "Face morphing between 4 expressions simultaneously, never settling.", "morphing, unstable", ("trickster", "shapeshifter", "morphing face")),
    Motif("archetype_05", "archetype", "Tyrannical father", "Large silhouette in doorway, belt in hand, authority absolute, fear worshipped.", "authority, fear", ("tyrannical father", "absolute authority", "fear worship")),
    Motif("archetype_06", "archetype", "Devouring mother", "Embrace becoming trap, arms multiplying, 'I only want what's best' suffocating.", "suffocating, trap", ("devouring mother", "suffocating embrace", "multiplied arms")),
    Motif("archetype_07", "archetype", "Golden child", "Perfect glowing figure on pedestal, cracks visible beneath gold leaf, fragile.", "fragile perfection", ("golden child", "fragile pedestal", "cracks beneath gold")),
    Motif("archetype_08", "archetype", "Scapegoat", "Figure covered in projected shadows - 'lazy' 'crazy' 'ungrateful' - carrying all.", "projected burden", ("scapegoat", "projected shadows", "carrying all")),
    Motif("archetype_09", "archetype", "Lost child", "Small figure in oversized adult clothes, invisible, hiding in plain sight.", "invisible, hidden", ("lost child", "invisible child", "hiding in plain sight")),
    Motif("archetype_10", "archetype", "Mascot/clown", "Painted smile, tears underneath, performing for survival, 'everything's fine'.", "performing, hidden pain", ("mascot clown", "painted smile", "tears underneath")),
    Motif("archetype_11", "archetype", "Wise old crone", "Ancient woman stirring pot, steam forming visions, knows your true name.", "ancient, knowing", ("wise crone", "ancient woman", "true name")),
    Motif("archetype_12", "archetype", "Dark magician", "Hands weaving threads of fate, audience unaware they're the puppets.", "fate weaving", ("dark magician", "fate threads", "unaware puppets")),
    Motif("archetype_13", "archetype", "Innocent fool", "Walking off cliff eyes closed, smile trusting, universe catches or doesn't.", "trusting, blind", ("innocent fool", "blind trust", "cliff edge")),
    Motif("archetype_14", "archetype", "Warrior shadow", "Battle-scarred, armor dented, fighting internal dragons, exhausted noble.", "exhausted noble", ("warrior shadow", "internal dragons", "battle scarred")),
    Motif("archetype_15", "archetype", "Siren/mermaid", "Half in water, song visible as golden waves, love equals drowning.", "drowning love", ("siren mermaid", "golden waves", "love drowning")),
    Motif("archetype_16", "archetype", "Hermit lantern", "Alone on mountain, light held high, not for others - for own path.", "solitary path", ("hermit lantern", "solitary light", "own path")),
    Motif("archetype_17", "archetype", "Justice blindfolded", "Scales tipped by gold coins, sword rusted, peek through blindfold.", "corrupted justice", ("blind justice", "tipped scales", "peek blindfold")),
    Motif("archetype_18", "archetype", "Death as therapist", "Skeleton in chair taking notes, 'and how did that make you feel?'.", "death therapist", ("death therapist", "skeleton therapist", "memento mori")),
    Motif("archetype_19", "archetype", "Inner child locked", "Small figure behind barred window in adult's chest, knocking, muffled.", "locked, muffled", ("inner child locked", "barred chest", "muffled knocking")),
    Motif("archetype_20", "archetype", "Integrated self", "All archetypes in circle around center figure, council, wholeness achieved.", "council, wholeness", ("integrated self", "archetype council", "wholeness achieved")),

    # ---- relationship (20) ----
    Motif("rel_01", "relationship", "Trauma bond chain", "Two figures chained together, walking into fire, 'we survive together'.", "chained, survival", ("trauma bond", "chained together", "survival together")),
    Motif("rel_02", "relationship", "Push-pull dance", "One reaching, one retreating, then reversing, never meeting, dizzying.", "dizzying, never meeting", ("push pull", "reaching retreating", "never meeting")),
    Motif("rel_03", "relationship", "Enmeshment vines", "Two bodies overgrown with same vines, roots tangled, where one ends unclear.", "tangled, unclear boundaries", ("enmeshment", "tangled vines", "unclear boundaries")),
    Motif("rel_04", "relationship", "Pedestal fall", "Figure on pedestal crumbling, other catching or watching, idealization shatters.", "crumbling, shattering", ("pedestal fall", "idealization shatters", "crumbling figure")),
    Motif("rel_05", "relationship", "Mirror neurons", "Two faces perfectly synced - micro-expressions identical - eerie resonance.", "synced, eerie", ("mirror neurons", "synced faces", "eerie resonance")),
    Motif("rel_06", "relationship", "Power struggle", "Tug of war with invisible rope, both exhausted, neither releasing.", "exhausted, stalemate", ("power struggle", "tug of war", "invisible rope")),
    Motif("rel_07", "relationship", "Caregiver exhaustion", "One figure carrying other (adult size), knees buckling, 'I can fix them'.", "buckling, fixing", ("caregiver exhaustion", "carrying adult", "fixing others")),
    Motif("rel_08", "relationship", "Narcissist supply", "Parasitic tendrils from hollow chest feeding on vibrant figure's light.", "parasitic, feeding", ("narcissist supply", "parasitic tendrils", "feeding on light")),
    Motif("rel_09", "relationship", "Codependent lock", "Two padlocks interlocked, no keys, each holds other's freedom.", "interlocked, no keys", ("codependent lock", "interlocked padlocks", "no keys")),
    Motif("rel_10", "relationship", "Boundaries fortress", "One behind high walls, other throwing flowers over - never reaching.", "walls, unreachable", ("boundaries fortress", "high walls", "flowers over wall")),
    Motif("rel_11", "relationship", "Gaslighting fog", "One holding flashlight under other's chin, 'you're imagining the light'.", "flashlight, denying", ("gaslighting fog", "flashlight chin", "denying reality")),
    Motif("rel_12", "relationship", "Love bombing shower", "One buried in rose petals/letters/phones, smiling predator above.", "buried, predator", ("love bombing", "buried in gifts", "smiling predator")),
    Motif("rel_13", "relationship", "Triangulation triangle", "Three figures - manipulator center pulling strings - divide conquer.", "divide conquer", ("triangulation", "divide conquer", "three figures")),
    Motif("rel_14", "relationship", "Hoovering vacuum", "Figure walking away, massive suction pulling back - 'I changed'.", "suction, pulling back", ("hoovering", "suction pull", "I changed")),
    Motif("rel_15", "relationship", "Discard phase", "Figure crumpled like used paper, new supply already being groomed.", "discarded, new supply", ("discard phase", "crumpled paper", "new supply groomed")),
    Motif("rel_16", "relationship", "Flying monkeys", "Swarm of masked figures attacking target on narcissist's command.", "swarm, commanded", ("flying monkeys", "masked swarm", "commanded attack")),
    Motif("rel_17", "relationship", "Gray rock defense", "Figure as featureless stone, attacker's tendrils sliding off, boredom.", "stone, boredom", ("gray rock", "featureless stone", "attacker boredom")),
    Motif("rel_18", "relationship", "No contact wall", "Concrete barrier dropping between figures, silence, peace, grief.", "barrier, silence", ("no contact", "concrete wall", "silence peace grief")),
    Motif("rel_19", "relationship", "Reconciliation trap", "Arms open, 'I'm sorry' - but hands hidden behind back holding knife.", "trap, hidden knife", ("reconciliation trap", "hidden knife", "false apology")),
    Motif("rel_20", "relationship", "Healthy bond", "Two separate trees, roots touching gently, space between, choosing together.", "separate, choosing", ("healthy bond", "separate trees", "choosing together")),

    # ---- healing (20) ----
    Motif("heal_01", "healing", "Kintsugi repair", "Broken gold-veined ceramic bowl - 'more beautiful for having been broken'.", "gold veins, beautiful", ("kintsugi", "gold repair", "beautiful broken")),
    Motif("heal_02", "healing", "Shadow integration", "Dark figure embraced by light figure, merging into wholeness, Jungian gold.", "merging, gold", ("shadow integration", "Jungian gold", "wholeness merge")),
    Motif("heal_03", "healing", "Inner child rescue", "Adult hands reaching through bars to small self, 'I'm here now, you're safe'.", "rescue, safety", ("inner child rescue", "reaching through bars", "safe now")),
    Motif("heal_04", "healing", "Boundary fence", "White picket fence going up - 'this far, no further' - peaceful not hostile.", "peaceful boundary", ("boundary fence", "peaceful boundary", "this far no further")),
    Motif("heal_05", "healing", "Voice returning", "Throat glowing, sound waves visible, first 'NO' spoken, power reclaimed.", "throat glowing, NO", ("voice returning", "first no", "power reclaimed")),
    Motif("heal_06", "healing", "Memory reconsolidation", "Painful memory burning, reforming gold - same event, new meaning.", "burning, gold reform", ("memory reconsolidation", "gold reform", "new meaning")),
    Motif("heal_07", "healing", "Body homecoming", "Figure settling into own skin, shoulders dropping, breath deepening, arrival.", "settling, arrival", ("body homecoming", "shoulders dropping", "breath deepening")),
    Motif("heal_08", "healing", "Compassion mirror", "Looking in mirror with kindness instead of criticism - first time.", "kindness, first time", ("compassion mirror", "kindness mirror", "first time kindness")),
    Motif("heal_09", "healing", "Grief wave", "Figure on shore, wave crashes, stands ground, 'I survive this feeling'.", "shore, standing", ("grief wave", "survive feeling", "standing ground")),
    Motif("heal_10", "healing", "Parts integration", "Fragmented selves at round table - protector, critic, child - all heard.", "round table, heard", ("parts integration", "internal family", "all parts heard")),
    Motif("heal_11", "healing", "Nervous system reset", "Vagus nerve glowing, heart rate slowing, safety signal received, ventral vagal.", "vagus glowing, safety", ("nervous system reset", "vagus nerve", "ventral vagal")),
    Motif("heal_12", "healing", "Forgiveness release", "Heavy stone dropped from hands into water, ripples, lightness.", "stone dropped, ripples", ("forgiveness release", "stone water", "lightness ripples")),
    Motif("heal_13", "healing", "Self-trust anchor", "Hand on own heart, 'I believe you' - internal attachment formed.", "hand heart, belief", ("self trust", "hand on heart", "internal attachment")),
    Motif("heal_14", "healing", "New neural path", "Brain lighting new golden pathway beside old red trauma highway.", "golden path, highway", ("new neural path", "golden pathway", "trauma highway")),
    Motif("heal_15", "healing", "Post-traumatic growth", "Tree growing through boulder - 'impossible' made possible.", "tree through boulder", ("post traumatic growth", "tree boulder", "impossible possible")),
    Motif("heal_16", "healing", "Community circle", "Diverse figures in circle, each holding piece of puzzle - belonging.", "puzzle pieces, circle", ("community circle", "puzzle pieces", "belonging circle")),
    Motif("heal_17", "healing", "Creative expression", "Paint-stained hands, canvas showing chaos becoming art - sublimation.", "chaos to art", ("creative expression", "sublimation", "chaos art")),
    Motif("heal_18", "healing", "Nature grounding", "Bare feet on earth, roots growing down, regulation through connection.", "bare feet, roots", ("nature grounding", "bare feet earth", "roots down")),
    Motif("heal_19", "healing", "Nightmare mastery", "Dreamer lucid in nightmare, turning monster into guide - dream alchemy.", "lucid, alchemy", ("nightmare mastery", "lucid dream", "dream alchemy")),
    Motif("heal_20", "healing", "Wisdom earned", "Older self handing lantern to younger - 'you'll survive this too'.", "lantern, survival", ("wisdom earned", "lantern pass", "you'll survive")),

    # ---- concept (20) ----
    Motif("concept_01", "concept", "Cognitive dissonance", "Two gears jammed - 'I'm good' vs 'I did bad' - smoke from friction.", "gears, smoke", ("cognitive dissonance", "jammed gears", "smoke friction")),
    Motif("concept_02", "concept", "Confirmation bias", "Figure only seeing evidence supporting belief, blinders on, reality filtered.", "blinders, filtered", ("confirmation bias", "blinders", "reality filtered")),
    Motif("concept_03", "concept", "Sunk cost fallacy", "Digging deeper in wrong hole, 'already invested so much' - shovel glowing.", "glowing shovel", ("sunk cost", "wrong hole", "glowing shovel")),
    Motif("concept_04", "concept", "Fundamental attribution", "Own mistake: 'circumstances' / Other's mistake: 'character' - double scale.", "double scale", ("fundamental attribution", "double scale", "circumstances vs character")),
    Motif("concept_05", "concept", "Halo effect", "Attractive figure glowing, 'good' words floating - intelligence/kindness assumed.", "glowing, assumed", ("halo effect", "attractive glowing", "assumed goodness")),
    Motif("concept_06", "concept", "Anchoring bias", "First number (1000) glows, all subsequent judgments orbit it - gravity.", "gravity, orbit", ("anchoring bias", "gravity anchor", "judgments orbit")),
    Motif("concept_07", "concept", "Availability heuristic", "Vivid memories (plane crash) loom large, statistics (car safety) invisible.", "looming, invisible", ("availability heuristic", "vivid memories", "invisible stats")),
    Motif("concept_08", "concept", "Dunning-Kruger peak", "Mount stupid: confidence high, competence low - valley of despair ahead.", "mount stupid", ("dunning kruger", "mount stupid", "valley despair")),
    Motif("concept_09", "concept", "Bystander effect", "Crowd watching figure fall, each thinking 'someone else will help' - diffusion.", "diffusion, crowd", ("bystander effect", "diffusion responsibility", "someone else help")),
    Motif("concept_10", "concept", "Stockholm syndrome", "Hostage defending captor, chains seen as jewelry - survival bonding.", "chains as jewelry", ("stockholm syndrome", "chains jewelry", "survival bonding")),
    Motif("concept_11", "concept", "Learned helplessness", "Dog in shuttle box not jumping barrier - door open but not tried.", "open door, not tried", ("learned helplessness", "open door", "not jumping")),
    Motif("concept_12", "concept", "Reactance", "Forbidden fruit glowing, 'don't push' sign - finger hovering, freedom threatened.", "finger hovering", ("reactance", "forbidden fruit", "freedom threatened")),
    Motif("concept_13", "concept", "Mere exposure", "Face repeating - 1x neutral, 5x familiar, 20x liked - repetition breeds comfort.", "repetition, comfort", ("mere exposure", "repetition comfort", "familiarity breeds liking")),
    Motif("concept_14", "concept", "Foot-in-door", "Small yes (sign petition) large yes (donate kidney) - commitment ladder.", "commitment ladder", ("foot in door", "commitment ladder", "small to large")),
    Motif("concept_15", "concept", "Door-in-face", "Huge request rejected small request accepted - reciprocal concession.", "reciprocal concession", ("door in face", "reciprocal concession", "rejection then accept")),
    Motif("concept_16", "concept", "Lowball technique", "Agreed price 'mistake' price doubled still buy - commitment sunk.", "commitment sunk", ("lowball", "price doubled", "sunk commitment")),
    Motif("concept_17", "concept", "Social proof", "Empty restaurant vs full - herd following herd - 'if others...'.", "herd following", ("social proof", "herd behavior", "empty vs full")),
    Motif("concept_18", "concept", "Authority obedience", "White coat says 'continue' - hand on shock lever - 65% comply.", "shock lever, comply", ("authority obedience", "milgram", "shock lever comply")),
    Motif("concept_19", "concept", "Scarcity urgency", "Countdown timer 00:03, 'only 2 left!' - panic buying - FOMO triggered.", "countdown, panic", ("scarcity urgency", "countdown timer", "FOMO panic")),
    Motif("concept_20", "concept", "Reciprocity trap", "Free gift obligation compliance - 'I gave, you owe' - ancient wiring.", "ancient wiring", ("reciprocity trap", "free gift obligation", "ancient wiring")),

    # ---- surreal (20) ----
    Motif("surreal_01", "surreal", "Brain as control room", "Tiny operators at consoles pulling levers labeled 'FEAR' 'DESIRE' 'ANGER'.", "operators, levers", ("brain control room", "tiny operators", "fear desire anger")),
    Motif("surreal_02", "surreal", "Heart in cage", "Anatomical heart in birdcage, singing beautifully, door open but won't leave.", "singing, won't leave", ("heart in cage", "birdcage heart", "won't leave open door")),
    Motif("surreal_03", "surreal", "Eyes on stalks", "Figure with eyes on long stalks watching own back - hypervigilance embodied.", "stalks, hypervigilance", ("eyes on stalks", "hypervigilance", "watching own back")),
    Motif("surreal_04", "surreal", "Mouth sewn shut", "Lips stitched with black thread, one stitch broken - 'I have things to say'.", "stitched, broken stitch", ("mouth sewn shut", "black thread", "broken stitch")),
    Motif("surreal_05", "surreal", "Head in cloud", "Literal cloud where head should be - thoughts weather, uncontrollable.", "cloud head", ("head in cloud", "thoughts weather", "uncontrollable mind")),
    Motif("surreal_06", "surreal", "Backpack of stones", "Figure carrying stones labeled '1998' '2003' '2011' - trauma weight.", "stones, years", ("backpack stones", "trauma weight", "labeled years")),
    Motif("surreal_07", "surreal", "Mask eating face", "Porcelain mask fusing with skin, becoming new face - identity consumed.", "fusing, consumed", ("mask eating face", "fusing skin", "identity consumed")),
    Motif("surreal_08", "surreal", "Puppeteer's hands", "Giant hands from clouds manipulating tiny world - but strings go UP further.", "hands, strings up", ("puppeteer hands", "giant hands", "strings go up")),
    Motif("surreal_09", "surreal", "Clockwork heart", "Open chest reveals gears - winding down - key missing - time borrowed.", "gears, missing key", ("clockwork heart", "winding down", "missing key")),
    Motif("surreal_10", "surreal", "Mirror maze infinite", "Infinite reflections - each slightly different - which is real?", "infinite, which real", ("mirror maze", "infinite reflections", "which is real")),
    Motif("surreal_11", "surreal", "Thought police", "Small uniformed figures in brain arresting 'UNAPPROVED THOUGHT' - censorship.", "arresting thoughts", ("thought police", "brain censorship", "unapproved thought")),
    Motif("surreal_12", "surreal", "Emotion chemistry", "Test tubes: 'LOVE' (pink) 'RAGE' (red) 'SHAME' (black) - mixing dangerous.", "test tubes, mixing", ("emotion chemistry", "test tubes", "mixing emotions")),
    Motif("surreal_13", "surreal", "Child driving car", "Adult body, child hands on wheel - 'I don't know how!' - regression.", "child hands, regression", ("child driving", "adult body", "regression")),
    Motif("surreal_14", "surreal", "Shadow therapist", "Patient on couch, shadow on wall doing therapy - unconscious healing.", "shadow therapy", ("shadow therapist", "unconscious healing", "shadow couch")),
    Motif("surreal_15", "surreal", "Memory library burning", "Bookshelves of memories aflame - 'forgetting is survival' - librarian weeping.", "burning, weeping", ("memory library", "forgetting survival", "librarian weeping")),
    Motif("surreal_16", "surreal", "Identity wardrobe", "Racks of person-suits - 'confident' 'victim' 'hero' - choosing daily costume.", "person suits, costume", ("identity wardrobe", "person suits", "daily costume")),
    Motif("surreal_17", "surreal", "Gaslight architecture", "MC Escher house - stairs to ceiling, doors to walls - reality unstable.", "escher, unstable", ("gaslight architecture", "escher house", "reality unstable")),
    Motif("surreal_18", "surreal", "Trauma time frozen", "Clock melted (Dali) at 3:17 - 'it's always 3:17 in here' - stuck.", "melted clock, stuck", ("trauma frozen", "melted clock", "always 3:17")),
    Motif("surreal_19", "surreal", "Soul retrieval", "Shaman figure catching glowing fragment from wind - 'you lost this'.", "shaman, fragment", ("soul retrieval", "shaman", "glowing fragment")),
    Motif("surreal_20", "surreal", "Phoenix ash", "Figure burning, ash forming new figure - not rebirth - TRANSFORMATION - different.", "transformation, different", ("phoenix ash", "transformation", "not rebirth different")),
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


assert len(MOTIFS) == 300, f"Expected 300 motifs, got {len(MOTIFS)}"
assert len(CATEGORIES) == 15, f"Expected 15 categories, got {len(CATEGORIES)}"