#!/usr/bin/env python3
"""
Focused psychology video image generator - EXACTLY what you specified.
4 categories, practical images for psychology shorts.
"""

import json
import os
import sys
import random
import time
import urllib.parse
from pathlib import Path

# Ensure experiments module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

# ─── CONFIG ────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

POLLINATIONS_API = "https://gen.pollinations.ai/image/"

API_KEYS = [
    os.environ.get("POLLINATIONS_API_KEY_FIRST"),
    os.environ.get("POLLINATIONS_API_KEY_TWO"),
    os.environ.get("POLLINATIONS_API_KEY_THREE"),
    os.environ.get("POLLINATIONS_API_KEY_FOUR"),
]
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise RuntimeError("No Pollinations API keys found. Set POLLINATIONS_API_KEY_FIRST, _TWO, _THREE, _FOUR")

MODELS = [
    "black-forest-labs/flux.1-schnell",
    "black-forest-labs/flux.1-dev",
    "black-forest-labs/flux.1.1-pro",
]
WIDTH = 1080
HEIGHT = 1920
OUT_DIR = Path("assets/psychology_motifs")
MANIFEST_PATH = OUT_DIR / "manifest.json"

_key_index = 0
_model_index = 0
_key_failures = {i: 0 for i in range(len(API_KEYS))}

STYLE = (
    "dark cinematic noir psychological, 9:16 vertical, near-black gradient, "
    "heavy vignette, film grain, centered subject, high contrast, "
    "shallow depth of field, colored accent rim light (cyan/amber/magenta), "
    "photorealistic 8k, no text, no watermark, no logos"
)

# ─── EXACTLY 4 CATEGORIES YOU SPECIFIED ────────────────────────────────

# 1. FACE EXPRESSIONS - Both men and women, all types
SCENARIOS = {
    "face_expressions": [
        # Micro-expressions (both genders)
        ("face_01", "Man micro-expression freeze", "Extreme close-up man mid-micro-expression, eyebrow raised, mouth twitching, suppressed emotion"),
        ("face_02", "Woman micro-expression freeze", "Extreme close-up woman mid-micro-expression, eyebrow raised, mouth twitching, suppressed emotion"),
        ("face_03", "Man dead-eye stare", "Unblinking stare into lens, dilated pupils, hollow eyes, clinical flat lighting"),
        ("face_04", "Woman dead-eye stare", "Unblinking stare into lens, dilated pupils, hollow eyes, clinical flat lighting"),
        ("face_05", "Man half-shadow face", "Face split by hard vertical shadow, one eye hidden, dramatic chiaroscuro"),
        ("face_06", "Woman half-shadow face", "Face split by hard vertical shadow, one eye hidden, dramatic chiaroscuro"),
        ("face_07", "Man fake smile", "Polite smile with cold empty eyes, no crow's feet, uncanny valley"),
        ("face_08", "Woman fake smile", "Polite smile with cold empty eyes, no crow's feet, uncanny valley"),
        ("face_09", "Man tears without emotion", "Tears rolling down blank face, no muscle movement in forehead/jaw"),
        ("face_10", "Woman tears without emotion", "Tears rolling down blank face, no muscle movement in forehead/jaw"),
        ("face_11", "Man skeptical raised brow", "One eyebrow raised, narrowed eyes, head tilted, 'prove it' expression"),
        ("face_12", "Woman skeptical raised brow", "One eyebrow raised, narrowed eyes, head tilted, 'prove it' expression"),
        ("face_13", "Man sudden realization", "Eyes widen, mouth parts, eyebrows shoot up, 'aha' moment frozen"),
        ("face_14", "Woman sudden realization", "Eyes widen, mouth parts, eyebrows shoot up, 'aha' moment frozen"),
        ("face_15", "Man manipulative charm", "Charismatic smile, head tilted, calculating eyes, predator disguise"),
        ("face_16", "Woman manipulative charm", "Charismatic smile, head tilted, calculating eyes, predator disguise"),
        ("face_17", "Man psychopathic flat affect", "Completely relaxed, zero tension, shark eyes, terrifyingly calm"),
        ("face_18", "Woman psychopathic flat affect", "Completely relaxed, zero tension, shark eyes, terrifyingly calm"),
        ("face_19", "Man gaslighting double reflection", "Mirror shows two expressions - face calm, reflection screaming"),
        ("face_20", "Woman gaslighting double reflection", "Mirror shows two expressions - face calm, reflection screaming"),
    ],

    # 2. CROWD SCENES - Protests, lines, shops, all types
    "crowd_scenes": [
        ("crowd_01", "Protest crowd wide", "Massive protest crowd wide shot, signs raised, diverse people, aerial perspective"),
        ("crowd_02", "Protest crowd close", "Protest crowd close-up, faces determined, chanting, banners, raw emotion"),
        ("crowd_03", "Shop queue", "Long line outside store, people waiting, shopping bags, impatient expressions"),
        ("crowd_04", "Black Friday stampede", "Chaotic store entrance, people pushing, frantic shopping, consumer frenzy"),
        ("crowd_05", "Rally speaker crowd", "Crowd facing stage, arms raised, listening intently, unified direction"),
        ("crowd_06", "Commuter rush hour", "Packed subway platform, briefcases, tired faces, synchronized movement"),
        ("crowd_07", "Festival crowd", "Music festival dense crowd, hands up, phones recording, collective euphoria"),
        ("crowd_08", "Evacuation crowd", "People fleeing, panic in eyes, carrying belongings, desperate movement"),
        ("crowd_09", "Voting line", "Long line at polling station, diverse ages, patient waiting, civic duty"),
        ("crowd_10", "Food bank line", "People queuing for food assistance, humble expressions, survival mode"),
        ("crowd_11", "Job fair crowd", "Job seekers in suits, resumes in hand, anxious hope, competitive atmosphere"),
        ("crowd_12", "Religious gathering", "Packed congregation, hands raised in prayer, unified belief, devotion"),
        ("crowd_13", "Sports riot", "Stadium chaos, fans clashing, team colors, tribal aggression"),
        ("crowd_14", "Pride parade", "Colorful diverse crowd, flags waving, celebration, visibility, acceptance"),
        ("crowd_15", "Refugee column", "Long line of displaced people, bags, children, exhaustion, displacement"),
        ("crowd_16", "Concert mosh pit", "Aggressive pushing, sweaty bodies, raw energy, collective abandon"),
        ("crowd_17", "Lecture hall", "Packed auditorium, students listening, notebooks, attention directed forward"),
        ("crowd_18", "Town hall meeting", "Community members seated, raised hands, democratic process, civic engagement"),
        ("crowd_19", "Black Friday campout", "Tents outside store overnight, sleeping bags, consumer dedication"),
        ("crowd_20", "Eviction protest", "Tenants with signs, landlords vs tenants, housing crisis, displacement"),
    ],

    # 3. CHARACTERS & ARCHETYPES - Society roles, both genders
    "characters_archetypes": [
        ("arch_01", "Machiavellian politician", "Sharp suit, cold eyes, slight smirk, calculating value, power broker"),
        ("arch_02", "Corporate psychopath CEO", "Corner office, expensive suit, cold gaze, decisions without conscience"),
        ("arch_03", "Wounded healer therapist", "Androgynous, hands glowing over client, own chest bleeding, empathy cost"),
        ("arch_04", "Trickster media personality", "Face morphing expressions, never settling, manufactured authenticity"),
        ("arch_05", "Tyrannical father figure", "Large silhouette, belt in hand, absolute authority, fear worshipped"),
        ("arch_06", "Devouring mother", "Embrace becoming trap, arms multiplying, 'only want what's best' suffocating"),
        ("arch_07", "Golden child prodigy", "Perfect glowing on pedestal, cracks beneath gold leaf, fragile perfection"),
        ("arch_08", "Family scapegoat", "Covered in projected shadows - 'lazy' 'crazy' - carrying all burden"),
        ("arch_09", "Lost child invisible", "Oversized adult clothes, hiding in plain sight, forgotten"),
        ("arch_10", "Mascot performer", "Painted smile, tears underneath, performing for survival, 'everything's fine'"),
        ("arch_11", "Wise grandmother elder", "Ancient woman, stirring pot, steam forming visions, knows true names"),
        ("arch_12", "Dark tech billionaire", "Hoodie, weaving algorithmic fate, audience unaware they're puppets"),
        ("arch_13", "Innocent idealist", "Walking off cliff eyes closed, smile trusting, universe catches or doesn't"),
        ("arch_14", "Battle-scarred veteran", "Armor dented, fighting internal dragons, exhausted noble, PTSD"),
        ("arch_15", "Siren influencer", "Half in water, song as golden waves, love equals drowning, algorithmic siren"),
        ("arch_16", "Hermit off-grid", "Alone on mountain, light held high, not for others - own path, autonomy"),
        ("arch_17", "Corrupted judge", "Scales tipped by gold, sword rusted, peek through blindfold, justice sold"),
        ("arch_18", "Death as therapist", "Skeleton in chair taking notes, 'and how did that make you feel?'"),
        ("arch_19", "Inner child locked", "Small figure behind barred window in adult chest, knocking, muffled"),
        ("arch_20", "Integrated whole human", "All archetypes in council around center, wholeness achieved, self-actualized"),
        ("arch_21", "Manipulative narcissist woman", "Perfect makeup, calculating eyes, love bombing then discarding"),
        ("arch_22", "Codependent enabler", "Carrying adult partner, knees buckling, 'I can fix them' exhaustion"),
        ("arch_23", "Gaslighting partner", "Flashlight under chin, 'you're imagining the light', denying reality"),
        ("arch_24", "Gray rock survivor", "Featureless stone, attacker's tendrils sliding off, boredom as armor"),
        ("arch_25", "Cult leader charismatic", "Hands raised, followers entranced, manufactured divinity, absolute control"),
    ],

    # 4. SOCIETAL DIAGRAMS - Inequality, politics, discrimination, both genders
    "societal_diagrams": [
        ("diag_01", "Income inequality pyramid", "Wealth pyramid: tiny top 1% glowing gold, massive base struggling, stark divide"),
        ("diag_02", "Wealth distribution chart", "Bar chart: 1% owns 50% wealth, 90% fights for crumbs, logarithmic scale"),
        ("diag_03", "Gender pay gap visual", "Two identical workers, male getting full dollar, female getting 82 cents"),
        ("diag_04", "Racial wealth gap", "Generational wealth transfer: white family compounding, Black family reset to zero"),
        ("diag_05", "Political polarization", "Two circles drifting apart, overlap shrinking, 'us vs them' polarization"),
        ("diag_06", "Media ownership concentration", "Network diagram: 6 corporations owning 90% media, arrows to outlets"),
        ("diag_07", "Lobbying money flow", "Money pipe from corporations to politicians, legislation output, corruption cycle"),
        ("diag_08", "School-to-prison pipeline", "Flowchart: underfunded schools → zero tolerance → juvenile detention → adult prison"),
        ("diag_09", "Housing cost vs wage", "Diverging lines: housing costs exponential, wages flat, gap widening"),
        ("diag_10", "Healthcare access map", "Map: dark areas = no hospitals, bright = elite care, zip code determines lifespan"),
        ("diag_11", "Algorithmic bias", "Facial recognition: white male 99% accurate, Black female 65%, discrimination coded"),
        ("diag_12", "Climate justice", "Map: Global North emissions, Global South consequences, inverted responsibility"),
        ("diag_13", "Union membership decline", "Line graph: union density falling 1950-present, wages stagnating in parallel"),
        ("diag_14", "Student debt crisis", "Ballooning debt bubble, degrees devaluing, generation delayed adulthood"),
        ("diag_15", "Retirement insecurity", "Three-legged stool broken: pension gone, Social Security threatened, 401k gambling"),
        ("diag_16", "Digital divide", "Split screen: high-speed fiber vs no internet, education/opportunity gap"),
        ("diag_17", "Food desert map", "Map: green = fresh food, red = only convenience stores, nutrition by zip code"),
        ("diag_18", "Maternal mortality race", "Bar chart: Black women 3x mortality rate, systemic bias in healthcare"),
        ("diag_19", "Mass incarceration", "Bar chart: US 4% world population, 25% prisoners, racial disproportionality"),
        ("diag_20", "Surveillance capitalism", "Data flow: every click → profile → prediction → behavior modification → profit"),
        ("diag_21", "Generational wealth transfer", "Family tree: compounding assets vs starting from zero each generation"),
        ("diag_22", "Corporate tax avoidance", "Flowchart: profits → tax havens → public services starved → inequality"),
        ("diag_23", "Disability employment gap", "Chart: employment rate by disability status, systemic barriers visualized"),
        ("diag_24", "LGBTQ+ rights patchwork", "Map: rights vary by state/country, human rights as geography lottery"),
        ("diag_25", "Ageism in hiring", "Resume stack: 'overqualified' stamp on 50+, 'lacks experience' on 20s"),
        ("diag_26", "Rent burden crisis", "Heat map: % income to rent, red zones = 50%+ income, working homeless"),
        ("diag_27", "Police violence disparity", "Chart: deaths per capita by race, unarmed, accountability near zero"),
        ("diag_28", "Voter suppression map", "Map: polling places closed, ID laws, gerrymandered districts, democracy erosion"),
        ("diag_29", "Pharma pricing", "Graph: R&D cost vs price markup, insulin $6 to make $300 to buy"),
        ("diag_29b", "Mental health access", "Map: therapist deserts, insurance barriers, crisis lines overwhelmed"),
        ("diag_30", "Intersectionality matrix", "Venn diagram: race × gender × class × sexuality = compounding disadvantage"),
    ],
}

# ─── GENERATION FUNCTIONS ─────────────────────────────────────────────

_key_index = 0
_model_index = 0
_key_failures = {i: 0 for i in range(len(API_KEYS))}

def get_next_key() -> str:
    global _key_index, _key_failures
    for _ in range(len(API_KEYS)):
        if _key_failures[_key_index] < 3:
            key = API_KEYS[_key_index]
            _key_index = (_key_index + 1) % len(API_KEYS)
            return key
        _key_index = (_key_index + 1) % len(API_KEYS)
    _key_failures = {i: 0 for i in range(len(API_KEYS))}
    return API_KEYS[0]

def get_next_model() -> str:
    global _model_index
    model = MODELS[_model_index]
    _model_index = (_model_index + 1) % len(MODELS)
    return model

def record_failure(key: str):
    global _key_failures
    try:
        idx = API_KEYS.index(key)
        _key_failures[idx] += 1
    except ValueError:
        pass


def generate_image(client: httpx.Client, category: str, img_id: str, name: str, description: str) -> bool:
    category_dir = OUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    out_path = category_dir / f"{img_id}.png"
    
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ✓ Exists: {category}/{img_id}.png")
        return True
    
    prompt = f"{description}, {name}. {STYLE}"
    encoded = urllib.parse.quote(prompt)
    
    for _ in range(len(API_KEYS) * len(MODELS)):
        key = get_next_key()
        model = get_next_model()
        
        url = f"{POLLINATIONS_API}{encoded}?model={model}&width={WIDTH}&height={HEIGHT}&nologo=true"
        headers = {"Authorization": f"Bearer {key}"}
        
        for attempt in range(1, 3):
            try:
                resp = client.get(url, headers=headers, timeout=180, follow_redirects=True)
                if resp.status_code == 402:
                    print(f"  ⚠ Key exhausted ({key[:12]}...), rotating...")
                    record_failure(key)
                    break
                if resp.status_code == 429:
                    wait = min(2 * (2 ** attempt) + random.uniform(0, 1), 60)
                    print(f"  ⏳ Rate limited, waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    print(f"  ⚠ API error {resp.status_code}: {resp.text[:200]}")
                    record_failure(key)
                    break
                resp.raise_for_status()
                
                out_path.write_bytes(resp.content)
                print(f"  ✓ Generated: {category}/{img_id}.png (model: {model})")
                return True
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (402, 429):
                    record_failure(key)
                    break
                if attempt == 2:
                    raise
                time.sleep(2)
            except Exception as e:
                if attempt == 2:
                    print(f"  ✗ Failed {category}/{img_id}: {e}")
                    return False
                time.sleep(2)
    
    print(f"  ✗ All keys/models exhausted for {category}/{img_id}")
    return False


def main():
    print(f"Generating psychology motifs to {OUT_DIR}")
    total = sum(len(v) for v in SCENARIOS.values())
    print(f"Total: {total} images across {len(SCENARIOS)} categories")
    
    # Skip completed
    completed = set()
    for cat_dir in OUT_DIR.iterdir():
        if cat_dir.is_dir() and cat_dir.name != "unknown":
            images = list(cat_dir.glob("*.png"))
            if len(images) >= 20:
                completed.add(cat_dir.name)
    
    if completed:
        print(f"Skipping completed: {sorted(completed)}")
    
    client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0), follow_redirects=True)
    manifest = {
        "generated_at": time.time(),
        "style": STYLE,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "model": MODELS[0],
        "categories": {}
    }
    
    try:
        # Load existing manifest to preserve completed categories
        existing_manifest = {"categories": {}}
        if MANIFEST_PATH.exists():
            try:
                existing_manifest = json.loads(MANIFEST_PATH.read_text())
            except:
                pass
        
        for category, items in SCENARIOS.items():
            if category in completed:
                print(f"\nSkipping {category} (already done)")
                # Preserve existing data for completed categories
                if category in existing_manifest.get("categories", {}):
                    manifest["categories"][category] = existing_manifest["categories"][category]
                continue
            
            print(f"\n{'='*50}")
            print(f"Category: {category} ({len(items)} images)")
            print(f"{'='*50}")
            
            manifest["categories"][category] = []
            
            for img_id, name, description in items:
                print(f"  [{img_id}] {name}")
                success = generate_image(client, category, img_id, name, description)
                
                manifest["categories"][category].append({
                    "id": img_id,
                    "name": name,
                    "description": description,
                    "filename": f"{img_id}.png",
                    "category": category,
                    "generated": success,
                    "usage": f"Use for: {category} - {name.lower()}"
                })
                
                # Update persistent description store
                from experiments.description_store import update_descriptions
                update_descriptions({
                    img_id: {
                        "id": img_id,
                        "name": name,
                        "description": description,
                        "category": category,
                    }
                })
                
                time.sleep(0.5)
        
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        print(f"\n✅ Manifest saved to {MANIFEST_PATH}")
        
        success = sum(1 for cat in manifest["categories"].values() for item in cat if item["generated"])
        print(f"\nComplete: {success}/{total} images generated")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()