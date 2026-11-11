#!/usr/bin/env python3
"""Organize flat images into category subdirs and create/update manifest.json."""

import json
import os
import sys
from pathlib import Path

# Ensure experiments module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Category mapping based on image ID prefixes (from NEW generator)
CATEGORY_MAP = {
    "face": "face_expressions",
    "crowd": "crowd_scenes",
    "arch": "characters_archetypes",
    "diag": "societal_diagrams",
    # legacy mappings
    "body": "body_language",
    "shadow": "shadows",
    "control": "manipulation_tactics",
    "space": "environments",
    "text": "text_overlays",
    "surv": "surveillance_tech",
    "abstract": "abstract",
    "weather": "weather",
    "motion": "motion",
    "archetype": "characters_archetypes",
    "rel": "relationships",
    "heal": "healing_recovery",
    "concept": "concepts",
    "surreal": "surreal",
    "manip": "manipulation_tactics",
    "env": "environments",
}

OUT_DIR = Path("assets/psychology_motifs")

# Move files into category subdirs
for img_path in OUT_DIR.glob("*.png"):
    img_id = img_path.stem
    prefix = img_id.split("_")[0]
    category = CATEGORY_MAP.get(prefix, "unknown")
    
    cat_dir = OUT_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    target = cat_dir / img_path.name
    if not target.exists():
        img_path.rename(target)
        print(f"Moved {img_path.name} -> {category}/")

# Load existing manifest for categories not on disk
old_manifest_path = OUT_DIR / "manifest.json"
existing = {}
if old_manifest_path.exists():
    try:
        existing = json.loads(old_manifest_path.read_text())
    except:
        pass

# Build a description lookup from persistent description store
from experiments.description_store import load_descriptions

existing_descriptions = load_descriptions()
print(f"Loaded {len(existing_descriptions)} descriptions from persistent store")

# Create merged manifest
manifest = {
    "generated_at": __import__("time").time(),
    "style": "dark cinematic noir psychological thriller, 9:16 vertical, near-black gradient, heavy vignette, film grain, centered subject, high contrast, shallow depth of field, colored accent rim light (cyan/amber/magenta), photorealistic 8k, no text, no watermark, no logos, original composition, unsettling atmosphere",
    "resolution": "1080x1920",
    "model": "flux",
    "categories": {}
}

# Category descriptions
CATEGORY_INFO = {
    "face_expressions": {"description": "Facial expressions - both men and women, micro-expressions, stares, skepticism, realization, manipulation"},
    "crowd_scenes": {"description": "Crowd scenarios - protests, queues, rallies, shops, festivals, evacuations, voting lines"},
    "characters_archetypes": {"description": "Societal archetypes - politicians, CEOs, healers, tricksters, tyrants, nurturers, scapegoats, survivors"},
    "societal_diagrams": {"description": "Data visualizations - inequality, discrimination, politics, economics, both genders"},
    # legacy
    "faces": {"description": "Facial expressions, micro-expressions, stares, masks"},
    "body_language": {"description": "Body language, power dynamics, postures, gestures"},
    "shadows": {"description": "Shadows, silhouettes, shadow manipulation"},
    "manipulation_tactics": {"description": "Control mechanisms, manipulation, gaslighting, tactics"},
    "environments": {"description": "Environments, rooms, architectural spaces"},
    "text_overlays": {"description": "Typography, written text, redacted documents, overlays"},
    "surveillance_tech": {"description": "CCTV, thermal, biometric, monitoring, tech"},
    "abstract": {"description": "Neural networks, fractals, metaphors, visual concepts"},
    "weather": {"description": "Atmospheric conditions, rain, fog, storms"},
    "motion": {"description": "Camera techniques, cinematic movements"},
    "relationships": {"description": "Interpersonal dynamics, trauma bonds, triangulation"},
    "healing_recovery": {"description": "Recovery, integration, growth, recovery symbols"},
    "concepts": {"description": "Psychological principles, cognitive biases"},
    "surreal": {"description": "Surreal metaphors, symbolic imagery"},
}

# Merge: use description store where available, supplement with filesystem
for cat_dir in sorted(OUT_DIR.iterdir()):
    if not cat_dir.is_dir() or cat_dir.name == "unknown":
        continue
    
    images = list(cat_dir.glob("*.png"))
    
    merged_images = []
    for img in sorted(images):
        img_id = img.stem
        # Get description from persistent store
        desc_data = existing_descriptions.get(img_id, {})
        
        if desc_data and desc_data.get("description"):
            merged_images.append({
                "id": img_id,
                "name": desc_data.get("name", img_id),
                "description": desc_data["description"],
                "filename": img.name,
                "category": desc_data.get("category", cat_dir.name),
            })
        else:
            merged_images.append({
                "id": img_id,
                "filename": img.name,
                "usage": f"Use for: {cat_dir.name} - {img_id}"
            })
    
    manifest["categories"][cat_dir.name] = {
        "description": CATEGORY_INFO.get(cat_dir.name, {}).get("description", ""),
        "count": len(merged_images),
        "images": merged_images
    }

# Also include categories that exist in manifest but not on disk yet
for cat_name, cat_data in existing.get("categories", {}).items():
    if cat_name not in manifest["categories"]:
        manifest["categories"][cat_name] = cat_data

(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
total = sum(c['count'] for c in manifest['categories'].values())
print(f"Manifest created with {total} images across {len(manifest['categories'])} categories")