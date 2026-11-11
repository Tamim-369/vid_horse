#!/usr/bin/env python3
"""Persistent description store - never overwritten, only appended."""

import json
from pathlib import Path

STORE_PATH = Path("assets/psychology_motifs/descriptions.json")


def load_descriptions() -> dict:
    """Load all known descriptions."""
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text(encoding="utf-8"))
        except:
            pass
    return {}


def save_descriptions(descriptions: dict):
    """Save descriptions atomically."""
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False))
    tmp.replace(STORE_PATH)


def update_descriptions(new_descriptions: dict):
    """Add new descriptions, preserving existing ones."""
    existing = load_descriptions()
    for img_id, data in new_descriptions.items():
        if img_id not in existing or not existing[img_id].get("description"):
            existing[img_id] = data
    save_descriptions(existing)


def get_description(img_id: str) -> str | None:
    """Get description for an image ID."""
    descriptions = load_descriptions()
    return descriptions.get(img_id, {}).get("description")


def get_full_data(img_id: str) -> dict | None:
    """Get full data for an image ID."""
    descriptions = load_descriptions()
    return descriptions.get(img_id)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Test lookup
        img_id = sys.argv[1]
        data = get_full_data(img_id)
        if data:
            print(json.dumps(data, indent=2))
        else:
            print(f"No data for {img_id}")
    else:
        # Print all
        desc = load_descriptions()
        print(f"Total stored: {len(desc)}")
        for k, v in list(desc.items())[:10]:
            print(f"  {k}: {v.get('description', '')[:80]}")