#!/usr/bin/env python3
"""Test Script 1: Basic Pipeline Test
Tests: lesson_picker -> script_engine -> sentence_assets -> voice -> render_black_style
Uses a simple hardcoded lesson to verify the full chain works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings, setup_logging
from blocks.lesson_picker import LessonIdea
from blocks.script_engine.block import ScriptEngineBlock
from blocks.sentence_assets import build_asset_pack
from blocks.music import annotate_music
from providers.llm import create_llm_provider
from render.black_style import render_black_style
from voice.tts import VoiceOver
from core.models import VideoScript


def main() -> int:
    setup_logging("INFO")
    settings = get_settings()

    # Create a test lesson idea (bypassing lesson_picker for speed)
    test_idea = LessonIdea(
        topic="The Illusion of Control",
        lesson="You overestimate your influence on outcomes, especially when chance plays a large role.",
        source_chunks=[
            "The illusion of control is the tendency for people to overestimate their ability to control events.",
            "When people are asked to predict the outcome of a coin toss, they feel more confident when they toss the coin themselves.",
            "In lottery experiments, people pay more for tickets they choose themselves versus randomly assigned ones."
        ],
        context_notes="Context: introductory chapter on cognitive biases"
    )

    # Create output directory
    slug = "test-01-illusion-of-control"
    out_dir = Path("output") / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TEST 1: Basic Pipeline - {test_idea.topic}")
    print(f"{'='*60}\n")

    # 1. Script Generation
    print("[1/5] Generating script...")
    llm = create_llm_provider(settings)
    engine = ScriptEngineBlock(llm, target_seconds=45)
    script = engine.run(
        test_idea.topic,
        source_chunks=test_idea.source_chunks,
        context_note="cognitive biases"
    )
    print(f"    Generated {len(script.scenes)} scenes, ~{script.target_seconds}s target")

    # Save script
    (out_dir / "short.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    print(f"    Script saved to {out_dir}/short.json")

    # 2. Music Annotation
    print("[2/5] Annotating music...")
    try:
        script = annotate_music(llm, script)
        print(f"    Music plan: base_role={script.music.base_role if script.music else 'none'}")
    except Exception as e:
        print(f"    Music annotation failed (continuing): {e}")

    # 3. Asset Pack (Pollinations images)
    print("[3/5] Building asset pack (generating Pollinations images)...")
    pack = build_asset_pack(script, slug=slug)
    (out_dir / "assets-short.json").write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    total_assets = sum(len(s.assets) for s in pack.scenes)
    print(f"    Generated {total_assets} visual assets across {len(pack.scenes)} scenes")

    # 4. Voice Generation
    print("[4/5] Generating voiceover (Chatterbox TTS)...")
    vo = VoiceOver()
    voice_path = vo.render_script(script, audio_dir, stem="voice")
    print(f"    Voice saved to {voice_path}")

    # 5. Render Black-Style Video
    print("[5/5] Rendering video (black-style with subtitles)...")
    video_path = out_dir / f"{slug}-short-black.mp4"
    render_black_style(
        script, pack, audio_dir, video_path,
        stem="voice", genre=None, music="", slug=slug
    )
    print(f"    Video saved to {video_path}")

    # Verify outputs
    print(f"\n{'='*60}")
    print("VERIFICATION:")
    print(f"{'='*60}")
    for f in ["short.json", "assets-short.json", "audio/voice.wav", f"{slug}-short-black.mp4"]:
        p = out_dir / f
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {f} MISSING")

    # Print script for review
    print(f"\nSCRIPT:")
    print(f"  Hook: {script.hook}")
    for i, scene in enumerate(script.scenes):
        print(f"  Scene {i+1}: {scene.narration[:80]}... | Visual: {scene.visual}")
    print(f"  Outro: {script.outro}")

    return 0


if __name__ == "__main__":
    sys.exit(main())