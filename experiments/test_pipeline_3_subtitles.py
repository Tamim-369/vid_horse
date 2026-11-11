#!/usr/bin/env python3
"""Test Script 3: Subtitle Timing & Alignment Test
Tests: Custom script -> voice -> whisper alignment -> subtitle burning -> render
Uses a hand-crafted script with known timing to verify subtitle accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings, setup_logging
from core.models import VideoScript, ScriptScene, ScriptFormat, MusicPlan
from blocks.sentence_assets import build_asset_pack
from blocks.music import annotate_music
from providers.llm import create_llm_provider
from render.black_style import render_black_style
from voice.tts import VoiceOver


def main() -> int:
    setup_logging("INFO")
    settings = get_settings()

    # Create a test script with precise, measurable sentences
    # Each sentence is designed to be ~3-4 seconds for easy timing verification
    test_script = VideoScript(
        title="Subtitle Timing Test",
        hook="Watch how the words appear exactly when spoken.",
        format="short",
        target_seconds=45,
        scenes=[
            ScriptScene(
                narration="This is the first sentence of the test. It should last about three seconds.",
                visual="timer counting seconds",
                seconds=4,
            ),
            ScriptScene(
                narration="Second sentence appears right after the first one ends. No overlap, no gap.",
                visual="clean transition",
                seconds=4,
            ),
            ScriptScene(
                narration="Third sentence tests multi-word groups. Each pair shows together.",
                visual="word pairs highlight",
                seconds=4,
            ),
            ScriptScene(
                narration="Fourth sentence verifies timing accuracy. Whisper aligns word by word.",
                visual="waveform sync",
                seconds=4,
            ),
            ScriptScene(
                narration="Final sentence confirms the system works. Subtitles match voice perfectly.",
                visual="check mark",
                seconds=4,
            ),
        ],
        outro="Test complete. Subtitle timing verified.",
        music=MusicPlan(base_role="default"),
    )

    slug = "test-03-subtitle-timing"
    out_dir = Path("output") / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TEST 3: Subtitle Timing & Alignment")
    print(f"{'='*60}\n")

    # Save script for reference
    (out_dir / "short.json").write_text(test_script.model_dump_json(indent=2), encoding="utf-8")

    # 1. Asset Pack
    print("[1/4] Building asset pack...")
    pack = build_asset_pack(test_script, slug=slug)
    (out_dir / "assets-short.json").write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    total_assets = sum(len(s.assets) for s in pack.scenes)
    print(f"    Generated {total_assets} visual assets")

    # 2. Voice Generation
    print("[2/4] Generating voiceover...")
    vo = VoiceOver()
    voice_path = vo.render_script(test_script, audio_dir, stem="voice")
    print(f"    Voice saved to {voice_path}")

    # 3. Render with Subtitles
    print("[3/4] Rendering video with word-by-word subtitles...")
    video_path = out_dir / f"{slug}-short-black.mp4"
    render_black_style(
        test_script, pack, audio_dir, video_path,
        stem="voice", genre=None, music="", slug=slug
    )
    print(f"    Video saved to {video_path}")

    # 4. Verify subtitle timing by checking individual line wavs
    print("[4/4] Verifying per-line audio files...")
    from render.media import _duration
    narration_lines = [test_script.hook] + [s.narration for s in test_script.scenes] + [test_script.outro]
    for i, line in enumerate(narration_lines):
        wav = audio_dir / f"voice-{i:02d}.wav"
        if wav.exists():
            dur = _duration(wav)
            print(f"  Line {i}: {dur:.2f}s - {line[:60]}...")
        else:
            print(f"  Line {i}: MISSING - {line[:60]}...")

    # Final verification
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

    # Print expected subtitle groups
    print(f"\nEXPECTED SUBTITLE GROUPS (2 words each):")
    for i, line in enumerate(narration_lines):
        words = line.split()
        groups = [" ".join(words[j:j+2]) for j in range(0, len(words), 2)]
        print(f"  Line {i}: {' | '.join(groups)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())