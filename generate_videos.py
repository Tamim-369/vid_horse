#!/usr/bin/env python3
"""Generate 3 videos using the pipeline API with resource-aware execution."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from catalog.store import TheoryCatalog
from config import get_settings, setup_logging
from core.models import ScriptFormat
from pipeline import build_content_pipeline
from render.assemble import render
from utils import computing
from voice.tts import VoiceOver

logger = logging.getLogger(__name__)


def generate_video(slug: str, fmt: ScriptFormat = "short") -> bool:
    """Generate one video through the full pipeline."""
    
    settings = get_settings()
    output_dir = Path(settings.output_dir)
    directory = output_dir / slug
    
    # Check if already completed
    video_path = directory / f"{slug}-{fmt}.mp4"
    if video_path.exists():
        logger.info(f"{slug}: video already exists at {video_path}")
        return True
    
    # Load theory from catalog
    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slug)
    
    if theory is None:
        logger.error(f"{slug}: theory not found in catalog")
        return False
    
    logger.info(f"=== Starting pipeline for {slug} ({fmt}) ===")
    started = time.monotonic()
    
    # Build pipeline
    pipeline = build_content_pipeline(settings)
    
    # Stage 1: Research
    brief_path = directory / "brief.json"
    if not brief_path.exists():
        logger.info(f"{slug}: running research...")
        stage_start = time.monotonic()
        brief = pipeline.research(theory)
        directory.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"{slug}: research complete ({time.monotonic() - stage_start:.1f}s)")
    else:
        from core.models import ResearchBrief
        brief = ResearchBrief.model_validate_json(brief_path.read_text(encoding="utf-8"))
        logger.info(f"{slug}: using cached research")
    
    # Stage 2: Script generation
    script_path = directory / f"{fmt}.json"
    if not script_path.exists():
        logger.info(f"{slug}: generating {fmt} script...")
        stage_start = time.monotonic()
        target_seconds = settings.script_short_seconds if fmt == "short" else settings.script_long_seconds
        script = pipeline.script(brief, fmt, target_seconds=target_seconds)
        
        # Add music annotation
        from blocks.music import annotate_music
        from providers.llm import create_llm_provider
        script = annotate_music(create_llm_provider(settings), script)
        
        script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"{slug}: script complete ({time.monotonic() - stage_start:.1f}s)")
    else:
        from core.models import VideoScript
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        logger.info(f"{slug}: using cached script")
    
    # Stage 3: Asset matching
    assets_path = directory / f"assets-{fmt}.json"
    if not assets_path.exists():
        logger.info(f"{slug}: matching assets...")
        stage_start = time.monotonic()
        trace: dict[str, str] = {}
        
        # Try with visual direction first, fallback to simple matching if it fails
        try:
            pack = pipeline.assets(theory, brief, script, trace=trace)
        except Exception as e:
            logger.warning(f"{slug}: visual direction failed ({e}), using simple asset matching")
            # Fallback: use simple asset research without visual direction
            if pipeline._asset_research is not None:
                pack = pipeline._asset_research.run(theory, brief, script, queries=None)
            else:
                raise
        
        # Save trace files
        for name, content in trace.items():
            trace_path = directory / name
            trace_path.write_text(content, encoding="utf-8")
        
        assets_path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"{slug}: assets complete ({time.monotonic() - stage_start:.1f}s)")
    else:
        from core.models import AssetPack
        pack = AssetPack.model_validate_json(assets_path.read_text(encoding="utf-8"))
        logger.info(f"{slug}: using cached assets")
    
    # Stage 4: Voice synthesis
    audio_dir = directory / "audio"
    stem = "voice" if fmt == "short" else f"voice-{fmt}"
    voice_path = audio_dir / f"{stem}.wav"
    
    if not voice_path.exists():
        logger.info(f"{slug}: synthesizing voice...")
        stage_start = time.monotonic()
        vo = VoiceOver()
        voice_path = vo.render_script(script, audio_dir, stem=stem)
        logger.info(f"{slug}: voice complete ({time.monotonic() - stage_start:.1f}s)")
    else:
        logger.info(f"{slug}: using cached voice")
    
    # Stage 5: Video render
    logger.info(f"{slug}: rendering video...")
    stage_start = time.monotonic()
    
    # Get genre for music selection
    genre = brief.genre if brief else None
    
    try:
        render(
            script,
            pack,
            audio_dir,
            video_path,
            stem=stem,
            genre=genre,
            music="",
        )
        logger.info(f"{slug}: render complete ({time.monotonic() - stage_start:.1f}s)")
        logger.info(f"✓ {slug}: VIDEO READY -> {video_path}")
        logger.info(f"  Total time: {time.monotonic() - started:.1f}s")
        return True
    except Exception as e:
        logger.error(f"{slug}: render failed: {e}")
        return False


def main():
    """Generate 3 videos on different topics."""
    
    settings = get_settings()
    setup_logging("INFO")
    
    # Log machine profile
    mode = computing.machine()
    budget = mode.total_gb * 0.8  # Reserve 20%
    logger.info(f"Machine: {mode.cores} cores / {mode.total_gb:.1f}G RAM")
    logger.info(f"Budget: {budget:.1f}G (reserving 20% for system)")
    logger.info(f"Stage weights: {computing.stage_weight('voice')}")
    
    # Selected topics (diverse and interesting)
    topics = [
        "flat-earth",           # Classic conspiracy
        "illuminati",           # Secret society
        "deep-state",           # Political conspiracy
    ]
    
    logger.info(f"\nGenerating {len(topics)} videos: {', '.join(topics)}")
    
    results = {}
    for slug in topics:
        try:
            success = generate_video(slug, fmt="short")
            results[slug] = "✓" if success else "✗"
        except Exception as e:
            logger.error(f"{slug}: pipeline failed: {e}")
            results[slug] = "✗"
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS:")
    for slug, status in results.items():
        logger.info(f"  {status} {slug}")
    
    succeeded = sum(1 for s in results.values() if s == "✓")
    logger.info(f"\nCompleted: {succeeded}/{len(topics)} videos")


if __name__ == "__main__":
    main()
