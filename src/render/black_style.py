"""Black-screen conceptual editing style.

Editing logic:
- Default: BLACK screen
- When narration describes a scene/concept → show the concept's visual asset
- When narration hits a KEY CONCEPT → cut to BLACK + key word(s) in white
- No subtitles ever
- Music bed per video from music assets
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from config.settings import get_settings
from config.video_style import load_style_config, VideoStyle
from core.errors import BlockError
from core.models import AssetPack, SfxPlan, VideoScript, VisualAsset
from utils import computing

logger = logging.getLogger(__name__)

FPS = 30
WIDTH = 1080
HEIGHT = 1920


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _black_frame(work: Path, index: int, length: float) -> Path:
    """Pure black frame for given length."""
    out = work / f"black-{index:02d}.mp4"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={WIDTH}x{HEIGHT}",
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _word_flash(work: Path, index: int, word: str, length: float, fontsize: int = 74) -> Path:
    """Black screen with single word/phrase centered in white."""
    out = work / f"flash-{index:02d}.mp4"
    safe_word = word.replace("'", "\\'").replace('"', '\\"')
    draw = f"text='{safe_word}':fontcolor=white:fontsize={fontsize}:fontfile=/usr/share/fonts/truetype/lato/Lato-Bold.ttf:x=(w-text_w)/2:y=(h-text_h)/2"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={WIDTH}x{HEIGHT}",
        "-vf", f"drawtext={draw}",
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _asset_clip(asset, work: Path, tag: str, length: float, style: VideoStyle) -> Path:
    """Render a concept asset with its concept-specific motion/grade."""
    from render.cards import _asset_source, _motion_filter, _style_for, _cover_prep

    clip = work / f"asset-{tag}.mp4"
    source = _asset_source(asset, work, tag)

    frames = max(1, round(length * FPS))
    motion_pool = style.image.motion_pool if style else list(range(24))
    motion_style = _style_for(tag, motion_pool)

    extra = getattr(asset, "extra", {}) or {}
    grade_emphasis = extra.get("grade_emphasis")

    tail = f"{_motion_filter(frames, motion_style, None)}"

    if grade_emphasis:
        grade_map = {
            "cold": "eq=contrast=1.15:saturation=0.15:brightness=-0.03:gamma=1.1,colorbalance=rs=-0.05:gs=-0.02:bs=0.08",
            "noir": "eq=contrast=1.25:saturation=0.35:brightness=-0.08:gamma=1.15,noise=alls=18:allf=t,vignette=PI/3.0",
            "raw": "eq=contrast=1.35:saturation=1.2:brightness=0.02:gamma=1.2,colorbalance=rs=0.15:gs=-0.1:bs=-0.15,vignette=PI/2.5",
            "academic": "eq=contrast=1.1:saturation=0.55:brightness=-0.05:gamma=1.08,colorbalance=rs=0.12:gs=0.06:bs=-0.04,noise=alls=12:allf=t",
            "tech": "eq=contrast=1.08:saturation=0.4:brightness=-0.02:gamma=1.05,colorbalance=rs=-0.08:gs=-0.04:bs=0.15",
            "neutral": "",
        }
        grade_filter = grade_map.get(extra.get("grade_emphasis"), "")
        if grade_filter:
            tail = f"{tail},{grade_filter}"

    vf = f"{_cover_prep()},{tail}"
    out = work / f"asset-{tag}.mp4"
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(source),
        "-vf", vf,
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _black_frame(work: Path, index: int, length: float) -> Path:
    out = work / f"black-{index:02d}.mp4"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={WIDTH}x{HEIGHT}",
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _word_flash(work: Path, index: int, word: str, length: float, fontsize: int = 74) -> Path:
    """Black screen with single word/phrase centered in white."""
    out = work / f"flash-{index:02d}.mp4"
    safe_word = word.replace("'", "\\'").replace('"', '\\"')
    draw = f"text='{safe_word}':fontcolor=white:fontsize={fontsize}:fontfile=/usr/share/fonts/truetype/lato/Lato-Bold.ttf:x=(w-text_w)/2:y=(h-text_h)/2"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={WIDTH}x{HEIGHT}",
        "-vf", f"drawtext={draw}",
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _asset_clip(asset, work: Path, tag: str, length: float, style: VideoStyle) -> Path:
    from render.cards import _asset_source, _motion_filter, _style_for, _cover_prep

    clip = work / f"asset-{tag}.mp4"
    source = _asset_source(asset, work, tag)

    frames = max(1, round(length * FPS))
    motion_pool = style.image.motion_pool if style else list(range(24))
    motion_style = _style_for(tag, motion_pool)

    extra = getattr(asset, "extra", {}) or {}
    grade_emphasis = extra.get("grade_emphasis")

    tail = f"{_motion_filter(frames, motion_style, None)}"

    if grade_emphasis:
        grade_map = {
            "cold": "eq=contrast=1.15:saturation=0.15:brightness=-0.03:gamma=1.1,colorbalance=rs=-0.05:gs=-0.02:bs=0.08",
            "noir": "eq=contrast=1.25:saturation=0.35:brightness=-0.08:gamma=1.15,noise=alls=18:allf=t,vignette=PI/3.0",
            "raw": "eq=contrast=1.35:saturation=1.2:brightness=0.02:gamma=1.2,colorbalance=rs=0.15:gs=-0.1:bs=-0.15,vignette=PI/2.5",
            "academic": "eq=contrast=1.1:saturation=0.55:brightness=-0.05:gamma=1.08,colorbalance=rs=0.12:gs=0.06:bs=-0.04,noise=alls=12:allf=t",
            "tech": "eq=contrast=1.08:saturation=0.4:brightness=-0.02:gamma=1.05,colorbalance=rs=-0.08:gs=-0.04:bs=0.15",
            "neutral": "",
        }
        grade_filter = grade_map.get(extra.get("grade_emphasis"), "")
        if grade_filter:
            tail = f"{tail},{grade_filter}"

    vf = f"{_cover_prep()},{tail}"
    out = work / f"asset-{tag}.mp4"
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(source),
        "-vf", vf,
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _join_cards(clips: list, work: Path) -> Path:
    concat_list = work / "concat.txt"
    concat_list.write_text("\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8")
    out = work / "cards.mp4"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(out),
    ])
    return out


def render_black_style(
    script,
    pack,
    audio_dir: Path,
    out_path: Path,
    *,
    stem: str = "voice",
    genre: str | None = None,
    music: str = "",
    slug: str | None = None,
) -> Path:
    """Render black-screen conceptual edit."""

    work = out_path.parent / "render-work"
    work.mkdir(parents=True, exist_ok=True)

    style_cfg = load_style_config(slug) if slug else VideoStyle.defaults()

    # Load narration audio
    narration_lines = [script.hook, *(s.narration for s in script.scenes), script.outro]
    wavs = []
    for i in range(len(narration_lines)):
        wav = audio_dir / f"voice-{i:02d}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"Missing narration {wav}")
        wavs.append(wav)

    # Get word-level timing from whisper
    from voice.subtitles import align_speech
    from render.media import _duration

    lengths = [_duration(w) for w in wavs]
    offsets = []
    start = 0.0
    for l in lengths:
        offsets.append(start)
        start += l
    line_ends = offsets[1:] + [start]

    word_lines, cues = align_speech(
        narration_lines, wavs, offsets, line_ends,
        model_size="tiny.en",
    )

    # Load music (variety: pick a track per video from the shelf so we never
    # ship the same bed for every video; content role biases the choice)
    from render.audio_mix import _shelf, _bed_tracks, _build_bed, _music_spans, _mix_audio, _voice_track
    import hashlib

    music_override = music
    if not music_override:
        shelf = _shelf()
        tracks = [t["file"] for t in shelf.get("tracks", []) if t.get("file") and t["file"] != "decrease_volume.mp3"]
        if tracks:
            seed = 0
            if slug:
                seed = int(hashlib.md5(slug.encode()).hexdigest(), 16)
            if getattr(script, "music", None) is not None and getattr(script.music, "base_role", None):
                role = script.music.base_role
                role_idx = {"threat": 0, "big-event": 1, "mystery": 2, "discovery": 3, "vast-reveal": 4, "dread-mission": 5, "default": 6}.get(role, 0)
                seed = seed + role_idx * 997
            music_override = tracks[seed % len(tracks)]
    cards = _bed_tracks(script, None, music_override)
    spans = _music_spans(cards, offsets, sum(_duration(w) for w in wavs))
    bed = _build_bed(spans, work, sum(_duration(w) for w in wavs))
    voice = _voice_track(wavs, lengths, work)
    mixed = _mix_audio(voice, bed, work)

    # Build concept-driven edit
    narration_lines_full = [script.hook, *(s.narration for s in script.scenes), script.outro]

    # Build asset lookup by concept (assets without a real url — fetch
    # failed — render as black cards instead of crashing the encode)
    asset_by_concept = {}
    for scene in pack.scenes:
        for asset in scene.assets:
            if not asset.url or not asset.file:
                continue  # placeholder (fetch failed) -> black card fallback
            if asset.extra and asset.extra.get("concept"):
                concept = asset.extra["concept"]
                if concept not in asset_by_concept:
                    asset_by_concept[concept] = []
                asset_by_concept[concept].append(asset)

    # Key words from emphasis
    from voice.emphasis import emphasize_cues
    from providers.llm import create_llm_provider
    from config.settings import get_settings as _gs

    _llm = create_llm_provider(_gs())
    narration_for_emphasis = [script.hook, *(s.narration for s in script.scenes), script.outro]
    cues = emphasize_cues(_llm, narration_for_emphasis, [])
    key_words = set()
    for cue in cues:
        for pop in cue.pops:
            key_words.add(pop.upper())

    # Build clips
    clips = []
    clip_idx = 0
    narration_lines_full = [script.hook, *(s.narration for s in script.scenes), script.outro]
    lengths = [0] * len(narration_lines_full)  # placeholder

    # Get actual durations
    from render.media import _duration
    for i, wav in enumerate([script.hook, *(s.narration for s in script.scenes), script.outro]):
        wav_path = out_path.parent / "audio" / f"voice-{i:02d}.wav"
        if wav_path.exists():
            lengths[i] = _duration(wav_path)

    # Track which asset files we've already shown so a video never
    # repeats the same image twice.
    _used_asset_files: set[str] = set()

    def _pick_asset_for_line(line_idx: int) -> VisualAsset | None:
        pool: list[VisualAsset] = []
        if line_idx == 0 and pack.scenes:
            pool.extend(pack.scenes[0].assets or [])
        elif 0 < line_idx < len(narration_lines_full) - 1 and line_idx < len(pack.scenes):
            pool.extend(pack.scenes[line_idx].assets or [])
        else:
            for sc in pack.scenes:
                pool.extend(sc.assets or [])
        # prefer unused
        for a in pool:
            if a.url and a.file and a.file not in _used_asset_files:
                _used_asset_files.add(a.file)
                return a
        # all used in this line's pool: search the whole pack for any unused image
        for sc in pack.scenes:
            for a in (sc.assets or []):
                if a.url and a.file and a.file not in _used_asset_files:
                    _used_asset_files.add(a.file)
                    return a
        for a in pool:
            if a.url:
                return a
        return None

    def _trim_clip(src: Path, work: Path, index: int, start: float, dur: float) -> Path:
        """Trim a clip to a specific time range."""
        out = work / f"trim-{index:02d}.mp4"
        _run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(out),
        ])
        return out

    def _overlay_subtitle_on_clip(src: Path, work: Path, index: int, text: str, length: float, fontsize: int = 74) -> Path:
        """Burn a centered subtitle over a clip segment. Auto-wraps long text."""
        out = work / f"sub-{index:02d}.mp4"
        # Wrap first, then escape for ffmpeg drawtext
        wrapped = _wrap_subtitle_text(text, fontsize)
        # Escape for ffmpeg drawtext: 
        # - Don't escape backslashes (needed for \n newlines)
        # - Escape colons
        # - Escape unescaped single quotes
        safe = wrapped.replace(":", "\\:")
        import re
        safe = re.sub(r"(?<!\\)'", "\\'", safe)
        print(f"DEBUG Overlay subtitle text sent to ffmpeg: {repr(safe)}")
        # No background box - just text with outline, centered
        vf = (
            f"drawtext=text='{safe}':fontcolor=white:fontsize={fontsize}:"
            f"fontfile=/usr/share/fonts/truetype/lato/Lato-Bold.ttf:"
            f"x=(w-text_w)/2:y=(h-text_h)/2"
        )
        _run([
            "ffmpeg", "-y", "-i", str(src),
            "-vf", vf,
            "-t", f"{length:.3f}", "-r", str(FPS),
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
            "-crf", "20", str(out),
        ])
        return out

    # For each narration line, burn word-by-word subtitles using the
    # word_lines from align_speech (which already has absolute timestamps).
    clips = []
    for line_idx, (line, line_dur) in enumerate(zip(narration_lines_full, lengths)):
        if line_dur <= 0:
            continue

        line_offset = offsets[line_idx]
        line_end = line_offset + line_dur

        asset = _pick_asset_for_line(line_idx)

        # Base visual clip for this line (image or black)
        if asset is not None and asset.url:
            base_clip = _asset_clip(asset, out_path.parent / "render-work", f"{len(clips)}", line_dur, style_cfg)
        else:
            base_clip = _black_frame(out_path.parent / "render-work", len(clips), line_dur)

        # Get word timings for this line from align_speech
        words_with_times = word_lines[line_idx] if line_idx < len(word_lines) else []

        # Burn subtitles word-by-word (groups of 2 words for readability)
        if words_with_times:
            segment_clips = []
            last_end = 0.0
            for i in range(0, len(words_with_times), 2):
                group = words_with_times[i:i+2]
                group_text = " ".join(w for w, _s, _e in group)
                print(f"DEBUG Subtitle group {i//2}: '{group_text}'")
                group_start = group[0][1] - line_offset
                group_end = group[-1][2] - line_offset
                if group_start < 0:
                    group_start = 0.0
                if group_end > line_dur:
                    group_end = line_dur
                if group_end - group_start < 0.1:
                    continue
                if group_start > last_end:
                    gap = group_start - last_end
                    segment_clips.append(
                        _trim_clip(base_clip, out_path.parent / "render-work", len(clips) + len(segment_clips), last_end, gap)
                    )
                segment_clips.append(
                    _overlay_subtitle_on_clip(
                        _trim_clip(base_clip, out_path.parent / "render-work", len(clips) + len(segment_clips), group_start, group_end - group_start),
                        out_path.parent / "render-work",
                        len(clips) + len(segment_clips),
                        group_text,
                        group_end - group_start,
                        style_cfg.subtitle.fontsize
                    )
                )
                last_end = group_end
            if last_end < line_dur:
                segment_clips.append(
                    _trim_clip(base_clip, out_path.parent / "render-work", len(clips) + len(segment_clips), last_end, line_dur - last_end)
                )
            clips.extend(segment_clips)
            continue

        # No word timings: fall back to full line with single subtitle
        base_clip = _overlay_subtitle(base_clip, out_path.parent / "render-work", len(clips), line, line_dur, style_cfg.subtitle.fontsize)
        clips.append(base_clip)

    # Build final video
    silent = _join_cards(clips, work)

    # Mux with audio
    total = sum(lengths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(silent), "-i", str(mixed),
        "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


def _wrap_subtitle_text(text: str, fontsize: int, max_width: int = 980) -> str:
    """Wrap subtitle text into multiple lines - max 2 words per line, stacked vertically."""
    words = text.split()
    if len(words) <= 2:
        return text
    # Group into pairs, each pair on its own line
    lines = []
    for i in range(0, len(words), 2):
        pair = words[i:i+2]
        lines.append(" ".join(pair))
    return "\\n".join(lines)


def _overlay_subtitle(src: Path, work: Path, index: int, text: str, length: float, fontsize: int = 74) -> Path:
    """Burn a centered subtitle over an existing clip."""

    out = work / f"sub-{index:02d}.mp4"
    # Wrap first, then escape for ffmpeg drawtext
    wrapped = _wrap_subtitle_text(text, fontsize)
    # Escape for ffmpeg drawtext: 
    # - Don't escape backslashes (needed for \n newlines)
    # - Escape colons
    # - Escape unescaped single quotes
    safe = wrapped.replace(":", "\\:")
    import re
    safe = re.sub(r"(?<!\\)'", "\\'", safe)
    logger.debug(f"Overlay subtitle text sent to ffmpeg: {repr(safe)}")
    logger.debug(f"Overlay subtitle text sent to ffmpeg: {repr(safe)}")
    # No background box - just text with outline
    vf = (
        f"drawtext=text='{safe}':fontcolor=white:fontsize={fontsize}:"
        f"fontfile=/usr/share/fonts/truetype/lato/Lato-Bold.ttf:"
        f"x=(w-text_w)/2:y=(h-text_h)/2"
    )
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out
    out = work / f"black-{index:02d}.mp4"
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920",
        "-t", f"{length:.3f}", "-r", "30", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(out)
    ])
    return out


def _word_flash(work, index, word, length, fontsize=106):
    out = work / f"flash-{index:02d}.mp4"
    safe_word = word.replace("'", "\\'").replace('"', '\\"')
    draw = f"text='{safe_word}':fontcolor=white:fontsize={fontsize}:fontfile=/usr/share/fonts/truetype/lato/Lato-Bold.ttf:x=(w-text_w)/2:y=(h-text_h)/2"
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920",
        "-vf", f"drawtext={draw}", "-t", f"{length:.3f}", "-r", "30",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(out)
    ])
    return out


def _asset_clip(asset, work, tag, length, style):
    from render.cards import _asset_source, _motion_filter, _style_for, _cover_prep
    clip = work / f"asset-{tag}.mp4"
    source = _asset_source(asset, work, tag)
    frames = max(1, round(length * FPS))
    motion_pool = style.image.motion_pool if style else list(range(24))
    motion_style = _style_for(tag, motion_pool)
    extra = getattr(asset, "extra", {}) or {}
    grade_emphasis = extra.get("grade_emphasis")
    tail = f"{_motion_filter(frames, motion_style, None)}"
    if grade_emphasis:
        grade_map = {
            "cold": "eq=contrast=1.15:saturation=0.15:brightness=-0.03:gamma=1.1,colorbalance=rs=-0.05:gs=-0.02:bs=0.08",
            "noir": "eq=contrast=1.25:saturation=0.35:brightness=-0.08:gamma=1.15,noise=alls=18:allf=t,vignette=PI/3.0",
            "raw": "eq=contrast=1.35:saturation=1.2:brightness=0.02:gamma=1.2,colorbalance=rs=0.15:gs=-0.1:bs=-0.15,vignette=PI/2.5",
            "academic": "eq=contrast=1.1:saturation=0.55:brightness=-0.05:gamma=1.08,colorbalance=rs=0.12:gs=0.06:bs=-0.04,noise=alls=12:allf=t",
            "tech": "eq=contrast=1.08:saturation=0.4:brightness=-0.02:gamma=1.05,colorbalance=rs=-0.08:gs=-0.04:bs=0.15",
            "neutral": "",
        }
        grade_filter = grade_map.get(extra.get("grade_emphasis"), "")
        if grade_filter:
            tail = f"{tail},{grade_filter}"
    vf = f"{_cover_prep()},{tail}"
    out = work / f"asset-{tag}.mp4"
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(source), "-vf", vf,
        "-t", f"{length:.3f}", "-r", str(FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])
    return clip


def _join_cards(clips, work):
    concat_list = work / "concat.txt"
    concat_list.write_text("\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8")
    out = work / "cards.mp4"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(out),
    ])
    return out