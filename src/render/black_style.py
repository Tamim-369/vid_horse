"""Black-screen conceptual editing style.

Editing logic:
- Each narration line is split into sentence-sized windows; each window gets
  its OWN image from the manifest-driven asset pack (no image twice in a row).
- Subtitles are word-timed: cues are groups of words (max 2 stacked rows),
  burned with drawtext `enable` windows — ONE ffmpeg encode per segment, so
  no flicker, no drift, no per-word re-encodes.
- Missing/unusable image -> random different image from the motif pool
  (never the same image twice in a row); blank card only if the pool is
  empty.
- Music bed per video from music assets.
"""

from __future__ import annotations

import logging
import random as _random
import subprocess
from pathlib import Path

from config.video_style import load_style_config, VideoStyle
from core.models import VisualAsset
from utils import computing

logger = logging.getLogger(__name__)

FPS = 30
WIDTH = 1080
HEIGHT = 1920

FONT = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"

# Subtitle look: word-group captions — only the 2-3 words being spoken
# right now, huge, dead center. Rows stack vertically when needed.
SUB_FONT = 84          # 120 was too loud on screen; -30%
_ROW_CHARS = 22         # Lato Bold @84 ≈ 44px/char; ~22 chars fit 980px
_MAX_ROWS = 3           # a 2-3 word group may need 2-3 rows
_CUE_MAX_WORDS = 3      # cue holds at most 3 words, then next group
_BORDER = 7             # outline scales with the font


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        tail = stderr[-2000:] if len(stderr) > 2000 else stderr
        raise RuntimeError(f"ffmpeg failed (code {result.returncode}):\n{tail}")


def _black_frame(work: Path, index: int, length: float) -> Path:
    """Pure black frame for given length."""
    out = work / f"black-{index}.mp4"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={WIDTH}x{HEIGHT}",
        "-t", f"{length:.3f}", "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(out),
    ])
    return out


def _choose_motion(asset, narration: str, fallback: int, pool: list[int]) -> int:
    """Pick a motion style that fits what is being shown/said."""
    t = (narration or "").lower()
    caption = (getattr(asset, "caption", "") or "").lower()
    page = (getattr(asset, "page_url", "") or "").lower()
    extra = getattr(asset, "extra", {}) or {}
    cat = (extra.get("concept") or page.split(":")[0] or "").lower()
    hay = f"{t} {caption} {cat}"

    preferred: int | None = None

    # Focus / intimate / screens / scrolling / thinking -> slow push-in
    if any(k in hay for k in ("scroll", "phone", "screen", "texting", "message", "reading", "doomscroll", "feed", "notification", "typing")):
        preferred = 0
    elif cat.startswith("face") or any(k in hay for k in ("face", "eyes", "tear", "mouth", "stare", "expression", "look")):
        # close/emotional -> push in, otherwise gentle drift
        if any(k in hay for k in ("close", "stare", "watch", "alone", "silent", "tear", "fear", "worry", "sad", "anger")):
            preferred = 0
        else:
            preferred = 16
    elif cat.startswith(("crowd", "group", "env", "space")) or any(k in hay for k in ("crowd", "people", "group", "everyone", "room", "street", "office", "meeting", "line", "walk", "together")):
        # movement / many people -> pan or sweep
        preferred = 1 if "together" in hay or "crowd" in hay else 12
    elif cat.startswith(("surv", "control", "diag", "abstract", "system")) or any(k in hay for k in ("watch", "camera", "monitor", "control", "power", "rule", "track", "data", "chart", "system")):
        preferred = 21  # cold, subtle
    elif any(k in t for k in ("cost", "trap", "stuck", "without noticing", "steer", "pull", "pressure", "quiet cost")):
        preferred = 3 if "notice" in t else 0
    elif any(k in t for k in ("why do you", "you feel", "in your head", "you notice")):
        preferred = 0

    if preferred is not None and preferred in pool:
        return preferred
    if preferred is not None and pool:
        # keep similar character when exact style not in pool
        families = [
            {0, 3, 6, 7, 20},
            {1, 2, 4, 5, 8, 9, 10, 11},
            {12, 13, 14, 15, 16, 17},
            {18, 19, 21, 22, 23},
        ]
        for fam in families:
            if preferred in fam:
                for cand in pool:
                    if cand in fam:
                        return cand
                break
    return fallback


def _asset_clip(asset, work: Path, tag: str, length: float, style: VideoStyle, narration: str = "") -> Path:
    """Render a concept asset with content-aware motion and concept grade."""
    from render.cards import _asset_source, _motion_filter, _style_for, _cover_prep

    source = _asset_source(asset, work, tag)

    frames = max(1, round(length * FPS))
    motion_pool = style.image.motion_pool if style else list(range(24))
    fallback = _style_for(tag, motion_pool)
    motion_style = _choose_motion(asset, narration, fallback, motion_pool)

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


# transition techniques (xfade names) - varied, smooth, suitable for image segments
_TRANSITION_TYPES = [
    "fade", "dissolve", "fadeblack", "fadegrays",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "slideleft", "slideright", "slideup", "slidedown",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "wipetl", "wipetr", "wipebl", "wipebr",
    "circleopen", "circleclose", "circlecrop",
    "horzopen", "horzclose", "vertopen", "vertclose",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "radial", "zoomin",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
    "squeezeh", "squeezev",
]


def _compose_with_transitions(
    clips: list[Path], durations: list[float], work: Path, *, slug: str | None, total: float
) -> Path:
    """Join segment clips with varied per-boundary xfade transitions.

    Uses freeze-handle composition so total duration stays sync-safe:
    bodies + transition pieces sum to the original total.
    """
    n = len(clips)
    if n == 1:
        out = work / "cards-transitions.mp4"
        _run([
            "ffmpeg", "-y", "-i", str(clips[0]),
            "-vf", "fade=t=in:st=0:d=0.35,fade=t=out:st={:.3f}:d=0.35".format(max(0.0, durations[0] - 0.35)),
            "-t", f"{total:.3f}", "-r", str(FPS),
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            str(out),
        ])
        return out

    rng = _random.Random(slug or "nebula")

    # frame-accurate planning
    cum = 0.0
    bounds = [0]
    for d in durations:
        cum += d
        bounds.append(round(cum * FPS))
    F = [bounds[i + 1] - bounds[i] for i in range(n)]
    # ensure exact total frames
    total_frames = round(total * FPS)
    diff = total_frames - sum(F)
    if diff != 0 and F:
        F[-1] += diff
        for i in range(1, len(bounds)):
            bounds[i] = bounds[i - 1] + F[i - 1]

    # choose transition length (frames, even) and type per boundary
    tf = [0] * (n + 1)
    ttype = ["fade"] * (n + 1)
    prev_type = None
    for i in range(1, n):
        want = 2 * round(rng.uniform(0.35, 0.70) * FPS / 2)
        cap = 2 * (min(F[i - 1], F[i]) // 4)
        t = min(want, cap)
        if t < 8:
            tf[i] = 0
            continue
        tf[i] = t
        choices = [x for x in _TRANSITION_TYPES if x != prev_type]
        chosen = rng.choice(choices or _TRANSITION_TYPES)
        ttype[i] = chosen
        prev_type = chosen

    enc = [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "30", "-bf", "0",
        "-an",
    ]

    pieces: list[Path] = []
    for i in range(n):
        s = tf[i] // 2
        e = F[i] - tf[i + 1] // 2
        length = e - s
        if length <= 0:
            length = max(1, F[i])
            s = 0
            e = F[i]
        out = work / f"body-{i:03d}.mp4"
        _run([
            "ffmpeg", "-y", "-i", str(clips[i]),
            "-vf", f"trim=start_frame={s}:end_frame={e},setpts=PTS-STARTPTS,fps={FPS},format=yuv420p",
            "-frames:v", str(length),
            *enc, str(out),
        ])
        pieces.append(out)

        if i < n - 1 and tf[i + 1] > 0:
            TF = tf[i + 1]
            D = (TF - 1) / FPS
            NA = F[i] - TF // 2
            NB_end = TF // 2
            tout = work / f"trans-{i + 1:03d}.mp4"
            trans = ttype[i + 1]
            fc = (
                f"[0:v]trim=start_frame={NA},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={TF/2/FPS:.6f},"
                f"fps={FPS},format=yuv420p[a];"
                f"[1:v]trim=end_frame={NB_end},setpts=PTS-STARTPTS,"
                f"tpad=start_mode=clone:start_duration={TF/2/FPS:.6f},"
                f"fps={FPS},format=yuv420p[b];"
                f"[a][b]xfade=transition={trans}:duration={D:.6f}:offset=0,format=yuv420p[v]"
            )
            _run([
                "ffmpeg", "-y", "-i", str(clips[i]), "-i", str(clips[i + 1]),
                "-filter_complex", fc,
                "-map", "[v]", "-frames:v", str(TF),
                *enc, str(tout),
            ])
            pieces.append(tout)

    concat_list = work / "trans-concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in pieces), encoding="utf-8")
    silent = work / "cards-transitions.mp4"
    # global fade in/out for polished edges
    fade_out_st = max(0.0, total - 0.4)
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-vf", f"fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out_st:.3f}:d=0.4",
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an", str(silent),
    ])
    return silent


def _wrap_subtitle_text(text: str, max_chars: int = _ROW_CHARS, max_rows: int = _MAX_ROWS) -> str:
    """Wrap subtitle text into stacked rows when it doesn't fit the width."""
    words = text.split()
    if not words:
        return ""
    rows: list[list[str]] = [[]]
    for word in words:
        row = rows[-1]
        test = word if not row else " ".join(row) + " " + word
        if len(test) <= max_chars:
            row.append(word)
        elif len(rows) < max_rows and row:
            rows.append([word])
        else:
            rows.append([word])
    return "\n".join(" ".join(r) for r in rows if r)


def _build_cues(words: list[tuple[str, float, float]], line_dur: float) -> list[tuple[str, float, float]]:
    """Group word timings into subtitle cues (max _CUE_MAX_WORDS words each).

    Each cue shows only the words being spoken right now; a cue lasts until
    the next cue replaces it. Returns [(text, start, end)] with
    line-relative times.
    """
    cues: list[tuple[str, float, float]] = []
    rows: list[list[str]] = [[]]
    start: float | None = None

    def word_count() -> int:
        return sum(len(r) for r in rows)

    def flush(end: float) -> None:
        nonlocal rows, start
        text = "\n".join(" ".join(r) for r in rows if r)
        if text and start is not None and end > start:
            cues.append((text, start, end))
        rows = [[]]
        start = None

    for word, s, e in words:
        s = max(0.0, s)
        e = min(max(e, s + 0.05), line_dur)
        if start is None:
            start = s
        row = rows[-1]
        test = word if not row else " ".join(row) + " " + word
        if len(test) <= _ROW_CHARS:
            row.append(word)
        elif len(rows) < _MAX_ROWS and row:
            rows.append([word])
        else:
            flush(s)
            start = s
            rows[0].append(word)
        # Word-count cap: show only the few words being spoken right now
        if word_count() >= _CUE_MAX_WORDS:
            flush(e)
    flush(line_dur)

    # A cue stays on screen until the next one replaces it.
    for i in range(len(cues) - 1):
        text, s, e = cues[i]
        cues[i] = (text, s, max(e, cues[i + 1][1]))
    return cues


def _subtitle_filter(cues: list[tuple[str, float, float]], seg_dur: float, tag: str, work: Path) -> str | None:
    """Build one drawtext chain with enable windows for all cues in a segment.

    Each cue's rows are drawn as separate drawtext entries so every row is
    centered individually — a stacked cue reads as a symmetric column.
    """
    filters: list[str] = []
    line_h = SUB_FONT * 1.25
    for ci, (text, cs, ce) in enumerate(cues):
        cs = max(0.0, cs)
        ce = min(seg_dur, ce)
        if ce - cs < 0.08:
            continue
        rows = [r for r in text.split("\n") if r.strip()]
        block_h = len(rows) * line_h
        base_y = (HEIGHT - block_h) / 2
        for ri, row in enumerate(rows):
            f = work / f"cue-{tag}-{ci:02d}-{ri}.txt"
            f.write_text(row, encoding="utf-8")
            y = base_y + ri * line_h
            filters.append(
                f"drawtext=textfile={f}:fontcolor=white:fontsize={SUB_FONT}:"
                f"fontfile={FONT}:borderw={_BORDER}:bordercolor=black@0.85:"
                f"x=(w-text_w)/2:y={y:.0f}:"
                f"enable='between(t,{cs:.2f},{ce:.2f})'"
            )
    return ",".join(filters) if filters else None


def _global_subtitle_filter(cues: list[tuple[str, float, float]], total: float, work: Path) -> str | None:
    """Build drawtext chain for all cues with absolute video timings."""
    filters: list[str] = []
    line_h = SUB_FONT * 1.25
    for ci, (text, cs, ce) in enumerate(cues):
        cs = max(0.0, cs)
        ce = min(total, ce)
        if ce - cs < 0.08:
            continue
        rows = [r for r in text.split("\n") if r.strip()]
        block_h = len(rows) * line_h
        base_y = (HEIGHT - block_h) / 2
        for ri, row in enumerate(rows):
            f = work / f"gcue-{ci:04d}-{ri}.txt"
            f.write_text(row, encoding="utf-8")
            y = base_y + ri * line_h
            filters.append(
                f"drawtext=textfile={f}:fontcolor=white:fontsize={SUB_FONT}:"
                f"fontfile={FONT}:borderw={_BORDER}:bordercolor=black@0.85:"
                f"x=(w-text_w)/2:y={y:.0f}:"
                f"enable='between(t,{cs:.2f},{ce:.2f})'"
            )
    return ",".join(filters) if filters else None


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

    word_lines, _cues = align_speech(
        narration_lines, wavs, offsets, line_ends,
        model_size="base.en",
    )

    # Load music (variety: different bed per video; with a MusicPlan the
    # director's base_role varies per video AND scene_roles change the track
    # inside the video, so no override is passed)
    from render.audio_mix import _shelf, _bed_tracks, _build_bed, _music_spans, _mix_audio, _voice_track

    music_override = music
    if not music_override and getattr(script, "music", None) is None:
        # plan-less legacy path: classify the narration's content by keyword
        # and use the matching role track — never a random shelf pick
        from blocks.music.director import classify_text_to_role
        from render.audio_mix import _role_track as _role_track_am
        role = classify_text_to_role(" ".join(narration_lines))
        music_override = _role_track_am(role)
    cards = _bed_tracks(script, None, music_override)
    spans = _music_spans(cards, offsets, sum(_duration(w) for w in wavs))
    bed = _build_bed(spans, work, sum(_duration(w) for w in wavs))
    voice = _voice_track(wavs, lengths, work)
    mixed = _mix_audio(voice, bed, work)

    # ------------------------------------------------------------------
    # Per-line assets: hook = pack.scenes[0], scenes map 1:1, outro gets
    # whatever good images are left over.
    # ------------------------------------------------------------------
    narration_lines_full = [script.hook, *(s.narration for s in script.scenes), script.outro]

    def _usable(a) -> bool:
        return bool(a.url and a.file)

    line_assets: list[list] = [[] for _ in narration_lines_full]
    if pack.scenes:
        if line_assets:
            line_assets[0] = [a for a in (pack.scenes[0].assets or []) if _usable(a)]
        for i in range(1, len(narration_lines_full) - 1):
            if i < len(pack.scenes):
                line_assets[i] = [a for a in (pack.scenes[i].assets or []) if _usable(a)]
        assigned_files = {a.file for lst in line_assets for a in lst}
        leftover = [
            a for sc in pack.scenes for a in (sc.assets or [])
            if _usable(a) and a.file not in assigned_files
        ]
        line_assets[-1] = leftover

    # Fallback pool: every motif image that actually exists on disk. When a
    # line has no usable assigned asset — or its file went missing — we pull
    # a RANDOM image from here instead of showing a blank card. The screen
    # must never be imageless, and never show the same image twice in a row.
    import random as _random

    from core.models import VisualAsset

    pool: list[VisualAsset] = []
    assets_root = Path("assets/psychology_motifs")
    if assets_root.exists():
        for p in sorted(assets_root.rglob("*.png")):
            pool.append(VisualAsset(
                kind="image",
                file=p.name,
                url=f"file://{p.resolve()}",
                page_url=f"fallback:{p.parent.name}/{p.name}",
                caption="",
            ))

    def _pool_pick(forbidden: str | None) -> VisualAsset | None:
        cands = [a for a in pool if a.file != forbidden] if forbidden else pool
        if not cands:
            cands = pool  # single-image pool: better a repeat than a blank
        return _random.choice(cands) if cands else None

    last_file: str | None = None

    # ------------------------------------------------------------------
    # Build clips: per line -> per asset window -> ONE encode per window
    # with all subtitle cues burned via enable windows.
    # ------------------------------------------------------------------
    clips = []
    clip_durations: list[float] = []
    shown_files: set[str] = set()
    tag_i = 0

    for line_idx, (line, line_dur) in enumerate(zip(narration_lines_full, lengths)):
        if line_dur <= 0:
            continue

        # Line-relative word timings
        abs_words = word_lines[line_idx] if line_idx < len(word_lines) else []
        rel_words = [(w, s - offsets[line_idx], e - offsets[line_idx]) for w, s, e in abs_words]
        line_cues = _build_cues(rel_words, line_dur)

        # Candidate images for this line: dedupe consecutive same-file
        # entries, then prefer files not already shown in this video.
        deduped: list[VisualAsset | None] = []
        for a in line_assets[line_idx]:
            if deduped and deduped[-1] is not None and a.file == deduped[-1].file:
                continue
            deduped.append(a)
        fresh = [a for a in deduped if a.file not in shown_files]
        seq = fresh if fresh else deduped
        if not seq:
            # never a blank screen: pull a random pool image (different
            # from the one just shown); black card only if the whole pool
            # is empty
            pick = _pool_pick(last_file)
            seq = [pick] if pick is not None else [None]

        # Show multiple images per line/sentence, but keep each on screen
        # long enough to register (don't flash). If the line is too short
        # for every assigned image, show as many as fit in order.
        MIN_IMG_DUR = 1.8
        if len(seq) > 1 and line_dur / len(seq) < MIN_IMG_DUR:
            keep = max(1, int(line_dur / MIN_IMG_DUR))
            seq = seq[:keep]

        bounds = [line_dur * i / len(seq) for i in range(len(seq) + 1)]

        for wi, asset in enumerate(seq):
            seg_start, seg_end = bounds[wi], bounds[wi + 1]
            seg_dur = seg_end - seg_start
            if seg_dur < 0.15:
                continue
            tag = f"{tag_i:03d}"
            tag_i += 1

            # cues whose midpoint falls inside this window
            seg_cues = [
                (t, cs - seg_start, ce - seg_start)
                for t, cs, ce in line_cues
                if seg_start <= (cs + ce) / 2 < seg_end or (wi == len(seq) - 1 and cs < seg_end and ce > seg_start)
            ]

            if asset is not None:
                # never the same image twice in a row, even inside a
                # repeated shown-image sequence
                if asset.file == last_file:
                    alt = _pool_pick(asset.file)
                    if alt is not None:
                        asset = alt
                try:
                    base = _asset_clip(asset, work, f"img-{tag}", seg_dur, style_cfg, line)
                except Exception as exc:  # noqa: BLE001 - a broken asset must
                    # never blank the screen: fall back to a pool image
                    logger.warning("asset clip failed for %s (%s); pool fallback", asset.file, exc)
                    alt = _pool_pick(asset.file)
                    if alt is not None:
                        asset = alt
                        base = _asset_clip(asset, work, f"img-{tag}", seg_dur, style_cfg, line)
                    else:
                        base = _black_frame(work, f"img-{tag}", seg_dur)
                shown_files.add(asset.file)
                last_file = asset.file
            else:
                base = _black_frame(work, f"img-{tag}", seg_dur)

            # render content clip (motion + subtitles), no inter-clip fades;
            # transitions are composed per-boundary below
            vf = _subtitle_filter(seg_cues, seg_dur, tag, work)
            out = work / f"seg-{tag}.mp4"
            args = [
                "ffmpeg", "-y", "-i", str(base),
            ]
            if vf:
                args += ["-vf", vf]
            args += [
                "-t", f"{seg_dur:.3f}", "-r", str(FPS),
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
                "-crf", "20", str(out),
            ]
            _run(args)
            clips.append(out)
            clip_durations.append(seg_dur)

    if not clips:
        raise RuntimeError("black_style: no clips produced (empty narration?)")

    total = sum(lengths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent = _compose_with_transitions(clips, clip_durations, work, slug=slug, total=total)

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
