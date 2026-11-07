"""Video assembly: timed scene stills + voiceover + music bed -> final MP4."""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.settings import get_settings
from core.errors import BlockError
from core.models import AssetPack, SfxPlan, VideoScript, VisualAsset
from utils import computing
from voice.subtitles import align_speech, build_plates, write_ass

from .audio_mix import (
    _apply_sfx,
    _bed_tracks,
    _build_bed,
    _mix_audio,
    _music_spans,
    _voice_track,
)
from .cards import (
    FPS,
    _blank_clip,
    _credit_draw,
    _fade_out_clip,
    _moving_still,
    _still_clip,
)
from .media import (
    _asset_source,
    _duration,
    _no_back_to_back,
    _run,
)

logger = logging.getLogger(__name__)

# Crossfade between CARDS (subject dissolves inside a card already exist).
# The transition runs over the incoming card's first FADE seconds while the
# outgoing card's silent padding tail blends out — so speech, card and
# subtitle hand over at exactly the same audio boundary, no drift.
CARD_FADE = 0.3

# Keyword-dim: while a keyword plate is on screen the frame dips slightly
# darker so the eye lands on the word (agreed design: keyword dimming).
DIM_OPACITY = 0.20


def _join_cards(
    clips: list[Path],
    lengths: list[float],
    work: Path,
    *,
    graded: bool = False,
    dim_spans: list[tuple[float, float]] | None = None,
) -> Path:
    """Concatenate cards with a CARD_FADE crossfade at each boundary.

    Every non-final card is built CARD_FADE seconds longer than its slot;
    each xfade offset is the AUDIO boundary of the next line, so the blend
    consumes the outgoing card's padding tail and the incoming card's
    content still starts exactly on its narration.
    """

    silent = work / "cards.mp4"
    n = len(clips)
    args = ["ffmpeg", "-y"]
    for clip in clips:
        args += ["-i", str(clip)]
    chain: list[str] = []
    previous = "[0:v]"
    if n > 1:
        boundary = 0.0
        for j, length in enumerate(lengths[:-1]):
            boundary += length
            label = f"[xc{j}]"
            chain.append(
                f"{previous}[{j + 1}:v]xfade=transition=fade"
                f":duration={CARD_FADE:.2f}:offset={boundary:.3f}{label}"
            )
            previous = label
    filters: list[str] = []
    for ps, pe in dim_spans or []:
        filters.append(
            f"drawbox=c=black@{DIM_OPACITY:.2f}:t=fill"
            f":enable='between(t,{ps:.2f},{pe:.2f})'"
        )
    if graded:
        filters.append(HORROR_GRADE)
    tail = (
        f"{previous}{','.join(filters)}[vout]" if filters
        else f"{previous}null[vout]"
    )
    graph = ";".join([*chain, tail])
    args += [
        "-filter_complex", graph, "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(silent),
    ]
    _run(args)
    return silent


def _mux(
    silent: Path,
    mixed: Path,
    timeline: float,
    out_path: Path,
    ass_path: Path | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-i", str(silent), "-i", str(mixed)]
    if ass_path is not None:
        args += ["-vf", f"ass='{ass_path}'"]
        video = [
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "fast",
            "-crf", "20", "-pix_fmt", "yuv420p",
        ]
    else:
        video = ["-map", "0:v", "-map", "1:a", "-c:v", "copy"]
    args += [
        "-t", f"{timeline:.3f}",
        *video,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(args)
    return out_path


def render(
    script: VideoScript,
    pack: AssetPack,
    audio_dir: Path,
    out_path: Path,
    *,
    stem: str = "voice",
    genre: str | None = None,
    music: str = "",
    subtitle_mode: str = "all",
) -> Path:
    """Assemble a vertical feed video: card per line, topical still per scene."""

    work = out_path.parent / "render-work"
    work.mkdir(parents=True, exist_ok=True)

    narration_lines: list[str] = [script.hook, *(scene.narration for scene in script.scenes), script.outro]
    lines = narration_lines  # compat
    wavs: list[Path] = []
    for index in range(len(narration_lines)):
        wav = audio_dir / f"{stem}-{index:02d}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"Missing narration {wav}; run 'ash_speaks voice' first.")
        wavs.append(wav)

    # Card length = the narration rounded UP to whole frames, so speech,
    # card and subtitle all hand over at the same frame — no drift, no
    # early subtitle switches.
    lengths = [(math.ceil(_duration(wav) * FPS) + 9) / FPS for wav in wavs]

    # Word-level cues: align the real voice audio (whisper), shift each
    # line's words onto the timeline at its card's start frame, and burn
    # 3-4 word cues over the assembled video in one libass pass.
    offsets = []
    start = 0.0
    for length in lengths:
        offsets.append(start)
        start += length
    line_ends = offsets[1:] + [start]
    word_lines, cues = align_speech(
        narration_lines, wavs, offsets, line_ends,
        model_size=get_settings().subtitle_whisper_model,
    )
    # Context-driven emphasis (deaf-viewer / no-audio test)
    try:
        from config.settings import get_settings as _gs
        from providers.llm import create_llm_provider
        from voice.emphasis import emphasize_cues
        _llm = create_llm_provider(_gs())
        cues = emphasize_cues(_llm, narration_lines, cues)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("subtitle emphasis skipped: %s", _exc)
    # Persistent keyword plates (agreed design): each main word appears once,
    # holds 2-2.5s (or until the next keyword), gentle fade, no motion. The
    # same timeline dims the frame while a plate is up (keyword dimming).
    plates = build_plates(cues)
    ass_path = write_ass(cues, work / "subs.ass", mode=subtitle_mode, plates=plates)
    # Only burn subtitles if the ASS file actually has cues
    _ass_lines = ass_path.read_text(encoding="utf-8").splitlines()
    if not any(line.startswith("Dialogue:") for line in _ass_lines):
        ass_path = None
    logger.info("Keyword plates: %d", len(plates))

    clips: list[Path] = []

    # the hook speaks for itself: it gets its own matched asset
    hook_asset_entry = next(
        (entry for entry in pack.scenes if entry.scene_index == -1), None
    )
    hook_visuals = [
        a
        for a in (
            hook_asset_entry.assets
            if hook_asset_entry is not None
            else (pack.scenes[0].assets if pack.scenes else [])
        )
        if a.kind in ("image", "video")
    ]
    scene_visuals: list[list[VisualAsset]] = []
    for scene_entry, _script_scene in zip(
        [e for e in pack.scenes if e.scene_index >= 0],
        script.scenes,
        strict=True,
    ):
        scene_visuals.append([a for a in scene_entry.assets if a.kind in ("image", "video")])
    _no_back_to_back([hook_visuals, *scene_visuals])

    # Every card except the outro runs CARD_FADE longer than its slot: the
    # crossfade at the next boundary consumes that tail, so each card's
    # content still starts exactly when its narration starts.
    def card_length(slot: int) -> float:
        return lengths[slot] + (CARD_FADE if slot < len(lengths) - 1 else 0.0)

    def hook_card() -> Path:
        if hook_visuals:
            try:
                return _moving_still(
                    hook_visuals, work, 0, card_length(0), line=narration_lines[0], word_line=word_lines[0]
                )
            except (RuntimeError, OSError, ValueError):
                logger.warning("hook card motion failed; using blank card")
        return _blank_clip(work, 0, card_length(0))

    build: list[tuple[int, Callable[..., Path]]] = [(0, hook_card)]
    for index, visuals in enumerate(scene_visuals):

        def scene_card(
            card=index + 1, vis=visuals, text=narration_lines[index + 1], words=word_lines[index + 1]
        ) -> Path:
            if vis:
                try:
                    return _moving_still(vis, work, card, card_length(card), line=text, word_line=words)
                except (RuntimeError, OSError, ValueError):
                    logger.warning("scene %d card motion failed; using blank card", card)
            return _blank_clip(work, card, card_length(card))

        build.append((index + 1, scene_card))
    # the per-card ffmpeg jobs are independent; three at a time roughly
    # triples this stage with no influence on what any card looks like
    with ThreadPoolExecutor(max_workers=computing.render_workers()) as pool:
        built = sorted(pool.map(lambda item: (item[0], item[1]()), build))
    clips.extend(path for _index, path in built)

    # Outro: no more text card. Fade out over the last scene's strongest
    # ending image (highest CLIP match — the outro's narration is the
    # emotional close, not a slot for the leftovers; borrowing whatever
    # sat last once put a ground telescope under "millions of years
    # later"); when a scene has nothing, keep a plain dark card.
    last_visuals = [
        a
        for a in (pack.scenes[-1].assets if pack.scenes else [])
        if a.kind in ("image", "video")
    ]
    if last_visuals:
        # the outro follows the last scene card, so the "strongest ending"
        # cannot be the image that card just led with (back-to-back rule)
        last_covered = scene_visuals[-1][0].url if scene_visuals and scene_visuals[-1] else None
        last_visuals.sort(key=lambda a: -(a.clip_score or 0.0))
        best = next((v for v in last_visuals if v.url != last_covered), last_visuals[0])
        try:
            source = _asset_source(best, work, "img-outro")
            outro_raw = _still_clip(
                source, work, len(build), lengths[-1],
                credit=_credit_draw(best, work, len(build)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("outro image failed (%s); using blank card", exc)
            outro_raw = _blank_clip(work, len(build), lengths[-1])
    else:
        outro_raw = _blank_clip(work, len(build), lengths[-1])
    clips.append(_fade_out_clip(outro_raw, lengths[-1], work, len(build)))

    # Each card must actually exist for its line's length (plus the
    # crossfade tail on non-final cards): the parallel builders once let
    # an off-by-one overwrite the hook clip, the missing scene went black
    # while its narration (and subs) kept playing.
    for index, (clip_path, expected) in enumerate(zip(clips, lengths, strict=True)):
        expected = expected + (CARD_FADE if index < len(lengths) - 1 else 0.0)
        actual = _probe_duration(clip_path)
        if abs(actual - expected) > 0.3:
            raise BlockError(
                f"Card {clip_path.name} is {actual:.2f}s but its narration "
                f"needs {expected:.2f}s; refusing to ship a broken cut."
            )

    silent = _join_cards(
        clips,
        lengths,
        work,
        graded=any(
            "original horror story" in note.lower() for note in script.source_notes
        ),
        dim_spans=[(ps, pe) for ps, pe, _word in plates] if ass_path else None,
    )
    voice = _voice_track(wavs, lengths, work)
    total = sum(lengths)
    sfx_path = out_path.parent / "sfx.json"
    if sfx_path.exists():
        plan = SfxPlan.model_validate_json(sfx_path.read_text(encoding="utf-8"))
        voice = _apply_sfx(voice, plan.cues, offsets, total, work)
    cards = _bed_tracks(script, genre, override=music)
    spans = _music_spans(cards, offsets, total)
    describe = ", ".join(
        f"{path.name} {start:.1f}-{end:.1f}s" for path, _gain, start, end in spans
    )
    logger.info("Music bed: %s", describe)
    bed = _build_bed(spans, work, total)
    mixed = _mix_audio(voice, bed, work)
    return _mux(silent, mixed, total, out_path, ass_path)

HORROR_GRADE = (
    "eq=contrast=1.08:saturation=0.42:brightness=-0.05:gamma=1.07,"
    "colorbalance=rs=-0.12:gs=0.02:bs=0.12,"
    "noise=alls=7:allf=t,"
    "vignette=PI/4.2"
)


def _probe_duration(clip: Path) -> float:
    """Seconds of video stream in an mp4 card, via ffprobe."""

    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration", "-of", "csv=p=0", str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())
