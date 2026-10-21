"""Video assembly: timed scene stills + voiceover + music bed -> final MP4."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import textwrap
import urllib.request
from pathlib import Path

import torch
import torchaudio

from core.models import AssetPack, SfxPlan, VideoScript
from voice.subtitles import align_cues, write_ass

logger = logging.getLogger(__name__)

WIDTH = 1080
HEIGHT = 1920
FPS = 25
MUSIC_GAIN = 0.35
MUSIC = Path("assets/music/normal.mp3")
CARD_BG = "0x0d1117"
CARD_TEXT_LINES = 26

# Working canvas for motion: 4x the output frame, keeps every animated crop
# sub-pixel smooth and lets loss in the wide-photo blur-fill look clean.
CANVAS_W = 4320
CANVAS_H = 7680

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
UA_HEADER = {"User-Agent": "videoai-pipeline/0.1"}


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {' '.join(args[:8])}...\n{result.stderr[-1500:]}"
        )


def _font() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _duration(wav_path: Path) -> float:
    info = torchaudio.info(str(wav_path))
    return info.num_frames / info.sample_rate


def _fetch_image(url: str, work: Path, name: str) -> Path:
    """Download ``url`` into ``work`` under a content-addressed name.

    Slot-named caching (img-00-0.jpg) made every re-render reuse the
    previous run's download, so an updated asset pack still showed stale
    pictures — the return of the grandpa. The digest name keeps repeat
    runs free of re-downloads without ever confusing two images.
    """

    digest = hashlib.sha1(url.encode()).hexdigest()[:10]
    stem = name.rsplit(".", 1)[0]
    path = work / f"{stem}-{digest}{Path(name).suffix or '.jpg'}"
    if not path.exists():
        request = urllib.request.Request(url, headers=UA_HEADER)
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _title_card(text: str, work: Path, index: int) -> Path:
    """Dark feed-style card; empty ``text`` yields a clean gradient nothing."""

    path = work / f"card-{index:02d}.png"
    if not text.strip():
        _run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c={CARD_BG}:s={WIDTH}x{HEIGHT}",
            "-frames:v", "1", str(path),
        ])
        return path
    wrapped = "\n".join(textwrap.wrap(text, CARD_TEXT_LINES))
    text_file = work / f"card-{index:02d}.txt"
    text_file.write_text(wrapped, encoding="utf-8")
    font = _font()
    draw = [
        f"textfile={text_file}",
        "fontcolor=white",
        "fontsize=76",
        "line_spacing=30",
        "x=(w-text_w)/2",
        "y=(h-text_h)/2",
    ]
    if font:
        draw.insert(1, f"fontfile={font}")
    path = work / f"card-{index:02d}.png"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={CARD_BG}:s={WIDTH}x{HEIGHT}",
        "-frames:v", "1", "-vf", f"drawtext={':'.join(draw)}",
        str(path),
    ])
    return path


# -- motion -------------------------------------------------------------------
# Motion runs on a 4x canvas (4320x7680): zoompan quantizes its window to
# whole pixels, so on a 4x source one rounding step is a quarter pixel at
# output — smooth instead of the shiver a 1x or 2x source shows.
def _motion_filter(frames: int, style: int, zoom: float = 0.05, drift: float = 0.5) -> str:
    """Zoompan window: 0 slow zoom-in, 1 pan L->R, 2 pan R->L.

    ``zoom`` is the total travel over the scene (keep it gentle), ``drift``
    the fraction of the pan slack a pan actually uses (1.0 = full sweep,
    far too quick; 0.5 reads as a slow cinematic drift).
    """

    end = 1.0 + zoom
    on_rate = frames
    if style == 0:
        z_expr = f"min(1+{zoom:.4f}*on/{on_rate},{end:.3f})"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"{end:.3f}"
        y = "ih/2-(ih/zoom/2)"
        travel = f"(iw-iw/zoom)*{drift:.2f}"
        center = "iw/2-(iw/zoom/2)"
        x = (
            f"{center}+{travel}*on/{on_rate}"
            if style == 1
            else f"{center}+{travel}*(1-on/{on_rate})"
        )
    return (
        f"zoompan=z='{z_expr}':x='{x}':y='{y}':"
        f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )


def _cover_prep() -> str:
    """Scale the source to the working canvas so no part is ever stretched."""

    return (
        f"scale=-2:{CANVAS_H}:force_original_aspect_ratio=increase,"
        f"crop={CANVAS_W}:{CANVAS_H},setsar=1"
    )


def _fill_prep() -> str:
    """Landscape sources: blurred copy covers, full photo floats centered.

    Nothing gets cropped so the photo never looks stretched; the motion
    still breathes because the whole canvas zooms and pans over it.
    """

    fg_w, fg_h = int(CANVAS_W * 0.92), int(CANVAS_H * 0.92)
    return (
        "[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale=-2:{CANVAS_H}:force_original_aspect_ratio=increase,"
        f"crop={CANVAS_W}:{CANVAS_H},boxblur=40:6,eq=brightness=-0.08[bg];"
        f"[fgsrc]scale={fg_w}:{fg_h}:force_original_aspect_ratio=decrease[fgs];"
        "[bg][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[pre]"
    )


def _still_clip(source: Path, work: Path, index: int, length: float, wide: bool) -> Path:
    """One animated clip from a still: cover/blur-fill prep + moving window."""

    style = index % 3
    clip = work / f"clip-{index:02d}.mp4"
    frames = max(1, round(length * FPS))
    tail = f"{_motion_filter(frames, style)}"
    if wide:
        args = [
            "-filter_complex", f"{_fill_prep()};[pre]{tail}[vout]",
            "-map", "[vout]",
        ]
    else:
        args = ["-vf", f"{_cover_prep()},{tail}"]
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(source), *args,
        "-t", f"{length:.3f}", "-r", str(FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(clip),
    ])
    return clip


def _blank_clip(work: Path, index: int, length: float) -> Path:
    """Plain dark card (hook before any subject image exists to show)."""

    return _still_clip(_title_card("", work, index), work, index, length, wide=False)


def _fade_out_clip(clip: Path, length: float, work: Path, index: int) -> Path:
    """Fade a finished clip to black over its last two seconds."""

    out = work / f"faded-{index:02d}.mp4"
    start = max(0.0, length - 2.0)
    _run([
        "ffmpeg", "-y", "-i", str(clip),
        "-vf", f"fade=t=out:st={start:.2f}:d=2.0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(out),
    ])
    return out


def _moving_still(visuals, work: Path, index: int, length: float) -> Path:
    """One scene slot: a single still in one unbroken motion, no switching.

    Multiple subjects per scene used to slice the scene into zoom/vanish
    sub-clips; that hard cut read as a glitch, so scenes now hand over
    subject choice to whoever built the asset list and play just the
    primary still for the whole scene.
    """

    scored = [asset for asset in visuals if asset.clip_score is not None]
    if scored:
        # what a shot DEPICTS outranks when its narration word lands
        primary = max(scored, key=lambda asset: asset.clip_score)
    else:
        primary = min(visuals, key=lambda asset: asset.position)
    source = _fetch_image(
        primary.thumb_url or primary.url, work, f"img-{index:02d}-0.jpg"
    )
    wide = (primary.width or 0) / max(primary.height or 1, 1) >= 1.0
    return _still_clip(source, work, index, length, wide)


HORROR_GRADE = (
    "eq=contrast=1.08:saturation=0.42:brightness=-0.05:gamma=1.07,"
    "colorbalance=rs=-0.12:gs=0.02:bs=0.12,"
    "noise=alls=7:allf=t,"
    "vignette=PI/4.2"
)


def _join_cards(clips: list[Path], work: Path, *, graded: bool = False) -> Path:
    concat_list = work / "cards.txt"
    concat_list.write_text(
        "\n".join(f"file '{clip.resolve()}'" for clip in clips),
        encoding="utf-8",
    )
    silent = work / "cards.mp4"
    filters = HORROR_GRADE if graded else None
    args = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list)]
    if filters:
        args += ["-vf", filters]
    args += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(silent),
    ]
    _run(args)
    return silent


def _voice_track(wavs: list[Path], lengths: list[float], work: Path) -> Path:
    """Narration lines padded to their card lengths, so audio and video
    hands over at exactly the same frames — subtitles can then trust the
    timeline instead of drifting."""

    tensors, sample_rate = [], None
    for wav, length in zip(wavs, lengths, strict=True):
        tensor, sample_rate = torchaudio.load(str(wav))
        target = int(sample_rate * length)
        if tensor.shape[1] < target:
            tensor = torch.nn.functional.pad(tensor, (0, target - tensor.shape[1]))
        tensors.append(tensor)
    voice = work / "voice-full.wav"
    torchaudio.save(str(voice), torch.cat(tensors, dim=1), sample_rate)
    return voice


def _sfx_shelf() -> dict[str, dict]:
    """Effect term -> {file, seconds} from the pulled SFX manifest."""

    manifest = Path("assets/sfx/sfx.json")
    if not manifest.exists():
        return {}
    shelf: dict[str, dict] = {}
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        term = entry.get("term", "")
        if term and term not in shelf and Path(entry.get("file", "")).exists():
            shelf[term] = {"file": entry["file"], "seconds": float(entry.get("seconds", 4))}
    return shelf


def _apply_sfx(
    voice: Path,
    cues: list,
    offsets: list[float],
    total: float,
    work: Path,
) -> Path:
    """Stamp the sound-design cues onto the padded narration track."""

    shelf = _sfx_shelf()
    if not cues or not shelf:
        return voice
    tensor, sample_rate = torchaudio.load(str(voice))
    out = torch.zeros_like(tensor)
    usable = min(tensor.shape[1], out.shape[1])
    out[:, :usable] = tensor[:, :usable]
    for cue in cues:
        item = shelf.get(cue.effect)
        if not item:
            continue
        effect, rate = torchaudio.load(str(item["file"]))
        if effect.shape[0] > 1:
            effect = effect.mean(dim=0, keepdim=True)
        if rate != sample_rate:
            effect = torchaudio.functional.resample(effect, rate, sample_rate)
        effect = effect[:, : int(min(5.0, item["seconds"]) * sample_rate)]
        if effect.shape[1] < 64:
            continue
        ramp = int(0.02 * sample_rate)
        effect[:, :ramp] *= torch.linspace(0.0, 1.0, ramp)
        effect[:, -ramp:] *= torch.linspace(1.0, 0.0, ramp)
        effect = effect * cue.gain
        place = offsets[min(cue.scene_index, len(offsets) - 1)] + cue.at
        start = max(0, int(place * sample_rate))
        end = min(start + effect.shape[1], out.shape[1])
        if end > start:
            out[:, start:end] += effect[:, : end - start]
    stamped = work / "voice-sfx.wav"
    torchaudio.save(str(stamped), out, sample_rate)
    return stamped


def _mix_audio(voice: Path, total: float, work: Path) -> Path:
    mixed = work / "mixed.m4a"
    fade_start = max(0.0, total - 2.5)
    _run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(MUSIC),
        "-i", str(voice),
        "-filter_complex",
        f"[0:a]volume={MUSIC_GAIN},afade=t=out:st={fade_start:.2f}:d=2.5[bg];"
        f"[1:a][bg]amix=inputs=2:duration=first:weights=1 1[mo]",
        "-map", "[mo]", "-c:a", "aac", "-b:a", "192k", str(mixed),
    ])
    return mixed


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
        video = ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        video = ["-c:v", "copy"]
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
    *, stem: str = "voice",
) -> Path:
    """Assemble a vertical feed video: card per line, topical still per scene."""

    work = out_path.parent / "render-work"
    work.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [script.hook, *(scene.narration for scene in script.scenes), script.outro]
    wavs: list[Path] = []
    for index in range(len(lines)):
        wav = audio_dir / f"{stem}-{index:02d}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"Missing narration {wav}; run 'videoai voice' first.")
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
    cues = align_cues(lines, wavs, offsets, line_ends)
    ass_path = write_ass(cues, work / "subs.ass")

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
        if a.kind == "image"
    ]
    if hook_visuals:
        clips.append(_moving_still(hook_visuals, work, 0, lengths[0]))
    else:
        clips.append(_blank_clip(work, 0, lengths[0]))
    for index, (scene, _script_scene) in enumerate(
        zip(
            [e for e in pack.scenes if e.scene_index >= 0],
            script.scenes,
            strict=True,
        )
    ):
        visuals = [a for a in scene.assets if a.kind == "image"]
        if visuals:
            clips.append(_moving_still(visuals, work, index + 1, lengths[index + 1]))
        else:
            clips.append(_blank_clip(work, index + 1, lengths[index + 1]))

    # Outro: no more text card. Fade out over the last scene's closing
    # image; when a scene has nothing, keep a plain dark card.
    last_visuals = [
        a for a in (pack.scenes[-1].assets if pack.scenes else []) if a.kind == "image"
    ]
    if last_visuals:
        source = _fetch_image(
            last_visuals[-1].thumb_url or last_visuals[-1].url,
            work,
            "img-outro.jpg",
        )
        wide = (last_visuals[-1].width or 0) / max(last_visuals[-1].height or 1, 1) >= 1.0
        outro_raw = _still_clip(source, work, len(pack.scenes) + 1, lengths[-1], wide)
    else:
        outro_raw = _blank_clip(work, len(pack.scenes) + 1, lengths[-1])
    clips.append(_fade_out_clip(outro_raw, lengths[-1], work, len(pack.scenes) + 1))

    silent = _join_cards(
        clips,
        work,
        graded=any(
            "original horror story" in note.lower() for note in script.source_notes
        ),
    )
    voice = _voice_track(wavs, lengths, work)
    total = sum(lengths)
    sfx_path = out_path.parent / "sfx.json"
    if sfx_path.exists():
        plan = SfxPlan.model_validate_json(sfx_path.read_text(encoding="utf-8"))
        voice = _apply_sfx(voice, plan.cues, offsets, total, work)
    mixed = _mix_audio(voice, total, work)
    return _mux(silent, mixed, total, out_path, ass_path)
