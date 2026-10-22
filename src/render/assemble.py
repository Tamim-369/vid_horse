"""Video assembly: timed scene stills + voiceover + music bed -> final MP4."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
import textwrap
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio

from core.errors import BlockError
from core.models import AssetPack, SfxPlan, VideoScript
from voice.subtitles import align_speech, write_ass

logger = logging.getLogger(__name__)

WIDTH = 1080
HEIGHT = 1920
FPS = 25
MUSIC_GAIN = 0.35
MUSIC = Path("assets/music/normal.mp3")
MUSIC_SHELF = Path("assets/music/music.json")
CARD_BG = "0x0d1117"
CARD_TEXT_LINES = 26

# Working canvas for motion: 4x the output frame, keeps every animated crop
# sub-pixel smooth and keeps a cropped wallpaper shot readable.
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


def _still_clip(source: Path, work: Path, tag: str | int, length: float) -> Path:
    """One animated clip from a still: cover-crop prep + moving window.

    The photo always fills the whole vertical frame (scaled to cover,
    sides cropped) so wallpaper-ratio shots show real detail instead of
    floating small on a blur bed. ``tag`` names the intermediate file
    (``clip-{tag}.mp4``); subject sub-clips pass ``{card}-{subject}`` so
    one card's subjects never collide.
    """

    style = sum(int(ch) for ch in str(tag) if ch.isdigit()) % 3
    clip = work / f"clip-{tag}.mp4"
    frames = max(1, round(length * FPS))
    tail = f"{_motion_filter(frames, style)}"
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

    return _still_clip(_title_card("", work, index), work, index, length)


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


_CROSSFADE = 0.45
_MIN_SWITCH_GAP = 0.9
_MAX_SUBJECTS = 4


def _narr_tokens(text: str) -> list[str]:
    """Lowercase narration/heard tokens; punctuation and dashes are noise."""

    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w]


def _find_window(trig: list[str], pool: list[str]) -> int | None:
    """First pool position of the trigger fragment, longest window first."""

    if not trig:
        return None
    window = min(len(trig), 6)
    while window >= 2:
        for position in range(len(pool) - window + 1):
            if pool[position : position + window] == trig[:window]:
                return position
        window -= 1
    return None


def _subject_arrivals(
    visuals, line: str, word_line
) -> list[float] | None:
    """Card-relative time each subject's trigger phrase is spoken at.

    ``word_line`` is the whisper word timeline (absolute seconds) for the
    line. The director's trigger is a consecutive narration fragment of up
    to six words. Matching runs against what whisper HEARD first — the
    voice says "two thousand five" for the written "2005", so heard words
    and word times always belong to the same list — and falls back to the
    written narration when spelling differs ("25" vs "twenty-five").
    Images whose trigger cannot be resolved get the midpoints of their
    resolved neighbours — rough timing beats none.
    """

    if len(visuals) < 2 or not word_line:
        return None
    tokens = _narr_tokens(line)
    heard: list[str] = []
    starts: list[float] = []
    for entry in word_line:
        word, at = entry[0], entry[1]
        parts = _narr_tokens(word)
        heard.extend(parts)
        starts.extend([at] * len(parts))

    def fraction_time(factor: float) -> float:
        first = starts[0] if starts else 0.0
        last = starts[-1] if len(starts) > 1 else first
        return first + factor * max(last - first, 0.0)

    resolved: list[float | None] = []
    for visual in visuals:
        trig = _narr_tokens(visual.trigger)
        at = None
        position = _find_window(trig, heard)
        if position is not None:
            at = starts[position]
        if at is None:
            position = _find_window(trig, tokens)
            if position is not None and tokens:
                at = fraction_time(position / len(tokens))
        if at is None and trig:  # first word heard through a distortion
            for index, token in enumerate(heard):
                if token == trig[0]:
                    at = starts[index]
                    break
        if at is None and trig and tokens:  # first word, written form
            for index, token in enumerate(tokens):
                if token == trig[0]:
                    at = fraction_time(index / len(tokens))
                    break
        resolved.append(None if at is None else at)

    if not any(value is not None for value in resolved):
        return None
    span = max((starts[-1] - starts[0]) if len(starts) > 1 else 0.1, 0.1)
    slots = len(resolved)
    for local_index, value in enumerate(resolved):
        if value is None:
            earlier = next(
                (
                    resolved[probe]
                    for probe in range(local_index - 1, -1, -1)
                    if resolved[probe] is not None
                ),
                None,
            )
            later = next(
                (
                    resolved[probe]
                    for probe in range(local_index + 1, slots)
                    if resolved[probe] is not None
                ),
                None,
            )
            if earlier is not None and later is not None:
                value = (earlier + later) / 2
            elif earlier is not None:
                value = earlier + span / (slots + 1)
            elif later is not None:
                value = max(0.0, later - span / (slots + 1))
            else:
                value = (local_index + 1) * span / (slots + 1)
            resolved[local_index] = value
    line_offset = starts[0] if starts else 0.0
    return [max(0.0, value - line_offset) for value in resolved]


def _moving_still(
    visuals,
    work: Path,
    index: int,
    length: float,
    line: str = "",
    word_line=None,
) -> Path:
    """One scene slot: crossfade each subject image in at its narration word.

    Every distinct subject the director found gets its own still, staged at
    the moment the narration names it (director triggers matched against
    the whisper word timeline). One short dissolve per subject replaces
    the old hard cuts. With a single visual or unreadable timing, the
    card keeps the old single primary still in one motion.
    """

    if not visuals:
        return _blank_clip(work, index, length)

    pairs: list[tuple[float, object]] | None = None
    if line and word_line:
        arrivals = _subject_arrivals(visuals, line, word_line)
        if arrivals is not None:
            ranked = sorted(zip(arrivals, visuals, strict=True), key=lambda pair: pair[0])
            pairs = [ranked[0]]
            for moment, asset in ranked[1:]:
                if len(pairs) >= _MAX_SUBJECTS:
                    break
                if moment < pairs[-1][0] + _MIN_SWITCH_GAP or moment > length - 1.2:
                    continue
                pairs.append((moment, asset))
    if pairs is not None and len(pairs) < 2:
        pairs = None

    if pairs is None:
        scored = [asset for asset in visuals if asset.clip_score is not None]
        primary = (
            max(scored, key=lambda asset: asset.clip_score)
            if scored
            else min(visuals, key=lambda asset: asset.position)
        )
        source = _fetch_image(
            primary.thumb_url or primary.url, work, f"img-{index:02d}-0.jpg"
        )
        return _still_clip(source, work, index, length)

    arrivals = [moment for moment, _asset in pairs]
    subject_assets = [asset for _moment, asset in pairs]
    switches = arrivals[1:]
    # xfade algebra: output total = sum(subject lengths) - k*_CROSSFADE;
    # each transition starts exactly at the arriving subject's word.
    subject_lengths = [switches[0] + _CROSSFADE]
    subject_lengths += [
        switches[j] + _CROSSFADE - switches[j - 1] for j in range(1, len(switches))
    ]
    subject_lengths.append(length - switches[-1])

    subpaths: list[Path] = []
    for slot, asset in enumerate(subject_assets):
        tag = f"{index:02d}-{slot}"
        source = _fetch_image(asset.thumb_url or asset.url, work, f"img-{tag}.jpg")
        subpaths.append(_still_clip(source, work, tag, subject_lengths[slot]))

    args: list[str] = ["ffmpeg", "-y"]
    for path in subpaths:
        args += ["-i", str(path)]
    chain: list[str] = []
    previous = "[0:v]"
    for j, moment in enumerate(switches):
        label = f"[vx{j}]"
        chain.append(
            f"{previous}[{j + 1}:v]xfade=transition=fade"
            f":duration={_CROSSFADE:.2f}:offset={moment:.3f}{label}"
        )
        previous = label
    out = work / f"clip-{index:02d}.mp4"
    args += [
        "-filter_complex", ";".join(chain), "-map", previous,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(out),
    ]
    _run(args)
    return out


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


def _shelf() -> dict:
    try:
        return json.loads(MUSIC_SHELF.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("Music shelf unreadable (%s); default bed.", error)
        return {"tracks": [], "roles": {}}


def _track_gain(shelf: dict, track_name: str) -> tuple[Path, float]:
    entry = next(
        (
            item
            for item in shelf.get("tracks", [])
            if item.get("file") == track_name
        ),
        {"file": track_name, "gain_db": 0.0},
    )
    path = Path("assets/music") / track_name
    if not path.exists():
        logger.warning("Music %s missing; default bed.", path)
        path = MUSIC
    gain = MUSIC_GAIN * (10 ** (entry.get("gain_db", 0.0) / 20))
    return path, gain


def _pick_music(
    script: VideoScript, genre: str | None, override: str = ""
) -> tuple[Path, float]:
    """The base bed (legacy path when the script carries no MusicPlan).

    auto: horror scripts get the deep drone, cosmic keeps the proven
    neutral pulse (it is the channel's sound), everything else gets the
    unease bed. An explicit override always wins. Per-track gain_db keeps
    quiet entries audible at the same relative level.
    """

    if override:
        return Path("assets/music") / override, MUSIC_GAIN
    plan = script.music
    if plan is not None:
        return _track_gain(_shelf(), _role_track(plan.base_role))
    if any("original horror story" in note.lower() for note in script.source_notes):
        lane = "horror"
    elif genre == "cosmic":
        lane = "cosmic"
    else:
        lane = "conspiracy"
    track = {"file": MUSIC.name, "gain_db": 0.0}
    shelf = _shelf()
    for entry in shelf.get("tracks", []):
        if lane in (entry.get("genres") or []):
            track = entry
            break
    return _track_gain(shelf, track["file"])


def _role_track(
    role: str, shelf: dict | None = None
) -> str:
    """Resolve one role to a track file name (unknown role -> default)."""

    shelf = shelf or _shelf()
    return (shelf.get("roles").get(role) or shelf["roles"]["default"])["track"]


def _bed_tracks(
    script: VideoScript, genre: str | None, override: str
) -> list[tuple[Path, float]]:
    """Music per card: card 0 = hook, 1..N = scenes, N+1 = outro.

    A MusicPlan gives one base track with occasional per-scene overrides —
    "one music through the video, some scenes get their own". The script's
    scene ``i`` is the card at index ``i + 1``. An explicit override
    flattens everything to one track.
    """

    if override:
        return [Path("assets/music") / override, MUSIC_GAIN] * 1
    plan = script.music
    if plan is None:
        path, gain = _pick_music(script, genre)
        return [(path, gain)]
    shelf = _shelf()
    base_path, base_gain = _track_gain(shelf, _role_track(plan.base_role, shelf))
    cards: list[tuple[Path, float]] = []
    for card in range(len(script.scenes) + 2):
        role = plan.scene_roles.get(str(card - 1)) if card >= 1 else None
        if role:
            cards.append(_track_gain(shelf, _role_track(role, shelf)))
        else:
            cards.append((base_path, base_gain))
    return cards


_CROSSFADE_MUSIC = 2.0
_SPAN_PAD = 1.0


def _music_spans(
    cards: list[tuple[Path, float]], offsets: list[float], total: float
) -> list[tuple[Path, float, float, float]]:
    """Collapse the per-card choices into continuous ``(track, gain, from, to)``.

    Adjacent cards sharing a track join silently: the base bed flows
    through them, and only a real override earns a junction.
    """

    spans: list[tuple[Path, float, float, float]] = []
    for card, (path, gain) in enumerate(cards):
        start = offsets[card]
        end = offsets[card + 1] if card + 1 < len(offsets) else total
        if spans and spans[-1][0] == path and spans[-1][1] == gain:
            spans[-1] = (*spans[-1][:3], end)
        else:
            spans.append((path, gain, start, end))
    return spans


def _build_bed(
    spans: list[tuple[Path, float, float, float]], work: Path, total: float
) -> Path:
    """One audio file: every span plays its track, junctions crossfade 2s.

    Each interior span is padded 1s into both neighbours so the 2s
    crossfade sits exactly on the scene boundary; the last span keeps the
    video's end so the sum of windows minus the junctions equals ``total``.
    """

    bed = work / "bed.wav"
    windows: list[tuple[Path, float, float, float]] = []
    for index, (path, gain, start, end) in enumerate(spans):
        first = index == 0
        last = index == len(spans) - 1
        w_start = start if first else max(0.0, start - _SPAN_PAD)
        w_end = end if last else end + _SPAN_PAD
        windows.append((path, gain, w_start, w_end))
    args: list[str] = ["ffmpeg", "-y"]
    chains: list[str] = []
    for index, (path, gain, w_start, w_end) in enumerate(windows):
        args += ["-stream_loop", "-1", "-i", str(path)]
        chains.append(
            f"[{index}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
            f"atrim=start={w_start:.3f}:end={w_end:.3f},volume={gain:.3f}[s{index}]"
        )
    label = "s0"
    for index in range(1, len(windows)):
        out = f"x{index}"
        chains.append(f"[{label}][s{index}]acrossfade=d={_CROSSFADE_MUSIC:.1f}[{out}]")
        label = out
    fade_start = max(0.0, total - 2.5)
    chains.append(f"[{label}]afade=t=out:st={fade_start:.3f}:d=2.5[bed]")
    args += ["-filter_complex", ";".join(chains), "-map", "[bed]", str(bed)]
    _run(args)
    return bed


def _mix_audio(
    voice: Path,
    bed: Path,
    work: Path,
) -> Path:
    mixed = work / "mixed.m4a"
    # Voice leads: weights push the narration above the bed, and the bed
    # sidechain-compresses under the voice so dynamic tracks (swells,
    # impacts) can never burst over a spoken line — while still rising
    # back to full level during narration gaps.
    _run([
        "ffmpeg", "-y",
        "-i", str(voice),
        "-i", str(bed),
        "-filter_complex",
        "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[nv][sc];"
        "[1:a]aformat=sample_rates=44100:channel_layouts=stereo[bg];"
        "[bg][sc]sidechaincompress=threshold=0.02:ratio=7:attack=0.12:release=0.9[duck];"
        "[nv]volume=1.6[vpad];[vpad][duck]amix=inputs=2:duration=first:normalize=0[mo];"
        "[mo]alimiter=limit=0.95:level=false[mo2]",
        "-map", "[mo2]", "-c:a", "aac", "-b:a", "192k", str(mixed),
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
        # Caption backdrop: the bottom 30% gets blurred and darkened so crisp
        # subtitles always read against busy imagery. The band's top edge
        # feathers over 96px via a blend ramp — no hard seam. The blend
        # expression needs its argument commas escaped (\,).
        top, low = 1344, 1440
        ramp = f"(Y-{top})/{low - top}"
        weight = (
            f"if(lt(Y\\,{top})\\,0\\,if(lt(Y\\,{low})\\,{ramp}\\,1))"
        )
        graph = (
            "[0:v]split=2[base][orig];"
            "[orig]crop=iw:ih*0.30:0:ih-ih*0.30[zone];"
            "[zone]boxblur=luma_radius=24:luma_power=2[soft];"
            "[soft]drawbox=w=iw:h=ih:color=black@0.40:t=fill[band];"
            "[base][band]overlay=0:main_h-overlay_h[shadowed];"
            "[shadowed][orig]"
            f"blend=all_expr='{weight}'[blended];"
            f"[blended]ass='{ass_path}'[tout]"
        )
        args += ["-filter_complex", graph]
        video = [
            "-map", "[tout]",
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "20", "-pix_fmt", "yuv420p",
        ]
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
    *,
    stem: str = "voice",
    genre: str | None = None,
    music: str = "",
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
    word_lines, cues = align_speech(lines, wavs, offsets, line_ends)
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

    def hook_card() -> Path:
        if hook_visuals:
            return _moving_still(
                hook_visuals, work, 0, lengths[0], line=lines[0], word_line=word_lines[0]
            )
        return _blank_clip(work, 0, lengths[0])

    build: list[tuple[int, Callable[..., Path]]] = [(0, hook_card)]
    for index, (scene, _script_scene) in enumerate(
        zip(
            [e for e in pack.scenes if e.scene_index >= 0],
            script.scenes,
            strict=True,
        )
    ):
        visuals = [a for a in scene.assets if a.kind == "image"]

        def scene_card(
            card=index + 1, vis=visuals, text=lines[index + 1], words=word_lines[index + 1]
        ) -> Path:
            if vis:
                return _moving_still(vis, work, card, lengths[card], line=text, word_line=words)
            return _blank_clip(work, card, lengths[card])

        build.append((index + 1, scene_card))
    # the per-card ffmpeg jobs are independent; three at a time roughly
    # triples this stage with no influence on what any card looks like
    with ThreadPoolExecutor(max_workers=3) as pool:
        built = sorted(pool.map(lambda item: (item[0], item[1]()), build))
    clips.extend(path for _index, path in built)

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
        outro_raw = _still_clip(source, work, len(build), lengths[-1])
    else:
        outro_raw = _blank_clip(work, len(build), lengths[-1])
    clips.append(_fade_out_clip(outro_raw, lengths[-1], work, len(build)))

    # Each card must actually exist for its line's length: the parallel
    # builders once let an off-by-one overwrite the hook clip, the missing
    # scene went black while its narration (and subs) kept playing.
    for clip_path, expected in zip(
        clips,
        [*lengths[:-1], lengths[-1]],
        strict=True,
    ):
        actual = _probe_duration(clip_path)
        if abs(actual - expected) > 0.3:
            raise BlockError(
                f"Card {clip_path.name} is {actual:.2f}s but its narration "
                f"needs {expected:.2f}s; refusing to ship a broken cut."
            )

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
    cards = _bed_tracks(script, genre, override=music)
    spans = _music_spans(cards, offsets, total)
    describe = ", ".join(
        f"{path.name} {start:.1f}-{end:.1f}s" for path, _gain, start, end in spans
    )
    logger.info("Music bed: %s", describe)
    bed = _build_bed(spans, work, total)
    mixed = _mix_audio(voice, bed, work)
    return _mux(silent, mixed, total, out_path, ass_path)
