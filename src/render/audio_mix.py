"""Audio mixing: voice track, SFX diegetic placement and the music bed."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


from core.models import VideoScript  # noqa: E402

from .media import MUSIC, MUSIC_GAIN, MUSIC_SHELF, _run  # noqa: E402


def _voice_track(wavs: list[Path], lengths: list[float], work: Path) -> Path:
    """Narration lines padded to their card lengths, so audio and video
    hands over at exactly the same frames — subtitles can then trust the
    timeline instead of drifting."""

    tensors, sample_rate = [], None
    for wav, length in zip(wavs, lengths, strict=True):
        tensor, rate = torchaudio.load(str(wav))
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            # lines synthesized at different times can carry different
            # rates (manual edits, older cached wavs): concatenating them
            # unresampled warps pitch and speed, so normalize first
            tensor = torchaudio.functional.resample(tensor, rate, sample_rate)
        target = int(sample_rate * length)
        if tensor.shape[1] < target:
            tensor = torch.nn.functional.pad(tensor, (0, target - tensor.shape[1]))
        tensors.append(tensor)
    if sample_rate is None:
        return work / "voice-full.wav"
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
        ramp = int(0.02 * sample_rate)
        if effect.shape[1] < 2 * ramp:
            continue
        effect[:, :ramp] *= torch.linspace(0.0, 1.0, ramp)
        effect[:, -ramp:] *= torch.linspace(1.0, 0.0, ramp)
        effect = effect * cue.gain
        # an LLM can return a negative scene index; clamp to the first card,
        # not Python's negative wrap onto the outro
        index = max(0, min(cue.scene_index, len(offsets) - 1))
        place = offsets[index] + cue.at
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
        return [(Path("assets/music") / override, MUSIC_GAIN)] * (len(script.scenes) + 2)
    plan = script.music
    if plan is None:
        path, gain = _pick_music(script, genre)
        return [(path, gain)] * (len(script.scenes) + 2)
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
