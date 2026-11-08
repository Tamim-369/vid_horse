"""Card visuals: title cards, motion stills, credits and subject timing."""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)


from .media import (  # noqa: E402  render shared constants
    CANVAS_H,
    CANVAS_W,
    CARD_BG,
    CARD_TEXT_LINES,
    FPS,
    HEIGHT,
    WIDTH,
    _asset_source,
    _font,
    _is_video_file,
    _run,
)


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
    """One of multiple window motions over the still.

    Styles 0-5: classic Ken Burns (zoom/pan/drift)
    Styles 6-11: subtle rotation + zoom combos
    Styles 12-17: slow diagonal sweeps
    Styles 18-23: gentle parallax-style moves
    All keep zoom/drift gentle so it reads as cinematic, not AI-slides.
    """

    end = 1.0 + zoom
    on_rate = frames
    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"
    travel = f"(iw-iw/zoom)*{drift:.2f}"
    v_travel = f"(ih-ih/zoom)*{drift:.2f}"

    # --- Classic Ken Burns (0-5) ---
    if style == 0:
        # slow zoom-in
        z_expr = f"min(1+{zoom:.4f}*on/{on_rate},{end:.3f})"
        x, y = center_x, center_y
    elif style == 1:
        # pan L->R
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*on/{on_rate}"
        y = center_y
    elif style == 2:
        # pan R->L
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*(1-on/{on_rate})"
        y = center_y
    elif style == 3:
        # zoom-out: start close and settle back
        z_expr = f"max(1+{zoom:.4f}*(1-on/{on_rate}),1.0)"
        x, y = center_x, center_y
    elif style == 4:
        # horizontal drift with slow vertical sink
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*(1-on/{on_rate})"
        y = f"{center_y}+{v_travel}*on/{on_rate}"
    elif style == 5:
        # vertical drift down (crane shot)
        z_expr = f"{end:.3f}"
        x = center_x
        y = f"{center_y}+{v_travel}*(1-on/{on_rate})"

    # --- Subtle rotation + zoom (6-11) ---
    elif style == 6:
        # slow zoom-in + tiny clockwise rotation (0.3° total)
        z_expr = f"min(1+{zoom:.4f}*on/{on_rate},{end:.3f})"
        x = f"{center_x}+(iw/zoom)*sin(0.005*on/{on_rate})"
        y = f"{center_y}-(ih/zoom)*cos(0.005*on/{on_rate})"
    elif style == 7:
        # slow zoom-out + tiny counter-clockwise rotation
        z_expr = f"max(1+{zoom:.4f}*(1-on/{on_rate}),1.0)"
        x = f"{center_x}-(iw/zoom)*sin(0.005*on/{on_rate})"
        y = f"{center_y}+(ih/zoom)*cos(0.005*on/{on_rate})"
    elif style == 8:
        # pan L->R with slight rotation
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*on/{on_rate}"
        y = f"{center_y}+(ih/zoom)*0.02*sin(3.14*on/{on_rate})"
    elif style == 9:
        # pan R->L with slight rotation
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*(1-on/{on_rate})"
        y = f"{center_y}-(ih/zoom)*0.02*sin(3.14*on/{on_rate})"
    elif style == 10:
        # diagonal sweep TL->BR
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*on/{on_rate}"
        y = f"{center_y}+{v_travel}*on/{on_rate}"
    elif style == 11:
        # diagonal sweep TR->BL
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*(1-on/{on_rate})"
        y = f"{center_y}+{v_travel}*on/{on_rate}"

    # --- Slow diagonal sweeps (12-17) ---
    elif style == 12:
        # gentle diagonal TL->BR, minimal zoom
        z_expr = f"1+{zoom*0.5:.4f}*on/{on_rate}"
        x = f"{center_x}+(iw/zoom)*0.15*on/{on_rate}"
        y = f"{center_y}+(ih/zoom)*0.15*on/{on_rate}"
    elif style == 13:
        # gentle diagonal TR->BL
        z_expr = f"1+{zoom*0.5:.4f}*on/{on_rate}"
        x = f"{center_x}-(iw/zoom)*0.15*on/{on_rate}"
        y = f"{center_y}+(ih/zoom)*0.15*on/{on_rate}"
    elif style == 14:
        # gentle diagonal BL->TR
        z_expr = f"1+{zoom*0.5:.4f}*on/{on_rate}"
        x = f"{center_x}+(iw/zoom)*0.15*on/{on_rate}"
        y = f"{center_y}-(ih/zoom)*0.15*on/{on_rate}"
    elif style == 15:
        # gentle diagonal BR->TL
        z_expr = f"1+{zoom*0.5:.4f}*on/{on_rate}"
        x = f"{center_x}-(iw/zoom)*0.15*on/{on_rate}"
        y = f"{center_y}-(ih/zoom)*0.15*on/{on_rate}"
    elif style == 16:
        # vertical slow zoom (portrait feel)
        z_expr = f"min(1+{zoom*0.8:.4f}*on/{on_rate},{1+zoom*0.8:.3f})"
        x = center_x
        y = f"{center_y}+(ih/zoom)*0.1*on/{on_rate}"
    elif style == 17:
        # horizontal slow zoom (landscape feel)
        z_expr = f"min(1+{zoom*0.8:.4f}*on/{on_rate},{1+zoom*0.8:.3f})"
        x = f"{center_x}+(iw/zoom)*0.1*on/{on_rate}"
        y = center_y

    # --- Parallax-style subtle moves (18-23) ---
    elif style == 18:
        # slow arc: pan with gentle curve
        z_expr = f"{end:.3f}"
        x = f"{center_x}+{travel}*sin(1.57*on/{on_rate})"
        y = f"{center_y}+{v_travel}*(1-cos(1.57*on/{on_rate}))"
    elif style == 19:
        # slow arc reverse
        z_expr = f"{end:.3f}"
        x = f"{center_x}-{travel}*sin(1.57*on/{on_rate})"
        y = f"{center_y}+{v_travel}*(1-cos(1.57*on/{on_rate}))"
    elif style == 20:
        # very slow zoom, almost static (breathing room)
        z_expr = f"1+{zoom*0.3:.4f}*sin(3.14*on/{on_rate})"
        x, y = center_x, center_y
    elif style == 21:
        # micro-pan: barely perceptible drift
        z_expr = f"1+{zoom*0.2:.4f}"
        x = f"{center_x}+(iw/zoom)*0.05*on/{on_rate}"
        y = center_y
    elif style == 22:
        # micro-vertical drift
        z_expr = f"1+{zoom*0.2:.4f}"
        x = center_x
        y = f"{center_y}+(ih/zoom)*0.05*on/{on_rate}"
    else:
        # style 23: almost static, just breathing zoom
        z_expr = f"1+{zoom*0.15:.4f}*sin(6.28*on/{on_rate})"
        x, y = center_x, center_y

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


def _video_clip(source: Path, work: Path, tag: str | int, length: float) -> Path:
    """One card built from real footage: cover-crop, native motion.

    The clip plays its own frames (no zoompan) at whatever trim starts
    deterministically; undersized sources loop via ``-stream_loop``.
    Source audio never leaks: the voice and the music bed are the only
    sound this pipeline speaks.
    """

    clip = work / f"clip-{tag}.mp4"
    _run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(source),
        "-vf", f"{_cover_prep()},scale={WIDTH}:{HEIGHT}",
        "-t", f"{length:.3f}", "-r", str(FPS), "-an",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", str(clip),
    ])
    return clip


_MOTION_POOL = list(range(24))  # 0-23: varied Ken Burns + rotation + diagonal + parallax


def _style_for(tag: str | int) -> int:
    """Deterministic but non-repeating motion pick for a card.

    The tag carries the card index (and the subject slot after it); we
    walk the motion pool one step per card and offset again per subject,
    so no two adjacent cards and no two subject switches inside a card
    ever use the same window motion. The old digit-sum hash let
    consecutive cards repeat the same zoom for whole videos, which is
    exactly the AI-slides look.
    """

    parts = [part for part in re.split(r"[-_]", str(tag)) if part.isdigit()]
    card = int(parts[0]) if parts else 0
    slot = int(parts[1]) if len(parts) > 1 else 0
    return _MOTION_POOL[(card + slot) % len(_MOTION_POOL)]


def _attribution(asset) -> tuple[str, int] | None:
    """Full credit text + corner fontsize, or None for self-free media.

    Public-domain and Pexels-free frames carry no obligation, so they get
    no text at all; CC-BY-style licenses legally expect a name, and the
    whole name goes in — clipped attribution ("…") is worse than none.
    Long credits buy completion with size instead: the font shrinks from
    22px toward a 12px whisper and word-wraps beside the frame edge, so
    the corner line stays skippable unless the viewer hunts for it.
    """

    license_text = (asset.license or "").lower()
    if "cc by" not in license_text:
        return None
    name = (asset.artist or "").strip()
    if not name:
        name = asset.file.rsplit(".", 1)[0].split("File:")[-1].strip()
    short = (asset.license or "").split(",")[0].split(";")[0].strip()
    text = f"{name} ({short})".replace(" , ", ", ")

    # ~0.55 em average advance at this face; 840px keeps the strip inside
    # a corner budget on the 1080-wide canvas
    size = max(12, min(22, int(840 / (0.55 * len(text)))))
    lines = [text]
    if 0.55 * size * len(text) > 870:
        lines = []
        line = ""
        for word in text.split():
            probe = f"{line} {word}".strip()
            if line and 0.55 * size * len(probe) > 900:
                lines.append(line)
                line = word
            else:
                line = probe
        if line:
            lines.append(line)
    return "\n".join(lines), size


def _credit_draw(asset, work: Path, tag: str | int) -> str | None:
    """The corner-credit drawtext filter piece, or None for self-free media.

    Writes the escaped text file so the card's own encode carries it —
    one ffmpeg pass total instead of an extra re-encode per credit.
    """

    credited = _attribution(asset)
    if not credited:
        return None
    credit, fontsize = credited
    text_file = work / f"credit-{tag}.txt"
    text_file.write_text(credit, encoding="utf-8")
    font = _font()
    draw = [
        f"textfile={text_file}",
        "fontcolor=white@0.35",
        f"fontsize={fontsize}",
        "x=w-text_w-28",
        "y=h-text_h-56",
    ]
    if font:
        draw.insert(1, f"fontfile={font}")
    return f"drawtext={':'.join(draw)}"


def _still_clip(
    source: Path,
    work: Path,
    tag: str | int,
    length: float,
    credit: str | None = None,
) -> Path:
    """One animated clip from a still: cover-crop prep + moving window.

    The photo always fills the whole vertical frame (scaled to cover,
    sides cropped) so wallpaper-ratio shots show real detail instead of
    floating small on a blur bed. ``tag`` names the intermediate file
    (``clip-{tag}.mp4``); subject sub-clips pass ``{card}-{subject}`` so
    one card's subjects never collide. ``credit`` is the corner drawtext
    baked into this same pass (saves a full re-encode per CC-BY card).
    """

    style = _style_for(tag)
    clip = work / f"clip-{tag}.mp4"
    if _is_video_file(source):
        return _video_clip(source, work, tag, length)
    frames = max(1, round(length * FPS))
    tail = f"{_motion_filter(frames, style)}"
    vf = f"{_cover_prep()},{tail}"
    if credit:
        vf = f"{vf},{credit}"
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(source), "-vf", vf,
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


def _split_sentences(text: str) -> list[str]:
    """Narration chunks per sentence; short lines stay whole."""

    parts = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+", text.strip())
        if piece.strip()
    ]
    meaty = [piece for piece in parts if len(piece.split()) >= 2]
    return meaty or ([text.strip()] if text.strip() else [])


def _sentence_spans(
    sentences: list[str], line: str, word_line, length: float
) -> list[tuple[float, float]]:
    """Card-relative (start, end) seconds for every sentence."""

    tokens = _narr_tokens(line)
    total = len(tokens) if tokens else 1
    if word_line:
        starts = [at for _word, at in ((e[0], e[1]) for e in word_line)]
        first = starts[0] if starts else 0.0
        last = starts[-1] if len(starts) > 1 else first
        span = max(last - first, 0.1)
        base = first
        scale = span / max(len(tokens), 1)
    else:
        base, scale = 0.0, length / max(total, 1)

    spans: list[tuple[float, float]] = []
    cursor = 0
    for index, sentence in enumerate(sentences):
        words = len(_narr_tokens(sentence))
        start = base + (cursor / total) * (scale * total)
        cursor += words
        stop = base + (cursor / total) * (scale * total)
        if word_line:
            start = base + (cursor - words) * scale
            stop = base + cursor * scale
        if index == len(sentences) - 1:
            stop = max(stop, length)
        spans.append((max(0.0, start), max(stop, start + 0.2)))
    return spans


def _caption_overlap(sentence: str, asset) -> int:
    """Content words the narration sentence shares with the asset's caption."""

    sentence_words = set(_narr_tokens(sentence))
    caption_words = set(_narr_tokens(asset.caption or ""))
    return len(sentence_words & caption_words)


def _sentence_schedule(
    line: str, word_line, visuals, length: float
) -> list[tuple[float, object]] | None:
    """Every sentence gets a shot; reuse allowed, never back-to-back.

    Director triggers anchor an asset to the moment its phrase is spoken;
    sentences without a trigger get a fill image chosen by caption
    overlap with the sentence, CLIP score, then least-recently-used —
    and the immediately preceding image is always skipped unless the
    pool is a single asset.
    """

    sentences = _split_sentences(line)
    if not sentences:
        return None
    spans = _sentence_spans(sentences, line, word_line, length)

    arrivals: list[float] | None = None
    if word_line:
        resolved = _subject_arrivals(visuals, line, word_line)
        if resolved is not None:
            arrivals = resolved

    anchor_times: dict[int, float] = {}
    for index in range(len(visuals)):
        moment = arrivals[index] if arrivals else None
        if moment is not None:
            anchor_times[index] = max(0.0, moment)

    schedule: list[tuple[float, object]] = []
    previous: object | None = None
    used_once: set[int] = set()

    def pick_fill(pool: list) -> int | None:
        best: tuple[int, int, int, int] | None = None
        best_index: int | None = None
        sentence_index = min(len(schedule), len(sentences) - 1)
        for index, asset in enumerate(pool):
            if asset is previous:
                continue
            score = (
                1 if index not in used_once else 0,
                _caption_overlap(sentences[sentence_index], asset),
                int(round((asset.clip_score or 0.0) * 100)),
                -index,
            )
            if best is None or score > best:
                best, best_index = score, index
        return best_index

    for slot, (start, end) in enumerate(spans):
        sentence = sentences[slot]
        anchors = [
            index
            for index, moment in anchor_times.items()
            if start - 0.4 <= moment <= end + 0.4
        ]
        ordered: list[int] = sorted(anchors, key=lambda i: anchor_times[i])
        # prefer anchors not repeating the previous shot
        ordered.sort(
            key=lambda i: (
                0 if visuals[i] is not previous else 1,
                -_caption_overlap(sentence, visuals[i]),
            )
        )
        picks = [i for i in ordered if visuals[i] is not previous][:2]
        if not picks:
            fills = pick_fill(visuals)
            picks = (
                [fills] if fills is not None
                else [ordered[0] if ordered else 0]
            )  # no fill and a single matching anchor: the pool allows repeat
        for index in picks:
            moment = max(start, anchor_times.get(index, start))
            if previous is not None and moment < (schedule[-1][0] + _MIN_SWITCH_GAP):
                moment = schedule[-1][0] + _MIN_SWITCH_GAP
            if moment >= length - 0.6:
                continue
            schedule.append((moment, visuals[index]))
            previous = visuals[index]
            used_once.add(index)
    if not schedule:
        return None
    schedule.sort(key=lambda pair: pair[0])
    if schedule[0][0] > 0.0:
        schedule[0] = (0.0, schedule[0][1])
    return schedule


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
        pairs = _sentence_schedule(line, word_line, visuals, length)
    if pairs is not None and len(pairs) < 2:
        pairs = None

    if pairs is None:
        scored = [asset for asset in visuals if asset.clip_score is not None]
        primary = (
            max(scored, key=lambda asset: asset.clip_score)
            if scored
            else min(visuals, key=lambda asset: asset.position)
        )
        source = _asset_source(primary, work, f"img-{index:02d}-0")
        credit = _credit_draw(primary, work, index)
        return _still_clip(source, work, index, length, credit=credit)

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
        source = _asset_source(asset, work, f"img-{tag}")
        subpaths.append(
            _still_clip(
                source, work, tag, subject_lengths[slot],
                credit=_credit_draw(asset, work, tag),
            )
        )

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
