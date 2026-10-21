"""Word-level subtitle cues for narration, aligned from the voice audio.

faster-whisper transcribes each narration line with word timestamps; those
timestamps are shifted by the line's position on the video timeline and
grouped into short cues (3-4 words that split at sentence punctuation).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

import torchaudio

logger = logging.getLogger(__name__)

CUE_MAX_WORDS = 4

# fmt: off
_FORMAT_LINE = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)
_STYLE_LINE = (
    "Style: Cue,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H00141414,"
    "&H90000000,1,0,0,0,100,100,0,0,1,3,2,2,60,60,240,1"
)
# fmt: on

_ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
{_FORMAT_LINE}
{_STYLE_LINE}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_TIMESTAMP = re.compile(r"[@,.!?;:']+$")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _cues(words: list[str], starts: list[float], ends: list[float]) -> list[Cue]:
    """Group narration words into plot-reel style cues."""

    groups: list[Cue] = []
    chunk: list[str] = []
    chunk_start: float | None = None
    chunk_end = 0.0

    def flush(last: float) -> None:
        if chunk and chunk_start is not None:
            groups.append(Cue(chunk_start, max(chunk_end, last), " ".join(chunk)))

    for index, word in enumerate(words):
        trimmed = word.strip(",")
        chunk.append(trimmed)
        if chunk_start is None:
            chunk_start = starts[index]
        chunk_end = ends[index]
        punctuation = word.rstrip().endswith(("!", ".", "?", ",", ";", ":"))
        if punctuation or len(chunk) >= CUE_MAX_WORDS:
            flush(chunk_end)
            chunk.clear()
            chunk_start = None
    flush(chunk_end)
    return groups


def _transcribe_words(
    line: str, wav: Path, offset: float, model
) -> tuple[list[str], list[float], list[float]]:
    """Word list and absolute (start, end) times for one narration line.

    Shared stream: subtitle cues and per-word subject timing both come
    from the SAME transcription pass so they can never disagree.
    """

    segments, _info = model.transcribe(
        str(wav),
        word_timestamps=True,
        language="en",
        initial_prompt=line,
    )
    words: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    metadata = torchaudio.info(str(wav))
    audio_end = metadata.num_frames / metadata.sample_rate
    seen: list[str] = []
    for segment in segments:
        words_in_segment = [w for w in (segment.words or []) if w.word.strip()]
        if not words_in_segment:
            continue
        # initial_prompt occasionally comes back as a hallucinated echo:
        # the whole line re-transcribed into the trailing silence with
        # slightly different words. Anything that repeats words we have
        # already heard in this line is an echo, not narration.
        tokens = [w.word.strip().lower().strip(".,!?;:'\"") for w in words_in_segment]
        cursor = 0
        matched = 0
        for token in tokens:
            scan = token in seen[cursor:]
            if scan:
                matched += 1
                cursor = seen.index(token, cursor) + 1
        if tokens and matched / len(tokens) >= 0.8:
            logger.debug("Dropping echo segment on %s", wav.name)
            continue
        seen.extend(tokens)
        # timestamps past the wav's own samples are padding artefacts
        if max(w.end for w in words_in_segment) > audio_end + 0.1:
            continue
        for word in words_in_segment:
            words.append(word.word.strip())
            starts.append(max(0.0, word.start) + offset)
            ends.append(max(0.0, word.end) + offset)
    if len(words) != len(line.split()):
        logger.debug(
            "ASR heard %d words for a %d-word line; trusting the audio",
            len(words),
            len(line.split()),
        )
    return words, starts, ends


def align_speech(
    lines: list[str],
    wavs: list[Path],
    offsets: list[float],
    line_ends: list[float] | None = None,
    *,
    model_size: str = "base.en",
    device: str = "cpu",
) -> tuple[list[list[tuple[str, float, float]]], list[Cue]]:
    """Transcribe the voice and return both word timings and subtitle cues.

    ``offsets`` are each line's start time in the assembled video — the card
    start frame times from the renderer, in seconds. ``line_ends`` are the
    matching card end times; cues never cross them. The script line itself is
    fed as ``initial_prompt`` so ASR reproduces the narration's own words
    (proper nouns included); whisper supplies only the timing.

    The word lists are absolute (line offset included) ``(word, start, end)``
    per line, newest callers use them to fire subject images at the narration
    word; the cues are the 3-4 word karaoke groups.
    """

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type="int8")
    word_lines: list[list[tuple[str, float, float]]] = []
    cues: list[Cue] = []
    for index, (line, wav, offset) in enumerate(
        zip(lines, wavs, offsets, strict=True)
    ):
        words, starts, ends = _transcribe_words(line, wav, offset, model)
        word_lines.append(list(zip(words, starts, ends, strict=False)))
        line_cues = _cues(words, starts, ends)
        if line_ends is not None:
            line_end = line_ends[index]
            line_cues = [
                replace(cue, end=min(cue.end, line_end - 0.02))
                for cue in line_cues
            ]
        cues.extend(line_cues)
    return word_lines, cues


def align_cues(
    lines: list[str],
    wavs: list[Path],
    offsets: list[float],
    line_ends: list[float] | None = None,
    *,
    model_size: str = "base.en",
    device: str = "cpu",
) -> list[Cue]:
    """Transcribe the voice segments and map their word cues onto the timeline."""

    return align_speech(
        lines, wavs, offsets, line_ends, model_size=model_size, device=device
    )[1]


def _ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cents = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cents:02d}"


def write_ass(cues: list[Cue], path: Path, *, end: float | None = None) -> Path:
    """Serialise the cues to hand-rendered ASS for libass."""

    body: list[str] = []
    for index, cue in enumerate(cues):
        # whisper occasionally reports a zero-length word; keep a floor so
        # every cue is actually readable
        start, stop = cue.start, max(cue.end, cue.start + 0.25)
        # end each cue when its words stop being spoken; only the seam
        # between two cues of continuous speech gets glued together.
        finish = stop
        if (
            index + 1 < len(cues)
            and cues[index + 1].start > stop
            and cues[index + 1].start - stop < 0.4
        ):
            finish = min(cues[index + 1].start - 0.08, stop)
        text = _TIMESTAMP.sub("", cue.text.replace("\n", " ")).strip()
        body.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(max(finish, start))},"
            f"Cue,,0,0,0,,{text}"
        )
    path.write_text(_ASS_HEADER + "\n".join(body) + "\n", encoding="utf-8")
    return path
