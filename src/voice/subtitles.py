"""Word-level subtitle cues for narration, aligned from the voice audio.

faster-whisper transcribes each narration line with word timestamps; those
timestamps are shifted by the line's position on the video timeline and
grouped into short cues (3-4 words that split at sentence punctuation).
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
from dataclasses import dataclass, field, replace
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
    "Style: Cue,DejaVu Sans Condensed,62,&H00C8C8C8,&H00C8C8C8,&H00000000,"
    "&HD0000000,0,0,0,0,100,100,0,0,1,2.5,1,2,60,60,420,1"
)
# Rare caption takeover (anchored low-center, feels like a magnified subtitle).
_HIT_LINE = (
    "Style: Hit,DejaVu Sans Condensed,96,&H00FFFFFF,&H00FFFFFF,&H00000000,"
    "&HE0000000,0,0,0,0,100,100,0,0,3,3.5,0,2,60,60,380,1"
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
{_HIT_LINE}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_TIMESTAMP = re.compile(r"[@,.!?;:']+$")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    pops: list[str] = field(default_factory=list)
    # (display_word, abs_start, abs_end) for word-timed rendering
    words: list[tuple[str, float, float]] = field(default_factory=list)


def _cues(words: list[str], starts: list[float], ends: list[float]) -> list[Cue]:
    """Group narration words into cues, retaining per-word timings."""

    groups: list[Cue] = []
    chunk: list[str] = []
    chunk_words: list[tuple[str, float, float]] = []
    chunk_start: float | None = None
    chunk_end = 0.0

    def flush(last: float) -> None:
        if chunk and chunk_start is not None:
            groups.append(
                Cue(
                    chunk_start,
                    max(chunk_end, last),
                    " ".join(chunk),
                    words=list(chunk_words),
                )
            )

    for index, word in enumerate(words):
        trimmed = word.strip(",")
        # whisper sometimes splits decimals into two tokens ("1", ".4");
        # keep the rendered cue looking like the number narration spoke.
        if chunk and trimmed.startswith((".", ",")) and chunk[-1][-1].isdigit():
            chunk[-1] += trimmed
            if chunk_words:
                w, ws, _we = chunk_words[-1]
                chunk_words[-1] = (w + trimmed, ws, ends[index])
            chunk_end = ends[index]
            continue
        chunk.append(trimmed)
        chunk_words.append((trimmed, starts[index], ends[index]))
        if chunk_start is None:
            chunk_start = starts[index]
        chunk_end = ends[index]
        punctuation = word.rstrip().endswith(("!", ".", "?", ",", ";", ":"))
        if punctuation or len(chunk) >= CUE_MAX_WORDS:
            flush(chunk_end)
            chunk.clear()
            chunk_words.clear()
            chunk_start = None
    flush(chunk_end)
    return groups


def _drop_repeat_cues(cues: list[Cue]) -> list[Cue]:
    """Remove hallucination echoes: the same phrase shown twice in a rush.

    Echo re-transcriptions replay the line's recently heard cues within a
    couple of seconds of the originals. Narration that deliberately repeats
    a phrase (rhetoric) never does so that fast, so any identical-text cue
    reappearing this close to an earlier one is an echo.
    """

    kept: list[Cue] = []
    for cue in cues:
        echoed = any(
            other.text == cue.text
            and -0.05 < cue.start - other.end < 2.0
            for other in kept[-5:]
        )
        # same phrase shown TWICE at the same instant (whisper echo at
        # the line boundary): overlapping spans never belong to narration
        simultaneous = any(
            other.text == cue.text
            and cue.start <= other.end + 0.02
            and other.start <= cue.end + 0.02
            and abs(cue.start - other.start) < 2.0
            for other in kept[-5:]
        )
        if not echoed and not simultaneous:
            kept.append(cue)
    return kept


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
    accepted_end = 0.0
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
        riff_header = (
            len(tokens) >= 2
            and any(
                seen[i] == tokens[0] and seen[i + 1] == tokens[1]
                for i in range(len(seen) - 1)
            )
        )
        # Echoes re-transcribe audio we already accepted, so their span
        # reaches back inside previously covered time — the decisive
        # signature (word repeats alone can be legitimate).
        seg_start = min(w.start for w in words_in_segment)
        retranscribed = seg_start < accepted_end - 0.12
        if retranscribed and matched / len(tokens) >= 0.5:
            logger.debug("Dropping echo segment on %s", wav.name)
            continue
        if tokens and matched / len(tokens) >= 0.8:
            logger.debug("Dropping echo segment on %s", wav.name)
            continue
        # riff: the hallucination CONTINUES a phrase the line already said
        # ("that still" -> "that still felouse wholeheartedly"), always
        # anchored after the real speech. A leading bigram the line spoke
        # verbatim is not narration
        if riff_header:
            logger.debug("Dropping riff segment on %s", wav.name)
            continue
        # hallucination anchored in the line's trailing silence: whisper
        # rambles past the speech end; anything that starts after the
        # accepted audio with repeated leading tokens is not narration
        # (real words never come after the line has fully ended)
        if accepted_end and seg_start > accepted_end + 0.55:
            logger.debug("Dropping tail segment on %s", wav.name)
            continue
        seen.extend(tokens)
        # timestamps past the wav's own samples are padding artefacts
        if max(w.end for w in words_in_segment) > audio_end + 0.1:
            continue
        accepted_end = max(accepted_end, max(w.end for w in words_in_segment))
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


_WHISPER_CACHE: dict[tuple[str, str], object] = {}
_WHISPER_LOCK = threading.Lock()


def _whisper_model(model_size: str, device: str):
    """One model per (size, device) per process; renders repeat cheaply."""

    from faster_whisper import WhisperModel

    key = (model_size, device)
    with _WHISPER_LOCK:
        if key not in _WHISPER_CACHE:
            _WHISPER_CACHE[key] = WhisperModel(
                model_size, device=device, compute_type="int8"
            )
    return _WHISPER_CACHE[key]


def _line_vocabulary(line: str) -> set[str]:
    """Word bag of one narration line, punctuation stripped."""

    return {
        w.strip(".,!?;:\"'").lower() for w in line.split() if w.strip(".,!?;:\"'")
    }


def _cue_belongs_to_line(text: str, vocab: set[str]) -> bool:
    """True when every cue word is (near-)vocabulary of its own line.

    Whisper hallucinations in the trailing silence mint cues the
    narration never spoke ("Two stars" after "...among the stars").
    The script text is known, so a cue carrying a word outside its
    line's vocabulary — even fuzzily — is not narration.
    """

    words = [w for w in text.rstrip(".,!?;:").lower().split()]
    for word in words:
        if word in vocab:
            continue
        near = any(
            difflib.SequenceMatcher(None, word, known).ratio() >= 0.8
            for known in vocab
        )
        if not near:
            return False
    return True


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

    model = _whisper_model(model_size, device)
    word_lines: list[list[tuple[str, float, float]]] = []
    cues: list[Cue] = []
    for index, (line, wav, offset) in enumerate(
        zip(lines, wavs, offsets, strict=True)
    ):
        words, starts, ends = _transcribe_words(line, wav, offset, model)
        word_lines.append(list(zip(words, starts, ends, strict=False)))
        line_cues = _drop_repeat_cues(_cues(words, starts, ends))
        line_cues = [
            cue for cue in line_cues if _cue_belongs_to_line(cue.text, _line_vocabulary(line))
        ]
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


def _cue_finish(cues: list[Cue], index: int, stop: float) -> float:
    finish = stop
    if (
        index + 1 < len(cues)
        and cues[index + 1].start > stop
        and cues[index + 1].start - stop < 0.4
    ):
        finish = min(cues[index + 1].start - 0.08, stop)
    return finish


def _norm_tok(s: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", s.lower())


def _pop_word_times(cue: Cue, pop: str) -> tuple[float, float] | None:
    words = getattr(cue, "words", None) or []
    if not words:
        return None
    toks = [_norm_tok(t) for t in str(pop).split() if _norm_tok(t)]
    if not toks:
        return None
    wnorm = [_norm_tok(w) for w, _s, _e in words]
    for i in range(len(wnorm)):
        if wnorm[i : i + len(toks)] == toks:
            return words[i][1], words[i + len(toks) - 1][2]
        if len(toks) == 1 and toks[0] and toks[0] in (wnorm[i] or ""):
            return words[i][1], words[i][2]
    return None


_TAKEOVER_TERMS = {
    "demand", "prove", "proof", "stop", "never", "lie", "lies", "trap",
    "trick", "control", "pressure", "manipulate", "fake", "test", "verify",
}


def _takeover_for_cue(cue: Cue) -> str | None:
    pops = [p for p in getattr(cue, "pops", []) or [] if str(p).strip()]
    if not pops:
        return None
    # prefer the strongest, shortest command/reveal phrase
    ranked = sorted(pops, key=lambda p: (len(str(p).split()), -sum(_norm_tok(t) in _TAKEOVER_TERMS for t in str(p).split())))
    for pop in ranked:
        words = str(pop).split()
        if len(words) > 3:
            continue
        normed = [_norm_tok(t) for t in words]
        if any(t in _TAKEOVER_TERMS for t in normed) or str(pop).strip().isupper():
            return str(pop).strip()
    return None


def _in_takeover(t: float, ranges: list[tuple[float, float]]) -> bool:
    return any(s - 0.02 <= t <= e + 0.02 for s, e in ranges)


def _progressive_bottom(
    body: list[str], cue: Cue, start: float, finish: float, takeovers: list[tuple[float, float]]
) -> None:
    words = [
        (w, max(start, ws), min(finish, we))
        for (w, ws, we) in (getattr(cue, "words", None) or [])
        if we > start and ws < finish
    ]
    if not words:
        if not _in_takeover((start + finish) / 2, takeovers):
            text = _TIMESTAMP.sub("", cue.text.replace("\n", " ")).strip()
            body.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(max(finish, start))},"
                f"Cue,,0,0,0,,{text}"
            )
        return
    plain = [w for w, _s, _e in words]
    pops_norm = {_norm_tok(p) for p in (getattr(cue, "pops", []) or []) for p in str(p).split()}
    for i, (_w, ws, we) in enumerate(words):
        if we - ws < 0.04 or _in_takeover((ws + we) / 2, takeovers):
            continue
        parts = []
        for j, tok in enumerate(plain):
            disp = _TIMESTAMP.sub("", tok).strip()
            is_pop = _norm_tok(tok) in pops_norm
            if j == i:
                if is_pop:
                    # in-line kinetic punch: lands crisp, bright, a touch larger
                    parts.append(r"{\c&H00FFFFFF&\3c&H00000000&\bord2.2\fs68\fscx94\fscy94\t(0,80,\fscx106\fscy106)\t(80,180,\fscx100\fscy100\fs66)}" + disp + r"{\r}")
                else:
                    parts.append(r"{\c&H00FFFFFF&\3c&H50000000&\bord1.8}" + disp + r"{\r}")
            elif is_pop:
                parts.append(r"{\c&H00F0F0F0&\3c&H70000000&\bord1.5}" + disp + r"{\r}")
            else:
                parts.append(r"{\c&H009E9E9E&\3c&H90000000&\bord1}" + disp + r"{\r}")
        line = " ".join(parts)
        body.append(
            f"Dialogue: 0,{_ass_time(max(start, ws))},{_ass_time(min(finish, max(we, ws + 0.06)))},"
            f"Cue,,0,0,0,,{line}"
        )


def write_ass(cues: list[Cue], path: Path, *, end: float | None = None) -> Path:
    """Serialise cues to ASS: progressive bottom + rare anchored takeovers."""

    # plan rare, density-capped takeovers (caption becomes the subtitle)
    takeovers: list[tuple[float, float, str]] = []
    last_t = -99.0
    for cue in cues:
        hit = _takeover_for_cue(cue)
        if not hit:
            continue
        wt = _pop_word_times(cue, hit)
        start, stop = cue.start, max(cue.end, cue.start + 0.25)
        finish = _cue_finish(cues, cues.index(cue), stop)
        if wt:
            ps, pe = wt[0], max(wt[1], wt[0] + 0.22)
        else:
            ps, pe = start, min(finish, start + 0.55)
        ps, pe = max(start, ps), min(finish, max(pe, ps + 0.28))
        if ps - last_t < 9.0:
            continue
        takeovers.append((ps, pe, hit.upper()))
        last_t = ps

    takeover_ranges = [(s, e) for s, e, _h in takeovers]
    body: list[str] = []
    for index, cue in enumerate(cues):
        start, stop = cue.start, max(cue.end, cue.start + 0.25)
        finish = _cue_finish(cues, index, stop)
        _progressive_bottom(body, cue, start, finish, takeover_ranges)

    for ps, pe, disp in takeovers:
        # anchored low-center: reads as the subtitle taking focus, not a sticker
        fx = r"{\an2\pos(540,1465)\fscx96\fscy96\fad(40,90)\t(0,90,\fscx103\fscy103)\t(90,200,\fscx100\fscy100)}"
        body.append(
            f"Dialogue: 2,{_ass_time(ps)},{_ass_time(max(pe, ps + 0.2))},"
            f"Hit,,0,0,0,,{fx}{disp}"
        )
    path.write_text(_ASS_HEADER + "\n".join(body) + "\n", encoding="utf-8")
    return path
