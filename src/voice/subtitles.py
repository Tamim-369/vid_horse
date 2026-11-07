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
# Normal progressive subtitles — bottom, white, no bounce, no gray
_STYLE_LINE = (
    "Style: Cue,DejaVu Sans Condensed,60,&H00F2F2F2,&H00F2F2F2,&H00000000,"
    "&H00000000,0,0,0,0,100,100,0,0,1,2.8,0,2,70,70,350,1"
)
# Keyword takeover — REPLACES subtitle at SAME position, bigger/bolder,
# centered, same margin. Gentle fade only.
_TAKEOVER_LINE = (
    "Style: Takeover,DejaVu Sans Condensed,85,&H00FFFFFF,&H00FFFFFF,&H00000000,"
    "&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,2,70,70,350,1"
)
# fmt: on

_ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{_STYLE_LINE}
{_TAKEOVER_LINE}

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
    "trick", "tricks", "control", "pressure", "manipulate", "manipulation",
    "fake", "test", "verify", "guilt", "shame", "fear", "doubt", "obey",
    "rush", "force", "blame", "bait", "threat", "power",
}
HIGHLIGHT_LEAD = 0.09  # show highlight a bit before the word is spoken

# Filler framing phrases (normalized same as _norm_tok) that should
# never plate over the actual concept
_PLATE_FILLER = {
    _norm_tok(p) for p in [
        "what we call", "what is", "what are", "we call", "is called",
        "they call", "this is", "that is", "known as", "referred to",
    ]
}


def _plate_pick(cue: Cue) -> tuple[str, bool] | None:
    """The keyword a plate should show for this cue, and whether it is strong.

    Strong terms (command/reveal vocabulary) may plate every
    ``PLATE_MIN_GAP`` seconds; ordinary emphasized words only every
    ``PLATEOrdinary_GAP`` so the big word stays an event, not a strobe.

    If ALL pops for this cue are filler framing phrases, skip the cue
    entirely — the next cue likely has the actual concept.
    """

    pops = [p for p in getattr(cue, "pops", []) or [] if str(p).strip()]
    if not pops:
        return None
    # If every pop is a filler phrase, this cue carries no concept — skip it
    if all(any(f in _norm_tok(p) for f in _PLATE_FILLER) for p in pops):
        return None
    ranked = sorted(
        pops,
        key=lambda p: (
            -sum(_norm_tok(t) in _TAKEOVER_TERMS for t in str(p).split()),
            # penalize filler framing phrases even when embedded in longer pops
            # e.g. "what we call crowd mind" should not beat "crowd mind"
            1 if any(f in _norm_tok(p) for f in _PLATE_FILLER) else 0,
            # then prefer concise concept words (2-3 words > 1, but filler is last)
            -len(str(p).split()),
        ),
    )
    for pop in ranked:
        words = str(pop).split()
        if not 0 < len(words) <= 3:
            continue
        strong = any(_norm_tok(t) in _TAKEOVER_TERMS for t in words)
        return " ".join(words).strip().upper(), strong
    return None


PLATE_MIN_GAP = 2.0  # min gap between ANY plate starts — ~8-10 plates per 40s
PLATE_ORDINARY_GAP = 2.0  # unified
PLATE_MAX_HOLD = 1.75  # hold window decreased ~30% from 2.5s
PLATE_MIN_HOLD = 0.9
PLATE_MAX_PER_VIDEO = 10  # cap


# Concept terms that boost a plate's priority (the lesson's core ideas)
_PLATE_CONCEPT_BOOST = {
    "crowd mind", "group think", "social proof", "herd", "conform",
    "manipulation", "influence", "persuasion", "compliance", "obedience",
    "authority", "reciprocity", "scarcity", "consistency", "liking",
    "foot in door", "door in face", "lowball", "bait", "anchoring",
}

def _plate_score(disp: str) -> int:
    """Higher = more important concept."""
    normed = _norm_tok(disp)
    words = disp.split()
    score = len(words)
    for term in _PLATE_CONCEPT_BOOST:
        if term.replace(" ", "") in normed:
            score += 5
    for w in words:
        if _norm_tok(w) in _TAKEOVER_TERMS:
            score += 3
    return score


PLATE_MIN_GAP = 2.0  # min gap between ANY plate starts
PLATE_ORDINARY_GAP = 2.0  # unified
PLATE_HOLD = 1.20  # exact hold: 1s 20ms (user spec)
PLATE_MAX_PER_VIDEO = 10  # cap


def _plate_pick(cue: Cue) -> tuple[str, bool] | None:
    """The keyword a takeover should show for this cue, and whether it is strong.

    If ALL pops for this cue are filler framing phrases, skip the cue
    entirely — the next cue likely has the actual concept.
    """

    pops = [p for p in getattr(cue, "pops", []) or [] if str(p).strip()]
    if not pops:
        return None
    # If every pop is a filler phrase, this cue carries no concept — skip it
    if all(any(f in _norm_tok(p) for f in _PLATE_FILLER) for p in pops):
        return None
    ranked = sorted(
        pops,
        key=lambda p: (
            -sum(_norm_tok(t) in _TAKEOVER_TERMS for t in str(p).split()),
            1 if any(f in _norm_tok(p) for f in _PLATE_FILLER) else 0,
            -len(str(p).split()),
        ),
    )
    for pop in ranked:
        words = str(pop).split()
        if not 0 < len(words) <= 3:
            continue
        strong = any(_norm_tok(t) in _TAKEOVER_TERMS for t in words)
        return " ".join(words).strip().upper(), strong
    return None


def build_plates(cues: list[Cue]) -> list[tuple[float, float, str]]:
    """Keyword takeovers: (start, end, DISPLAY).

    A takeover appears when its word starts being spoken (HIGHLIGHT_LEAD
    early) and holds for EXACTLY PLATE_HOLD (1.20s). It REPLACES the
    subtitle at the same screen position for that window. No overlap
    between takeovers — higher-scoring concepts win.
    """

    candidates: list[tuple[float, str, bool, int]] = []
    for cue in cues:
        picked = _plate_pick(cue)
        if not picked:
            continue
        disp, strong = picked
        wt = _pop_word_times(cue, disp)
        if not wt:
            continue
        ps = max(0.0, wt[0] - HIGHLIGHT_LEAD)
        score = _plate_score(disp)
        candidates.append((ps, disp, strong, score))
    # sort by score DESC, then time ASC
    candidates.sort(key=lambda item: (-item[3], item[0]))

    chosen: list[list] = []  # [start, display, score]
    for ps, disp, strong, score in candidates:
        if len(chosen) >= PLATE_MAX_PER_VIDEO:
            break
        # check interval overlap: [ps, ps+PLATE_HOLD] vs [cp, cp+PLATE_HOLD]
        overlap = False
        for cp, cd, cs in chosen:
            if ps < cp + PLATE_HOLD and cp < ps + PLATE_HOLD:
                overlap = True
                break
        if overlap:
            continue
        chosen.append([ps, disp, score])

    # sort chosen by time for output
    chosen.sort(key=lambda item: item[0])

    plates: list[tuple[float, float, str]] = []
    for ps, disp, _score in chosen:
        pe = ps + PLATE_HOLD
        plates.append((ps, pe, disp))
    return plates


def _progressive_bottom(
    body: list[str],
    cue: Cue,
    start: float,
    finish: float,
    takeover_ranges: list[tuple[float, float]] | None = None,
) -> None:
    raw = [
        (w, max(start, ws), min(finish, we))
        for (w, ws, we) in (getattr(cue, "words", None) or [])
        if we > start and ws < finish
    ]
    if not raw:
        # check if this whole cue falls in a takeover window
        if takeover_ranges:
            mid = (start + finish) / 2
            if any(ps <= mid <= pe for ps, pe in takeover_ranges):
                return  # skip — takeover covers this
        text = _TIMESTAMP.sub("", cue.text.replace("\n", " ")).strip()
        body.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(max(finish, start))},"
            f"Cue,,0,0,0,,{text}"
        )
        return
    # lead highlight slightly before speech for snappier reading
    lead = HIGHLIGHT_LEAD
    bounds = [max(start, raw[i][1] - lead) for i in range(len(raw))]
    bounds.append(min(finish, raw[-1][2]))
    for k in range(1, len(bounds)):
        if bounds[k] <= bounds[k - 1]:
            bounds[k] = bounds[k - 1] + 0.02
    plain = [w for w, _s, _e in raw]
    pops_norm = {_norm_tok(p) for p in (getattr(cue, "pops", []) or []) for p in str(p).split()}
    for i in range(len(raw)):
        ws, we = bounds[i], bounds[i + 1]
        if we - ws < 0.04:
            continue
        # skip this word event if it falls in a takeover window
        if takeover_ranges:
            mid = (ws + we) / 2
            if any(ps <= mid <= pe for ps, pe in takeover_ranges):
                continue
        parts = []
        for j, tok in enumerate(plain):
            disp = _TIMESTAMP.sub("", tok).strip()
            is_pop = _norm_tok(tok) in pops_norm
            if j == i:
                parts.append(r"{\c&H00FFFFFF&\bord3\3c&H00000000&}" + disp + r"{\r}")
            elif is_pop:
                parts.append(r"{\c&H00FFFFFF&\bord2.2\3c&H00000000&}" + disp + r"{\r}")
            else:
                parts.append(r"{\c&H00F2F2F2&\bord2\3c&H00000000&}" + disp + r"{\r}")
        line = " ".join(parts)
        body.append(
            f"Dialogue: 0,{_ass_time(max(start, ws))},{_ass_time(min(finish, max(we, ws + 0.06)))},"
            f"Cue,,0,0,0,,{line}"
        )


def write_ass(
    cues: list[Cue],
    path: Path,
    *,
    end: float | None = None,
    mode: str = "all",
    plates: list[tuple[float, float, str]] | None = None,
) -> Path:
    """Serialise cues to ASS.

    mode: 'all' = word-synced subtitles WITH TAKEOVER GAPS + keyword takeovers
          (takeover REPLACES subtitle at same position for 1.20s),
          'highlights' = takeovers only, 'none' = nothing.
    """

    if plates is None:
        plates = build_plates(cues)

    # takeover time ranges for skipping subtitle generation
    takeover_ranges = [(ps, pe) for ps, pe, _d in plates]

    body: list[str] = []
    if mode == "all":
        for index, cue in enumerate(cues):
            start, stop = cue.start, max(cue.end, cue.start + 0.25)
            finish = _cue_finish(cues, index, stop)
            _progressive_bottom(body, cue, start, finish, takeover_ranges)

    if mode != "none":
        for ps, pe, disp in plates:
            # REPLACES subtitle at SAME position (MarginV 350), bigger/bolder
            fx = r"{\an2\pos(540,1180)\fad(150,250)}"
            body.append(
                f"Dialogue: 2,{_ass_time(ps)},{_ass_time(max(pe, ps + 0.2))},"
                f"Takeover,,0,0,0,,{fx}{disp}"
            )
    path.write_text(_ASS_HEADER + "\n".join(body) + "\n", encoding="utf-8")
    return path
