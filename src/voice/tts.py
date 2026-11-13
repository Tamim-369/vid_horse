"""Voiceover generation for scripts (Chatterbox Nano + voice clone)."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio as ta

from config.settings import get_settings
from core.models import VideoScript
from utils import computing
from utils.tts_text import speakable

logger = logging.getLogger(__name__)

# a clean 8.5s slice of the clone source: the 45s original carries
# dramatic pauses the zero-shot model copies into every line (the
# staccato "stops after five words" artefact)
VOICE_PROMPT = Path("assets/voice/voice_prompt_short.wav")

# when the key of a rendered asset changes shape, older cached wavs must
# be re-rendered, not silently shipped
_KEY_VERSION = "s7"  # slower, more deliberate pacing and longer pauses

# sentence boundary: punctuation that ends a spoken sentence, then space.
# speakable() has already folded colons/semicolons into commas, so these
# are true sentence ends
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# a piece that ends with a single capital letter + period is an
# abbreviation fragment ("U." of "U.S."), not a sentence end
_ABBREVIATION_TAIL = re.compile(r"\b[A-Z][.]$")
# words that OPEN a sentence with the kind of turn a storyteller lets
# breathe for a beat before saying it
_TURNING_WORDS = frozenset(
    {"and", "but", "then", "yet", "so", "until", "unless", "because"}
)


def _sentences(text: str) -> list[str]:
    """Split spoken text into sentence chunks, merging abbreviation tails."""

    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and _ABBREVIATION_TAIL.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _text_key(text: str, voice_tag: str = "", speed: float = 1.0) -> str:
    """Content key for a narration line: the exact text the model sees.

    ``voice_tag`` distinguishes alternative reference voices so cached
    wavs from a different voice are never re-shipped for this one.
    """

    return hashlib.sha256(
        f"{_KEY_VERSION}|{voice_tag}|{speed}\n{speakable(text)}".encode()
    ).hexdigest()[:16]




def _atempo(path: Path, speed: float) -> None:
    """Time-stretch in place, preserving pitch. No-op when speed is 1.0."""
    if abs(speed - 1.0) < 0.001 or not path.exists():
        return
    tmp = path.with_suffix(".tempo.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-filter:a", f"atempo={speed}",
         "-ar", "24000", str(tmp)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    tmp.replace(path)

class VoiceOver:
    """Chatterbox-Nano narration with the project's cloned voice.

    Lines synthesize on a worker pool sized by the machine profile: each
    thread keeps its own model instance so no shared weights race, its
    torch threads stay pinned to the profile's slice, and the voice
    recipe stays identical — same prompt file, same settings, per line.

    Each line is synthesized PER SENTENCE, then the sentence wavs are
    joined with a controlled silence gap. One blob for a two-sentence
    scene made the model decide the sentence boundary for itself: two
    lines slammed together with no pause while a comma mid-sentence got
    a dramatic beat. Fixed units + fixed gaps give every sentence the
    same audible shape — and the gap is tuned PER SENTENCE so story
    beats land before turning words, not at metronome intervals.
    """

    def __init__(
        self,
        prompt_path: Path | None = None,
        device: str = "cpu",
        *,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> None:
        settings = get_settings()
        self._device = device
        self._prompt_path = prompt_path or settings.voice_prompt
        # only non-default reference voices need their own cache namespace
        self._key_tag = (
            ""
            if self._prompt_path.resolve() == settings.voice_prompt.resolve()
            else self._prompt_path.stem
        )
        self._exaggeration = (
            settings.voice_exaggeration if exaggeration is None else exaggeration
        )
        self._cfg_weight = (
            settings.voice_cfg_weight if cfg_weight is None else cfg_weight
        )
        self._sentence_gap = settings.voice_sentence_gap
        self._line_gap = settings.voice_line_gap
        self._speed = settings.voice_speed
        self._local = threading.local()
        # Don't reset torch threads if parallel work has already started
        try:
            torch.set_num_threads(computing.voice_torch_threads())
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Threading already initialized by CLIP or other components
            pass
        self._model = self._model_for_thread()

    def _model_for_thread(self):
        model = getattr(self._local, "model", None)
        if model is None:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            model = ChatterboxTurboTTS.from_pretrained(device=self._device)
            self._local.model = model
        return model

    def _speak(self, text: str):
        """One raw Chatterbox call with the identical voice recipe."""

        return self._model_for_thread().generate(
            speakable(text),
            audio_prompt_path=str(self._prompt_path),
            exaggeration=self._exaggeration,
            cfg_weight=self._cfg_weight,
            norm_loudness=False,
        )

    def _beat_gap(self, previous: str, current: str) -> float:
        """The pause between two sentences of one line, tuned like a storyteller.

        A monotone gap between every sentence reads as a paragraph recital.
        A sentence that OPENS on a turn ("And", "But", "Then", "So...") is a
        story beat — the listener gets half a breath of anticipation first.
        An exclamation closes heavier, so it holds a beat too.
        """

        gap = self._sentence_gap
        if current.split() and current.split()[0].lower().strip(",") in _TURNING_WORDS:
            gap += 0.08
        if previous.rstrip().endswith("!"):
            gap += 0.06
        return round(gap, 3)

    def _trimmed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Cut leading/trailing silence so the inserted gaps are the only pauses.

        Chatterbox pads each utterance with its own variable-length
        silence; left untouched, that silence stacks with the fixed gap
        and the rhythm drifts. A loudness threshold rather than zero
        detection keeps breaths and decays intact.
        """

        amp = wav.abs().amax(dim=0)
        voiced = (amp > 0.004).nonzero(as_tuple=True)[0]
        if voiced.numel() == 0:
            return wav
        margin = int(0.10 * sr)
        return wav[..., max(0, voiced[0] - margin) : min(wav.shape[-1], voiced[-1] + margin)]

    def _sentence(
        self, index: int, position: int, text: str, sr: int, out_dir: Path, stem: str
    ) -> torch.Tensor:
        wav_path = out_dir / f"{stem}-{index:02d}.s{position:02d}.wav"
        sidecar = out_dir / f"{stem}-{index:02d}.s{position:02d}.key"
        key = _text_key(text, self._key_tag, self._speed)
        if wav_path.exists() and sidecar.exists() and sidecar.read_text(encoding="utf-8") == key:
            return ta.load(str(wav_path))[0]
        wav = self._trimmed(self._speak(text), sr)
        ta.save(str(wav_path), wav, sr)
        sidecar.write_text(key, encoding="utf-8")
        logger.info("Line %d.%d rendered (%.1fs)", index, position, wav.shape[-1] / sr)
        return wav

    def _render_line(
        self, index: int, text: str, out_dir: Path, stem: str
    ) -> tuple[torch.Tensor, int]:
        """Render one script line as sentence-sized units with fixed gaps."""

        spoken = speakable(text)
        sentences = _sentences(spoken) or [spoken]
        sr = self._model_for_thread().sr
        pieces: list[torch.Tensor] = []
        for position, sentence in enumerate(sentences):
            if position:
                gap = self._beat_gap(sentences[position - 1], sentence)
                pieces.append(torch.zeros(1, int(sr * gap)))
            pieces.append(self._sentence(index, position, sentence, sr, out_dir, stem))
        return torch.cat(pieces, dim=1), sr

    def render(self, text: str, out_path: Path) -> float:
        """Generate speech for one line; return its duration in seconds."""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        wav, _sr = self._render_line(-1, text, out_path.parent, out_path.stem)
        ta.save(str(out_path), wav, self._model_for_thread().sr)
        _atempo(out_path, self._speed)
        if abs(self._speed - 1.0) >= 0.001:
            wav, _sr = ta.load(str(out_path))
        return wav.shape[-1] / self._model_for_thread().sr

    def render_script(
        self, script: VideoScript, out_dir: Path, *, stem: str = "voice"
    ) -> Path:
        """Render hook, every scene and the outro; return the joined track."""

        segments: list[str] = [script.hook]
        segments.extend(scene.narration for scene in script.scenes)
        segments.append(script.outro)

        out_dir.mkdir(parents=True, exist_ok=True)

        def synthesize(index: int) -> tuple[int, torch.Tensor, int]:
            line_path = out_dir / f"{stem}-{index:02d}.wav"
            sidecar = out_dir / f"{stem}-{index:02d}.key"
            key = _text_key(segments[index], self._key_tag, self._speed)
            # content-keyed cache: reuse the line only when the wav exists
            # AND it was synthesized from exactly this text; a stale wav
            # from an edited script must re-render, not silently ship
            if (
                line_path.exists()
                and sidecar.exists()
                and sidecar.read_text(encoding="utf-8") == key
            ):
                wav, sr = ta.load(str(line_path))
                return index, wav, sr
            wav, sr = self._render_line(index, segments[index], out_dir, stem)
            ta.save(str(line_path), wav, sr)
            _atempo(line_path, self._speed)
            wav, sr = ta.load(str(line_path))
            # append inter-line pause (after speed change so duration is consistent)
            line_gap = 0.0 if index == len(segments) - 1 else self._line_gap
            if line_gap > 0:
                gap = torch.zeros(wav.shape[0], int(sr * line_gap))
                wav = torch.cat([wav, gap], dim=1)
                ta.save(str(line_path), wav, sr)
            sidecar.write_text(key, encoding="utf-8")
            logger.info("Line %d rendered (%.1fs)", index, wav.shape[-1] / sr)
            return index, wav, sr

        with ThreadPoolExecutor(max_workers=computing.voice_workers()) as pool:
            results = sorted(pool.map(synthesize, range(len(segments))))

        sr = results[0][2]
        wavs: list[torch.Tensor] = [wav for _index, wav, _sr in results]

        joined = torch.cat(wavs, dim=1)
        joined_path = out_dir / f"{stem}.wav"
        ta.save(str(joined_path), joined, sr)
        return joined_path
