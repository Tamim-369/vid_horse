"""Voiceover generation for scripts (Chatterbox Nano + voice clone)."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio as ta

from core.models import VideoScript
from utils import computing
from utils.tts_text import speakable

logger = logging.getLogger(__name__)

VOICE_PROMPT = Path("assets/voice/dark_gaming_voice_prompt.wav")
SEGMENT_GAP_SECONDS = 0.35


class VoiceOver:
    """Chatterbox-Nano narration with the project's cloned voice.

    Lines synthesize on a worker pool sized by the machine profile: each
    thread keeps its own model instance so no shared weights race, its
    torch threads stay pinned to the profile's slice, and the voice
    recipe stays identical — same prompt file, same settings, per line.
    """

    def __init__(self, prompt_path: Path = VOICE_PROMPT, device: str = "cpu") -> None:
        self._device = device
        self._prompt_path = prompt_path
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

            model = ChatterboxTurboTTS.from_pretrained(
                device=self._device, nano=True
            )
            self._local.model = model
        return model

    def render(self, text: str, out_path: Path) -> float:
        """Generate speech for one line; return its duration in seconds."""

        wav = self._model_for_thread().generate(
            speakable(text),
            audio_prompt_path=str(self._prompt_path),
            norm_loudness=False,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(out_path), wav, self._model_for_thread().sr)
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
            if line_path.exists():
                wav, sr = ta.load(str(line_path))
                return index, wav, sr
            duration = self.render(segments[index], line_path)
            wav, sr = ta.load(str(line_path))
            logger.info("Line %d rendered (%.1fs)", index, duration)
            return index, wav, sr

        with ThreadPoolExecutor(max_workers=computing.voice_workers()) as pool:
            results = sorted(pool.map(synthesize, range(len(segments))))

        sr = results[0][2]
        wavs: list[torch.Tensor] = []
        for _index, wav, _sr in results:
            wavs.append(wav)
            wavs.append(torch.zeros(1, int(sr * SEGMENT_GAP_SECONDS)))

        joined = torch.cat(wavs, dim=1)
        joined_path = out_dir / f"{stem}.wav"
        ta.save(str(joined_path), joined, sr)
        return joined_path
