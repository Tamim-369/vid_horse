"""Voiceover generation for scripts (Chatterbox Nano + voice clone)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchaudio as ta

from core.models import VideoScript
from utils.tts_text import speakable

logger = logging.getLogger(__name__)

VOICE_PROMPT = Path("assets/voice/dark_gaming_voice_prompt.wav")
SEGMENT_GAP_SECONDS = 0.35


class VoiceOver:
    """Chatterbox-Nano narration with the project's cloned voice."""

    def __init__(self, prompt_path: Path = VOICE_PROMPT, device: str = "cpu") -> None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self._model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        self._prompt_path = prompt_path

    def render(self, text: str, out_path: Path) -> float:
        """Generate speech for one line; return its duration in seconds."""

        wav = self._model.generate(
            speakable(text),
            audio_prompt_path=str(self._prompt_path),
            norm_loudness=False,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(out_path), wav, self._model.sr)
        return wav.shape[-1] / self._model.sr

    def render_script(
        self, script: VideoScript, out_dir: Path, *, stem: str = "voice"
    ) -> Path:
        """Render hook, every scene and the outro; return the joined track."""

        segments: list[str] = [script.hook]
        segments.extend(scene.narration for scene in script.scenes)
        segments.append(script.outro)

        out_dir.mkdir(parents=True, exist_ok=True)
        wavs: list[torch.Tensor] = []
        sr = None
        for index, text in enumerate(segments):
            line_path = out_dir / f"{stem}-{index:02d}.wav"
            if line_path.exists():
                wav, sr = ta.load(str(line_path))
            else:
                duration = self.render(text, line_path)
                wav, sr = ta.load(str(line_path))
                logger.info("Line %d rendered (%.1fs)", index, duration)
            wavs.append(wav)
            gap = torch.zeros(1, int(sr * SEGMENT_GAP_SECONDS))
            wavs.append(gap)

        joined = torch.cat(wavs, dim=1)
        joined_path = out_dir / f"{stem}.wav"
        ta.save(str(joined_path), joined, sr)
        return joined_path
