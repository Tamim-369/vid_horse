#!/usr/bin/env python3
"""Test music director with a gaslighting script."""

from __future__ import annotations

import json
from providers.llm import create_llm_provider
from blocks.music.director import annotate_music
from core.models import VideoScript, ScriptScene, ScriptFormat
from config import get_settings

settings = get_settings()
llm = create_llm_provider(settings)

script = VideoScript(
    title='Gaslighting Tactics',
    hook='They make you doubt your own memory.',
    format='short',
    target_seconds=45,
    scenes=[
        ScriptScene(narration='The abuser denies things they said, making you question your memory.', visual='gaslighting', seconds=10),
        ScriptScene(narration='They isolate you from friends, saying everyone is against you.', visual='isolation', seconds=10),
        ScriptScene(narration='But you can spot the pattern: deny, attack, reverse victim and offender.', visual='reveal', seconds=10),
    ],
    outro='Trust your perception. You are not crazy.'
)

result = annotate_music(llm, script)
print('Base role:', result.music.base_role if result.music else 'none')
print('Scene roles:', result.music.scene_roles if result.music else 'none')