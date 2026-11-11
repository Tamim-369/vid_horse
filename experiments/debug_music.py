#!/usr/bin/env python3
"""Debug music director LLM response."""

from __future__ import annotations

import json
import sys
sys.path.insert(0, 'src')

from providers.llm import create_llm_provider
from config import get_settings

# Import directly from the module files
from blocks.music.director import MOOD_SYSTEM
from blocks.music.schema import MusicAnnotation

settings = get_settings()
llm = create_llm_provider(settings)

# Create script with proper ScriptScene objects
from core.models import VideoScript, ScriptScene, ScriptFormat

script = VideoScript(
    title='Gaslighting Tactics',
    hook='They make you doubt your own memory.',
    format='short',
    target_seconds=45,
    scenes=[
        __import__('core.models').models.ScriptScene(narration='The abuser denies things they said, making you question your memory.', visual='gaslighting', seconds=10),
        __import__('core.models').models.ScriptScene(narration='They isolate you from friends, saying everyone is against you.', visual='isolation', seconds=10),
        __import__('core.models').models.ScriptScene(narration='But you can spot the pattern: deny, attack, reverse victim and offender.', visual='reveal', seconds=10),
    ],
    outro='Trust your perception. You are not crazy.'
)

# Build payload
payload = {
    "hook": script.hook,
    "scenes": [
        {"index": i, "narration": s.narration, "visual": s.visual}
        for i, s in enumerate(script.scenes)
    ],
    "outro": script.outro,
}

print("Testing LLM response...")
try:
    from config import get_settings
    from providers.llm import create_llm_provider
    from blocks.music.schema import MusicAnnotation
    from blocks.music.director import MOOD_SYSTEM

    settings = get_settings()
    llm = create_llm_provider(settings)

    answer = llm.complete_json(
        [
            {"role": "system", "content": MOOD_SYSTEM},
            {"role": "user", "content": json.dumps({
                "hook": script.hook,
                "scenes": [{"index": i, "narration": s.narration, "visual": s.visual} for i, s in enumerate(script.scenes)],
                "outro": script.outro
            }, ensure_ascii=False)},
        ],
        MusicAnnotation,
        temperature=0.25,
    )
    print("Raw answer:", answer)
    print("Answer type:", type(answer))
    if hasattr(answer, '__dict__'):
        print("Answer dict:", answer.__dict__)
    else:
        print("Answer attributes:", dir(answer))
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()