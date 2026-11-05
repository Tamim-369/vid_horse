"""Generate a dark-psychology short script from a topic (no hardcoded brief).

Usage:
    uv run python experiments/generate_script.py "Love Bombing"
    uv run python experiments/generate_script.py   # uses default topic
Output:
    output/sample_script.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocks.script_generation.block import ScriptGenerationBlock
from blocks.topic_generation import TopicGenerationBlock
from config.settings import get_settings
from providers.llm import create_llm_provider

DEFAULT_TOPIC = "Love Bombing"


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    settings = get_settings()
    llm = create_llm_provider(settings)

    topic_block = TopicGenerationBlock(llm)
    brief = topic_block.run(topic)
    print("brief ready:", brief.title)

    script_block = ScriptGenerationBlock(
        llm, hook_candidates=3, critic=True, critic_threshold=80, max_revisions=1, scene_count=6
    )
    script = script_block.run(brief, format="short", target_seconds=45)

    out = Path("output/sample_script.txt")
    out.parent.mkdir(exist_ok=True)
    lines = ["TITLE: " + script.title, "HOOK: " + script.hook, ""]
    for i, sc in enumerate(script.scenes):
        lines.append(f"SCENE {i} ({sc.seconds}s): {sc.narration}")
        lines.append(f"  VISUAL: {sc.visual}")
    lines += ["", "OUTRO: " + script.outro]
    out.write_text("\n".join(lines))
    print("saved", out)


if __name__ == "__main__":
    main()
