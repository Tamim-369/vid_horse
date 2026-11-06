"""Generate 10 dark-psychology short scripts into one file.

Usage:
    uv run python experiments/generate_10_scripts.py
Output:
    output/ten_scripts.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocks.script_engine.block import ScriptEngineBlock
from config.settings import get_settings
from providers.llm import create_llm_provider

TOPICS = [
    "Love Bombing",
    "Gaslighting",
    "Negging",
    "Mirroring",
    "Anchoring",
    "Scarcity",
    "Foot in the Door",
    "Silent Treatment",
    "DARVO",
    "Trauma Bonding",
]


def main() -> None:
    settings = get_settings()
    llm = create_llm_provider(settings)
    engine = ScriptEngineBlock(llm, target_seconds=45)

    out = Path("output/ten_scripts.txt")
    out.parent.mkdir(exist_ok=True)
    out.write_text("")

    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i}/10] {topic}", flush=True)
        script = engine.run(topic, format="short")

        lines = [f"===== SCRIPT {i}: {script.title} =====", f"HOOK: {script.hook}", ""]
        for j, sc in enumerate(script.scenes):
            lines.append(f"SCENE {j} ({sc.seconds}s): {sc.narration}")
            lines.append(f"  VISUAL: {sc.visual}")
        lines += ["", "OUTRO: " + script.outro, "", ""]
        with out.open("a") as f:
            f.write("\n".join(lines))

    print("saved", out)


if __name__ == "__main__":
    main()
