#!/usr/bin/env python3
"""Generate the same script with several Groq models and compare the results.

The research brief is produced once and reused for every model, so the only
thing that changes between runs is the scripting model. Outputs land in
``output/<slug>/compare/<model>/`` plus a ``<format>-comparison.md`` summary.

Usage:
    uv run python experiments/compare_models.py baron-52
    uv run python experiments/compare_models.py baron-52 --format long
"""

from __future__ import annotations

import argparse
import logging

from blocks.script_generation import ScriptGenerationBlock, render_markdown
from blocks.topic_research import TopicResearchBlock
from catalog.store import TheoryCatalog
from config.settings import Settings, get_settings
from core.errors import VideoAIError
from core.models import Theory, VideoScript
from providers.llm.groq import GroqProvider
from providers.sources import MediaWikiArticleFetcher, build_http_client
from utils.cache import DiskCache
from utils.text import slugify

DEFAULT_MODELS = (
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()

    if not settings.groq_api_keys:
        raise SystemExit("No Groq API keys configured.")

    theory = _load_theory(settings, args.slug)
    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)

    def provider(model: str) -> GroqProvider:
        return GroqProvider(
            model=model,
            api_keys=settings.groq_api_keys,
            base_url=settings.groq_base_url,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            timeout=settings.groq_timeout,
            max_retries=settings.max_retries,
            cache=cache,
        )

    fetcher = MediaWikiArticleFetcher(
        client=build_http_client(settings),
        language=settings.wikipedia_language,
        delay=settings.request_delay,
        max_chars=settings.research_max_chars,
    )

    print(f"Researching '{theory.name}' once with {args.models[0]}...")
    brief = TopicResearchBlock(
        provider(args.models[0]),
        fetcher,
        max_chars=settings.research_max_chars,
    ).run(theory)

    root = settings.output_dir / theory.slug / "compare"
    root.mkdir(parents=True, exist_ok=True)
    (root / "brief.json").write_text(brief.model_dump_json(indent=2))

    seconds = (
        settings.script_short_seconds
        if args.format == "short"
        else settings.script_long_seconds
    )
    results: list[tuple[str, VideoScript]] = []

    for model in args.models:
        print(f"Writing {args.format} script with {model}...")
        block = ScriptGenerationBlock(
            provider(model),
            max_attempts=settings.max_retries + 1,
            chunk_size=settings.script_long_chunk_size,
            scene_count=settings.script_long_scenes,
            hook_candidates=settings.script_hook_candidates,
        )
        try:
            script = block.run(brief, format=args.format, target_seconds=seconds)
        except VideoAIError as exc:
            print(f"  failed: {exc}")
            continue

        folder = root / slugify(model)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{args.format}.json").write_text(script.model_dump_json(indent=2))
        (folder / f"{args.format}.md").write_text(render_markdown(script))

        results.append((model, script))
        words, total = _stats(script)
        print(f"  title: {script.title}")
        print(f"  hook:  {script.hook}")
        print(f"  {len(script.scenes)} scenes · {words} words · ~{total}s")

    if not results:
        raise SystemExit("Every model failed; nothing to compare.")

    summary = _write_summary(root, theory.name, args.format, results)
    print(f"\nSaved comparison to {summary}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Catalog slug, e.g. baron-52")
    parser.add_argument("--format", choices=("short", "long"), default="short")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def _load_theory(settings: Settings, slug: str) -> Theory:
    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slug)
    if theory is None:
        raise SystemExit(f"No theory '{slug}' in the catalog.")
    return theory


def _stats(script: VideoScript) -> tuple[int, int]:
    words = sum(len(scene.narration.split()) for scene in script.scenes)
    seconds = sum(scene.seconds for scene in script.scenes)
    return words, seconds


def _write_summary(root, name: str, fmt: str, results: list[tuple[str, VideoScript]]):
    lines = [f"# {name} — {fmt} model comparison", ""]
    for model, script in results:
        words, seconds = _stats(script)
        lines.extend(
            [
                f"## {model}",
                "",
                f"- **Title:** {script.title}",
                f"- **Hook:** {script.hook}",
                f"- **Scenes:** {len(script.scenes)}",
                f"- **Words:** {words}",
                f"- **Estimated:** ~{seconds}s",
                "",
            ]
        )
    path = root / f"{fmt}-comparison.md"
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    main()
