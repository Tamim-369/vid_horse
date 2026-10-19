#!/usr/bin/env python3
"""Audit the single-shot hook the real pipeline produces, across theories.

Runs the same research + short-script path the CLI uses, once per theory, and
prints the hook plus any rule problems. Retries triggered by the hook quality
gate are logged, so you can see how often one shot was not enough.

Usage:
    uv run python experiments/hook_audit.py baron-52 odessa new-swabia
    uv run python experiments/hook_audit.py --format long baron-52
"""

from __future__ import annotations

import argparse
import logging

from blocks.script_generation import ScriptGenerationBlock, hook_problems
from blocks.topic_research import TopicResearchBlock
from catalog.store import TheoryCatalog
from config.settings import Settings, get_settings
from core.errors import VideoAIError
from core.models import ResearchBrief, Theory
from providers.llm.groq import GroqProvider
from providers.sources import MediaWikiArticleFetcher, build_http_client
from utils.cache import DiskCache


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("blocks").setLevel(logging.INFO)
    settings = get_settings()
    if not settings.groq_api_keys:
        raise SystemExit("No Groq API keys configured.")

    cache = DiskCache(settings.cache_dir, enabled=settings.cache_enabled)
    llm = GroqProvider(
        model=args.model,
        api_keys=settings.groq_api_keys,
        base_url=settings.groq_base_url,
        temperature=args.temperature,
        max_tokens=settings.groq_max_tokens,
        timeout=settings.groq_timeout,
        max_retries=settings.max_retries,
        cache=cache,
    )

    for slug in args.slugs:
        theory = _load_theory(settings, slug)
        try:
            brief = _load_or_research(settings, theory, llm)
            script = ScriptGenerationBlock(
                llm, hook_candidates=settings.script_hook_candidates
            ).run(brief, format=args.format)
        except VideoAIError as exc:
            print(f"\n{slug}\n  failed: {exc}")
            continue
        problems = hook_problems(script.hook) or ["(clean)"]
        print(f"\n{slug}")
        print(f"  hook: {script.hook}")
        print(f"  gate: {'; '.join(problems)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="+")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--format", choices=("short", "long"), default="short")
    parser.add_argument("--temperature", type=float, default=0.6)
    return parser.parse_args()


def _load_theory(settings: Settings, slug: str) -> Theory:
    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slug)
    if theory is None:
        raise SystemExit(f"No theory '{slug}' in the catalog.")
    return theory


def _load_or_research(settings: Settings, theory: Theory, llm) -> ResearchBrief:
    path = settings.output_dir / theory.slug / "brief.json"
    if path.exists():
        return ResearchBrief.model_validate_json(path.read_text())

    fetcher = MediaWikiArticleFetcher(
        client=build_http_client(settings),
        language=settings.wikipedia_language,
        delay=settings.request_delay,
        max_chars=settings.research_max_chars,
    )
    brief = TopicResearchBlock(
        llm, fetcher, max_chars=settings.research_max_chars
    ).run(theory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief.model_dump_json(indent=2))
    return brief


if __name__ == "__main__":
    main()
