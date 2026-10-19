#!/usr/bin/env python3
"""Generate many hook candidates for one theory, across several models.

Reuses the research brief from ``output/<slug>/compare/brief.json`` when it
exists, otherwise researches once. Prints numbered candidates per model so the
best "grab + kick" line can be chosen.

Usage:
    uv run python experiments/hook_lab.py baron-52
    uv run python experiments/hook_lab.py baron-52 --count 10 --temperature 0.95
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re

from blocks.script_generation import parse_hook_scores, parse_hooks
from blocks.script_generation.prompts import (
    build_hook_messages,
    build_hook_score_messages,
)
from blocks.topic_research import TopicResearchBlock
from catalog.store import TheoryCatalog
from config.settings import Settings, get_settings
from core.errors import VideoAIError
from core.models import ResearchBrief, Theory
from providers.llm.groq import GroqProvider
from providers.sources import MediaWikiArticleFetcher, build_http_client
from utils.cache import DiskCache

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

    def provider(
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> GroqProvider:
        return GroqProvider(
            model=model,
            api_keys=settings.groq_api_keys,
            base_url=settings.groq_base_url,
            temperature=args.temperature,
            max_tokens=max_tokens or settings.groq_max_tokens,
            timeout=settings.groq_timeout,
            max_retries=settings.max_retries,
            reasoning_effort=reasoning_effort,
            cache=cache,
        )

    brief = _load_or_research(settings, theory, provider(args.models[0]))

    pool: list[str] = []
    for model in args.models:
        print(f"\n===== {model} =====")
        try:
            raw = provider(model).complete(
                build_hook_messages(brief, args.count),
                temperature=args.temperature,
            )
            hooks = parse_hooks(raw)
        except VideoAIError as exc:
            print(f"  failed: {exc}")
            continue
        for index, hook in enumerate(hooks, start=1):
            print(f"  {index:>2}. {hook}")
        pool.extend(hooks)

    if args.rank and pool:
        _rank(pool, args.judges, provider, top=args.top)


def _rank(pool: list[str], judges: list[str], provider, *, top: int) -> None:
    """Score every hook with each judge and print the averaged ranking."""

    by_key: dict[str, str] = {_norm(hook): hook for hook in pool}
    scores: dict[str, list[int]] = {key: [] for key in by_key}

    for model in judges:
        print(f"\n===== judge: {model} =====")
        try:
            raw = provider(model, reasoning_effort="low", max_tokens=8192).complete(
                build_hook_score_messages(pool),
                temperature=0.0,
            )
            scored = parse_hook_scores(raw)
        except VideoAIError as exc:
            print(f"  scoring failed: {exc}")
            continue
        for score, hook in scored:
            key = _match_key(_norm(hook), by_key)
            if key is None:
                continue
            scores[key].append(score)
            print(f"  {score:>3} | {hook}")

    print(f"\n===== averaged ranking ({len(judges)} judges) =====")
    ranked = sorted(
        scores.items(),
        key=lambda item: (sum(item[1]) / len(item[1])) if item[1] else -1.0,
        reverse=True,
    )
    for key, values in ranked[:top]:
        if not values:
            continue
        mean = sum(values) / len(values)
        spread = ",".join(str(v) for v in values)
        print(f"  {mean:5.1f}  [{spread}]  {by_key[key]}")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _match_key(key: str, by_key: dict[str, str]) -> str | None:
    if key in by_key:
        return key
    close = difflib.get_close_matches(key, by_key.keys(), n=1, cutoff=0.8)
    return close[0] if close else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Catalog slug, e.g. baron-52")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        help="Models used to rank the candidates; scores are averaged.",
    )
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--no-rank",
        dest="rank",
        action="store_false",
        help="Print candidates only, without scoring.",
    )
    return parser.parse_args()


def _load_theory(settings: Settings, slug: str) -> Theory:
    with TheoryCatalog(settings.catalog_path) as catalog:
        theory = catalog.get(slug)
    if theory is None:
        raise SystemExit(f"No theory '{slug}' in the catalog.")
    return theory


def _load_or_research(settings: Settings, theory: Theory, llm) -> ResearchBrief:
    path = settings.output_dir / theory.slug / "compare" / "brief.json"
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
