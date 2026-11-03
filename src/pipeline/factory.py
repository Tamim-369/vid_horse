"""Build pipelines from settings."""

from __future__ import annotations

from blocks import ScriptGenerationBlock, TopicResearchBlock
from blocks.asset_research import AssetOptions, AssetResearchBlock
from config.settings import Settings, get_settings
from core.interfaces import ArticleFetcher, LLMProvider
from pipeline.orchestrator import ContentPipeline
from providers.llm import create_llm_provider
from providers.sources import (
    MediaWikiArticleFetcher,
    NasaImagesClient,
    OpenverseClient,
    build_http_client,
)
from providers.sources.mediawiki import MediaWikiClient
from providers.sources.webimages import WebImagesClient
from providers.sources.yandex import build_yandex_client
from providers.vision import build_vision_provider
from utils import computing
from utils.clip import ClipScorer
from utils.shelf import AssetShelf


def _web_http(settings: Settings):
    """A browser-flavoured client just for scraped image engines."""

    import httpx

    return httpx.Client(
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
            ),
            "Accept-Language": "en",
        },
        timeout=settings.http_timeout,
        follow_redirects=True,
    )


def _vision(settings: Settings):
    """Vision relevance checker for picks the CLIP scorer is unsure about.

    Judge of last resort, on by default: results are cached on disk, so
    each verdict is only ever paid for once.
    """

    provider = build_vision_provider(
        model=settings.vision_model, host=settings.ollama_host
    )
    provider.attach_cache(settings.cache_dir / "vision.sqlite3")
    # stage 1: the raw fast caption judge (d3 decision model); the slow
    # pixel look only gets the cases this stage is not confident about
    provider.attach_decision(model=settings.vision_decision_model)
    return provider


def build_content_pipeline(
    settings: Settings | None = None,
    *,
    llm: LLMProvider | None = None,
    fetcher: ArticleFetcher | None = None,
) -> ContentPipeline:
    """Wire up the default research + script pipeline."""

    settings = settings or get_settings()
    llm = llm or create_llm_provider(settings)
    http = build_http_client(settings)
    if fetcher is None:
        fetcher = MediaWikiArticleFetcher(
            client=http,
            language=settings.wikipedia_language,
            delay=settings.request_delay,
            max_chars=settings.research_max_chars,
        )
    commons = MediaWikiClient(
        "https://commons.wikimedia.org/w/api.php",
        client=http,
        delay=settings.request_delay,
    )
    wiki = MediaWikiClient(
        f"https://{settings.wikipedia_language}.wikipedia.org/w/api.php",
        client=http,
        delay=settings.request_delay,
    )
    return ContentPipeline(
        topic_research=TopicResearchBlock(
            llm,
            fetcher,
            max_chars=settings.research_max_chars,
            max_attempts=settings.max_retries + 1,
            race=computing.race_attempts(),
        ),
        script_generation=ScriptGenerationBlock(
            llm,
            max_attempts=settings.max_retries + 1,
            race=computing.race_attempts(),
            chunk_size=settings.script_long_chunk_size,
            scene_count=settings.script_long_scenes,
            hook_candidates=settings.script_hook_candidates,
        ),
        asset_research=AssetResearchBlock(
            wiki,
            commons,
            llm=llm,
            http=http,
            clip=(
                ClipScorer(settings.http_user_agent, threads=computing.torch_threads())
                if settings.asset_judge != "d3"
                else None
            ),
            options=AssetOptions(
                assets_max=settings.asset_max_per_scene,
                words_per_asset=settings.asset_words_per_asset,
                search_limit=settings.asset_search_limit,
                nasa_limit=settings.asset_nasa_limit,
                entity_limit=settings.asset_entity_limit,
                thumb_width=settings.asset_thumb_width,
                min_score=settings.asset_min_score,
                judge=settings.asset_judge,
            ),
            nasa=NasaImagesClient(client=http, delay=settings.request_delay),
            openverse=OpenverseClient(client=http, delay=settings.request_delay),
            yandex=build_yandex_client(delay=max(1.0, settings.request_delay)),
            vision=_vision(settings),
            web=WebImagesClient(
                client=_web_http(settings),
                delay=max(1.5, settings.request_delay),
                license_image=settings.web_license,
            ),
            shelf=AssetShelf(settings.cache_dir / "shelf.sqlite3"),
        ),
        asset_direction=None,
        query_routing=None,
    )
