"""The asset research block: topical media for every script scene."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from core.errors import BlockError, ProviderError
from core.models import (
    AssetPack,
    QueryPack,
    ResearchBrief,
    SceneAssets,
    ScriptScene,
    Theory,
    VideoScript,
)
from providers.sources.mediawiki import MediaWikiClient
from utils import computing

from .chooser import ChooseMixin
from .claims import _Claims
from .judging import JudgingMixin
from .media import _VIDEO_SLOTS_PER_VIDEO, AssetOptions
from .packing import PackingMixin
from .pools import PoolsMixin
from .routing import RoutingMixin
from .staging import StagingMixin

logger = logging.getLogger(__name__)




class AssetResearchBlock(
    PoolsMixin,
    StagingMixin,
    RoutingMixin,
    JudgingMixin,
    ChooseMixin,
    PackingMixin,
):
    """Build an :class:`AssetPack` of topical media for every script scene."""

    name = "asset_research"

    def __init__(
        self,
        wiki: MediaWikiClient,
        commons: MediaWikiClient,
        *,
        options: AssetOptions | None = None,
        nasa=None,
        openverse=None,
        yandex=None,
        vision=None,
        llm=None,
        http=None,
        clip=None,
        web=None,
        shelf=None,
    ) -> None:
        self._wiki = wiki
        self._commons = commons
        self._options = options or AssetOptions()
        self._nasa = nasa
        self._openverse = openverse
        self._yandex = yandex
        self._vision = vision
        self._llm = llm
        self._http = http
        self._clip = clip
        self._web = web
        self._shelf = shelf
        self._fiction = False
        self._chooser_pool = 6
        self._video_slots = _VIDEO_SLOTS_PER_VIDEO
        self._run_title = ""

    def run(
        self,
        theory: Theory,
        brief: ResearchBrief,
        script: VideoScript,
        queries: QueryPack | None = None,
    ) -> AssetPack:
        """Fetch and match media for every scene of ``script``.

        With ``queries`` the directed agents + nasa/openverse sources drive
        the search; without them the block falls back to its own phrase
        mining and Commons-only results.
        """

        source = self._article_source(theory)
        article_title = source.title if source is not None else theory.name
        article_url = source.url if source is not None else ""
        self._fiction = any(
            "original horror story" in note.lower() for note in script.source_notes
        )
        self._run_title = theory.name
        try:
            article_pool = [info for info in self._page_pool([article_title]) if self._usable(info)]
            entity_pool = self._entity_pool(brief)
        except ProviderError as exc:
            raise BlockError(f"Could not fetch media for '{theory.name}': {exc}") from exc

        query_map = {
            scene.scene_index: scene.queries for scene in (queries.scenes if queries else [])
        }
        used = _Claims()
        scenes: list[SceneAssets] = []

        # the hook is a sentence too: its own query, its own asset
        hook_scene = ScriptScene(
            narration=script.hook, visual=f"hook: {script.hook}", seconds=6
        )
        hook_assets = self._assets_for_scene(
            hook_scene,
            self._query_for(hook_scene, brief),
            query_map.get(-1),
            self._target_for(hook_scene),
            brief,
            entity_pool,
            article_pool,
            used,
        )
        scenes.append(
            SceneAssets(
                scene_index=-1,
                visual="hook",
                query=script.hook,
                assets=hook_assets,
            )
        )

        tasks: list[tuple[int, ScriptScene, str, object, int]] = []
        for index, scene in enumerate(script.scenes):
            query = self._query_for(scene, brief)
            actions = query_map.get(index)
            target = len(actions) if actions else self._target_for(scene)
            tasks.append((index, scene, query, actions, target))

        def _run_scene(task) -> SceneAssets:
            _index, scene, query, actions, target = task
            return SceneAssets(
                scene_index=_index,
                visual=scene.visual,
                query=query,
                assets=self._assets_for_scene(
                    scene, query, actions, target, brief, entity_pool,
                    article_pool, used,
                ),
            )

        workers = computing.scene_workers()
        if workers > 1 and len(tasks) > 1:
            # independent scenes: fetched and CLIP-scored in parallel, so the
            # stage paces at its slowest scene, not the sum of scenes; every
            # quality gate (floor, junk veto, chooser, claims) runs per scene
            # exactly like the serial version did
            with ThreadPoolExecutor(max_workers=workers) as pool:
                built = list(pool.map(_run_scene, tasks))
        else:
            built = [_run_scene(task) for task in tasks]
        scenes.extend(built)

        self.resolve_races(scenes, tasks, brief, entity_pool, article_pool, used)
        self.unique_within(scenes)
        self.settle_adjacency(scenes, tasks, brief, entity_pool, article_pool)
        self.fill_blanks(scenes)

        return AssetPack(
            slug=theory.slug,
            topic=theory.name,
            article_title=article_title,
            article_url=article_url,
            scenes=scenes,
        )
