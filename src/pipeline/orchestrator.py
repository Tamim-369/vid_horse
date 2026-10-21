"""Orchestration of blocks into an end-to-end pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from blocks import ScriptGenerationBlock, TopicResearchBlock
from blocks.asset_research import AssetResearchBlock
from blocks.asset_research.director import QueryBuilderBlock, VisualDirectorBlock
from core.models import (
    AssetPack,
    ResearchBrief,
    ScriptFormat,
    Theory,
    VideoScript,
)


@dataclass
class PipelineResult:
    """Everything produced for one theory."""

    theory: Theory
    brief: ResearchBrief
    scripts: dict[str, VideoScript] = field(default_factory=dict)


class ContentPipeline:
    """Run topic research followed by script generation."""

    def __init__(
        self,
        topic_research: TopicResearchBlock,
        script_generation: ScriptGenerationBlock,
        asset_research: AssetResearchBlock | None = None,
        asset_direction: VisualDirectorBlock | None = None,
        query_routing: QueryBuilderBlock | None = None,
    ) -> None:
        self._topic_research = topic_research
        self._script_generation = script_generation
        self._asset_research = asset_research
        self._asset_direction = asset_direction
        self._query_routing = query_routing

    def research(self, theory: Theory, *, max_chars: int | None = None) -> ResearchBrief:
        """Run only the research block for ``theory``."""

        return self._topic_research.run(theory, max_chars=max_chars)

    def script(
        self,
        brief: ResearchBrief,
        format: ScriptFormat = "short",
        *,
        target_seconds: int | None = None,
    ) -> VideoScript:
        """Run only the script block for ``brief``."""

        return self._script_generation.run(
            brief, format=format, target_seconds=target_seconds
        )

    def assets(
        self,
        theory: Theory,
        brief: ResearchBrief,
        script: VideoScript,
        *,
        trace: dict[str, str] | None = None,
    ) -> AssetPack:
        """Direct, route and fetch topical media for every scene of ``script``.

        ``trace`` collects the intermediary agent outputs (their model dumps,
        keyed by filename) so the caller can persist them for inspection.
        """

        if self._asset_research is None:
            raise NotImplementedError("No asset block configured for this pipeline.")
        queries = None
        if self._asset_direction is not None and self._query_routing is not None:
            intents = self._asset_direction.run(theory, script)
            queries = self._query_routing.run(brief, intents)
            if trace is not None:
                trace["direction.json"] = intents.model_dump_json(indent=2)
                trace["queries.json"] = queries.model_dump_json(indent=2)
        return self._asset_research.run(theory, brief, script, queries)

    def run(
        self,
        theory: Theory,
        *,
        formats: Iterable[ScriptFormat] = ("short", "long"),
        target_seconds: dict[str, int] | None = None,
    ) -> PipelineResult:
        """Research ``theory`` then write a script for each requested format."""

        brief = self.research(theory)
        overrides = target_seconds or {}
        scripts = {
            fmt: self.script(brief, fmt, target_seconds=overrides.get(fmt))
            for fmt in formats
        }
        return PipelineResult(theory=theory, brief=brief, scripts=scripts)
