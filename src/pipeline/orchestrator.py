"""Orchestration of blocks into an end-to-end pipeline."""

from blocks import ScriptGenerationBlock, TopicResearchBlock


class ContentPipeline:
    """Run topic research followed by script generation."""

    def __init__(
        self,
        topic_research: TopicResearchBlock,
        script_generation: ScriptGenerationBlock,
    ) -> None:
        self._topic_research = topic_research
        self._script_generation = script_generation
