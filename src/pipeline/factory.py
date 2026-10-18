"""Build pipelines from settings."""

from pipeline.orchestrator import ContentPipeline


def build_content_pipeline() -> ContentPipeline:
    """Wire up the default content pipeline."""
    ...
