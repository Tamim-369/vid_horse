"""Structured output for the title/description agent."""

from __future__ import annotations

from core.models import StrictModel


class TitlePack(StrictModel):
    """YouTube Shorts metadata for one video."""

    title: str
    description: str
    tags: list[str] = []
    hashtags: list[str] = []
    pinned_comment: str = ""
