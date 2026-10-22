"""Pydantic answer shape the music director must return."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MusicAnnotation(BaseModel):
    """Top-level JSON the music director agent returns."""

    base_role: str = Field(description="The base bed's role for the whole video.")
    scene_roles: dict[str, str] = Field(
        default_factory=dict,
        description='Rare overrides: scene index -> role, e.g. {"2": "vast-reveal"}.',
    )
