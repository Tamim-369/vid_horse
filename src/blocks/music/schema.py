"""Pydantic answer shape the music director must return."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MusicAnnotation(BaseModel):
    """Top-level JSON the music director agent returns.

    The director chooses TRACKS by vibe (any track in ``assets/music``),
    so the track fields carry the decision. The role fields stay for the
    keyword fallback path.
    """

    base_track: str = Field(description="Base bed track file name, chosen by vibe.")
    scene_tracks: dict[str, str] = Field(
        default_factory=dict,
        description='Rare overrides: scene index -> track file name, e.g. {"2": "swell-tension.mp3"}.',
    )
    base_role: str = Field(default="default", description="Fallback role label (keyword path).")
    scene_roles: dict[str, str] = Field(
        default_factory=dict,
        description='Rare overrides: scene index -> role, e.g. {"2": "vast-reveal"}.',
    )
