"""Motif asset bridge — delegates to per-sentence assignment."""

from __future__ import annotations

from blocks.sentence_assets import assign_sentence_assets, build_asset_pack, choose_concepts as choose_motifs
from core.models import AssetPack, VideoScript  # noqa: F401

__all__ = ["assign_sentence_assets", "build_asset_pack", "choose_motifs"]
