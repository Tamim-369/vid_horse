"""Video style configuration — per-video JSON control over all visual aspects.

Config file location (in order of precedence):
  1. output/<slug>/style.json          (per-video override)
  2. config/video_style_default.json   (global defaults)
  3. Built-in DEFAULTS (final fallback)

Any key not provided falls back to the next level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in defaults (mirror current pipeline behavior)
# ---------------------------------------------------------------------------

@dataclass
class SubtitleStyle:
    font: str = "DejaVu Sans Condensed"
    fontsize: int = 74               # 30% smaller than 106
    primary_color: str = "&H00FFFFFF"  # pure white
    outline_color: str = "&H00000000"  # black
    back_color: str = "&H00000000"
    bold: int = 1                    # bold
    italic: int = 0
    underline: int = 0
    strikeout: int = 0
    border_style: int = 1
    outline: float = 4.0             # thicker outline
    shadow: float = 0
    alignment: int = 10              # middle center
    margin_l: int = 70
    margin_r: int = 70
    margin_v: int = 0                # center vertically
    encoding: int = 1
    scale_x: float = 100.0
    scale_y: float = 100.0
    spacing: float = 0.0
    angle: float = 0.0

@dataclass
class MotionStyle:
    """Single motion definition for zoompan."""
    name: str = ""
    zoom: float = 0.05
    drift: float = 0.5
    # Custom zoompan expression pieces (advanced)
    z_expr: str | None = None
    x_expr: str | None = None
    y_expr: str | None = None

@dataclass
class ImageStyle:
    motion_pool: list[int] = field(default_factory=lambda: list(range(24)))
    # Per-style overrides (index -> MotionStyle)
    motions: dict[int, MotionStyle] = field(default_factory=dict)
    # Crossfade between subject images within a card (seconds)
    subject_crossfade: float = 0.45
    # Crossfade between cards (seconds)
    card_crossfade: float = 0.30
    # Ken Burns canvas multiplier (4 = 4x for sub-pixel smoothness)
    canvas_multiplier: int = 4
    # Credit styling
    credit_font: str = "DejaVu Sans Condensed"
    credit_color: str = "white@0.35"
    credit_fontsize_min: int = 12
    credit_fontsize_max: int = 22

@dataclass
class ColorGrade:
    enabled: bool = False
    # Applied only to horror genre
    eq: str = "contrast=1.08:saturation=0.42:brightness=-0.05:gamma=1.07"
    colorbalance: str = "rs=-0.12:gs=0.02:bs=0.12"
    noise: str = "alls=7:allf=t"
    vignette: str = "PI/4.2"

@dataclass
class AudioStyle:
    voice_speed: float = 0.97
    voice_gap: float = 0.15
    line_gap: float = 0.30
    music_bed_volume: float = 0.18
    sfx_volume: float = 0.40
    music_bed: str = ""

@dataclass
class VideoStyle:
    """Complete style for one video render."""
    subtitle: SubtitleStyle = field(default_factory=SubtitleStyle)
    image: ImageStyle = field(default_factory=ImageStyle)
    color_grade: ColorGrade = field(default_factory=ColorGrade)
    audio: AudioStyle = field(default_factory=AudioStyle)

    # FPS and canvas (rarely changed)
    fps: int = 30
    width: int = 1080
    height: int = 1920

    @classmethod
    def defaults(cls) -> "VideoStyle":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoStyle":
        # Recursively build nested dataclasses
        subtitle = SubtitleStyle(**data.get("subtitle", {}))
        image_data = data.get("image", {})
        motions = {}
        for k, v in image_data.get("motions", {}).items():
            motions[int(k)] = MotionStyle(**v)
        image = ImageStyle(
            motion_pool=image_data.get("motion_pool", list(range(24))),
            motions=motions,
            subject_crossfade=image_data.get("subject_crossfade", 0.45),
            card_crossfade=image_data.get("card_crossfade", 0.30),
            canvas_multiplier=image_data.get("canvas_multiplier", 4),
            credit_font=image_data.get("credit_font", "DejaVu Sans Condensed"),
            credit_color=image_data.get("credit_color", "white@0.35"),
            credit_fontsize_min=image_data.get("credit_fontsize_min", 12),
            credit_fontsize_max=image_data.get("credit_fontsize_max", 22),
        )
        color_grade = ColorGrade(**data.get("color_grade", {}))
        audio = AudioStyle(**data.get("audio", {}))
        return cls(
            subtitle=subtitle,
            image=image,
            color_grade=color_grade,
            audio=audio,
            fps=data.get("fps", 30),
            width=data.get("width", 1080),
            height=data.get("height", 1920),
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/video_style_default.json")
PRESETS_PATH = Path("config/style_presets.json")


def _load_presets() -> dict:
    """Load style presets from JSON file."""
    if PRESETS_PATH.exists():
        try:
            with PRESETS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("presets", {})
        except Exception as e:
            logger.warning("Failed to load style presets: %s", e)
    return {}


def _apply_preset(base: dict, preset_name: str) -> dict:
    """Apply a named preset on top of base config."""
    presets = _load_presets()
    if preset_name in presets:
        preset = presets[preset_name]
        # Remove metadata fields
        preset_cfg = {k: v for k, v in preset.items() if k not in ("name", "description")}
        logger.info("Applied style preset: %s", preset_name)
        return _deep_merge(base, preset_cfg)
    else:
        logger.warning("Style preset '%s' not found, ignoring", preset_name)
        return base


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_style_config(slug: str | None = None) -> VideoStyle:
    """Load style config for a video slug.

    Precedence: built-in defaults -> global default file -> per-video file.
    """
    # Start with built-in defaults
    merged: dict = VideoStyle.defaults().to_dict()

    # Global default file
    if DEFAULT_CONFIG_PATH.exists():
        try:
            with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
                global_cfg = json.load(f)
            merged = _deep_merge(merged, global_cfg)
            logger.info("Loaded global style config from %s", DEFAULT_CONFIG_PATH)
        except Exception as e:
            logger.warning("Failed to load global style config: %s", e)

    # Per-video file
    if slug:
        video_cfg_path = Path("output") / slug / "style.json"
        if video_cfg_path.exists():
            try:
                with video_cfg_path.open("r", encoding="utf-8") as f:
                    video_cfg = json.load(f)
                # Check for preset first
                preset_name = video_cfg.get("style_preset")
                if preset_name:
                    merged = _apply_preset(merged, preset_name)
                merged = _deep_merge(merged, video_cfg)
                logger.info("Loaded per-video style config from %s", video_cfg_path)
            except Exception as e:
                logger.warning("Failed to load per-video style config: %s", e)

    return VideoStyle.from_dict(merged)


def save_style_config(style: VideoStyle, slug: str) -> Path:
    """Save style config for a video slug (creates per-video override)."""
    path = Path("output") / slug / "style.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(style.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def create_example_config(path: Path | str = "config/video_style_example.json") -> Path:
    """Write a fully-documented example config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = {
        "_comment": "Video style configuration — all keys optional. See src/config/video_style.py for schema.",
        "_presets": "Available presets: clinical_cold, noir_grain, documentary_clean, dark_academic, tech_minimal, visceral_raw. Use 'style_preset' key to apply one.",
        "style_preset": "noir_grain",
        "subtitle": {
            "font": "DejaVu Sans Condensed",
            "fontsize": 76,
            "primary_color": "&H00F2F2F2",
            "active_color": "&H00FFFFFF",
            "key_color": "&H00FFFFFF",
            "margin_v": 310,
            "outline": 2.8,
            "active_outline": 3.0,
            "key_outline": 2.5,
        },
        "image": {
            "motion_pool": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            "subject_crossfade": 0.45,
            "card_crossfade": 0.30,
            "motions": {
                "0": {"name": "slow_zoom_in", "zoom": 0.05, "drift": 0.5},
                "6": {"name": "zoom_rotate_cw", "zoom": 0.05, "drift": 0.3},
            }
        },
        "color_grade": {"enabled": False},
        "audio": {"voice_speed": 0.97, "music_bed_volume": 0.18},
        "fps": 30,
        "width": 1080,
        "height": 1920,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)
    return path