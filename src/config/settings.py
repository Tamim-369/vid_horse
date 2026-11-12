"""Runtime configuration.

Settings are read from environment variables (optionally a local ``.env``).
There is a single source of truth, so no component reaches for ``os.environ``
directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VIDEOAI_",
        extra="ignore",
    )

    # -- Local LLM (Ollama) -------------------------------------------------
    llm_provider: str = Field(default="ollama")
    ollama_host: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("VIDEOAI_OLLAMA_HOST", "OLLAMA_HOST"),
    )
    # Role-routed local models (all <=4B, already pulled):
    #   reader/writer/reviewer -> ollama_model        (gemma3:4b: reads book
    #       chunks, reasons about lessons, writes and reviews scripts)
    #   decision -> ollama_decision_model             (d3:2b: fast yes/no and
    #       pick-one judgements; also the vision judge via vision_decision_model)
    #   assets -> ollama_asset_model                  (gemma3:1b: which asset
    #       goes into which sentence; flip to d3:2b to test it in this role)
    ollama_model: str = Field(default="gemma3:4b")
    ollama_decision_model: str = Field(
        default="d3:2b",
        validation_alias=AliasChoices(
            "VIDEOAI_OLLAMA_DECISION_MODEL", "OLLAMA_DECISION_MODEL"
        ),
    )
    ollama_asset_model: str = Field(
        default="gemma3:1b",
        validation_alias=AliasChoices(
            "VIDEOAI_OLLAMA_ASSET_MODEL", "OLLAMA_ASSET_MODEL"
        ),
    )
    # Script writing benefits from a model that follows instructions and
    # keeps a coherent thread. Falls back to ollama_model when empty.
    # llama3.2 writes the most coherent, grounded, plain-word scripts in
    # local comparisons (qwen3.5/3 4B often return empty or incoherent output).
    ollama_writer_model: str = Field(
        default="llama3.2:latest",
        validation_alias=AliasChoices(
            "VIDEOAI_OLLAMA_WRITER_MODEL", "OLLAMA_WRITER_MODEL",
            "VIDEOAI_WRITER_MODEL", "WRITER_MODEL",
        ),
    )
    ollama_num_ctx: int = Field(
        default=8192, gt=0,
        description=(
            "Prompt context window in tokens. 8192 fits a lesson chunk plus "
            "surrounding context; kept moderate to avoid high RAM/CPU use. "
            "Lower values make Ollama silently truncate the text."
        ),
    )
    ollama_num_thread: int = Field(
        default=4,
        description="CPU threads per ollama request. Limit to 4 to prevent thermal shutdown.",
    )
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ollama_num_predict: int | None = Field(default=None, gt=0)
    ollama_repeat_penalty: float = Field(default=1.3, ge=0.0)
    ollama_repeat_last_n: int = Field(default=256, ge=0)
    ollama_keep_alive: str = Field(
        default="30s",
        description="Unload the model quickly after use to free RAM.",
    )
    ollama_timeout: float = Field(default=120.0, gt=0)

    # -- Pollinations --------------------------------------------------------
    pollinations_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("POLLINATIONS_API_KEY"),
    )
    # -- Pexels / Pixels -----------------------------------------------------
    pixels_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PIXELS_API_KEY", "PEXELS_API_KEY"),
    )
    # -- HF Token ------------------------------------------------------------
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN"),
    )

    # -- Reliability --------------------------------------------------------
    max_retries: int = Field(default=4, ge=0, le=6)
    # DuckDuckGo image-search license chip; vocabulary:
    # any | Public | Share | ShareCommercially | Modify | ModifyCommercially
    web_license: str = Field(
        default="ShareCommercially", validation_alias=AliasChoices("WEB_LICENSE")
    )

    # -- HTTP / discovery sources ------------------------------------------
    http_user_agent: str = Field(
        default="videoai/0.1 (https://github.com/; contact@example.org)"
    )
    http_timeout: float = Field(default=30.0, gt=0)
    request_delay: float = Field(default=0.2, ge=0.0)
    pageviews_months: int = Field(default=12, ge=1, le=60)

    # Popularity tiers (average monthly pageviews)
    popularity_mainstream: int = Field(default=100_000, ge=0)
    popularity_popular: int = Field(default=20_000, ge=0)
    popularity_moderate: int = Field(default=2_000, ge=0)

    # Discovery sources
    wikipedia_language: str = Field(default="en")
    discovery_categories: list[str] = Field(
        default_factory=lambda: ["Conspiracy theories"]
    )
    discovery_seed_pages: list[str] = Field(
        default_factory=lambda: ["List of conspiracy theories"]
    )
    fandom_wikis: list[str] = Field(default_factory=lambda: ["conspiracy.fandom.com"])

    # -- Research / scripting ----------------------------------------------
    research_max_chars: int = Field(default=6000, gt=500)
    script_short_seconds: int = Field(default=45, gt=0)
    script_long_seconds: int = Field(default=600, gt=0)
    script_long_scenes: int = Field(default=10, ge=3, le=20)
    script_long_chunk_size: int = Field(default=3, ge=1, le=10)
    script_hook_candidates: int = Field(default=3, ge=1, le=6)

    # -- Asset fetch --------------------------------------------------------
    asset_max_per_scene: int = Field(default=3, ge=1, le=6)
    asset_words_per_asset: int = Field(default=18, ge=8, le=60)
    asset_search_limit: int = Field(default=8, ge=1, le=20)
    asset_nasa_limit: int = Field(default=6, ge=1, le=20)
    asset_entity_limit: int = Field(default=5, ge=0, le=20)
    asset_thumb_width: int = Field(default=1280, gt=0)
    asset_min_score: int = Field(default=1, ge=1, le=9)
    asset_judge: str = Field(default="d3", validation_alias=AliasChoices("ASSET_JUDGE"))
    # "llm": the ollama_asset_model decides which candidate image goes into
    # which sentence (CLIP only pre-filters); "ask": d3-style yes/no decision
    # model via ollama_decision_model; "clip": pure CLIP scoring.
    asset_sentence_chooser: str = Field(
        default="llm",
        validation_alias=AliasChoices(
            "VIDEOAI_ASSET_SENTENCE_CHOOSER", "ASSET_SENTENCE_CHOOSER"
        ),
    )

    # -- Caching / output ---------------------------------------------------
    cache_enabled: bool = Field(default=True)
    cache_dir: Path = Field(default=Path(".cache/videoai"))
    catalog_path: Path = Field(default=Path(".cache/videoai/catalog.sqlite3"))
    output_dir: Path = Field(default=Path("output"))
    log_level: str = Field(default="INFO")

    # -- Voice --------------------------------------------------------------
    voice_prompt: Path = Field(
        default=Path("assets/voice/voice_prompt_short.wav"),
        validation_alias=AliasChoices("VIDEOAI_VOICE_PROMPT", "VOICE_PROMPT"),
    )
    # Chatterbox expressiveness: 0 = a flat, newsread-ish read; ~0.4 lands a
    # dramatic storyteller delivery without shouting. Tune in 0.1 steps.
    voice_exaggeration: float = Field(default=0.4, ge=0.0, le=1.0)
    voice_cfg_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    # Silence between sentences inside one line, and between script lines
    voice_sentence_gap: float = Field(default=0.24, ge=0.0, le=2.0)
    voice_line_gap: float = Field(default=0.35, ge=0.0, le=2.0)
    voice_speed: float = Field(
        default=0.97, ge=0.5, le=2.0,
        description="Playback speed for TTS (atempo). <1 slower, >1 faster.",
    )

    # -- Vision / subtitles ---------------------------------------------------
    vision_model: str = Field(
        default="qwen3-vl:2b",
        validation_alias=AliasChoices("VIDEOAI_VISION_MODEL", "VISION_MODEL"),
    )
    vision_decision_model: str = Field(
        default="d3:2b",
        validation_alias=AliasChoices("VIDEOAI_VISION_DECISION_MODEL", "VISION_DECISION_MODEL"),
    )
    subtitle_whisper_model: str = Field(
        default="base.en",
        validation_alias=AliasChoices(
            "VIDEOAI_SUBTITLE_WHISPER_MODEL", "SUBTITLE_WHISPER_MODEL"
        ),
    )

    # -- YouTube publishing ----------------------------------------------
    youtube_client_secrets_path: Path = Field(
        default=Path("client_secrets.json"),
        validation_alias=AliasChoices("YOUTUBE_CLIENT_SECRETS"),
    )
    youtube_token_path: Path = Field(
        default=Path("token.json"),
        validation_alias=AliasChoices("YOUTUBE_TOKEN"),
    )
    youtube_privacy_status: str = Field(
        default="private",
        validation_alias=AliasChoices("YOUTUBE_PRIVACY_STATUS"),
    )
    youtube_redirect_port: int = Field(
        default=8080, gt=0, le=65535,
        validation_alias=AliasChoices("YOUTUBE_REDIRECT_PORT"),
    )
    youtube_category_id: str = Field(
        default="22",
        validation_alias=AliasChoices("YOUTUBE_CATEGORY_ID"),
    )
    auto_publish: bool = Field(
        default=False,
        validation_alias=AliasChoices("AUTO_PUBLISH"),
    )


@lru_cache
def get_settings() -> Settings:
    """Return process-wide settings (cached)."""

    return Settings()
