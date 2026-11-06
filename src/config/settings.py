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

    # -- Ollama -------------------------------------------------------------
    llm_provider: str = Field(default="ollama")
    ollama_host: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("VIDEOAI_OLLAMA_HOST", "OLLAMA_HOST"),
    )
    ollama_model: str = Field(default="gemma3:4b")
    ollama_num_ctx: int = Field(default=4096, gt=0)
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ollama_num_predict: int | None = Field(default=None, gt=0)
    ollama_repeat_penalty: float = Field(default=1.3, ge=0.0)
    ollama_repeat_last_n: int = Field(default=256, ge=0)
    ollama_keep_alive: str = Field(default="5m")
    ollama_timeout: float = Field(default=600.0, gt=0)

    # -- OpenRouter -----------------------------------------------------------
    openrouter_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY")
    )
    openrouter_models: str = Field(
        default="openrouter/free,nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-super-vl:free,nvidia/nemotron-3.5-lightning:free,google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free,recursive/flint-v1.0:free,xingchen-agi/xing4.0-29b-a4b:free"
    )
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openrouter_max_tokens: int | None = Field(default=None, gt=0)
    openrouter_timeout: float = Field(default=120.0, gt=0)

    # -- Ollama Cloud --------------------------------------------------------
    ollama_cloud_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VIDEOAI_OLLAMA_CLOUD_API_KEY", "OLLAMA_CLOUD_API_KEY", "OLLAMA_API_KEY"
        ),
    )
    ollama_cloud_host: str = Field(
        default="https://ollama.com/v1",
        validation_alias=AliasChoices("VIDEOAI_OLLAMA_CLOUD_HOST", "OLLAMA_CLOUD_HOST"),
    )
    ollama_cloud_models: str = Field(
        default="gemma4:31b,gpt-oss:120b,gpt-oss:20b,nemotron-3-nano:30b,nemotron-3-super,nemotron-3-ultra",
        validation_alias=AliasChoices("VIDEOAI_OLLAMA_CLOUD_MODELS", "OLLAMA_CLOUD_MODELS"),
    )
    ollama_cloud_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ollama_cloud_max_tokens: int | None = Field(default=None, gt=0)
    ollama_cloud_timeout: float = Field(default=120.0, gt=0)
    ollama_cloud_fallback_to_local: bool = Field(default=True)

    # -- Groq ---------------------------------------------------------------
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    groq_max_tokens: int | None = Field(default=None, gt=0)
    groq_timeout: float = Field(default=120.0, gt=0)
    groq_key_mode: Literal["single", "pool"] = Field(default="pool")
    groq_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY")
    )
    groq_api_key_first: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY_FIRST")
    )
    groq_api_key_second: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY_SECOND")
    )
    groq_api_key_third: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY_THIRD")
    )
    groq_api_key_fourth: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY_FOURTH")
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

    @property
    def groq_api_keys(self) -> list[str]:
        """The Groq keys to use, in rotation order.

        ``pool`` mode prefers the numbered ``GROQ_API_KEY_*`` keys and falls
        back to ``GROQ_API_KEY``; ``single`` mode uses only ``GROQ_API_KEY``.
        """

        if self.groq_key_mode == "single":
            return [self.groq_api_key] if self.groq_api_key else []
        candidates = [
            self.groq_api_key_first,
            self.groq_api_key_second,
            self.groq_api_key_third,
            self.groq_api_key_fourth,
            self.groq_api_key,
        ]
        return [key for key in candidates if key]


@lru_cache
def get_settings() -> Settings:
    """Return process-wide settings (cached)."""

    return Settings()
