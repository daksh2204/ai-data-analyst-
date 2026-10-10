"""Application settings, loaded from environment variables (and an optional ``.env``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["auto", "openai", "anthropic", "gemini", "groq", "openrouter", "ollama", "offline"]

ROOT_DIR = Path(__file__).resolve().parents[2]

# Default model per provider. Override with LLM_MODEL.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openai/gpt-5.4-mini",
    "ollama": "qwen2.5:7b",
    "offline": "heuristic-planner",
}

# Base URLs for providers that speak the OpenAI chat-completions protocol.
OPENAI_COMPATIBLE_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App -----------------------------------------------------------------
    app_name: str = "AI Data Analyst"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    frontend_dist: Path = ROOT_DIR / "frontend" / "dist"
    sample_data_dir: Path = ROOT_DIR / "sample_data"

    # --- Auth (optional) -----------------------------------------------------
    # When set, every /api request (except health) must send `Authorization: Bearer <token>`.
    app_access_token: str | None = None

    # --- LLM -----------------------------------------------------------------
    llm_provider: ProviderName = "auto"
    llm_model: str | None = None
    llm_base_url: str | None = None  # override for OpenAI-compatible providers (e.g. a proxy)
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str | None = None
    # Fall back to the offline planner when the LLM call fails (quota, network...).
    llm_fallback_to_offline: bool = True

    # --- Agent ---------------------------------------------------------------
    agent_max_steps: int = 8
    agent_history_turns: int = 6  # turns kept verbatim; older turns are compacted
    tool_result_max_rows: int = 40  # rows of a tool result shown to the LLM

    # --- Data limits ---------------------------------------------------------
    max_upload_mb: int = 50
    max_files_per_session: int = 10
    max_rows_per_file: int = 2_000_000
    max_columns_per_file: int = 500
    query_timeout_seconds: float = 15.0
    query_max_rows: int = 5_000
    pandas_timeout_seconds: float = 10.0
    chat_rate_limit_per_minute: int = 30  # per client; 0 disables

    # --- Sessions & cache ----------------------------------------------------
    session_ttl_minutes: int = 120
    max_sessions: int = 200
    cache_ttl_seconds: int = 900
    cache_max_entries: int = 512

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip().startswith("["):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    # --- Derived -------------------------------------------------------------
    def resolved_provider(self) -> str:
        """Pick a provider: the configured one, or the first one with credentials."""
        if self.llm_provider != "auto":
            return self.llm_provider
        for name, key in (
            ("anthropic", self.anthropic_api_key),
            ("openai", self.openai_api_key),
            ("gemini", self.gemini_api_key),
            ("groq", self.groq_api_key),
            ("openrouter", self.openrouter_api_key),
        ):
            if key:
                return name
        if self.ollama_base_url:
            return "ollama"
        return "offline"

    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.resolved_provider()]

    def api_key_for(self, provider: str) -> str | None:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": "ollama",  # Ollama ignores the key but the SDK requires one
        }.get(provider)

    def base_url_for(self, provider: str) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        if provider == "ollama" and self.ollama_base_url:
            return self.ollama_base_url.rstrip("/") + "/v1"
        return OPENAI_COMPATIBLE_BASE_URLS.get(provider)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
