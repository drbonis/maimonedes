"""Runtime configuration loaded from environment / .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "sqlite:///./maimonedes.db"

    # Ollama serving the supervised + judge models (see docs/roadmap.md).
    # Default points at the GPU laptop on the LAN.
    ollama_base_url: str = "http://192.168.1.30:11434/v1"
    ollama_api_key: str = "ollama"  # Ollama ignores this; OpenAI SDK requires non-empty.
    ollama_request_timeout_s: float = 60.0
    ollama_max_retries: int = 3

    # Initial model tags from docs/roadmap.md.
    ollama_supervised_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_judge_model: str = "medgemma1.5:4b-it-q4_K_M"


def get_settings() -> Settings:
    return Settings()
