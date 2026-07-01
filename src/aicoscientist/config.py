"""Environment-driven settings for the AI Co-Scientist."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Sources
    contact_email: str = Field(default="anonymous@example.com", alias="CONTACT_EMAIL")
    semantic_scholar_api_key: str | None = Field(
        default=None, alias="SEMANTIC_SCHOLAR_API_KEY"
    )
    max_results_per_source: int = Field(default=8, alias="MAX_RESULTS_PER_SOURCE")

    # Run tuning
    max_domains: int = Field(default=4, alias="MAX_DOMAINS")
    num_hypotheses: int = Field(default=8, alias="NUM_HYPOTHESES")
    artifacts_dir: str = Field(default="artifacts", alias="ARTIFACTS_DIR")

    # Layer 3 — bounded reflection / closed-loop refinement budget
    max_validation_iters: int = Field(default=2, alias="MAX_VALIDATION_ITERS")

    @property
    def artifacts_path(self) -> Path:
        path = Path(self.artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def user_agent(self) -> str:
        return f"AI-Co-Scientist/0.1 (mailto:{self.contact_email})"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
