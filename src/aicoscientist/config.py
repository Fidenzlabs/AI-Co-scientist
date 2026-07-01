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

    # Layer 3 — AS-ALD surface-reactivity engine (ADR-004/009)
    compute_tier: int = Field(
        default=0,
        alias="COMPUTE_TIER",
        description="0 = pure-python (literature/xTB dE), 1 = foundation MLIP, 2 = +spot-checks",
    )
    mlip_model: str = Field(default="mace-mp", alias="MLIP_MODEL")
    mlip_device: str = Field(
        default="auto", alias="MLIP_DEVICE", description="auto|cuda|mps|cpu"
    )
    ald_temperature_k: float = Field(default=423.0, alias="ALD_TEMPERATURE_K")
    surface_ensemble_n: int = Field(
        default=5, alias="SURFACE_ENSEMBLE_N", description="slabs per surface condition"
    )
    selection_criteria_path: str = Field(
        default="selection_criteria.md", alias="SELECTION_CRITERIA_PATH"
    )

    @property
    def artifacts_path(self) -> Path:
        path = Path(self.artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_mlip_device(self) -> str:
        """Resolve 'auto' to the best available torch device.

        MACE energy differences require float64, which the MPS backend does not
        support, so Apple-silicon runs fall back to CPU for the MLIP tier.
        """
        if self.mlip_device != "auto":
            return self.mlip_device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

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
