"""Application settings, sourced only from the environment (and an optional .env).

The Anthropic key is read from ``ANTHROPIC_API_KEY`` (the conventional name) and held as a
``SecretStr`` so it never leaks into logs or reprs. All other settings use the ``TRIAGEMCP_``
prefix, e.g. ``TRIAGEMCP_MODEL``.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Fields are populated from the environment."""

    model_config = SettingsConfigDict(env_prefix="TRIAGEMCP_", env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr = Field(validation_alias="ANTHROPIC_API_KEY")
    model: str = "claude-sonnet-4-6"
    max_tokens: int = Field(default=2048, ge=1)
    concurrency: int = Field(default=5, ge=1)
    max_iterations: int = Field(default=8, ge=1)
    per_alert_timeout_s: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=4, ge=1)
    vote_samples: int = Field(default=1, ge=1)
    db_path: str = ":memory:"
