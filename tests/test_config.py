"""Spec for Settings: key from ANTHROPIC_API_KEY, app config from TRIAGEMCP_* prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from triagemcp.config import Settings


def test_reads_key_and_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    for var in ("TRIAGEMCP_MODEL", "TRIAGEMCP_CONCURRENCY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-123"
    assert settings.model == "claude-sonnet-4-6"
    assert settings.concurrency == 5


def test_prefixed_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("TRIAGEMCP_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("TRIAGEMCP_CONCURRENCY", "9")
    settings = Settings()
    assert settings.model == "claude-haiku-4-5"
    assert settings.concurrency == 9


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_secret_is_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    settings = Settings()
    assert "sk-super-secret" not in repr(settings)
