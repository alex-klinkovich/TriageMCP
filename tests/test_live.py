"""Optional live smoke test against the real Anthropic API.

Skipped by default. Run with::

    pytest --run-live

with ``ANTHROPIC_API_KEY`` set. Proves the real wiring (SDK adapter, tool-use loop,
schema-validated output) works end-to-end against a live model.
"""

from __future__ import annotations

import datetime as dt

import aiosqlite
import pytest
from anthropic import AsyncAnthropic

from triagemcp.agent.llm import AnthropicLLMClient
from triagemcp.agent.loop import AgentConfig, TriageAgent
from triagemcp.config import Settings
from triagemcp.models import Alert, Severity, TriageResult
from triagemcp.tools.registry import build_default_registry


@pytest.mark.live
async def test_live_triage_produces_valid_result() -> None:
    settings = Settings()  # requires ANTHROPIC_API_KEY
    alert = Alert(
        id="LIVE-1",
        title="Encoded PowerShell spawned by Microsoft Word",
        description=(
            "winword.exe spawned powershell.exe with a base64 -EncodedCommand that decodes to a "
            "downloader fetching a second-stage payload from an external host."
        ),
        source="EDR",
        severity_reported=Severity.HIGH,
        timestamp=dt.datetime(2026, 5, 28, 12, 0, tzinfo=dt.UTC),
    )
    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    try:
        async with aiosqlite.connect(":memory:") as conn:
            registry = await build_default_registry(conn)
            agent = TriageAgent(
                AnthropicLLMClient(client), registry, AgentConfig(model=settings.model)
            )
            result = await agent.triage(alert)
    finally:
        await client.close()

    assert isinstance(result, TriageResult)
    assert result.alert_id == "LIVE-1"
    assert result.mitre_technique_id.startswith("T")
