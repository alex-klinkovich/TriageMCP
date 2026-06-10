"""Spec for the FastMCP server: triage_alert is exposed and runs end-to-end (mocked LLM)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import aiosqlite
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from triagemcp.agent.loop import AgentConfig, TriageAgent
from triagemcp.config import Settings
from triagemcp.models import Alert, Severity, TriageResult
from triagemcp.server import build_runtime, create_server
from triagemcp.server import main as server_main
from triagemcp.testing import FakeLLMClient, submit, tool_use
from triagemcp.tools.registry import ToolRegistry, build_default_registry


def _alert_dict() -> dict[str, Any]:
    return Alert(
        id="A-1",
        title="Encoded PowerShell",
        description="powershell -enc",
        source="EDR",
        severity_reported=Severity.MEDIUM,
        timestamp=dt.datetime(2026, 5, 28, 12, 0, tzinfo=dt.UTC),
    ).model_dump(mode="json")


def _valid_args() -> dict[str, Any]:
    return {
        "severity": "high",
        "confidence": 0.8,
        "mitre_technique_id": "T1059.001",
        "mitre_technique_name": "PowerShell",
        "recommended_action": "contain",
        "rationale": "Encoded PowerShell spawned by Office; known initial-access TTP.",
    }


async def test_server_exposes_triage_alert_tool() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        registry = await build_default_registry(conn)
        agent = TriageAgent(FakeLLMClient([submit(_valid_args())]), registry, AgentConfig())
        server = create_server(agent)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            listing = await session.list_tools()
            names = [tool.name for tool in listing.tools]
    assert "triage_alert" in names


async def test_server_triage_alert_returns_validated_verdict() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        registry = await build_default_registry(conn)
        agent = TriageAgent(
            FakeLLMClient([tool_use("map_to_mitre", {"text": "x"}), submit(_valid_args())]),
            registry,
            AgentConfig(),
        )
        server = create_server(agent)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            result = await session.call_tool("triage_alert", {"alert": _alert_dict()})
    assert result.isError is False
    assert result.structuredContent is not None
    verdict = TriageResult.model_validate(result.structuredContent)
    assert verdict.alert_id == "A-1"
    assert verdict.severity is Severity.HIGH


async def test_build_runtime_yields_a_triage_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    settings = Settings()  # db_path defaults to :memory:
    async with build_runtime(settings) as triager:
        assert isinstance(triager, TriageAgent)


async def test_build_runtime_forwards_injected_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    captured: dict[str, object] = {}
    real_build = build_default_registry

    async def _spy(
        conn: aiosqlite.Connection, *, clock: Callable[[], dt.datetime] | None = None
    ) -> ToolRegistry:
        captured["clock"] = clock
        return await real_build(conn, clock=clock)

    monkeypatch.setattr("triagemcp.server.build_default_registry", _spy)
    fixed = dt.datetime(2026, 5, 29, 4, 55, tzinfo=dt.UTC)
    async with build_runtime(Settings(), clock=lambda: fixed) as _triager:
        pass
    assert captured["clock"] is not None
    assert captured["clock"]() == fixed  # type: ignore[operator]


def test_main_without_key_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        server_main()
    assert excinfo.value.code == 1
