"""Behavioral spec for the TriageAgent tool-use loop (LLM fully mocked, offline)."""

from __future__ import annotations

import datetime as dt

import pytest

from triagemcp.agent.llm import AssistantTurn, ToolCall
from triagemcp.agent.loop import AgentConfig, TriageAgent
from triagemcp.errors import (
    AgentTimeoutError,
    MaxIterationsError,
    ModelRefusedToSubmitError,
    TransientLLMError,
)
from triagemcp.models import Alert, RecommendedAction, Severity, TriageResult
from triagemcp.testing import FakeLLMClient, end_turn, submit, tool_use
from triagemcp.tools.mitre import MapToMitreTool, MitreTechnique
from triagemcp.tools.registry import ToolRegistry


def _alert() -> Alert:
    return Alert(
        id="A-1",
        title="Encoded PowerShell",
        description="winword.exe spawned powershell.exe -enc",
        source="EDR",
        severity_reported=Severity.MEDIUM,
        timestamp=dt.datetime(2026, 5, 28, 12, 0, tzinfo=dt.UTC),
    )


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            MapToMitreTool(
                [
                    MitreTechnique(
                        id="T1059.001",
                        name="PowerShell",
                        tactic="Execution",
                        keywords=("powershell", "-enc"),
                    )
                ]
            )
        ]
    )


def _valid_args(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "severity": "high",
        "confidence": 0.86,
        "mitre_technique_id": "T1059.001",
        "mitre_technique_name": "PowerShell",
        "recommended_action": "contain",
        "rationale": "Encoded PowerShell spawned by Office is a known initial-access TTP.",
    }
    return {**base, **over}


async def _noop_sleep(_delay: float) -> None:
    return None


def _agent(client: FakeLLMClient, *, config: AgentConfig | None = None) -> TriageAgent:
    return TriageAgent(
        client, _registry(), config or AgentConfig(max_iterations=5), sleep=_noop_sleep
    )


async def test_happy_path_investigates_then_submits() -> None:
    client = FakeLLMClient(
        [
            tool_use("map_to_mitre", {"text": "encoded powershell -enc"}),
            submit(_valid_args()),
        ]
    )
    result = await _agent(client).triage(_alert())
    assert isinstance(result, TriageResult)
    assert result.severity is Severity.HIGH
    assert result.mitre_technique_id == "T1059.001"
    assert result.recommended_action is RecommendedAction.CONTAIN
    assert result.alert_id == "A-1"
    assert client.calls == 2
    assert "submit_triage" in client.seen_tools


async def test_alert_id_is_forced_to_the_real_alert() -> None:
    client = FakeLLMClient([submit(_valid_args(alert_id="WRONG-ID"))])
    result = await _agent(client).triage(_alert())
    assert result.alert_id == "A-1"


async def test_invalid_submission_is_rejected_then_retried() -> None:
    client = FakeLLMClient(
        [
            submit(_valid_args(confidence=5.0)),  # out of [0, 1] -> rejected
            submit(_valid_args()),  # corrected resubmission
        ]
    )
    result = await _agent(client).triage(_alert())
    assert result.confidence == 0.86
    assert client.calls == 2


async def test_runs_out_of_iterations_without_a_submission() -> None:
    client = FakeLLMClient([tool_use("map_to_mitre", {"text": "x"}) for _ in range(10)])
    agent = _agent(client, config=AgentConfig(max_iterations=3))
    with pytest.raises(MaxIterationsError):
        await agent.triage(_alert())
    assert client.calls == 3


async def test_end_turn_is_nudged_once_then_succeeds() -> None:
    client = FakeLLMClient([end_turn("I am unsure."), submit(_valid_args())])
    result = await _agent(client).triage(_alert())
    assert result.severity is Severity.HIGH
    assert client.calls == 2


async def test_repeated_end_turn_raises_refused_to_submit() -> None:
    client = FakeLLMClient([end_turn("nope"), end_turn("still nope")])
    with pytest.raises(ModelRefusedToSubmitError):
        await _agent(client).triage(_alert())


async def test_transient_errors_are_retried() -> None:
    client = FakeLLMClient(
        [TransientLLMError("429"), TransientLLMError("503"), submit(_valid_args())]
    )
    agent = _agent(client, config=AgentConfig(max_iterations=5, max_retries=4))
    result = await agent.triage(_alert())
    assert result.severity is Severity.HIGH
    assert client.calls == 3


async def test_timeout_raises_agent_timeout_error() -> None:
    client = FakeLLMClient([submit(_valid_args())], delay=5.0)
    agent = TriageAgent(
        client, _registry(), AgentConfig(per_alert_timeout_s=0.05), sleep=_noop_sleep
    )
    with pytest.raises(AgentTimeoutError):
        await agent.triage(_alert())


async def test_run_reports_result_and_iteration_count() -> None:
    client = FakeLLMClient([tool_use("map_to_mitre", {"text": "x"}), submit(_valid_args())])
    run = await _agent(client).run(_alert())
    assert run.result.severity is Severity.HIGH
    assert run.iterations == 2


async def test_temperature_is_passed_to_the_client() -> None:
    client = FakeLLMClient([submit(_valid_args())])
    agent = TriageAgent(client, _registry(), AgentConfig(temperature=0.7), sleep=_noop_sleep)
    await agent.triage(_alert())
    assert client.last_temperature == 0.7


def _multi_tool_turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(stop_reason="tool_use", text="", tool_calls=tuple(calls))


async def test_submit_alongside_another_tool_call_in_one_turn_finalizes() -> None:
    # Models can emit parallel tool calls. A single turn that BOTH investigates and submits must
    # finalize on the valid submit (returning within that one turn).
    client = FakeLLMClient(
        [
            _multi_tool_turn(
                ToolCall(id="c1", name="map_to_mitre", arguments={"text": "powershell -enc"}),
                ToolCall(id="c2", name="submit_triage", arguments=_valid_args()),
            )
        ]
    )
    result = await _agent(client).triage(_alert())
    assert result.alert_id == "A-1"
    assert result.severity is Severity.HIGH
    assert client.calls == 1  # finalized within the single multi-call turn


async def test_invalid_submit_with_sibling_tool_call_recovers() -> None:
    # First turn: an investigation call PLUS an invalid submit. Both tool_use blocks must be
    # answered, and the loop continues to a corrected resubmission.
    client = FakeLLMClient(
        [
            _multi_tool_turn(
                ToolCall(id="c1", name="map_to_mitre", arguments={"text": "x"}),
                ToolCall(id="c2", name="submit_triage", arguments=_valid_args(confidence=5.0)),
            ),
            submit(_valid_args()),
        ]
    )
    result = await _agent(client).triage(_alert())
    assert result.confidence == 0.86
    assert client.calls == 2
