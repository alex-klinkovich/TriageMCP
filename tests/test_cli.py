"""Spec for the Typer CLI surface (offline: sample, help, and missing-key handling)."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from triagemcp.agent.loop import AgentRun
from triagemcp.cli import app
from triagemcp.config import Settings
from triagemcp.models import Alert, RecommendedAction, Severity, TriageResult

runner = CliRunner()


class _FakeTriager:
    async def run(self, alert: Alert) -> AgentRun:
        result = TriageResult(
            alert_id=alert.id,
            severity=Severity.HIGH,
            confidence=0.8,
            mitre_technique_id="T1110",
            mitre_technique_name="Brute Force",
            recommended_action=RecommendedAction.INVESTIGATE,
            rationale="prior sightings + malicious IP",
        )
        return AgentRun(result=result, iterations=2)


def _alert_dict(alert_id: str) -> dict[str, object]:
    return Alert(
        id=alert_id,
        title="t",
        description="d",
        source="EDR",
        severity_reported=Severity.MEDIUM,
        timestamp=dt.datetime(2026, 5, 28, 12, 0, tzinfo=dt.UTC),
    ).model_dump(mode="json")


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("triage", "serve", "eval", "sample"):
        assert command in result.output


def test_sample_emits_bundled_labeled_alerts() -> None:
    result = runner.invoke(app, ["sample"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) >= 30
    assert "alert" in data[0]
    assert "label" in data[0]


def test_sample_writes_to_file(tmp_path: Path) -> None:
    out = tmp_path / "alerts.json"
    result = runner.invoke(app, ["sample", "--out", str(out)])
    assert result.exit_code == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))) >= 30


def test_triage_without_key_exits_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["triage"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_eval_without_key_exits_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_triage_renders_json_report_with_injected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    @contextlib.asynccontextmanager
    async def _fake_runtime(_settings: Settings) -> AsyncIterator[_FakeTriager]:
        yield _FakeTriager()

    monkeypatch.setattr("triagemcp.cli.build_runtime", _fake_runtime)

    alerts_file = tmp_path / "alerts.json"
    alerts_file.write_text(json.dumps([_alert_dict("A-1"), _alert_dict("A-2")]), encoding="utf-8")

    result = runner.invoke(app, ["triage", str(alerts_file), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [item["alert_id"] for item in data] == ["A-1", "A-2"]
    assert data[0]["result"]["mitre_technique_id"] == "T1110"
    assert data[0]["error"] is None


def test_experiment_lists_variants_in_help() -> None:
    result = runner.invoke(app, ["experiment", "--help"])
    assert result.exit_code == 0
    assert "variant" in result.output


def test_experiment_without_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["experiment", "--variant", "baseline"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_eval_vote_samples_wraps_and_samples_n_times(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    class _Counting:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, alert: Alert) -> AgentRun:
            self.calls += 1
            return AgentRun(
                result=TriageResult(
                    alert_id=alert.id,
                    severity=Severity.HIGH,
                    confidence=0.8,
                    mitre_technique_id="T1110",
                    mitre_technique_name="Brute Force",
                    recommended_action=RecommendedAction.INVESTIGATE,
                    rationale="r",
                ),
                iterations=1,
            )

    fake = _Counting()

    @contextlib.asynccontextmanager
    async def _fake_runtime(
        _settings: Settings, *, temperature: float = 0.0, clock: object = None
    ) -> AsyncIterator[_Counting]:
        captured["temperature"] = temperature
        yield fake

    monkeypatch.setattr("triagemcp.cli.build_runtime", _fake_runtime)
    result = runner.invoke(app, ["eval", "--vote-samples", "3"])
    assert result.exit_code == 0, result.output
    assert captured["temperature"] == 0.7  # voting temperature applied when N > 1
    assert fake.calls == 38 * 3  # 3 samples per alert -> VotingTriager active
