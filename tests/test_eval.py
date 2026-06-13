"""Spec for the eval harness: scoring math, full-data wiring, and README headline writing."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import aiosqlite
import pytest

from triagemcp.agent.loop import AgentRun
from triagemcp.datasets import load_sample_alerts
from triagemcp.eval.harness import eval_reference_clock, run_eval, write_headline_to_readme
from triagemcp.eval.metrics import score
from triagemcp.models import (
    Alert,
    AlertLabel,
    RecommendedAction,
    Severity,
    TriageOutcome,
    TriageResult,
)
from triagemcp.tools.mitre import load_mitre_techniques
from triagemcp.tools.registry import build_default_registry

TACTIC = {"T1059.001": "Execution", "T1059.003": "Execution", "T1110": "Credential Access"}


def _result(
    alert_id: str, severity: Severity, tech: str, action: RecommendedAction, conf: float = 0.8
) -> TriageResult:
    return TriageResult(
        alert_id=alert_id,
        severity=severity,
        confidence=conf,
        mitre_technique_id=tech,
        mitre_technique_name="x",
        recommended_action=action,
        rationale="r",
    )


def _label(severity: Severity, tech: str, action: RecommendedAction) -> AlertLabel:
    return AlertLabel(severity=severity, mitre_technique_id=tech, recommended_action=action)


def test_score_computes_each_accuracy() -> None:
    outcomes = [
        TriageOutcome.success(
            _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
            iterations=2,
            latency_ms=1.0,
        ),
        TriageOutcome.success(
            _result("A2", Severity.MEDIUM, "T1110", RecommendedAction.MONITOR),
            iterations=2,
            latency_ms=1.0,
        ),
        TriageOutcome.success(
            _result("A3", Severity.LOW, "T1059.003", RecommendedAction.MONITOR),
            iterations=2,
            latency_ms=1.0,
        ),
        TriageOutcome.failure("A4", "boom", iterations=0, latency_ms=1.0),
    ]
    labels = {
        "A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),  # all correct
        "A2": _label(
            Severity.HIGH, "T1110", RecommendedAction.INVESTIGATE
        ),  # sev +-1, action wrong
        "A3": _label(Severity.LOW, "T1059.001", RecommendedAction.MONITOR),  # tech wrong, tactic ok
    }
    report = score(outcomes, labels, TACTIC)

    assert report.total == 4
    assert report.scored == 3
    assert report.errors == 1
    assert report.severity_exact_accuracy == pytest.approx(2 / 3)
    assert report.severity_within_one_accuracy == pytest.approx(1.0)
    assert report.mitre_technique_accuracy == pytest.approx(2 / 3)
    assert report.mitre_tactic_accuracy == pytest.approx(1.0)
    assert report.action_accuracy == pytest.approx(2 / 3)
    assert report.overall_accuracy == pytest.approx(2 / 3)
    assert report.mean_confidence == pytest.approx(0.8)
    assert report.severity_confusion["high"] == {"high": 1, "medium": 1}


def test_score_reports_within_one_action() -> None:
    outcomes = [
        TriageOutcome.success(
            _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
            iterations=1,
            latency_ms=1.0,
        ),
        TriageOutcome.success(
            _result("A2", Severity.HIGH, "T1059.001", RecommendedAction.ESCALATE),
            iterations=1,
            latency_ms=1.0,
        ),
    ]
    labels = {
        "A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.ESCALATE),  # 1 rung off
        "A2": _label(Severity.HIGH, "T1059.001", RecommendedAction.MONITOR),  # 2 rungs off
    }
    report = score(outcomes, labels, TACTIC)
    assert report.action_accuracy == pytest.approx(0.0)  # neither exact
    assert report.action_within_one_accuracy == pytest.approx(0.5)  # A1 within one, A2 not


def test_score_with_all_errors_is_zero() -> None:
    outcomes = [TriageOutcome.failure("A1", "x", iterations=0, latency_ms=1.0)]
    report = score(outcomes, {}, TACTIC)
    assert report.scored == 0
    assert report.errors == 1
    assert report.overall_accuracy == 0.0


class _OracleTriager:
    """A triager that returns the ground-truth label as its verdict (perfect run)."""

    def __init__(self, labels: dict[str, AlertLabel]) -> None:
        self._labels = labels

    async def run(self, alert: Alert) -> AgentRun:
        label = self._labels[alert.id]
        result = _result(
            alert.id, label.severity, label.mitre_technique_id, label.recommended_action, 0.9
        )
        return AgentRun(result=result, iterations=1)


async def test_run_eval_over_full_sample_set_with_oracle_scores_100() -> None:
    labeled = load_sample_alerts()
    labels = {item.alert.id: item.label for item in labeled}
    tactic_by_id = {technique.id: technique.tactic for technique in load_mitre_techniques()}

    report = await run_eval(
        labeled, _OracleTriager(labels), concurrency=4, tactic_by_id=tactic_by_id
    )

    assert report.scored == len(labeled)
    assert report.errors == 0
    assert report.severity_exact_accuracy == pytest.approx(1.0)
    assert report.mitre_technique_accuracy == pytest.approx(1.0)
    assert report.action_accuracy == pytest.approx(1.0)
    assert report.overall_accuracy == pytest.approx(1.0)


def test_write_headline_to_readme_replaces_marked_section(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- EVAL:START -->\nold\n<!-- EVAL:END -->\n\nfooter\n", encoding="utf-8"
    )
    report = score(
        [
            TriageOutcome.success(
                _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
                iterations=1,
                latency_ms=1.0,
            )
        ],
        {"A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN)},
        TACTIC,
    )
    write_headline_to_readme(report, readme)

    text = readme.read_text(encoding="utf-8")
    assert "Overall triage accuracy: 100.0%" in text
    assert "old" not in text
    assert text.startswith("# Title")
    assert "footer" in text


def test_write_headline_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here", encoding="utf-8")
    report = score([], {}, TACTIC)
    with pytest.raises(ValueError, match="markers"):
        write_headline_to_readme(report, readme)


def test_wilson_interval_known_values() -> None:
    from triagemcp.eval.metrics import wilson_interval

    low, high = wilson_interval(25, 38)
    assert low == pytest.approx(0.499, abs=0.01)
    assert high == pytest.approx(0.788, abs=0.01)


def test_wilson_interval_edges() -> None:
    from triagemcp.eval.metrics import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(38, 38)[1] == pytest.approx(1.0, abs=0.001)


def test_score_reports_confidence_intervals() -> None:
    outcomes = [
        TriageOutcome.success(
            _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
            iterations=1,
            latency_ms=1.0,
        )
    ]
    labels = {"A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN)}
    report = score(outcomes, labels, TACTIC)
    assert (
        report.mitre_technique_ci[0]
        <= report.mitre_technique_accuracy
        <= report.mitre_technique_ci[1]
    )
    assert report.severity_exact_ci[1] <= 1.0


def test_eval_reference_clock_anchors_just_after_newest_alert() -> None:
    labeled = load_sample_alerts()
    clock = eval_reference_clock(labeled)
    newest = max(item.alert.timestamp for item in labeled)
    assert clock() == newest + dt.timedelta(hours=1)
    assert clock() == clock()  # frozen: same value on every call


async def test_recurring_observable_is_in_default_window_under_eval_clock() -> None:
    labeled = load_sample_alerts()
    clock = eval_reference_clock(labeled)
    async with aiosqlite.connect(":memory:") as conn:
        registry = await build_default_registry(conn, clock=clock)
        # No explicit lookback_hours -> the DEFAULT 168h window. Under utcnow this returns 0
        # today; under the dataset-anchored clock it deterministically returns the prior sighting.
        result = await registry.dispatch(
            "query_recent_alerts", {"observable": "FIN-WS-118", "observable_type": "host"}
        )
    assert result.content["match_count"] >= 1
