"""Spec for self-consistency majority voting (pure, offline)."""

from __future__ import annotations

import datetime as dt

import pytest

from triagemcp.agent.loop import AgentRun
from triagemcp.agent.voting import VotingTriager, majority_vote
from triagemcp.models import Alert, RecommendedAction, Severity, TriageResult


def _result(
    severity: Severity,
    technique: str,
    action: RecommendedAction,
    *,
    conf: float = 0.8,
    name: str = "x",
    rationale: str = "r",
) -> TriageResult:
    return TriageResult(
        alert_id="A-1",
        severity=severity,
        confidence=conf,
        mitre_technique_id=technique,
        mitre_technique_name=name,
        recommended_action=action,
        rationale=rationale,
    )


def test_majority_vote_takes_the_modal_field_values() -> None:
    results = [
        _result(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
        _result(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
        _result(Severity.LOW, "T1110", RecommendedAction.MONITOR),
    ]
    voted = majority_vote(results)
    assert voted.severity is Severity.HIGH
    assert voted.mitre_technique_id == "T1059.001"
    assert voted.recommended_action is RecommendedAction.CONTAIN


def test_unanimous_returns_that_verdict() -> None:
    r = _result(Severity.MEDIUM, "T1110", RecommendedAction.INVESTIGATE)
    voted = majority_vote([r, r, r])
    assert voted.severity is Severity.MEDIUM
    assert voted.mitre_technique_id == "T1110"
    assert voted.recommended_action is RecommendedAction.INVESTIGATE


def test_ordinal_tie_breaks_to_the_more_severe_rung() -> None:
    results = [
        _result(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
        _result(Severity.CRITICAL, "T1059.001", RecommendedAction.ESCALATE),
    ]
    voted = majority_vote(results)
    assert voted.severity is Severity.CRITICAL  # 1-1 tie -> the more severe rung
    assert (
        voted.recommended_action is RecommendedAction.ESCALATE
    )  # tie -> the more escalated action


def test_technique_tie_breaks_to_the_earliest_sample() -> None:
    results = [
        _result(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
        _result(Severity.HIGH, "T1110", RecommendedAction.CONTAIN),
    ]
    voted = majority_vote(results)
    assert voted.mitre_technique_id == "T1059.001"  # 1-1 tie -> the earliest sample


def test_voted_verdict_fills_confidence_name_and_rationale() -> None:
    results = [
        _result(
            Severity.HIGH,
            "T1059.001",
            RecommendedAction.CONTAIN,
            conf=0.9,
            name="PowerShell",
            rationale="from sample 1",
        ),
        _result(
            Severity.HIGH,
            "T1059.001",
            RecommendedAction.CONTAIN,
            conf=0.7,
            name="PowerShell",
            rationale="from sample 2",
        ),
        _result(
            Severity.LOW,
            "T1110",
            RecommendedAction.MONITOR,
            conf=0.5,
            name="Brute Force",
            rationale="from sample 3",
        ),
    ]
    voted = majority_vote(results)
    assert voted.confidence == pytest.approx((0.9 + 0.7 + 0.5) / 3)  # mean of the samples
    assert voted.mitre_technique_name == "PowerShell"  # the name for the voted technique
    assert voted.rationale == "from sample 1"  # representative: full-match, highest confidence


def test_empty_results_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        majority_vote([])


def _alert() -> Alert:
    return Alert(
        id="A-1",
        title="Encoded PowerShell",
        description="powershell -enc",
        source="EDR",
        severity_reported=Severity.MEDIUM,
        timestamp=dt.datetime(2026, 5, 28, 12, 0, tzinfo=dt.UTC),
    )


class _ScriptedTriager:
    """A base triager that replays scripted AgentRuns (or raises a scripted exception)."""

    def __init__(self, runs: list[AgentRun | Exception]) -> None:
        self._runs = runs
        self._i = 0

    async def run(self, alert: Alert) -> AgentRun:
        run = self._runs[self._i]
        self._i += 1
        if isinstance(run, Exception):
            raise run
        return run


def _run(severity: Severity, technique: str, action: RecommendedAction, *, iters: int) -> AgentRun:
    return AgentRun(result=_result(severity, technique, action), iterations=iters)


async def test_voting_triager_samples_n_times_and_votes() -> None:
    base = _ScriptedTriager(
        [
            _run(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN, iters=3),
            _run(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN, iters=3),
            _run(Severity.LOW, "T1110", RecommendedAction.MONITOR, iters=2),
        ]
    )
    out = await VotingTriager(base, samples=3).run(_alert())
    assert out.result.severity is Severity.HIGH  # voted
    assert out.iterations == 8  # sum across the three samples


async def test_voting_excludes_a_failed_sample() -> None:
    base = _ScriptedTriager(
        [
            _run(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN, iters=3),
            RuntimeError("sample 2 failed"),
            _run(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN, iters=3),
        ]
    )
    out = await VotingTriager(base, samples=3).run(_alert())
    assert out.result.severity is Severity.HIGH  # voted over the two successes
    assert out.iterations == 6


async def test_voting_all_samples_fail_propagates() -> None:
    base = _ScriptedTriager([RuntimeError("a"), RuntimeError("b")])
    with pytest.raises(RuntimeError, match="b"):
        await VotingTriager(base, samples=2).run(_alert())
