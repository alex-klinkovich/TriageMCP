"""Spec for the async batch pipeline: isolation, order preservation, concurrency cap."""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from triagemcp.agent.loop import AgentRun
from triagemcp.models import Alert, RecommendedAction, Severity, TriageResult
from triagemcp.pipeline import triage_batch


def _alert(alert_id: str) -> Alert:
    return Alert(
        id=alert_id,
        title="t",
        description="d",
        source="EDR",
        severity_reported=Severity.LOW,
        timestamp=dt.datetime(2026, 5, 28, 12, 0, tzinfo=dt.UTC),
    )


def _result(alert_id: str) -> TriageResult:
    return TriageResult(
        alert_id=alert_id,
        severity=Severity.MEDIUM,
        confidence=0.7,
        mitre_technique_id="T1110",
        mitre_technique_name="Brute Force",
        recommended_action=RecommendedAction.INVESTIGATE,
        rationale="ok",
    )


class _RecordingTriager:
    """Fake Triager that records concurrency and can fail selected alert ids."""

    def __init__(self, *, fail_ids: frozenset[str] = frozenset(), delay: float = 0.0) -> None:
        self._fail_ids = fail_ids
        self._delay = delay
        self.active = 0
        self.max_active = 0

    async def run(self, alert: Alert) -> AgentRun:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if alert.id in self._fail_ids:
                raise RuntimeError(f"boom for {alert.id}")
            return AgentRun(result=_result(alert.id), iterations=2)
        finally:
            self.active -= 1


async def test_returns_one_outcome_per_alert_in_input_order() -> None:
    alerts = [_alert(f"A-{i}") for i in range(5)]
    outcomes = await triage_batch(alerts, _RecordingTriager(), concurrency=3)
    assert [o.alert_id for o in outcomes] == [a.id for a in alerts]
    assert all(o.result is not None and o.error is None for o in outcomes)


async def test_one_failing_alert_is_isolated() -> None:
    alerts = [_alert("A-0"), _alert("A-1"), _alert("A-2")]
    triager = _RecordingTriager(fail_ids=frozenset({"A-1"}))
    outcomes = await triage_batch(alerts, triager, concurrency=2)

    by_id = {o.alert_id: o for o in outcomes}
    assert by_id["A-0"].result is not None
    assert by_id["A-2"].result is not None
    assert by_id["A-1"].result is None
    assert by_id["A-1"].error is not None
    assert "boom for A-1" in by_id["A-1"].error


async def test_failure_in_the_middle_preserves_input_order() -> None:
    # The load-bearing guarantee: a mid-batch failure (which completes out of order) must not
    # shuffle the outcomes. Assert positional order AND isolation together.
    alerts = [_alert(f"A-{i}") for i in range(5)]
    triager = _RecordingTriager(fail_ids=frozenset({"A-2"}))
    outcomes = await triage_batch(alerts, triager, concurrency=3)
    assert [o.alert_id for o in outcomes] == [a.id for a in alerts]
    assert outcomes[2].error is not None and outcomes[2].result is None
    assert all(outcomes[i].result is not None for i in (0, 1, 3, 4))


async def test_concurrency_is_capped_by_the_semaphore() -> None:
    alerts = [_alert(f"A-{i}") for i in range(8)]
    triager = _RecordingTriager(delay=0.02)
    await triage_batch(alerts, triager, concurrency=2)
    assert triager.max_active <= 2


async def test_success_outcome_carries_iterations() -> None:
    outcomes = await triage_batch([_alert("A-9")], _RecordingTriager(), concurrency=1)
    assert outcomes[0].iterations == 2
    assert outcomes[0].latency_ms >= 0.0


async def test_empty_batch_returns_empty_list() -> None:
    assert await triage_batch([], _RecordingTriager(), concurrency=4) == []


async def test_invalid_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        await triage_batch([_alert("A-0")], _RecordingTriager(), concurrency=0)
