"""Spec for the experiment runner (offline; fake triager via patched build_runtime)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import pytest

from triagemcp.agent.loop import AgentRun
from triagemcp.config import Settings
from triagemcp.eval.experiments import PROMPT_VARIANTS, run_experiment
from triagemcp.models import Alert, AlertLabel, TriageResult


class _LabelEchoTriager:
    def __init__(self, labels: dict[str, AlertLabel]) -> None:
        self._labels = labels

    async def run(self, alert: Alert) -> AgentRun:
        lab = self._labels[alert.id]
        result = TriageResult(
            alert_id=alert.id,
            severity=lab.severity,
            confidence=0.9,
            mitre_technique_id=lab.mitre_technique_id,
            mitre_technique_name="x",
            recommended_action=lab.recommended_action,
            rationale="echo",
        )
        return AgentRun(result=result, iterations=1)


def test_prompt_variants_cover_the_ladder() -> None:
    assert set(PROMPT_VARIANTS) == {"baseline", "catalog", "catalog+map", "catalog+map+fewshot"}


async def test_run_experiment_scores_against_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from triagemcp.datasets import load_sample_alerts

    labels = {la.alert.id: la.label for la in load_sample_alerts()}

    @contextlib.asynccontextmanager
    async def _fake_runtime(
        _settings: Settings, **_kwargs: object
    ) -> AsyncIterator[_LabelEchoTriager]:
        yield _LabelEchoTriager(labels)

    monkeypatch.setattr("triagemcp.eval.experiments.build_runtime", _fake_runtime)
    report = await run_experiment(
        "catalog", settings=Settings(), model="claude-haiku-4-5", concurrency=2
    )
    assert report.scored == len(labels)
    assert report.mitre_technique_accuracy == pytest.approx(1.0)
