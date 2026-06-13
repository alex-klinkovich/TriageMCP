"""Spec for leave-one-out cross-validated variant selection (pure, offline)."""

from __future__ import annotations

import pytest

from triagemcp.eval.crossval import cross_validate
from triagemcp.models import AlertLabel, RecommendedAction, Severity, TriageResult

TACTIC = {"T1059.001": "Execution", "T1110": "Credential Access"}


def _label() -> AlertLabel:
    return AlertLabel(
        severity=Severity.HIGH,
        mitre_technique_id="T1059.001",
        recommended_action=RecommendedAction.CONTAIN,
    )


def _verdict(alert_id: str, *, correct: bool) -> TriageResult:
    if correct:
        return TriageResult(
            alert_id=alert_id,
            severity=Severity.HIGH,
            confidence=0.9,
            mitre_technique_id="T1059.001",
            mitre_technique_name="x",
            recommended_action=RecommendedAction.CONTAIN,
            rationale="r",
        )
    return TriageResult(
        alert_id=alert_id,
        severity=Severity.LOW,
        confidence=0.9,
        mitre_technique_id="T1110",
        mitre_technique_name="x",
        recommended_action=RecommendedAction.MONITOR,
        rationale="r",
    )


def test_loo_detects_a_non_generalizing_variant_choice() -> None:
    # va is correct on A1,A2 and wrong on A3,A4; vb is the mirror. In-sample they tie at 0.5, but
    # leave-one-out the winning variant flips per fold and is always wrong on the held-out alert.
    labels = {f"A{i}": _label() for i in range(1, 5)}
    va = {f"A{i}": _verdict(f"A{i}", correct=i in (1, 2)) for i in range(1, 5)}
    vb = {f"A{i}": _verdict(f"A{i}", correct=i in (3, 4)) for i in range(1, 5)}

    report = cross_validate({"va": va, "vb": vb}, labels, TACTIC)

    assert report.in_sample_best_variant == "va"  # tie -> alphabetical
    assert report.in_sample_best.overall_accuracy == pytest.approx(0.5)
    assert report.out_of_fold.overall_accuracy == pytest.approx(0.0)
    assert report.selection_optimism == pytest.approx(0.5)
    assert set(report.selected_variant_by_alert.values()) == {"va", "vb"}
    assert report.out_of_fold.scored == 4


def test_loo_falls_back_when_selected_variant_errored_on_held_out() -> None:
    labels = {f"A{i}": _label() for i in range(1, 4)}
    # va wins selection everywhere but has NO verdict for A1; vb covers A1.
    va = {"A2": _verdict("A2", correct=True), "A3": _verdict("A3", correct=True)}
    vb = {f"A{i}": _verdict(f"A{i}", correct=True) for i in range(1, 4)}
    report = cross_validate({"va": va, "vb": vb}, labels, TACTIC)
    assert report.out_of_fold.scored == 3  # A1 falls back to vb, not dropped
    assert report.selected_variant_by_alert["A1"] == "vb"
