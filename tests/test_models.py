"""Behavioral spec for the Pydantic data models."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from triagemcp.models import (
    Alert,
    AlertLabel,
    LabeledAlert,
    Observables,
    RecommendedAction,
    Severity,
    TriageOutcome,
    TriageResult,
)

AWARE = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)


def _alert_kwargs() -> dict[str, object]:
    return {
        "id": "A-1",
        "title": "Suspicious PowerShell",
        "description": "Encoded command observed.",
        "source": "EDR",
        "severity_reported": Severity.MEDIUM,
        "timestamp": AWARE,
    }


def _valid_alert(**over: object) -> Alert:
    return Alert.model_validate({**_alert_kwargs(), **over})


def _result_kwargs() -> dict[str, object]:
    return {
        "alert_id": "A-1",
        "severity": Severity.HIGH,
        "confidence": 0.9,
        "mitre_technique_id": "T1059.001",
        "mitre_technique_name": "PowerShell",
        "recommended_action": RecommendedAction.ESCALATE,
        "rationale": "Encoded PowerShell launched from Office; matches known TTP.",
    }


# --- Severity ---------------------------------------------------------------


def test_severity_ranks_informational_to_critical() -> None:
    assert [s.level for s in Severity] == [0, 1, 2, 3, 4]


def test_severity_distance_is_absolute_rank_gap() -> None:
    assert Severity.CRITICAL.distance(Severity.HIGH) == 1
    assert Severity.INFORMATIONAL.distance(Severity.CRITICAL) == 4
    assert Severity.MEDIUM.distance(Severity.MEDIUM) == 0


def test_severity_parses_from_lowercase_value() -> None:
    assert Severity("high") is Severity.HIGH


# --- RecommendedAction ------------------------------------------------------


def test_recommended_action_value_set() -> None:
    assert RecommendedAction("escalate") is RecommendedAction.ESCALATE
    assert {a.value for a in RecommendedAction} == {
        "close_false_positive",
        "monitor",
        "investigate",
        "contain",
        "escalate",
    }


# --- Observables ------------------------------------------------------------


def test_observables_default_to_empty_lists() -> None:
    obs = Observables()
    assert obs.ips == []
    assert obs.file_hashes == []


def test_observables_coerce_ip_strings() -> None:
    obs = Observables.model_validate({"ips": ["10.0.0.1", "2001:db8::1"]})
    assert str(obs.ips[0]) == "10.0.0.1"


def test_observables_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Observables(unknown=["x"])  # type: ignore[call-arg]


# --- Alert ------------------------------------------------------------------


def test_alert_is_immutable() -> None:
    alert = _valid_alert()
    with pytest.raises(ValidationError):
        alert.title = "changed"  # type: ignore[misc]


def test_alert_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        _valid_alert(timestamp=dt.datetime(2026, 1, 1, 0, 0, 0))


def test_alert_rejects_blank_required_string() -> None:
    with pytest.raises(ValidationError):
        _valid_alert(id="")


def test_alert_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        Alert.model_validate({**_alert_kwargs(), "weird": 1})


# --- TriageResult -----------------------------------------------------------


def test_triage_result_accepts_valid_technique_ids() -> None:
    for tid in ("T1059", "T1059.001", "T1110"):
        result = TriageResult.model_validate({**_result_kwargs(), "mitre_technique_id": tid})
        assert result.mitre_technique_id == tid


@pytest.mark.parametrize("bad", ["1059", "T123", "T10590", "T1059.1", "t1059", "foo", ""])
def test_triage_result_rejects_malformed_technique_id(bad: str) -> None:
    with pytest.raises(ValidationError):
        TriageResult.model_validate({**_result_kwargs(), "mitre_technique_id": bad})


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_triage_result_rejects_out_of_range_confidence(bad: float) -> None:
    with pytest.raises(ValidationError):
        TriageResult.model_validate({**_result_kwargs(), "confidence": bad})


def test_triage_result_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        TriageResult.model_validate({**_result_kwargs(), "nope": 1})


def test_triage_result_rejects_empty_rationale() -> None:
    with pytest.raises(ValidationError):
        TriageResult.model_validate({**_result_kwargs(), "rationale": ""})


# --- AlertLabel / LabeledAlert ---------------------------------------------


def test_labeled_alert_round_trips_through_json() -> None:
    labeled = LabeledAlert(
        alert=_valid_alert(),
        label=AlertLabel(
            severity=Severity.HIGH,
            mitre_technique_id="T1110",
            recommended_action=RecommendedAction.INVESTIGATE,
        ),
    )
    again = LabeledAlert.model_validate_json(labeled.model_dump_json())
    assert again == labeled


def test_alert_label_rejects_bad_mitre_id() -> None:
    with pytest.raises(ValidationError):
        AlertLabel(
            severity=Severity.LOW,
            mitre_technique_id="nope",
            recommended_action=RecommendedAction.MONITOR,
        )


# --- TriageOutcome ----------------------------------------------------------


def test_outcome_success_factory_sets_result_only() -> None:
    result = TriageResult.model_validate(_result_kwargs())
    outcome = TriageOutcome.success(result, iterations=3, latency_ms=12.5)
    assert outcome.result is result
    assert outcome.error is None
    assert outcome.alert_id == result.alert_id


def test_outcome_failure_factory_sets_error_only() -> None:
    outcome = TriageOutcome.failure("A-9", "boom", iterations=2, latency_ms=5.0)
    assert outcome.error == "boom"
    assert outcome.result is None
    assert outcome.alert_id == "A-9"


def test_outcome_rejects_both_result_and_error() -> None:
    result = TriageResult.model_validate(_result_kwargs())
    with pytest.raises(ValidationError):
        TriageOutcome(alert_id="A-1", result=result, error="x", iterations=1, latency_ms=1.0)


def test_outcome_rejects_neither_result_nor_error() -> None:
    with pytest.raises(ValidationError):
        TriageOutcome(alert_id="A-1", iterations=1, latency_ms=1.0)
