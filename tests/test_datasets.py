"""The bundled datasets load, validate, and stay internally consistent."""

from __future__ import annotations

import json

from triagemcp.datasets import load_sample_alerts, read_data_text
from triagemcp.models import RecommendedAction, Severity


def test_sample_alerts_load_and_validate() -> None:
    alerts = load_sample_alerts()
    assert 30 <= len(alerts) <= 50


def test_sample_alert_ids_are_unique() -> None:
    ids = [labeled.alert.id for labeled in load_sample_alerts()]
    assert len(ids) == len(set(ids))


def test_every_label_technique_exists_in_mitre_dataset() -> None:
    known_ids = {entry["id"] for entry in json.loads(read_data_text("mitre_techniques.json"))}
    used = {labeled.label.mitre_technique_id for labeled in load_sample_alerts()}
    assert used <= known_ids, f"labels reference unknown techniques: {sorted(used - known_ids)}"


def test_sample_alerts_span_all_severities() -> None:
    severities = {labeled.label.severity for labeled in load_sample_alerts()}
    assert severities == set(Severity)


def test_sample_alerts_span_most_actions() -> None:
    actions = {labeled.label.recommended_action for labeled in load_sample_alerts()}
    assert len(actions) >= 4
    assert RecommendedAction.CLOSE_FALSE_POSITIVE in actions
