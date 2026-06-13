"""Spec for report rendering: JSON, Markdown, and rich tables."""

from __future__ import annotations

import io
import json
import re

from rich.console import Console

from triagemcp.eval.metrics import EvalReport
from triagemcp.models import RecommendedAction, Severity, TriageOutcome, TriageResult
from triagemcp.report import (
    eval_report_table,
    outcomes_table,
    outcomes_to_json,
    outcomes_to_markdown,
)


def _ok(alert_id: str) -> TriageOutcome:
    return TriageOutcome.success(
        TriageResult(
            alert_id=alert_id,
            severity=Severity.HIGH,
            confidence=0.8,
            mitre_technique_id="T1110",
            mitre_technique_name="Brute Force",
            recommended_action=RecommendedAction.INVESTIGATE,
            rationale="r",
        ),
        iterations=2,
        latency_ms=12.3,
    )


def _fail(alert_id: str) -> TriageOutcome:
    return TriageOutcome.failure(alert_id, "AgentTimeoutError: boom", iterations=0, latency_ms=5.0)


def _render(renderable: object) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=200).print(renderable)
    return buf.getvalue()


def test_outcomes_to_json_is_valid_and_complete() -> None:
    data = json.loads(outcomes_to_json([_ok("A1"), _fail("A2")]))
    assert len(data) == 2
    assert data[0]["result"]["severity"] == "high"
    assert data[1]["result"] is None
    assert data[1]["error"].startswith("AgentTimeoutError")


def test_outcomes_to_markdown_has_header_rows_and_errors_section() -> None:
    md = outcomes_to_markdown([_ok("A1"), _fail("A2")])
    assert "| A1 |" in md
    assert "OK" in md
    assert "FAILED" in md
    assert "T1110" in md
    assert "### Errors" in md
    assert "AgentTimeoutError" in md


def test_outcomes_to_markdown_escapes_pipes_and_newlines() -> None:
    # An alert id / error text containing a pipe or newline must not break the table layout.
    bad = TriageOutcome.failure(
        "A|3\ninjected", "line1\nline2 | piped", iterations=0, latency_ms=1.0
    )
    md = outcomes_to_markdown([_ok("A1"), bad])
    # Every rendered table row has the same number of *delimiter* pipes (escaped \| don't count).
    table_rows = [line for line in md.splitlines() if line.startswith("| ")]
    delims = [len(re.findall(r"(?<!\\)\|", row)) for row in table_rows]
    assert len(set(delims)) == 1
    # Raw control/pipe characters are not emitted verbatim into a cell.
    assert "A|3" not in md
    assert "line1\nline2" not in md


def test_outcomes_table_has_one_row_per_outcome() -> None:
    table = outcomes_table([_ok("A1"), _ok("A2")])
    assert table.row_count == 2
    assert "A1" in _render(table)


def test_eval_report_table_shows_overall_percentage() -> None:
    report = EvalReport(
        total=1,
        scored=1,
        errors=0,
        severity_exact_accuracy=1.0,
        severity_within_one_accuracy=1.0,
        mitre_technique_accuracy=1.0,
        mitre_tactic_accuracy=1.0,
        action_accuracy=1.0,
        overall_accuracy=1.0,
        mean_confidence=0.9,
    )
    out = _render(eval_report_table(report))
    assert "Overall" in out
    assert "100.0%" in out


def test_eval_report_table_shows_mitre_ci() -> None:
    report = EvalReport(
        total=38,
        scored=38,
        errors=0,
        severity_exact_accuracy=0.5,
        severity_within_one_accuracy=1.0,
        mitre_technique_accuracy=0.658,
        mitre_tactic_accuracy=0.737,
        action_accuracy=0.447,
        overall_accuracy=0.535,
        mean_confidence=0.9,
        mitre_technique_ci=(0.50, 0.79),
    )
    out = _render(eval_report_table(report))
    assert "50.0" in out
    assert "79.0" in out


def test_eval_report_table_shows_action_within_one_and_ece() -> None:
    report = EvalReport(
        total=1,
        scored=1,
        errors=0,
        severity_exact_accuracy=1.0,
        severity_within_one_accuracy=1.0,
        mitre_technique_accuracy=1.0,
        mitre_tactic_accuracy=1.0,
        action_accuracy=1.0,
        action_within_one_accuracy=1.0,
        overall_accuracy=1.0,
        mean_confidence=0.9,
        ece=0.25,
    )
    out = _render(eval_report_table(report))
    assert "Action within one" in out
    assert "ECE" in out
    assert "0.25" in out
