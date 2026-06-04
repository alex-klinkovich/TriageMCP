"""Render triage outcomes and eval reports as JSON, Markdown, or rich tables."""

from __future__ import annotations

import json
from collections.abc import Sequence

from rich.table import Table

from triagemcp.eval.metrics import EvalReport
from triagemcp.models import TriageOutcome

_COLUMNS = ("Alert", "Status", "Severity", "Confidence", "MITRE", "Action", "Iters", "Latency (ms)")


def _row(outcome: TriageOutcome) -> list[str]:
    if outcome.result is not None:
        result = outcome.result
        return [
            outcome.alert_id,
            "OK",
            result.severity.value,
            f"{result.confidence:.2f}",
            result.mitre_technique_id,
            result.recommended_action.value,
            str(outcome.iterations),
            f"{outcome.latency_ms:.1f}",
        ]
    return [
        outcome.alert_id,
        "FAILED",
        "-",
        "-",
        "-",
        "-",
        str(outcome.iterations),
        f"{outcome.latency_ms:.1f}",
    ]


def outcomes_to_json(outcomes: Sequence[TriageOutcome]) -> str:
    """Serialize outcomes as a pretty JSON array (full result/error per alert)."""
    return json.dumps([outcome.model_dump(mode="json") for outcome in outcomes], indent=2)


def outcomes_to_markdown(outcomes: Sequence[TriageOutcome]) -> str:
    """Render outcomes as a Markdown table, with an Errors section for failures."""
    lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _COLUMNS) + " |",
        *("| " + " | ".join(_row(outcome)) + " |" for outcome in outcomes),
    ]
    errors = [outcome for outcome in outcomes if outcome.error is not None]
    if errors:
        lines.append("")
        lines.append("### Errors")
        lines.extend(f"- **{outcome.alert_id}**: {outcome.error}" for outcome in errors)
    return "\n".join(lines)


def outcomes_table(outcomes: Sequence[TriageOutcome]) -> Table:
    """Build a rich table of outcomes (failed rows highlighted)."""
    table = Table(title="Triage results")
    for column in _COLUMNS:
        table.add_column(column)
    for outcome in outcomes:
        style = None if outcome.result is not None else "red"
        table.add_row(*_row(outcome), style=style)
    return table


def eval_report_table(report: EvalReport) -> Table:
    """Build a rich table summarizing an :class:`EvalReport`."""
    table = Table(title="Evaluation report")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = (
        ("Scored / errors", f"{report.scored} / {report.errors}"),
        ("Severity exact", f"{report.severity_exact_accuracy:.1%}"),
        ("Severity within one", f"{report.severity_within_one_accuracy:.1%}"),
        (
            "MITRE technique",
            f"{report.mitre_technique_accuracy:.1%} "
            f"[{report.mitre_technique_ci[0]:.1%}-{report.mitre_technique_ci[1]:.1%}]",
        ),
        ("MITRE tactic", f"{report.mitre_tactic_accuracy:.1%}"),
        ("Action", f"{report.action_accuracy:.1%}"),
        ("Overall", f"{report.overall_accuracy:.1%}"),
        ("Mean confidence", f"{report.mean_confidence:.2f}"),
    )
    for name, value in rows:
        table.add_row(name, value)
    return table
