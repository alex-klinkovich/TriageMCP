"""Drive the agent over a labeled set, score it, and record the headline in the README."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from triagemcp.eval.metrics import EvalReport, score
from triagemcp.models import LabeledAlert
from triagemcp.pipeline import Triager, triage_batch

_MARKER = re.compile(r"(<!-- EVAL:START -->)(.*?)(<!-- EVAL:END -->)", re.DOTALL)


async def run_eval(
    labeled: Sequence[LabeledAlert],
    triager: Triager,
    *,
    concurrency: int,
    tactic_by_id: Mapping[str, str],
) -> EvalReport:
    """Triage every labeled alert and score the results against the labels."""
    alerts = [item.alert for item in labeled]
    outcomes = await triage_batch(alerts, triager, concurrency=concurrency)
    labels = {item.alert.id: item.label for item in labeled}
    return score(outcomes, labels, tactic_by_id)


def write_headline_to_readme(report: EvalReport, readme_path: Path) -> None:
    """Replace the content between the README's EVAL markers with the headline line."""
    text = readme_path.read_text(encoding="utf-8")
    if not _MARKER.search(text):
        raise ValueError("README is missing the <!-- EVAL:START -->/<!-- EVAL:END --> markers")

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{report.summary_line()}\n{match.group(3)}"

    readme_path.write_text(_MARKER.sub(_replace, text), encoding="utf-8")
