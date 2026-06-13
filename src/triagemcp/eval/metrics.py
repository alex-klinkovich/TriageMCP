"""Score triage outcomes against ground-truth labels.

Accuracy is computed only over alerts that produced a verdict; alerts that errored are
counted separately so a high accuracy can never hide a high failure rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from triagemcp.models import AlertLabel, TriageOutcome, TriageResult


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (analytic, no sampling)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


class EvalReport(BaseModel):
    """The computed evaluation metrics."""

    model_config = ConfigDict(extra="forbid")

    total: int
    scored: int
    errors: int
    severity_exact_accuracy: float
    severity_within_one_accuracy: float
    mitre_technique_accuracy: float
    mitre_tactic_accuracy: float
    action_accuracy: float
    action_within_one_accuracy: float = 0.0
    overall_accuracy: float
    mean_confidence: float
    severity_exact_ci: tuple[float, float] = (0.0, 0.0)
    mitre_technique_ci: tuple[float, float] = (0.0, 0.0)
    action_ci: tuple[float, float] = (0.0, 0.0)
    severity_confusion: dict[str, dict[str, int]] = Field(default_factory=dict)

    def summary_line(self) -> str:
        """One-line headline suitable for the README."""
        return (
            f"**Overall triage accuracy: {self.overall_accuracy:.1%}** "
            f"(severity {self.severity_exact_accuracy:.1%} exact / "
            f"{self.severity_within_one_accuracy:.1%} within one level, "
            f"MITRE technique {self.mitre_technique_accuracy:.1%}, "
            f"action {self.action_accuracy:.1%}) over {self.scored} scored alerts "
            f"with {self.errors} errors."
        )


def _tactic(tactic_by_id: Mapping[str, str], technique_id: str) -> str:
    # Fall back to the id itself so two unknown techniques never spuriously "match" tactics.
    return tactic_by_id.get(technique_id, technique_id)


def score(
    outcomes: Sequence[TriageOutcome],
    labels: Mapping[str, AlertLabel],
    tactic_by_id: Mapping[str, str],
) -> EvalReport:
    """Compare ``outcomes`` to ``labels`` and produce an :class:`EvalReport`."""
    pairs: list[tuple[TriageResult, AlertLabel]] = [
        (outcome.result, labels[outcome.alert_id])
        for outcome in outcomes
        if outcome.result is not None
    ]
    total = len(outcomes)
    scored = len(pairs)
    errors = total - scored

    if scored == 0:
        return EvalReport(
            total=total,
            scored=0,
            errors=errors,
            severity_exact_accuracy=0.0,
            severity_within_one_accuracy=0.0,
            mitre_technique_accuracy=0.0,
            mitre_tactic_accuracy=0.0,
            action_accuracy=0.0,
            action_within_one_accuracy=0.0,
            overall_accuracy=0.0,
            mean_confidence=0.0,
        )

    sev_exact_n = sum(pred.severity == lab.severity for pred, lab in pairs)
    tech_n = sum(pred.mitre_technique_id == lab.mitre_technique_id for pred, lab in pairs)
    action_n = sum(pred.recommended_action == lab.recommended_action for pred, lab in pairs)
    sev_exact = sev_exact_n / scored
    sev_within = sum(pred.severity.distance(lab.severity) <= 1 for pred, lab in pairs) / scored
    technique = tech_n / scored
    tactic = (
        sum(
            _tactic(tactic_by_id, pred.mitre_technique_id)
            == _tactic(tactic_by_id, lab.mitre_technique_id)
            for pred, lab in pairs
        )
        / scored
    )
    action = action_n / scored
    action_within = (
        sum(pred.recommended_action.distance(lab.recommended_action) <= 1 for pred, lab in pairs)
        / scored
    )
    mean_conf = sum(pred.confidence for pred, _ in pairs) / scored
    overall = (sev_exact + technique + action) / 3

    confusion: dict[str, dict[str, int]] = {}
    for pred, lab in pairs:
        row = confusion.setdefault(lab.severity.value, {})
        row[pred.severity.value] = row.get(pred.severity.value, 0) + 1

    return EvalReport(
        total=total,
        scored=scored,
        errors=errors,
        severity_exact_accuracy=sev_exact,
        severity_within_one_accuracy=sev_within,
        mitre_technique_accuracy=technique,
        mitre_tactic_accuracy=tactic,
        action_accuracy=action,
        action_within_one_accuracy=action_within,
        overall_accuracy=overall,
        mean_confidence=mean_conf,
        severity_exact_ci=wilson_interval(sev_exact_n, scored),
        mitre_technique_ci=wilson_interval(tech_n, scored),
        action_ci=wilson_interval(action_n, scored),
        severity_confusion=confusion,
    )
