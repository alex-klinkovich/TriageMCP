"""Self-consistency voting: aggregate N stochastic verdicts into one by per-field majority.

Pure and deterministic given the samples. Ordinal fields (severity, action) break ties toward the
more severe / more escalated rung (their enum definition order); the nominal technique breaks ties
toward the earliest sample. Non-voted display fields come from a representative sample.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

from triagemcp.agent.loop import AgentRun
from triagemcp.models import Alert, RecommendedAction, Severity, TriageResult
from triagemcp.pipeline import Triager


def _vote_mode[E: Hashable](values: Sequence[E], order: Sequence[E] | None = None) -> E:
    counts = Counter(values)
    top = max(counts.values())
    if order is None:
        # nominal: tie -> the earliest value in sample order
        return next(value for value in values if counts[value] == top)
    tied = [value for value, count in counts.items() if count == top]
    return max(tied, key=order.index)  # ordinal: tie -> later in order (more severe / escalated)


def majority_vote(results: Sequence[TriageResult]) -> TriageResult:
    """Combine N per-alert verdicts into one by per-field majority (see module docstring)."""
    if not results:
        raise ValueError("majority_vote requires at least one result")

    severity = _vote_mode([r.severity for r in results], list(Severity))
    action = _vote_mode([r.recommended_action for r in results], list(RecommendedAction))
    technique = _vote_mode([r.mitre_technique_id for r in results])

    def matches(result: TriageResult) -> int:
        return (
            (result.severity == severity)
            + (result.mitre_technique_id == technique)
            + (result.recommended_action == action)
        )

    representative = max(results, key=lambda r: (matches(r), r.confidence))
    name = next(r.mitre_technique_name for r in results if r.mitre_technique_id == technique)
    return TriageResult(
        alert_id=results[0].alert_id,
        severity=severity,
        confidence=sum(r.confidence for r in results) / len(results),
        mitre_technique_id=technique,
        mitre_technique_name=name,
        recommended_action=action,
        rationale=representative.rationale,
        supporting_observations=representative.supporting_observations,
    )


class VotingTriager:
    """Sample a base triager N times per alert and majority-vote the verdicts (temperature > 0)."""

    def __init__(self, base: Triager, samples: int) -> None:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self._base = base
        self._samples = samples

    async def run(self, alert: Alert) -> AgentRun:
        runs: list[AgentRun] = []
        last_error: Exception | None = None
        for _ in range(self._samples):
            try:
                runs.append(await self._base.run(alert))
            except Exception as exc:  # a failed sample must not sink the vote
                last_error = exc
        if not runs:
            assert last_error is not None  # samples >= 1, so a failure was recorded
            raise last_error
        voted = majority_vote([run.result for run in runs])
        return AgentRun(result=voted, iterations=sum(run.iterations for run in runs))
