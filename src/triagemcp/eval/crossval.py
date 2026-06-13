"""Leave-one-out cross-validated prompt-variant selection over persisted per-alert verdicts.

Selection on a subset counts a variant's errored (missing) alerts as incorrect, so a variant is
never rewarded for skipping hard alerts. The pooled out-of-fold and the in-sample per-variant
reports use the normal scoring (errors excluded), consistent with the rest of the eval.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict

from triagemcp.agent.prompts import build_system_prompt
from triagemcp.config import Settings
from triagemcp.datasets import load_sample_alerts
from triagemcp.eval.experiments import PROMPT_VARIANTS
from triagemcp.eval.harness import eval_reference_clock
from triagemcp.eval.metrics import EvalReport, score
from triagemcp.models import AlertLabel, TriageOutcome, TriageResult
from triagemcp.pipeline import triage_batch
from triagemcp.server import build_runtime


class CrossValReport(BaseModel):
    """Out-of-fold accuracy plus the in-sample best variant and the selection-optimism gap."""

    model_config = ConfigDict(extra="forbid")

    out_of_fold: EvalReport
    in_sample_best_variant: str
    in_sample_best: EvalReport
    selection_optimism: float
    selected_variant_by_alert: dict[str, str]


def _per_alert_overall(verdict: TriageResult | None, label: AlertLabel) -> float:
    if verdict is None:
        return 0.0
    sev = verdict.severity == label.severity
    tech = verdict.mitre_technique_id == label.mitre_technique_id
    act = verdict.recommended_action == label.recommended_action
    return (sev + tech + act) / 3


def _selection_score(
    verdicts: Mapping[str, TriageResult], labels: Mapping[str, AlertLabel], alert_ids: list[str]
) -> float:
    if not alert_ids:
        return 0.0
    return sum(_per_alert_overall(verdicts.get(aid), labels[aid]) for aid in alert_ids) / len(
        alert_ids
    )


def _outcomes_for(
    picks: Mapping[str, TriageResult | None], labels: Mapping[str, AlertLabel]
) -> list[TriageOutcome]:
    outcomes: list[TriageOutcome] = []
    for aid in labels:
        verdict = picks.get(aid)
        if verdict is None:
            outcomes.append(TriageOutcome.failure(aid, "no verdict", iterations=0, latency_ms=0.0))
        else:
            outcomes.append(TriageOutcome.success(verdict, iterations=0, latency_ms=0.0))
    return outcomes


def cross_validate(
    verdicts_by_variant: Mapping[str, Mapping[str, TriageResult]],
    labels: Mapping[str, AlertLabel],
    tactic_by_id: Mapping[str, str],
) -> CrossValReport:
    alert_ids = sorted(labels)
    variants = sorted(verdicts_by_variant)

    in_sample: dict[str, EvalReport] = {}
    for variant in variants:
        picks: dict[str, TriageResult | None] = {
            aid: verdicts_by_variant[variant].get(aid) for aid in alert_ids
        }
        in_sample[variant] = score(_outcomes_for(picks, labels), labels, tactic_by_id)
    # max() returns the first element achieving the max in iteration order; variants is sorted,
    # so ties go to the alphabetically-first variant (deterministic).
    best_variant = max(variants, key=lambda v: in_sample[v].overall_accuracy)

    selected_by_alert: dict[str, str] = {}
    of_picks: dict[str, TriageResult | None] = {}
    for held in alert_ids:
        train_ids = [aid for aid in alert_ids if aid != held]
        # Precompute scores, then sort tuples for a deterministic, closure-free ranking (B023).
        scores = {
            variant: _selection_score(verdicts_by_variant[variant], labels, train_ids)
            for variant in variants
        }
        ranked = [name for _, name in sorted((-scores[variant], variant) for variant in variants)]
        verdict: TriageResult | None = None
        chosen = ranked[0]
        for variant in ranked:
            candidate = verdicts_by_variant[variant].get(held)
            if candidate is not None:
                verdict, chosen = candidate, variant
                break
        of_picks[held] = verdict
        selected_by_alert[held] = chosen
    out_of_fold = score(_outcomes_for(of_picks, labels), labels, tactic_by_id)

    return CrossValReport(
        out_of_fold=out_of_fold,
        in_sample_best_variant=best_variant,
        in_sample_best=in_sample[best_variant],
        selection_optimism=in_sample[best_variant].overall_accuracy - out_of_fold.overall_accuracy,
        selected_variant_by_alert=selected_by_alert,
    )


class VerdictArtifact(BaseModel):
    """Persisted per-alert verdicts per variant plus the labels, for offline re-scoring."""

    model_config = ConfigDict(extra="forbid")

    verdicts_by_variant: dict[str, dict[str, TriageResult]]
    labels: dict[str, AlertLabel]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.model_validate_json(text)


async def collect_variant_verdicts(
    settings: Settings, *, model: str, concurrency: int
) -> dict[str, dict[str, TriageResult]]:
    """Run every prompt variant over the labeled set (live) and collect per-alert verdicts."""
    labeled = load_sample_alerts()
    alerts = [item.alert for item in labeled]
    clock = eval_reference_clock(labeled)
    collected: dict[str, dict[str, TriageResult]] = {}
    for variant, options in PROMPT_VARIANTS.items():
        system_prompt = build_system_prompt(options)
        async with build_runtime(
            settings, model=model, system_prompt=system_prompt, temperature=0.0, clock=clock
        ) as triager:
            outcomes = await triage_batch(alerts, triager, concurrency=concurrency)
        collected[variant] = {o.alert_id: o.result for o in outcomes if o.result is not None}
    return collected
