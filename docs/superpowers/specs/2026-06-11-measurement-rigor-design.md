# Design: a trustworthy accuracy-measurement harness (Phase 1)

**Date:** 2026-06-11
**Status:** approved (design), pending implementation plan

## Context

This is **Phase 1 of a two-phase accuracy program**. The goal of the program is to *increase
accuracy credibly* — gains that are real (generalizable) and honestly measured — rather than to
push the 38-alert headline number up. The multi-agent audit showed why the number alone can't be
trusted:

- the eval has **no train/test split** (the prompt was tuned in-sample on the same 38 it's scored on);
- the "catalog" experiment win is partly **answer-key overlap** (the 36-technique catalog is a
  superset of the 29 label techniques);
- the **action metric is broken**: `recommended_action` is scored single-valid (44.7%) even though
  several actions are legitimately defensible per alert;
- **confidence is uncalibrated** (mean 0.93 vs ~54% accuracy), reported but never quantified.

Phase 1 builds the measurement rigor that makes any later technique gain (Phase 2: self-consistency
voting / critique-revise) believable. Phase 1 **does not aim to raise the headline** — it makes the
headline defensible, fixes the one metric that is genuinely undercounting, and quantifies calibration.

**Key property exploited throughout:** at temperature 0 each prompt variant's per-alert verdict is
deterministic. So the only step that costs API spend is running each variant once over the 38 alerts
(the experiment ladder already does this: 4 variants × 38 = 152 calls). If those per-alert verdicts
are **persisted**, then cross-validated selection, within-one action scoring, and ECE are all
**deterministic offline recomputations** — free, reproducible, and unit-testable.

## Goals

- An **ordinal "within-one" action accuracy** metric, mirroring the existing within-one severity
  metric, so the action dimension is scored against SOC reality (one rung off ≠ a blunder).
- A **calibration metric (ECE)** plus a reliability table that quantifies the confidence gap.
- **Leave-one-out cross-validated variant selection**, reporting out-of-fold accuracy vs in-sample
  best-variant accuracy — the gap *is* the selection optimism.
- **Persisted per-alert verdicts** so all of the above recompute offline from one live run.
- All new scoring logic is **pure and offline-tested**; the existing suite stays green.

## Non-goals

- Raising the headline accuracy. Phase 1 makes it credible, not bigger.
- Fabricating new alerts/labels (rejected: AI-authored test data carries its own validity problem).
- Per-alert "acceptable action sets" (cherry-picking risk). The ordinal within-one metric is the
  principled, non-gameable substitute.
- The technique improvements themselves (self-consistency voting, critique-revise) — that is Phase 2,
  a separate spec, measured *on this harness*.

## Constraints

- Live API cost must not exceed running the variants once (152 calls at temp 0); everything else is
  offline. The old 54.4% run did not persist per-alert verdicts, so one fresh live run is needed to
  regenerate them under the new harness.
- Determinism: no `Math.random`-style fold shuffling — LOO has no fold-assignment freedom. All
  scoring is a pure function of (verdicts, labels).
- `mypy --strict` and ruff stay clean.

## Design

### 1. Ordinal action metric (`models.py`, `eval/metrics.py`)

`RecommendedAction`'s definition order already encodes the escalation ladder
(`close_false_positive` < `monitor` < `investigate` < `contain` < `escalate`). Give it the same
`.level` (rank 0–4) and `.distance(other)` helpers `Severity` has. In `score()`, add
`action_within_one_accuracy = mean(pred.action.distance(label.action) <= 1)` and surface it on
`EvalReport`. `overall_accuracy` is unchanged (still exact-based) to avoid moving the goalposts; the
within-one action number is reported alongside, exactly as severity already reports both.

### 2. Calibration / ECE (`eval/metrics.py`)

Add `expected_calibration_error(pairs, *, bins=5) -> float` and a reliability table. Define per-alert
**verdict correctness** as the strict conjunction `severity exact AND technique exact AND action
exact` (the model's `confidence` is its confidence in the *whole* verdict). Bin predictions by
confidence into `bins` equal-width buckets; ECE = Σ (n_b / N) · |acc_b − conf_b|. Report ECE and the
per-bin table `(conf_range, count, mean_confidence, accuracy)` on `EvalReport`. Use 5 bins (not 10)
given n=38, and document the small-n caveat. A near-0.9 mean confidence against a much lower
full-correct rate yields a large ECE — the honest, quantified version of the known overconfidence.

### 3. Leave-one-out cross-validated variant selection (`eval/crossval.py`, new)

A pure function over per-variant per-alert verdicts:

```
cross_validate(
    verdicts_by_variant: Mapping[str, Mapping[str, TriageResult]],   # variant -> {alert_id -> verdict}
    labels: Mapping[str, AlertLabel],
    tactic_by_id: Mapping[str, str],
) -> CrossValReport
```

For each held-out alert *i*: compute each variant's selection score on the **other 37** (selection
score = `overall_accuracy` over those 37, **counting alerts that variant errored on as incorrect**, so
a variant is never rewarded for skipping hard alerts), pick the argmax variant (ties broken by a fixed
variant order for determinism), and take that variant's verdict for *i*. If the selected variant has
no verdict for *i* (it errored on *i*), fall back to the next-best variant that *did* score *i*; if no
variant scored *i*, record an out-of-fold error for *i*. Pool the 38 held-out outcomes into a single
`EvalReport` (the **out-of-fold** report). Also compute, for reference, each variant's full
in-sample `EvalReport` and identify the in-sample best. `CrossValReport` carries: the out-of-fold
report, the in-sample best-variant report, the **selection-optimism gap** (in-sample best overall −
out-of-fold overall), and a per-alert record of which variant was selected. Expected honest result:
the gap is small (only 4 variants to overfit), which itself is the finding — variant selection
generalizes.

### 4. Verdict persistence + CLI (`cli.py`, `eval/crossval.py`)

A `triagemcp crossval` command:

- default (live): build the runtime per variant (reusing `run_experiment`'s wiring with the
  dataset-anchored clock), triage the labeled set under each variant, collect per-alert verdicts,
  **persist them to a JSON artifact** (`{variant: {alert_id: TriageResult}}` plus the label set), then
  print the cross-validation + within-one + ECE report.
- `--from <file>`: skip the live run and recompute the entire report from a persisted artifact
  (free, deterministic) — the reproducibility path.

Serialization uses the existing Pydantic models (`TriageResult`, `AlertLabel`).

### 5. Reporting (`report.py`, docs)

Extend the eval report rendering to show within-one action and ECE. Add a write-up under
`docs/experiments/` describing the methodology and (once run) the numbers; refresh the README metrics
language to include within-one action and the calibration figure.

### Data flow

```
run variants (LIVE, once) ─► persist per-alert verdicts (JSON)
                                      │
        ┌─────────────────────────────┴───────────── all offline, free, deterministic ─┐
        ▼                              ▼                                  ▼
  cross_validate(...)          score() within-one action          expected_calibration_error(...)
   -> out-of-fold report        -> action_within_one              -> ECE + reliability table
   -> selection-optimism gap
```

## Testing (all offline, deterministic)

- `RecommendedAction.level`/`.distance` — known ranks and gaps (mirror the severity tests).
- `action_within_one_accuracy` — a constructed outcome/label set with a known one-rung-off case.
- `expected_calibration_error` — hand-computed inputs (e.g., two bins with known conf/acc) matching a
  hand-derived ECE; edge cases (single bin, perfect calibration → 0).
- `cross_validate` — a tiny synthetic grid: 2–3 variants × a handful of alerts with verdicts rigged so
  a *different* variant wins on different held-out alerts, asserting the pooled out-of-fold report and
  the selection-optimism gap are computed correctly (this is the one case worth engineering carefully
  so the LOO logic is provably right).
- Persistence round-trip: dump → `--from` reload → identical report.
- CLI: `crossval --from <fixture>` prints a report offline (no key, via a committed tiny fixture or a
  monkeypatched runtime); `crossval` without a key exits with the standard clear message.
- The existing 156 tests stay green; `overall_accuracy` and severity metrics are unchanged.

## Success criteria

- `triagemcp crossval` produces, from one live run, a report with: out-of-fold accuracy + the
  selection-optimism gap, within-one action accuracy, and ECE + a reliability table — and the same
  report recomputes byte-identically from the persisted artifact with `--from`.
- New scoring logic is `mypy --strict`/ruff clean and covered by offline tests; the headline
  `overall_accuracy` is unchanged by this phase.
- A reader can see that the reported accuracy is out-of-fold (not in-sample), that action is scored
  ordinally, and that calibration is quantified — i.e., the number is now defensible.

## Out of scope / future (Phase 2 and beyond)

- **Phase 2 — real technique gains** measured on this harness: self-consistency voting (sample N at
  temp>0, majority-vote per field) and/or critique-revise (a second pass that checks the verdict
  against the evidence). These raise accuracy; this harness makes the raise believable.
- Dataset expansion (would need real, not AI-authored, labels to add value).
- Per-dimension ECE and confidence recalibration (temperature scaling).
- Multi-seed determinism caveats for any temp>0 Phase-2 work.
