# Measurement rigor: cross-validation, ordinal action scoring, calibration

Phase 1 of the accuracy program makes the eval number trustworthy rather than bigger.

**Why.** The headline eval is in-sample (the prompt was tuned on the same 38 alerts), the action
metric is single-valid (undercounting defensible alternatives), and confidence is reported but its
calibration was never quantified.

**What this adds.**
- **Within-one action accuracy** — actions are ordered (close_false_positive < monitor < investigate
  < contain < escalate); being one rung off is credited, mirroring within-one severity. Principled,
  not per-alert cherry-picking.
- **ECE + reliability table** — bins verdicts by confidence and reports the expected calibration
  error against strict (all-three-exact) correctness. Quantifies the 0.93-confidence gap.
- **Leave-one-out cross-validation** — for each alert, the best prompt variant is chosen on the
  other 37 and used to predict the held-out one. The pooled out-of-fold accuracy vs the in-sample
  best-variant accuracy is the **selection-optimism gap**.

**How to run.** `triagemcp crossval --out verdicts.json` runs the four prompt variants once over the
labelled set (the only API spend; ~$6 on Sonnet 4.6, reported exactly by the run), persists the
per-alert verdicts, and prints the report. `triagemcp crossval --from verdicts.json` recomputes
everything offline for free. **On a rate-limited or low-tier account, pass `--concurrency 1` (or 2)** —
the default of five concurrent triages provoked sustained HTTP 429s that errored most alerts, while a
serial run completes cleanly. The committed run reproduces from
[`measurement-rigor-verdicts.json`](measurement-rigor-verdicts.json) with `--from`.

## Results — first clean run (2026-06-13, Sonnet 4.6, temperature 0)

38/38 alerts scored, **0 errors** on every variant. The four prompt variants cluster within ~5 points:

| Variant | Overall | MITRE technique | Action exact / within-one | ECE |
|---|---|---|---|---|
| baseline | 62.3% | 79.0% | 42.1% / 94.7% | 0.74 |
| catalog | 64.0% | 92.1% | 36.8% / 94.7% | 0.71 |
| catalog+map | 63.2% | 86.8% | 39.5% / 97.4% | 0.74 |
| catalog+map+fewshot | **67.5%** | 94.7% | 42.1% / 97.4% | 0.63 |

**Cross-validation.** `catalog+map+fewshot` is the in-sample best (67.5%) and wins all 38 leave-one-out
folds, so the pooled out-of-fold accuracy is also **67.5% — a selection-optimism gap of 0.0%**. With
one variant robustly dominant there is nothing to overfit; the variant choice generalizes.

**Within-one action.** Exact action accuracy is only 42.1%, but **97.4% of verdicts are within one
rung** of the labelled action. The dimension the audit flagged as "broken" (single-valid 44.7%) is
near-miss, not blunder — the ordinal metric working as intended.

**Calibration.** Mean confidence is 92.0% while the strict whole-verdict-correct rate (severity AND
technique AND action all exact) is just 29.0%, giving **ECE 0.63** — severe, now-quantified
overconfidence. 36 of 38 alerts sit in the [0.8, 1.0) confidence bin at 93% mean confidence but 31%
accuracy.

**Two honest caveats.**
- The 94.7% MITRE-technique figure is **partly answer-key overlap**: the catalog prompt lists a
  superset of the label techniques, so the model chooses from a menu that contains the answer.
  Leave-one-out does not correct this — every fold uses the same catalog — so read it as "with catalog
  assistance," not unaided recall.
- This run's baseline measured 62.3% vs the **54.4% in the README headline recorded earlier**. Even at
  temperature 0, verdicts drift across model and code revisions — which is exactly why the harness
  persists per-alert verdicts and recomputes offline rather than trusting a one-off number.

**Reading it.** A small selection-optimism gap means the variant choice generalizes; here it is exactly
zero. Within-one action and severity are reported alongside the exact figures. The `overall_accuracy`
metric is unchanged by this phase *by construction* — Phase 1 makes the number defensible (out-of-fold,
ordinally scored, calibration-quantified), not bigger.
