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

**How to run.** `triagemcp crossval --out verdicts.json` runs the variants once (the only API spend),
persists the per-alert verdicts, and prints the report. `triagemcp crossval --from verdicts.json`
recomputes everything offline for free.

**Reading it.** A small selection-optimism gap means the variant choice generalizes (with only four
variants there is little to overfit). Within-one action is reported alongside exact action; the
headline `overall_accuracy` is unchanged by this phase by design.
