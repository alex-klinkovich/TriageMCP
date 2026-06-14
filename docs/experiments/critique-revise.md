# Experiment log: critique-revise

**Method.** Run the eval twice on the same 38 labeled alerts at temperature 0 (deterministic),
default `triagemcp eval` system prompt. The BASE arm is a single triage (`critique_rounds=0`); the
CRITIQUE arm adds one self-critique-and-revise pass (`critique_rounds=1`, Reflexion-style — the
agent is shown its own verdict and asked to find and fix errors). The only difference between arms
is that extra round. Model `claude-sonnet-4-6`, `--concurrency 1`. `overall` is the unweighted
mean of the three per-field exact-match rates (severity, MITRE technique, action). With n=38 a
one-alert change is ~2.6 points and inside the Wilson 95% CI.

Both arms are temperature 0, so the run is deterministic and re-runnable with
`triagemcp eval --critique-rounds 0` / `--critique-rounds 1`. The verdict artifact
`critique-revise-verdicts.json` records both arms' per-alert verdicts plus labels.

## Results

| Arm                          | Overall   | Severity (within-1) | MITRE technique (95% CI) | Action (95% CI)        | Mean conf | Scored | Err |
| ---------------------------- | --------- | ------------------- | ------------------------ | ---------------------- | --------- | ------ | --- |
| BASE (`critique_rounds=0`)   | **63.2%** | 63.2% (100%)        | 78.9% [63.7%, 88.9%]     | 47.4% [32.5%, 62.7%]   | 0.919     | 38     | 0   |
| CRITIQUE (`critique_rounds=1`) | 61.4%   | 60.5% (100%)        | 73.7% [58.0%, 85.0%]     | 50.0% [34.8%, 65.2%]   | 0.920     | 38     | 0   |
| lift (critique − base)       | **−1.8**  | −2.6                | −5.3                     | +2.6                   | +0.001    | —      | —   |

Verdicts changed by the critique round: **8/38** (improved 2, hurt 4, neutral 2).

## Reading

One critique round **moves** verdicts but **net-hurts** at this n. It helped the action field
(+2.6) at the expense of technique (−5.3) and severity (−2.6), netting −1.8 overall. The technique
regressions dominate and have a clear mechanism: the critique talks the model into *over-refining*
a correct answer. Example — alert A-0001: BASE returns `T1110` (correct); the critique revises it
to the more specific `T1110.001`, which no longer exact-matches the `T1110` label. The extra round
spends tokens second-guessing answers that were already right more often than it rescues wrong
ones.

Confidence is essentially unchanged (0.919 → 0.920): the model is **no better calibrated** after
revising — it is equally sure of the slightly-worse answers.

## Conclusion

Critique-revise (1 round, temp 0) is a **null / slightly-negative** result on this task at n=38,
and not worth its ~1.6–2× cost here. Every per-field delta sits inside the other arm's Wilson 95%
CI (the intervals overlap heavily), so the honest reading is **"no improvement,"** not "harm
proven" — the −1.8 is within noise. A different critique prompt, or more rounds, might behave
differently, but there is no signal in this run that they would clear the cost bar. As with the
self-consistency-voting experiment, the value delivered is an honest measurement of a well-known
technique that did not pay off, rather than a headline number.

The cost knob (`critique_rounds`) is a first-class, user-configurable setting, not hardcoded — an
operator can spend more rounds where it is justified.

**Caveats.** (1) n=38: the deltas are within overlapping CIs; direction, not proof. (2) A single
round at temperature 0 — narrow by construction. (3) Open-ended baseline prompt, so absolute
technique accuracy is honest open recall (not lifted by the `catalog` experiment's answer-key
overlap).

Spec:
[`docs/superpowers/specs/2026-06-14-critique-revise-design.md`](../superpowers/specs/2026-06-14-critique-revise-design.md).
