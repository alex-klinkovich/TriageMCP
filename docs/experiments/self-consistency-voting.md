# Experiment log: self-consistency voting

**Method.** Sample the triage agent N=5 times independently at temperature 0.7 on the same 38
labeled alerts (the default `triagemcp eval` system prompt — open-ended baseline, *not* the
`catalog` experiment variant), then majority-vote per field. Ordinal fields (severity, action)
break ties toward the more severe / more escalated rung; the nominal MITRE technique breaks ties
toward the earliest sample; confidence is the mean across samples. Model `claude-sonnet-4-6`,
`--concurrency 1` (Tier-1 rate limits). `overall` is the unweighted mean of the three per-field
exact-match rates (severity, MITRE technique, action). With n=38 a one-alert change is ~2.6 points
and sits inside the Wilson 95% CI, so single-alert moves are suggestive, not definitive.

Because temperature is 0.7 with no API seed, this run is **not reproducible** — the verdict
artifact `voting-verdicts.json` (all 5 samples + the voted verdict + labels) is the record.
Re-run a fresh sample set with `triagemcp eval --vote-samples 5`.

## Single-sample accuracy (per pass)

| Pass     | Overall   | Severity | MITRE technique | Action | Scored | Errors |
| -------- | --------- | -------- | --------------- | ------ | ------ | ------ |
| 1        | 57.9%     | 63.2%    | 73.7%           | 36.8%  | 38     | 0      |
| 2        | 60.5%     | 60.5%    | 78.9%           | 42.1%  | 38     | 0      |
| 3        | 62.3%     | 65.8%    | 71.1%           | 50.0%  | 38     | 0      |
| 4        | 61.4%     | 65.8%    | 78.9%           | 39.5%  | 38     | 0      |
| 5        | 63.0%     | 63.9%    | 80.6%           | 44.4%  | 36     | 2      |
| **mean** | **61.0%** | —        | —               | —      | —      | —      |

Pass 5 hit the per-alert timeout / rate limit on 2 alerts (Tier-1); per-alert isolation recorded
them as errors, so that pass scored 36/38. The voted line below still covers all 38 (every alert
has ≥3 samples).

## Voted vs single

| Config                       | Overall   | Severity | MITRE technique | Action | Cost |
| ---------------------------- | --------- | -------- | --------------- | ------ | ---- |
| temp-0.7 single (mean of 5)  | 61.0%     | —        | 78.9%\*         | —      | 1×   |
| **temp-0.7 voted (N=5)**     | **62.3%** | 63.2%    | **81.6%**       | 42.1%  | 5×   |
| lift (voted − mean-single)   | +1.3      | −0.7     | **+4.9**        | −0.5   | —    |

\*best single-pass technique at full n=38 (passes 2 and 4 = 30/38).

Voted MITRE technique = **31/38, 81.6% [66.6%, 90.8%]** — one alert above the best single pass
(30/38, 78.9% [63.7%, 88.9%]) and above every individual sample. Voting denoises the one field
that is genuinely variance-driven here: the model flip-flops on technique granularity (e.g.
`T1110` vs `T1110.001`) between samples, and the majority recovers the stable answer. Severity and
action, which barely move across samples, are left flat-to-slightly-negative.

## The decisive comparison (why this does not ship)

| Config                  | Overall accuracy | Relative cost |
| ----------------------- | ---------------- | ------------- |
| temp-0 greedy single    | **63.2%**        | 1×            |
| temp-0.7 single (mean)  | 61.0%            | 1×            |
| temp-0.7 voted (N=5)    | 62.3%            | 5×            |

The temp-0 greedy single baseline — 63.2%, measured under identical prompt, model, and alerts in
the critique-revise experiment's BASE arm — is **both the most accurate and the cheapest.** Moving
to temperature 0.7 costs ~2 points of overall accuracy; voting at 5× the cost recovers most of
that self-inflicted penalty but never climbs back above the free temperature-0 number.

## Conclusion

Self-consistency voting is **not worth deploying on this task at n=38.** Its only clean signal is
+1 alert on the noisy MITRE-technique field (within the CI, so suggestive). The honest headline:
the simplest configuration — a single temperature-0 triage — is simultaneously the most accurate
*and* the cheapest here. The technique is implemented, measured, and reported rather than shipped
on a cherry-picked sample; that honest null/modest result is the deliverable.

The cost/quality knob (`vote_samples`) is a first-class, user-configurable setting, not hardcoded
to the developer's budget — an operator running a real SOC can raise N where the technique field
matters and the spend is justified.

**Caveats.** (1) n=38: the voted-vs-single deltas all sit inside overlapping Wilson 95% CIs; read
them as direction, not proof. (2) Not reproducible (temp 0.7, no seed). (3) Open-ended baseline
prompt: the absolute technique numbers here are *not* inflated by the answer-key overlap the
`catalog` experiment carries (that variant injects a near-superset of the label set; this run does
not). (4) The temp-0 63.2% reference comes from a separate run (the critique BASE arm), not this
one — same prompt, model, and alerts, but a different invocation.

Spec:
[`docs/superpowers/specs/2026-06-14-self-consistency-voting-design.md`](../superpowers/specs/2026-06-14-self-consistency-voting-design.md).
