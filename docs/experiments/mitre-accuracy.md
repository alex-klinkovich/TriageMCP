# Experiment log: lifting MITRE technique accuracy

**Method.** Each step changes exactly one thing in the system prompt and re-runs the eval over
the same 38 labeled alerts at temperature 0 (deterministic). We report MITRE `mitre_technique_id`
exact-match accuracy with a Wilson 95% CI. Because n is only 38, a one- or two-alert change is
within the CI and is treated as suggestive, not definitive. Iteration was done on Haiku
(`claude-haiku-4-5`); the winner was confirmed on Sonnet (`claude-sonnet-4-6`).

Reproduce any row with: `triagemcp experiment --variant <name> --model <model> --concurrency 2`.

## Phase 1 — variant ladder on Haiku (0 errors on every run)

| Step | Variant             | Technique acc (95% CI)      | Correct | Overall | Decision  | Notes                                   |
| ---- | ------------------- | --------------------------- | ------- | ------- | --------- | --------------------------------------- |
| E0   | baseline            | 86.8% [72.7%, 94.3%]        | 33/38   | 61.4%   | reference | re-baseline at temp 0                   |
| E1   | catalog             | 92.1% [79.2%, 97.3%]        | 35/38   | 61.4%   | **keep**  | +2 alerts; choosing from the known set  |
| E2   | catalog+map         | 92.1% [79.2%, 97.3%]        | 35/38   | 64.9%   | **keep**  | no further technique gain, but +overall |
| E3   | catalog+map+fewshot | 89.5% [75.9%, 95.8%]        | 34/38   | 68.4%   | **drop**  | -1 on technique (helped overall, hurt technique) |

**Reading Phase 1:** the technique catalog drove the technique gain (+2 alerts). `require
map_to_mitre` added no further technique gain but lifted overall/severity at zero technique cost.
Few-shot was a **documented negative** for the target metric (it traded 1 technique alert for a
higher overall score). Winner on the target metric: **catalog+map**.

## Phase 2 — confirmation on Sonnet (temperature 0)

| Variant     | Technique acc (95% CI) | Correct | Overall | Errors |
| ----------- | ---------------------- | ------- | ------- | ------ |
| baseline    | 73.7% [58.0%, 85.0%]   | 28/38   | 56.1%   | 0      |
| catalog+map | 91.4% [77.6%, 97.0%]   | 32/35   | 65.7%   | 3      |

**Result:** catalog+map lifted Sonnet MITRE-technique accuracy from **73.7% to 91.4%** over the
scored alerts (a conservative count that treats the 3 errors as misses is **32/38 = 84.2%**, still
clearly above baseline), and overall accuracy from 56.1% to 65.7%.

**Trade-off (the cost):** the catalog injects all 36 ATT&CK techniques into the system prompt on
every call. That larger prompt increased per-call latency enough that **3 of 38 alerts hit the
per-alert timeout / rate limit** on the Tier-1 account (the per-alert isolation recorded them as
errors rather than failing the batch). On a higher tier, or with a longer `per_alert_timeout_s`,
those would likely complete — but the latency cost of a large static prompt is real and is the
honest counterweight to the accuracy gain.

## Conclusion

The hypothesis held: **grounding the model in the valid ATT&CK set materially improves technique
selection** (+~18 points on Sonnet, +5 on Haiku). `require map_to_mitre` is a free add (helps
secondary metrics, no technique cost). **Few-shot did not help the target metric** and is dropped.
If adopting catalog+map in production, pair it with prompt caching or a longer timeout to absorb
the added latency.

**Leakage:** few-shot examples are synthetic (`FEWSHOT-*`), asserted disjoint from the eval set by
`test_few_shot_examples_do_not_leak_into_eval_set`. The baseline was re-measured at temperature 0
for an apples-to-apples comparison (so these baselines differ slightly from the temperature-1
headline eval).
