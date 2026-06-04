# Experiment log: lifting MITRE technique accuracy

**Method.** Each step changes exactly one thing in the system prompt and re-runs the eval over
the same 38 labeled alerts at temperature 0. We compare on the *same* alerts (paired), counting
how many flipped incorrect->correct vs correct->incorrect on `mitre_technique_id`, and report the
Wilson 95% CI. A change of one or two alerts on 38 is within the CI and is treated as noise.
Iteration is on Haiku (`claude-haiku-4-5`); the final winner is confirmed once on Sonnet.

Reproduce a row with: `triagemcp experiment --variant <name> --model <model>`.

| Step  | Variant             | Model  | Technique acc (95% CI) | Net flips vs prev   | Decision  | Notes                    |
| ----- | ------------------- | ------ | ---------------------- | ------------------- | --------- | ------------------------ |
| E0    | baseline            | haiku  | _TBD by run_           | --                  | reference | re-baseline at temp 0    |
| E1    | catalog             | haiku  | _TBD_                  | _TBD_               | _TBD_     | technique list in prompt |
| E2    | catalog+map         | haiku  | _TBD_                  | _TBD_               | _TBD_     | require map_to_mitre     |
| E3    | catalog+map+fewshot | haiku  | _TBD_                  | _TBD_               | _TBD_     | + few-shot               |
| Final | winner              | sonnet | _TBD_                  | vs 54.4% baseline   | _TBD_     | confirmation             |

**Leakage:** few-shot examples are synthetic (`FEWSHOT-*`), asserted disjoint from the eval set
by `test_few_shot_examples_do_not_leak_into_eval_set`.

_Rows marked TBD are filled by running the experiments with a real key; the table records the
actual outcome, including any negative results._
