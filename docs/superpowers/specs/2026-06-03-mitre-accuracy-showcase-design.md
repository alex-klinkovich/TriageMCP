# Design: MITRE-accuracy improvement showcase (eval-driven iteration)

**Date:** 2026-06-03
**Status:** approved (design), pending implementation plan

## Context

TriageMCP is built and shipped: an agentic triage engine over MCP, with `mypy --strict`/ruff
clean, 112 offline tests, and a self-scoring eval. The first live eval (Sonnet, 38 labeled
alerts, 0 errors) recorded:

| dimension | exact | note |
|---|---|---|
| severity | 52.6% | 100% within one level (well-calibrated, judgment-call gap) |
| MITRE technique | 65.8% | most *objective* dimension, clear headroom |
| action | 44.7% | most *subjective* dimension |
| overall | 54.4% | mean of the three |
| mean confidence | 0.93 | overconfident vs exact rates |

The goal of this work is **not** a bigger headline number — it is to **showcase eval-driven
iteration as an engineering skill**: a documented baseline → hypothesis → controlled test →
keep/discard loop, including honest handling of small-sample noise and negative results. A
senior reviewer values demonstrated ability to *reason about and improve a system's quality
with evidence* over a suspiciously high percentage.

We focus on **MITRE technique accuracy** because it is the most objective dimension (least
label subjectivity) and has the clearest headroom — the cleanest place to show defensible gains.

## Goals

- A reproducible, controlled experiment loop that varies **one** thing at a time on a fixed
  labeled set.
- Honest statistics: confidence intervals + paired comparison, so we can tell signal from noise
  on only 38 alerts.
- A written experiment log capturing every hypothesis, its measured effect, and the keep/discard
  decision — negatives included.

## Non-goals

- Chasing a target percentage. The number may move modestly; the *method* is the deliverable.
- Improving the subjective dimensions (action, exact severity) — explicitly out of scope here.
- Expanding the labeled dataset or changing the scoring of existing dimensions.

## Constraints

- **Lean budget (~$3–6 total).** Develop and compare interventions on **Haiku**
  (`claude-haiku-4-5`, ~1/10th cost); confirm the final winner **once** on Sonnet
  (`claude-sonnet-4-6`) against the 54.4% baseline. Runs use concurrency 2 (fresh-account
  rate limits).
- **Offline test suite stays free and green.** All new code is covered by mocked tests; only
  the human-run experiments cost money, exactly like today's `triagemcp eval`.
- **No eval-set leakage.** Few-shot examples must be separate synthetic alerts.

## Design

### 1. Harness changes (make three constants into parameters)

- **Parameterized system prompt.** `agent/prompts.py` gains
  `build_system_prompt(options: PromptOptions) -> str`, where `PromptOptions` is a frozen
  dataclass of booleans: `include_technique_catalog`, `require_map_to_mitre`,
  `include_few_shot`. The current prompt is `PromptOptions()` (all False). The technique-catalog
  text is rendered from `load_mitre_techniques()`; the few-shot text from synthetic examples.
  `AgentConfig` gains `system_prompt: str` (default = the baseline prompt); `TriageAgent` uses
  `self._config.system_prompt` instead of the module constant.
- **Temperature control.** `AgentConfig` gains `temperature: float = 0.0`. The `LLMClient`
  protocol's `create(...)` gains a `temperature` parameter; `AnthropicLLMClient` passes it to
  `messages.create`; `FakeLLMClient` accepts and ignores it. Evals run at temperature 0 so an
  intervention's effect is not masked by sampling noise. (This re-establishes the baseline under
  controlled conditions — E0 below.)
- **Confidence intervals (zero API cost).** `eval/metrics.py` gains
  `wilson_interval(successes, n) -> (low, high)` (95%, z=1.96). `EvalReport` reports a CI for each
  binomial dimension (`severity_exact`, `mitre_technique`, `action`). Rendered as
  `65.8% [50.0–79.1%]`.
- **Experiment runner.** A `triagemcp experiment` CLI subcommand (and an `eval/experiments.py`
  function it calls): given a variant name + model, it builds the agent with that variant's
  `PromptOptions`, runs the eval over the 38 alerts, and prints/saves one result row. Reuses
  `run_eval`; offline-testable with the fake triager (assert the right prompt is wired; LLM
  mocked).

### 2. Experiment ladder (cumulative, Haiku-first)

| Step | Adds to previous winner | Hypothesis |
|---|---|---|
| **E0** baseline | nothing (`PromptOptions()`) | reference point + CI, at temp 0 |
| **E1** | `include_technique_catalog` | choosing from the known ATT&CK set beats free recall |
| **E2** | `require_map_to_mitre` | grounding technique choice in the deterministic tool helps |
| **E3** | `include_few_shot` (synthetic) | a worked example raises verdict quality |

Then re-run the cumulative winner **once on Sonnet** vs. the Sonnet baseline.

### 3. Decision rule (signal vs noise on 38 alerts)

Because every step scores the **same** 38 alerts, use a **paired comparison**: count how many
alerts flipped *incorrect→correct* vs *correct→incorrect* on MITRE technique. Keep an
intervention only if net flips are positive and not plausibly noise (report the counts and the
binomial CI overlap; a couple of net flips on 38 is explicitly flagged as within noise). This is
more honest and more powerful than comparing two wide independent CIs.

### 4. Deliverable: the write-up

`docs/experiments/mitre-accuracy.md` — methodology + a results table populated as experiments
run: `step | change | technique acc (95% CI) | net flips | decision | notes`, including
negatives and the noise caveat. README links to it from the eval section.

### 5. Leakage safety

Few-shot examples are 2–3 **synthetic** alerts authored separately (distinct ids, e.g.
`FEWSHOT-*`). A test asserts no few-shot id or title collides with the 38-alert eval set, and the
write-up states this explicitly.

### 6. Testing

- `build_system_prompt` options produce the expected prompt fragments (catalog present/absent,
  etc.).
- `temperature` is threaded to the client (assert via fake client capturing the value).
- `wilson_interval` matches known values; `EvalReport` exposes CIs.
- Experiment runner wires the requested variant (offline, fake triager).
- Leakage test for few-shot examples.
- CI stays offline and green; no new live calls in the suite.

## Success criteria

- A reader of `docs/experiments/mitre-accuracy.md` can see the baseline, each hypothesis, its
  measured paired effect with a CI, and a clear keep/discard call — negatives included.
- The harness changes are `mypy --strict`/ruff clean and covered by offline tests.
- The final cumulative winner is confirmed once on Sonnet, and the README eval section links to
  the write-up. (Whether the number rises a little or not, the documented method is the result.)

## Out of scope / future

- Action and exact-severity improvement (subjective; would need label rigor first).
- Dataset expansion and multi-valid-action scoring (Approach C — a separate effort).
- Self-consistency/voting and extended-thinking experiments (could extend the ladder later).
