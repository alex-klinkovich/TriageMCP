# Design: self-consistency voting for higher-accuracy triage (Phase 2-A)

**Date:** 2026-06-14
**Status:** approved to build (offline); the live measurement is gated on explicit user go.
**Branch:** `feat/self-consistency-voting` (off `main`; independent of B and the Phase-1 PR).

## Context

Phase 1 made the eval credible; Phase 2-B (critique-revise) is the cheaper, deterministic technique.
A = self-consistency voting (Wang et al.) — the most expensive option, hard-gated, built only after
B. It samples N verdicts at temperature > 0 and majority-votes each field, trading **N× cost** for
variance reduction. Best when the model is right "on average" but noisy.

## Goal

Reduce per-field error variance: sample N independent verdicts and take the modal value of each
field. Measured on the harness vs the single-sample baseline; **N is the user-configurable cost cap**.

## Non-goals

- Critique-revise (B, separate). Prompt caching (separate PR — but it's the cost-enabler for A).
- Tuning N or temperature on the eval set.

## Design

### 1. Pure majority-vote (`agent/voting.py`)

`majority_vote(results: Sequence[TriageResult]) -> TriageResult` — pure, deterministic, offline.
- **Per-field mode** of `severity`, `mitre_technique_id`, `recommended_action` across the N results.
- **Tie-break (deterministic):** ordinal fields (severity, action) break ties to the *more severe /
  more escalated* rung — SOC-conservative: when the samples split, don't under-call. The nominal
  technique breaks ties to the value from the earliest sample (stable, reproducible).
- **Non-voted fields:** pick a *representative* sample — the one whose `(severity, technique, action)`
  matches the voted triple on the most fields (ties → highest confidence) — and take its
  `mitre_technique_name`, `rationale`, `supporting_observations`. `confidence` = mean of the N
  confidences. `alert_id` from the results.

### 2. `VotingTriager` wrapper (`agent/voting.py`)

Implements the `Triager` protocol (`run(alert) -> AgentRun`). Wraps a base triager; runs it N times
per alert (bounded concurrency), collects the successful `TriageResult`s, and returns
`AgentRun(majority_vote(results), iterations=sum)`. If some samples error, vote over the successes;
if **all** N error, propagate the last error so the alert is recorded as failed (consistent with the
harness). Because it satisfies the `Triager` protocol, it drops into `triage_batch`/`run_eval`
unchanged.

### 3. Wiring (`config.py`, `server.py`, `cli.py`)

- `Settings.vote_samples: int = 1` (env `TRIAGEMCP_VOTE_SAMPLES`) — the cost knob; 1 = off.
- `build_runtime(..., vote_samples=N)` → when N > 1, wrap the agent in `VotingTriager`. Voting needs
  temperature > 0 for diversity, so `build_runtime` applies a voting temperature (default 0.7) when
  vote_samples > 1; a single sample keeps the caller's temperature.
- `eval --vote-samples N` runs the labelled set with N-sample voting.

### 4. Measurement (paid, gated)

Run the labelled set with vote_samples=N at temp > 0; compare voted accuracy vs single-sample
accuracy (sample 1 of the N). Persist all N samples so the *aggregation* recomputes offline. N× cost.
**Gated on explicit user go** — the offline build stops before it.

## Determinism caveat (the honest wrinkle)

Unlike B (temp 0, deterministic, reproducible), voting needs temp > 0 → the samples are stochastic
and the Anthropic API exposes **no seed**, so a voting run is **not reproducible** (re-running yields
different samples). We persist the samples so the aggregation is reproducible, but the samples
themselves cannot be regenerated identically. Report the result with this caveat; consider 2–3 runs
to bound the variance.

## Testing (all offline, deterministic, `FakeLLMClient`)

- `majority_vote`: clear majority; all-agree; ordinal tie → more severe; technique tie → earliest;
  representative-sample field-fill; confidence = mean.
- `VotingTriager`: N scripted results → voted result; a failing sample is excluded; all-fail
  propagates the error.
- Wiring: `Settings.vote_samples`; `build_runtime` wraps when N > 1 and uses temp > 0;
  `eval --vote-samples` flows the value.
- Full offline suite green; `mypy --strict` and `ruff` clean.

## Cost

N× base per alert (the configurable cap). N=5 ≈ $7.2 to measure once. Prompt caching (separate PR)
is the enabler — it amortizes the fixed prefix that voting multiplies by N. Gated.

## Success criteria

- **Offline (this build):** `majority_vote` + `VotingTriager` + wiring + CLI, fully unit-tested;
  suite green; types/lint clean.
- **Live (gated, later):** voted accuracy beats single-sample with clearly separated Wilson intervals
  — a real gain — or an honest null/negative, reported with the determinism caveat.
