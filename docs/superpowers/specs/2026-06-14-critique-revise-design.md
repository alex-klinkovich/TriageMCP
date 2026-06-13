# Design: critique-revise for higher-accuracy triage (Phase 2-B)

**Date:** 2026-06-14
**Status:** approved to build (offline); the live measurement is gated on explicit user go.
**Branch:** `feat/critique-revise` (off `main`; independent of the in-flight Phase-1 PR).

## Context

Phase 1 (separate, in-flight PR) made the eval credible — leave-one-out out-of-fold accuracy,
ordinal within-one action, and ECE. Its first clean run (best variant `catalog+map+fewshot`) scored
**67.5% out-of-fold** with **0.0% selection optimism**, but the strict whole-verdict-correct rate
(severity AND technique AND action all exact) is only **29%** and **ECE is 0.63**: the model is
confidently wrong on a large share of whole verdicts. Phase 2 raises real accuracy.

We chose **critique-revise** over self-consistency voting (Phase 2-A, hard-gated) and a
prompt-variant ensemble (dropped — the four variants are nested/correlated, partly answer-key
contaminated, and n=38 is too small to separate a gain from noise).

## Goal

A second, evidence-anchored self-critique pass: after the model proposes a verdict it re-examines
each field against the evidence it gathered and resubmits — revised if warranted, unchanged if it
holds. Temperature 0 (deterministic), bounded cost (~1.6x, one extra pass), with the cost ceiling
exposed as a user-configurable knob.

## Non-goals

- Self-consistency voting (Phase 2-A) — hard-gated behind explicit user go.
- Prompt caching and other cost/robustness infra — separate focused PRs.
- Changing how the metrics are computed — Phase 1 owns that.

## Design

### 1. In-loop critique rounds (`agent/loop.py`)

Add `critique_rounds: int = 0` to `AgentConfig`. The agent loop already iterates, dispatches tools,
validates the `submit_triage` verdict, and retries — critique-revise is "one more pass with a
nudge," so it reuses that machinery rather than duplicating it in a separate wrapper.

When the model submits its first valid verdict, instead of returning, the loop:
- records it as a **draft**,
- answers the `submit_triage` tool call with an evidence-anchored **critique prompt** ("re-examine
  each field — severity, technique, action — against the evidence; investigate further if useful;
  then call submit_triage once more, revised if warranted or unchanged if it holds"),
- and continues. The model re-examines (optionally re-investigating via tools) and resubmits.

After `critique_rounds` critiques, the latest verdict is returned.

- `critique_rounds = 0` (default) → today's behavior, byte-for-byte unchanged.
- **Additive-only robustness:** if a critique round does not yield a resubmission (the model ends
  its turn or hits the iteration cap), the most recent **draft** is returned rather than raising —
  critique can only improve on, never lose, a verdict the base run already produced.
- **Multi-tool turns** (a turn that both investigates and submits) are handled: sibling tool calls
  are dispatched and answered, and the submit is answered with the critique prompt.

(Alternative considered: a separate `CritiqueReviseTriager` wrapper. More isolated, but it would
duplicate the LLM-call / tool-dispatch / retry plumbing and require exposing the loop's transcript.
Rejected in favor of the smaller in-loop change.)

### 2. Wiring (`server.py`, `config.py`)

`build_runtime` gains a `critique_rounds: int = 0` parameter, threaded into `AgentConfig`.
`Settings` gains `critique_rounds: int = 0` (env `TRIAGEMCP_CRITIQUE_ROUNDS`) — the user-configurable
cost knob: an operator dials it up, default 0 keeps it off.

### 3. Measurement (`cli.py`) — the paid step, gated

Add `--critique-rounds N` to the `eval` command, running the labelled set with N critique rounds.
The lift is `eval(critique) - eval(base)` on the same alerts. **The confound Phase 1 worried about
(the prompt was tuned in-sample) is held constant across both arms, so the *difference* is a fair
measure of critique's marginal effect even in-sample**; the harness reports Wilson confidence
intervals to judge whether it is real. The critique *prompt* must not be tuned on the eval alerts.
Once Phase 1 is merged, the same comparison can be run through the out-of-fold crossval harness for
full rigor.

This live run is the **only API spend** and is **gated on explicit user go** — the offline build
stops before it.

## Testing (all offline, deterministic, `FakeLLMClient`)

- Loop: critique revises a wrong draft (submit V1 → critique → submit V2 ⇒ V2 returned); critique
  confirms (V1 → V1); `critique_rounds=0` unchanged; **draft fallback** when no resubmission;
  multi-round; iteration count includes critique turns; multi-tool-turn submit + critique.
- Wiring: `build_runtime` / `Settings` carry `critique_rounds`; the CLI `--critique-rounds` flag
  flows into the config (monkeypatched runtime — no API).
- The full offline suite stays green; `mypy --strict` and `ruff` clean.

## Cost

~1.6x base per alert (one extra pass), bounded by `critique_rounds` (the configurable cap), no
determinism loss (temperature 0). A cheap measurement (best variant, base vs critique) is ~$2-3.

## Success criteria

- **Offline (this build):** `critique_rounds` implemented, wired, CLI-exposed, fully unit-tested;
  suite green; types/lint clean.
- **Live (gated, later):** `eval --critique-rounds 1` on `catalog+map+fewshot` shows an accuracy
  lift over base with clearly separated Wilson intervals — a real, honestly-measured gain — or an
  honest null result, reported either way.
