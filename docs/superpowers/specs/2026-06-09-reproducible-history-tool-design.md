# Design: reproducible, realistic alert-history tool

**Date:** 2026-06-09
**Status:** approved (design), pending implementation plan

## Context

An audit of the shipped project found that `query_recent_alerts` is **silently inert at its
default lookback** and **not reproducible across dates**. Three facts interact:

- The bundled sample alerts have fixed timestamps (2026-05-23 → 2026-05-29).
- `build_default_registry` wires the history tool with the default `utcnow` clock
  (`tools/recent_alerts.py:_utcnow`) in every non-test path.
- The tool's default lookback is 168h (7 days).

So as wall-clock time advances past the data, the seeded "prior sightings" age out of the
window. Empirically, on 2026-06-08 **0 of 34** alerts with an IP/host observable returned any
match at the default lookback (34/34 at a 1-year lookback). The unit tests don't catch this
because they **inject a fixed clock** — CI stays green while the real eval and live server decay.

A second, pre-existing smell compounds it: `_history_from_samples` fabricates one prior sighting
for **every** alert (at its own `timestamp − 24h`), so whenever the window does catch them, every
alert looks like a repeat offender — unrealistic, and a signal the agent could lean on spuriously.

## Goals

- **Reproducible:** `query_recent_alerts` returns the same results in the eval/experiment
  regardless of the calendar date the run happens.
- **Realistic:** a meaningful repeat-vs-novel mix — some observables genuinely recur across
  alerts (repeat offenders), others are unique (novel, no history) — instead of a 100% repeat
  rate.
- **Correct seams:** the live `serve` path (and the general `triage` path) keep **real time**
  (`utcnow`); only the fixed-benchmark paths (`eval`, `experiment`) use a fixed clock.

## Non-goals

- Per-alert "as-of" history (triaging each alert as of its own timestamp). The eval uses one
  global anchor; the temporal-ordering nuance is an accepted simplification (see below).
- Expanding the dataset, adding new observable types, or changing the scoring.
- Changing the tool's public input schema or its default lookback.

## Constraints

- Offline test suite stays free, deterministic, and green; no new live calls.
- `mypy --strict` and ruff stay clean.
- No change to the `Alert`/`TriageResult` contracts or the MCP tool surface.

## Design

Two independent changes. Change 1 (seeding) is clock-independent and applies to all paths; change
2 (clock) is injected only into the eval/experiment paths.

### 1. Recurrence-based history seeding (`tools/registry.py`)

Rewrite `_history_from_samples` to derive "repeat offenders" from genuine cross-alert overlap in
the sample set rather than fabricating a self-sighting for every alert.

- Build a recurrence map: each **typed** observable value `(value, type)` → the set of alert IDs
  that contain it (types: ip, hash, domain, user, host — as today).
- For every observable appearing in **≥2 distinct alerts**, seed **one** `HistoryEntry`:
  - `seen_at` = *(earliest occurrence timestamp among its alerts) − 1h* — unambiguously prior,
    and comfortably inside the default 168h window under the anchored clock (verified by test).
  - `observable` / `observable_type` = the recurring value and its type.
  - `severity` = the earliest containing alert's `severity_reported`.
  - `title` = `f"Prior activity involving {value}"`.
  - `alert_id` = `f"HIST-{observable_type}-{value}"` (synthetic, clearly a historical row).
- Observables appearing in exactly one alert are **not seeded** → genuinely novel.

Effect (data as bundled today): 12 recurring observables across 22 of 38 alerts → querying a
recurring observable returns 1 prior sighting; a novel observable returns 0. ~22 repeat / ~16
novel, deterministic.

### 2. Dataset-anchored clock for the eval/experiment paths

- Add `eval_reference_clock(labeled: Sequence[LabeledAlert]) -> Callable[[], datetime]` in
  `eval/harness.py`. It computes `anchor = max(item.alert.timestamp for item in labeled) + 1h`
  and returns a frozen `lambda: anchor`. The +1h buffer only needs to be > 0; the repeat/novel
  classification is insensitive to its exact value (the seeded sightings sit well inside the
  window for any small positive buffer).
- `server.build_runtime(...)` gains a keyword-only `clock: Callable[[], datetime] | None = None`,
  forwarded as `build_default_registry(conn, clock=clock)` (which already accepts it).
- `cli._evaluate` and `eval/experiments.run_experiment` compute the clock from the labeled set and
  pass it to `build_runtime`.
- `serve_stdio` and `cli._triage` pass **nothing** → `None` → `utcnow`. Real-alert paths keep real
  time.

### Data flow

```
eval / experiment:
  load_sample_alerts() ─► eval_reference_clock(labeled) ─► build_runtime(settings, clock=…)
                                                              └► build_default_registry(conn, clock=…)
                                                                   ├► _history_from_samples()  [change 1, clock-independent]
                                                                   └► QueryRecentAlertsTool(clock=…)  [change 2]

serve / triage:
  build_runtime(settings)  ─► build_default_registry(conn)  ─► QueryRecentAlertsTool(clock=_utcnow)
```

### Deliberate non-issues

- **Not leakage.** Repeat-offender status is derived from observable recurrence — a legitimate
  feature available at real inference time — not from the ground-truth label. An agent *should*
  weigh "this host appears in several alerts." (Whether recurrence happens to correlate with
  severity in this set is irrelevant: recurrence is a real signal, not the label.)
- **Single global anchor.** Because the eval uses one fixed "now," a recurring observable returns
  its prior sighting for *every* alert that contains it, without per-alert temporal ordering. This
  is an accepted simplification of the fixed-clock eval and is documented in the README.

## Testing (TDD, all offline)

- `_history_from_samples` seeds a known recurring observable (`FIN-WS-118`, in A-0003 & A-0023)
  and **omits** a known singleton observable.
- The seeded prior sighting for a recurring observable falls inside the default 168h window under
  `eval_reference_clock` (guards the −1h / +1h margins against regression).
- With the anchored clock + default registry: querying a recurring observable returns
  `match_count ≥ 1`; querying a novel observable returns `0`. Date-independent (the clock is
  injected, so the assertion does not depend on `utcnow`).
- `eval_reference_clock(labeled)` returns `max(timestamp) + 1h` deterministically.
- `build_runtime` forwards an injected clock to the query tool (assert the eval path uses the
  injected clock, not `utcnow`); `serve`/`triage` paths still default to `utcnow`.
- Existing suite stays green; fix any test that implicitly assumed every alert has history.

## Docs

- README "Known simplifications": replace the "synthetic prior sightings derived from the sample
  set" sentence with the recurrence-based seeding + dataset-anchored eval clock description.
- `docs/experiments/mitre-accuracy.md`: one line noting the eval's history results are
  reproducible (date-independent) via the anchored clock.

## Success criteria

- Running `triagemcp eval` on any date yields the same `query_recent_alerts` behavior:
  ~22 repeat / ~16 novel, never all-empty and never all-repeat.
- The live `serve` path is behaviorally unchanged (still `utcnow`).
- `mypy --strict` and ruff clean; offline suite green with the new tests.

## Out of scope / future

- Per-alert as-of-its-own-timestamp history (richer temporal realism).
- Sourcing history from a real feed (the `AlertHistoryStore` seam already supports it).
- Excluding the alert-under-triage from its own observable's matches (needs the tool to know the
  current alert id — a cross-cutting change not justified here).
