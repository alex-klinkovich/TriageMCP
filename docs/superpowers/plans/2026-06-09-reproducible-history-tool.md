# Reproducible, Realistic Alert-History Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `query_recent_alerts` deterministic and date-independent in the eval/experiment paths, and make it return a realistic repeat-vs-novel mix instead of flagging every alert as a repeat offender — while the live `serve`/`triage` paths keep real time.

**Architecture:** Two independent changes. (1) Rewrite `_history_from_samples` so only observables that genuinely recur across ≥2 sample alerts get a seeded prior sighting; singletons stay novel. (2) Add an injectable, dataset-anchored clock (`max(alert.timestamp) + 1h`) and thread it through `build_runtime` into the eval and experiment call sites only; `serve`/`triage` pass nothing and keep `utcnow`.

**Tech Stack:** Python 3.12, Pydantic v2, aiosqlite, pytest/pytest-asyncio, mypy --strict, ruff.

**Commit convention:** end every commit message with
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Spec:** `docs/superpowers/specs/2026-06-09-reproducible-history-tool-design.md`

---

## File map

- Modify: `src/triagemcp/eval/harness.py` — add `eval_reference_clock(labeled)`.
- Modify: `src/triagemcp/server.py` — `build_runtime` gains a keyword-only `clock` param, forwarded to `build_default_registry`.
- Modify: `src/triagemcp/tools/registry.py` — rewrite `_history_from_samples` to seed only recurring observables.
- Modify: `src/triagemcp/eval/experiments.py` — `run_experiment` injects the eval clock.
- Modify: `src/triagemcp/cli.py` — `_evaluate` injects the eval clock.
- Modify: `README.md` — "Known simplifications" wording.
- Modify: `docs/experiments/mitre-accuracy.md` — one line on history reproducibility.
- Test: `tests/test_eval.py`, `tests/test_server.py`, `tests/test_registry.py`, `tests/test_experiments.py`.

---

## Task 1: `eval_reference_clock` helper

**Files:**
- Modify: `src/triagemcp/eval/harness.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py` (add `import datetime as dt` near the top imports, and add `eval_reference_clock` to the existing `from triagemcp.eval.harness import ...` line):

```python
def test_eval_reference_clock_anchors_just_after_newest_alert() -> None:
    labeled = load_sample_alerts()
    clock = eval_reference_clock(labeled)
    newest = max(item.alert.timestamp for item in labeled)
    assert clock() == newest + dt.timedelta(hours=1)
    assert clock() == clock()  # frozen: same value on every call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval.py::test_eval_reference_clock_anchors_just_after_newest_alert -v`
Expected: FAIL — `ImportError: cannot import name 'eval_reference_clock'`.

- [ ] **Step 3: Write minimal implementation**

In `src/triagemcp/eval/harness.py`, add `import datetime as dt` to the imports, add `Callable` to the `from collections.abc import Mapping, Sequence` line (→ `from collections.abc import Callable, Mapping, Sequence`), then add:

```python
def eval_reference_clock(labeled: Sequence[LabeledAlert]) -> Callable[[], dt.datetime]:
    """A fixed clock anchored just after the newest labeled alert.

    Injected into the eval/experiment paths so ``query_recent_alerts`` returns the same results
    regardless of the calendar date the run happens. The +1h buffer only needs to be > 0; the
    repeat/novel classification is insensitive to its exact size.
    """
    anchor = max(item.alert.timestamp for item in labeled) + dt.timedelta(hours=1)
    return lambda: anchor
```

(`LabeledAlert` is already imported in this module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval.py::test_eval_reference_clock_anchors_just_after_newest_alert -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/eval/harness.py tests/test_eval.py
git commit -m "feat: add eval_reference_clock for reproducible eval history"
```

---

## Task 2: thread an injectable clock through `build_runtime`

**Files:**
- Modify: `src/triagemcp/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py` (`dt`, `Settings`, `build_default_registry`, and `build_runtime` are already imported):

```python
async def test_build_runtime_forwards_injected_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    captured: dict[str, object] = {}
    real_build = build_default_registry

    async def _spy(conn: Any, *, clock: Any = None) -> Any:
        captured["clock"] = clock
        return await real_build(conn, clock=clock)

    monkeypatch.setattr("triagemcp.server.build_default_registry", _spy)
    fixed = dt.datetime(2026, 5, 29, 4, 55, tzinfo=dt.UTC)
    async with build_runtime(Settings(), clock=lambda: fixed) as _triager:
        pass
    assert captured["clock"] is not None
    assert captured["clock"]() == fixed  # type: ignore[operator]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py::test_build_runtime_forwards_injected_clock -v`
Expected: FAIL — `TypeError: build_runtime() got an unexpected keyword argument 'clock'`.

- [ ] **Step 3: Write minimal implementation**

In `src/triagemcp/server.py`:
- Add `import datetime as dt` to the stdlib imports.
- Change `from collections.abc import AsyncIterator` to `from collections.abc import AsyncIterator, Callable`.
- Update `build_runtime`'s signature and the registry build call:

```python
@contextlib.asynccontextmanager
async def build_runtime(
    settings: Settings,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    clock: Callable[[], dt.datetime] | None = None,
) -> AsyncIterator[TriageAgent]:
    """Construct the production triager and guarantee its resources are released."""
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    conn = await aiosqlite.connect(settings.db_path)
    try:
        registry = await build_default_registry(conn, clock=clock)
        config = AgentConfig(
            model=model or settings.model,
            max_tokens=settings.max_tokens,
            max_iterations=settings.max_iterations,
            per_alert_timeout_s=settings.per_alert_timeout_s,
            max_retries=settings.max_retries,
            temperature=temperature,
        )
        if system_prompt is not None:
            config = replace(config, system_prompt=system_prompt)
        agent = TriageAgent(AnthropicLLMClient(anthropic_client), registry, config)
        yield agent
    finally:
        await conn.close()
        await anthropic_client.close()
```

(`build_default_registry(conn, clock=None)` already defaults the query tool to `utcnow`, so `serve_stdio`/`triage` — which pass no clock — are unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -v`
Expected: PASS (new test and the existing `test_build_runtime_yields_a_triage_agent`).

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/server.py tests/test_server.py
git commit -m "feat: forward an injectable clock through build_runtime"
```

---

## Task 3: recurrence-based history seeding

**Files:**
- Modify: `src/triagemcp/tools/registry.py:69-98` (`_history_from_samples`)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registry.py` (extend the registry import to `from triagemcp.tools.registry import ToolRegistry, _history_from_samples, build_default_registry`, and add `from triagemcp.datasets import load_sample_alerts`):

```python
def test_history_seeds_exactly_the_recurring_observables() -> None:
    # Independently compute which (value, type) appear in >= 2 distinct alerts.
    seen_in: dict[tuple[str, str], set[str]] = {}
    for labeled in load_sample_alerts():
        a = labeled.alert
        o = a.observables
        typed = [
            *((str(ip), "ip") for ip in o.ips),
            *((h, "hash") for h in o.file_hashes),
            *((d, "domain") for d in o.domains),
            *((u, "user") for u in o.users),
            *((host, "host") for host in o.hosts),
        ]
        for value, typ in typed:
            seen_in.setdefault((value, typ), set()).add(a.id)

    expected_recurring = {key for key, ids in seen_in.items() if len(ids) >= 2}
    expected_singletons = {key for key, ids in seen_in.items() if len(ids) == 1}

    seeded = {(e.observable, e.observable_type) for e in _history_from_samples()}

    assert seeded == expected_recurring
    assert seeded.isdisjoint(expected_singletons)
    assert ("FIN-WS-118", "host") in seeded  # sanity anchor: appears in A-0003 and A-0023
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_registry.py::test_history_seeds_exactly_the_recurring_observables -v`
Expected: FAIL — current `_history_from_samples` seeds *every* observable (singletons too), so `seeded == expected_recurring` is False (seeded is a superset).

- [ ] **Step 3: Write minimal implementation**

Replace the entire `_history_from_samples` function in `src/triagemcp/tools/registry.py` with:

```python
def _history_from_samples() -> list[HistoryEntry]:
    """Seed prior-sighting rows for observables that genuinely recur across multiple alerts.

    A "repeat offender" is an observable (IP, host, hash, user, domain) that appears in two or
    more *distinct* sample alerts. Each such observable gets exactly one synthetic prior-sighting
    row, timestamped just before its earliest occurrence. Observables unique to a single alert
    are left unseeded, so they read as genuinely novel. This avoids fabricating a self-sighting
    for every alert (which made every alert look like a repeat) while staying fully deterministic.
    """
    # (value, type) -> {alert_id: (timestamp, severity_reported)} for each distinct containing alert
    by_observable: dict[tuple[str, str], dict[str, tuple[dt.datetime, str]]] = {}
    for labeled in load_sample_alerts():
        alert = labeled.alert
        obs = alert.observables
        typed_values: list[tuple[str, str]] = [
            *((str(ip), "ip") for ip in obs.ips),
            *((file_hash, "hash") for file_hash in obs.file_hashes),
            *((domain, "domain") for domain in obs.domains),
            *((user, "user") for user in obs.users),
            *((host, "host") for host in obs.hosts),
        ]
        for value, observable_type in typed_values:
            by_observable.setdefault((value, observable_type), {})[alert.id] = (
                alert.timestamp,
                alert.severity_reported.value,
            )

    entries: list[HistoryEntry] = []
    for (value, observable_type), occurrences in by_observable.items():
        if len(occurrences) < 2:
            continue  # appears in only one alert -> genuinely novel, no seeded history
        earliest_ts, earliest_severity = min(occurrences.values(), key=lambda pair: pair[0])
        entries.append(
            HistoryEntry(
                alert_id=f"HIST-{observable_type}-{value}",
                observable=value,
                observable_type=observable_type,
                title=f"Prior activity involving {value}",
                severity=earliest_severity,
                seen_at=earliest_ts - dt.timedelta(hours=1),
            )
        )
    return entries
```

(No new imports needed: `dt`, `HistoryEntry`, and `load_sample_alerts` are already imported in this module.)

- [ ] **Step 4: Run the registry tests to verify the new test passes and the existing ones still pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_registry.py -v`
Expected: PASS for all, including the pre-existing `test_default_history_records_are_distinct_prior_sightings` (it queries `203.0.113.45`, which recurs in A-0001 & A-0002 → still seeded as `HIST-ip-203.0.113.45`, within its 336h lookback) and `test_build_default_registry_wires_all_four_tools`.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/tools/registry.py tests/test_registry.py
git commit -m "feat: seed alert history from genuine cross-alert recurrence only"
```

---

## Task 4: inject the eval clock into the eval and experiment paths

**Files:**
- Modify: `src/triagemcp/eval/experiments.py`
- Modify: `src/triagemcp/cli.py` (`_evaluate`)
- Test: `tests/test_experiments.py`, `tests/test_eval.py`

- [ ] **Step 1: Write the failing wiring test**

Add to `tests/test_experiments.py` (add `import datetime as dt`):

```python
async def test_run_experiment_injects_dataset_anchored_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from triagemcp.datasets import load_sample_alerts

    labels = {la.alert.id: la.label for la in load_sample_alerts()}
    captured: dict[str, object] = {}

    @contextlib.asynccontextmanager
    async def _fake_runtime(
        _settings: Settings, *, clock: object = None, **_kwargs: object
    ) -> AsyncIterator[_LabelEchoTriager]:
        captured["clock"] = clock
        yield _LabelEchoTriager(labels)

    monkeypatch.setattr("triagemcp.eval.experiments.build_runtime", _fake_runtime)
    await run_experiment(
        "baseline", settings=Settings(), model="claude-haiku-4-5", concurrency=2
    )

    newest = max(la.alert.timestamp for la in load_sample_alerts())
    assert captured["clock"] is not None
    assert captured["clock"]() == newest + dt.timedelta(hours=1)  # type: ignore[operator]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_experiments.py::test_run_experiment_injects_dataset_anchored_clock -v`
Expected: FAIL — `run_experiment` does not pass `clock`, so `captured["clock"]` is `None` and `assert ... is not None` fails.

- [ ] **Step 3: Implement the experiment-path wiring**

In `src/triagemcp/eval/experiments.py`, change the harness import to include the helper and inject the clock:

```python
from triagemcp.eval.harness import eval_reference_clock, run_eval
```

```python
async def run_experiment(
    variant: str, *, settings: Settings, model: str, concurrency: int
) -> EvalReport:
    """Triage the labeled set with the given prompt variant and score the result."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    labeled = load_sample_alerts()
    tactic_by_id = {t.id: t.tactic for t in load_mitre_techniques()}
    system_prompt = build_system_prompt(PROMPT_VARIANTS[variant])
    clock = eval_reference_clock(labeled)
    async with build_runtime(
        settings, model=model, system_prompt=system_prompt, temperature=0.0, clock=clock
    ) as triager:
        return await run_eval(labeled, triager, concurrency=concurrency, tactic_by_id=tactic_by_id)
```

- [ ] **Step 4: Implement the eval-path (CLI) wiring**

In `src/triagemcp/cli.py`, add `eval_reference_clock` to the harness import:

```python
from triagemcp.eval.harness import eval_reference_clock, run_eval, write_headline_to_readme
```

and update `_evaluate`:

```python
async def _evaluate(settings: Settings, concurrency: int) -> EvalReport:
    labeled = load_sample_alerts()
    tactic_by_id = {technique.id: technique.tactic for technique in load_mitre_techniques()}
    clock = eval_reference_clock(labeled)
    async with build_runtime(settings, clock=clock) as triager:
        return await run_eval(labeled, triager, concurrency=concurrency, tactic_by_id=tactic_by_id)
```

(Leave `_triage` unchanged — `triage` keeps `utcnow` for real alerts.)

- [ ] **Step 5: Write the integration behavioral test**

Add to `tests/test_eval.py` (add `import aiosqlite` and `from triagemcp.tools.registry import build_default_registry`; `eval_reference_clock` is imported from Task 1):

```python
async def test_recurring_observable_is_in_default_window_under_eval_clock() -> None:
    labeled = load_sample_alerts()
    clock = eval_reference_clock(labeled)
    async with aiosqlite.connect(":memory:") as conn:
        registry = await build_default_registry(conn, clock=clock)
        # No explicit lookback_hours -> the DEFAULT 168h window. Under utcnow this returns 0
        # today; under the dataset-anchored clock it deterministically returns the prior sighting.
        result = await registry.dispatch(
            "query_recent_alerts", {"observable": "FIN-WS-118", "observable_type": "host"}
        )
    assert result.content["match_count"] >= 1
```

- [ ] **Step 6: Run the affected suites to verify everything passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_experiments.py tests/test_eval.py tests/test_cli.py -v`
Expected: PASS. (The existing `test_run_experiment_scores_against_labels` still passes — its fake runtime already absorbs extra kwargs; the CLI `triage` test is unaffected because `_triage` is unchanged.)

- [ ] **Step 7: Commit**

```bash
git add src/triagemcp/eval/experiments.py src/triagemcp/cli.py tests/test_experiments.py tests/test_eval.py
git commit -m "feat: inject the dataset-anchored clock into eval and experiment runs"
```

---

## Task 5: documentation

**Files:**
- Modify: `README.md` ("Known simplifications" section)
- Modify: `docs/experiments/mitre-accuracy.md`

- [ ] **Step 1: Update the README "Known simplifications" wording**

In `README.md`, replace this sentence:

```
`query_recent_alerts` is seeded with synthetic prior
sightings derived from the sample set.
```

with:

```
`query_recent_alerts` is seeded from genuine cross-alert overlap in the sample set — an
observable that recurs across two or more alerts gets one prior-sighting row, while observables
unique to a single alert stay novel. The eval and experiment runs use a clock anchored to the
dataset (just after the newest alert), so this tool's results are deterministic and independent
of the calendar date; the live server uses real time.
```

- [ ] **Step 2: Add a reproducibility note to the experiment write-up**

In `docs/experiments/mitre-accuracy.md`, in the `**Method.**` paragraph, append this sentence:

```
The `query_recent_alerts` tool runs against a clock anchored to the dataset, so its history
results are reproducible across dates rather than decaying as wall-clock time advances.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/experiments/mitre-accuracy.md
git commit -m "docs: describe recurrence-based seeding and the dataset-anchored eval clock"
```

---

## Task 6: full verification gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full offline suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all tests pass (128 prior + 4 new = 132), 1 skipped (live). If any pre-existing test regressed, fix it before proceeding — do not weaken assertions to make them pass.

- [ ] **Step 2: Type-check**

Run: `.venv\Scripts\python.exe -m mypy`
Expected: `Success: no issues found`. If `by_observable`'s annotation trips strict mode, keep the explicit `dict[tuple[str, str], dict[str, tuple[dt.datetime, str]]]` annotation shown in Task 3 (it is the mypy-clean form).

- [ ] **Step 3: Lint + format check**

Run: `.venv\Scripts\python.exe -m ruff check . ; .venv\Scripts\python.exe -m ruff format --check .`
Expected: `All checks passed!` and no formatting diffs.

- [ ] **Step 4: Manual reproducibility sanity check (optional, no API key needed)**

Confirm the behavioral win end-to-end: the eval-clock path returns history matches for a recurring observable and nothing for a novel one, regardless of today's date. This is already locked by `test_recurring_observable_is_in_default_window_under_eval_clock`; no extra script required.

- [ ] **Step 5: Final commit (only if Steps 1–3 required fixes)**

```bash
git add -A -- src tests README.md docs
git commit -m "fix: address verification findings for the history-tool change"
```

Note: never `git add -A` from the repo root without the `-- src tests README.md docs` pathspec — the untracked personal files (`before_interview.md`, `cv_info.md`) must never be staged.

---

## Self-review notes

- **Spec coverage:** recurrence seeding (Task 3), dataset-anchored clock + threading (Tasks 1–2), eval/experiment-only injection with serve/triage unchanged (Task 4), tests for each (Tasks 1–4), docs (Task 5). All spec sections map to a task.
- **Type consistency:** `eval_reference_clock(labeled) -> Callable[[], dt.datetime]` is defined in Task 1 and consumed identically in Task 2's `clock` param and Task 4's call sites. `build_default_registry(conn, *, clock=...)` already exists and is called with the same keyword in Tasks 2–4.
- **No placeholders:** every code step shows complete code; every run step shows the exact command and expected result.
- **Watch-item:** the pre-existing `test_default_history_records_are_distinct_prior_sightings` is expected to keep passing (verified in Task 3 Step 4); if it fails, the cause is the chosen recurring anchor observable, not a weakened assertion — investigate before changing it.
