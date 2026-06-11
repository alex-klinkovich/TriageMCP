# Accuracy-Measurement Harness (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trustworthy accuracy-measurement harness — ordinal within-one action accuracy, ECE/calibration, and leave-one-out cross-validated variant selection over persisted per-alert verdicts — so any later technique gain is believable.

**Architecture:** Add an ordinal `.distance` to `RecommendedAction` (mirroring `Severity`); extend `score()`/`EvalReport` with within-one action accuracy and an ECE + reliability table; add a pure `eval/crossval.py` that does leave-one-out variant selection over a `{variant: {alert_id: TriageResult}}` grid and persists/reloads it; expose a `triagemcp crossval` CLI. All scoring is pure and offline-tested; the only live step is running the variants once.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, rich, pytest, mypy --strict, ruff.

**Commit convention:** end every commit message with
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Spec:** `docs/superpowers/specs/2026-06-11-measurement-rigor-design.md`

**Run commands** (Windows; venv-prefixed): tests `\.venv\Scripts\python.exe -m pytest`, types `\.venv\Scripts\python.exe -m mypy`, lint `\.venv\Scripts\python.exe -m ruff check .`.

---

## File map

- Modify: `src/triagemcp/models.py` — `RecommendedAction.level` / `.distance`.
- Modify: `src/triagemcp/eval/metrics.py` — `action_within_one_accuracy`, `ReliabilityBin`, `calibration()`, `ece` + `reliability` on `EvalReport`.
- Create: `src/triagemcp/eval/crossval.py` — `CrossValReport`, `cross_validate`, `VerdictArtifact`, `collect_variant_verdicts`.
- Modify: `src/triagemcp/report.py` — within-one action + ECE rows; `crossval_report_table`.
- Modify: `src/triagemcp/cli.py` — `crossval` command.
- Modify: `README.md`; Create: `docs/experiments/measurement-rigor.md`.
- Tests: `tests/test_models.py`, `tests/test_eval.py`, `tests/test_crossval.py` (new), `tests/test_report.py`, `tests/test_cli.py`.

---

## Task 1: ordinal `RecommendedAction`

**Files:**
- Modify: `src/triagemcp/models.py` (the `RecommendedAction` class)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
def test_recommended_action_ranks_close_fp_to_escalate() -> None:
    assert [a.level for a in RecommendedAction] == [0, 1, 2, 3, 4]


def test_recommended_action_distance_is_absolute_rank_gap() -> None:
    assert RecommendedAction.ESCALATE.distance(RecommendedAction.CONTAIN) == 1
    assert RecommendedAction.CLOSE_FALSE_POSITIVE.distance(RecommendedAction.ESCALATE) == 4
    assert RecommendedAction.MONITOR.distance(RecommendedAction.MONITOR) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_models.py::test_recommended_action_ranks_close_fp_to_escalate -v`
Expected: FAIL — `AttributeError: 'RecommendedAction' object has no attribute 'level'`.

- [ ] **Step 3: Write minimal implementation**

In `src/triagemcp/models.py`, replace the `RecommendedAction` class body's trailing members with the same `.level`/`.distance` helpers `Severity` has. The full class becomes:

```python
class RecommendedAction(StrEnum):
    """The action a SOC analyst should take, from benign-close to full escalation.

    Definition order encodes the escalation ladder (rank 0 = close, 4 = escalate)."""

    CLOSE_FALSE_POSITIVE = "close_false_positive"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    CONTAIN = "contain"
    ESCALATE = "escalate"

    @property
    def level(self) -> int:
        """Rank from 0 (close_false_positive) to 4 (escalate)."""
        return list(RecommendedAction).index(self)

    def distance(self, other: RecommendedAction) -> int:
        """Absolute number of rungs between two actions (used by the eval)."""
        return abs(self.level - other.level)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_models.py -q`
Expected: PASS (all model tests).

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/models.py tests/test_models.py
git commit -m "feat: give RecommendedAction an ordinal escalation ladder"
```

---

## Task 2: within-one action accuracy

**Files:**
- Modify: `src/triagemcp/eval/metrics.py` (`EvalReport`, `score`)
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py` (uses the existing `_result`, `_label`, `score`, `TACTIC` helpers):

```python
def test_score_reports_within_one_action() -> None:
    outcomes = [
        TriageOutcome.success(
            _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN),
            iterations=1,
            latency_ms=1.0,
        ),
        TriageOutcome.success(
            _result("A2", Severity.HIGH, "T1059.001", RecommendedAction.ESCALATE),
            iterations=1,
            latency_ms=1.0,
        ),
    ]
    labels = {
        "A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.ESCALATE),  # 1 rung off
        "A2": _label(Severity.HIGH, "T1059.001", RecommendedAction.MONITOR),  # 2 rungs off
    }
    report = score(outcomes, labels, TACTIC)
    assert report.action_accuracy == pytest.approx(0.0)  # neither exact
    assert report.action_within_one_accuracy == pytest.approx(0.5)  # A1 within one, A2 not
```

- [ ] **Step 2: Run test to verify it fails**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_eval.py::test_score_reports_within_one_action -v`
Expected: FAIL — `AttributeError: 'EvalReport' object has no attribute 'action_within_one_accuracy'`.

- [ ] **Step 3: Write minimal implementation**

In `src/triagemcp/eval/metrics.py`:

(a) Add the field to `EvalReport`, immediately after `action_accuracy: float`:

```python
    action_accuracy: float
    action_within_one_accuracy: float = 0.0
```

(b) In `score()`, in the **`if scored == 0:`** early-return `EvalReport(...)`, add `action_within_one_accuracy=0.0,` after `action_accuracy=0.0,`.

(c) In `score()`, after the line `action = action_n / scored`, add:

```python
    action_within = (
        sum(pred.recommended_action.distance(lab.recommended_action) <= 1 for pred, lab in pairs)
        / scored
    )
```

(d) In the final `return EvalReport(...)`, add `action_within_one_accuracy=action_within,` after `action_accuracy=action,`.

- [ ] **Step 4: Run test to verify it passes**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/eval/metrics.py tests/test_eval.py
git commit -m "feat: add ordinal within-one action accuracy to the eval"
```

---

## Task 3: ECE / calibration

**Files:**
- Modify: `src/triagemcp/eval/metrics.py` (`ReliabilityBin`, `calibration`, `EvalReport`, `score`)
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
def test_calibration_computes_ece_and_bins() -> None:
    from triagemcp.eval.metrics import calibration

    # All three correct -> "correct"; any mismatch -> "incorrect". Confidences chosen to land in
    # two 5-bins: 0.9 (bin [0.8,1.0)) and 0.1 (bin [0.0,0.2)).
    correct = (_result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN, conf=0.9),
               _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN))
    wrong = (_result("A2", Severity.LOW, "T1110", RecommendedAction.MONITOR, conf=0.9),
             _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN))
    ece, bins = calibration([correct, wrong], bins=5)
    # One bin (conf 0.9, two items, accuracy 0.5): |0.5 - 0.9| = 0.4 over the whole set.
    assert ece == pytest.approx(0.4, abs=1e-9)
    assert len(bins) == 1
    assert bins[0].count == 2
    assert bins[0].accuracy == pytest.approx(0.5)


def test_score_exposes_ece() -> None:
    outcomes = [
        TriageOutcome.success(
            _result("A1", Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN, conf=0.9),
            iterations=1,
            latency_ms=1.0,
        )
    ]
    labels = {"A1": _label(Severity.HIGH, "T1059.001", RecommendedAction.CONTAIN)}
    report = score(outcomes, labels, TACTIC)
    assert report.ece == pytest.approx(0.1)  # fully correct, conf 0.9 -> |1.0 - 0.9|
    assert report.reliability[0].count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_eval.py::test_calibration_computes_ece_and_bins -v`
Expected: FAIL — `ImportError: cannot import name 'calibration'`.

- [ ] **Step 3: Write minimal implementation**

In `src/triagemcp/eval/metrics.py`:

(a) Define `ReliabilityBin` **above** `EvalReport`:

```python
class ReliabilityBin(BaseModel):
    """One confidence bucket of the reliability table."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
```

(b) Add the function (place it after `wilson_interval`):

```python
def calibration(
    pairs: Sequence[tuple[TriageResult, AlertLabel]], *, bins: int = 5
) -> tuple[float, list[ReliabilityBin]]:
    """Expected Calibration Error + reliability table over confidence buckets.

    Verdict correctness is the strict conjunction (severity exact AND technique AND action exact),
    since ``confidence`` is the model's confidence in the whole verdict. ECE = sum over non-empty
    bins of (n_b / N) * |accuracy_b - mean_confidence_b|. Equal-width bins; the top bin includes 1.0.
    """
    records = [
        (
            pred.confidence,
            pred.severity == lab.severity
            and pred.mitre_technique_id == lab.mitre_technique_id
            and pred.recommended_action == lab.recommended_action,
        )
        for pred, lab in pairs
    ]
    n = len(records)
    if n == 0:
        return 0.0, []
    table: list[ReliabilityBin] = []
    ece = 0.0
    for b in range(bins):
        lower = b / bins
        upper = (b + 1) / bins
        in_bin = [
            (conf, ok)
            for conf, ok in records
            if (lower <= conf < upper) or (b == bins - 1 and conf == upper)
        ]
        if not in_bin:
            continue
        count = len(in_bin)
        mean_conf = sum(conf for conf, _ in in_bin) / count
        accuracy = sum(1 for _, ok in in_bin if ok) / count
        ece += (count / n) * abs(accuracy - mean_conf)
        table.append(
            ReliabilityBin(
                lower=lower, upper=upper, count=count, mean_confidence=mean_conf, accuracy=accuracy
            )
        )
    return ece, table
```

(c) Add fields to `EvalReport` after `severity_confusion`:

```python
    severity_confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    ece: float = 0.0
    reliability: list[ReliabilityBin] = Field(default_factory=list)
```

(d) In `score()`'s `if scored == 0:` branch the defaults already apply (`ece`/`reliability` have field defaults), so no change there. In the final `return EvalReport(...)`, compute and pass it — add before the `return`:

```python
    ece, reliability = calibration(pairs)
```

and add `ece=ece, reliability=reliability,` to the `EvalReport(...)` kwargs.

- [ ] **Step 4: Run test to verify it passes**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/eval/metrics.py tests/test_eval.py
git commit -m "feat: add ECE and a reliability table to the eval"
```

---

## Task 4: leave-one-out cross-validation

**Files:**
- Create: `src/triagemcp/eval/crossval.py`
- Test: `tests/test_crossval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_crossval.py`:

```python
"""Spec for leave-one-out cross-validated variant selection (pure, offline)."""

from __future__ import annotations

import pytest

from triagemcp.eval.crossval import cross_validate
from triagemcp.models import AlertLabel, RecommendedAction, Severity, TriageResult

TACTIC = {"T1059.001": "Execution", "T1110": "Credential Access"}


def _label() -> AlertLabel:
    return AlertLabel(
        severity=Severity.HIGH, mitre_technique_id="T1059.001", recommended_action=RecommendedAction.CONTAIN
    )


def _verdict(alert_id: str, *, correct: bool) -> TriageResult:
    if correct:
        return TriageResult(
            alert_id=alert_id, severity=Severity.HIGH, confidence=0.9, mitre_technique_id="T1059.001",
            mitre_technique_name="x", recommended_action=RecommendedAction.CONTAIN, rationale="r",
        )
    return TriageResult(
        alert_id=alert_id, severity=Severity.LOW, confidence=0.9, mitre_technique_id="T1110",
        mitre_technique_name="x", recommended_action=RecommendedAction.MONITOR, rationale="r",
    )


def test_loo_detects_a_non_generalizing_variant_choice() -> None:
    # va is correct on A1,A2 and wrong on A3,A4; vb is the mirror. In-sample they tie at 0.5, but
    # leave-one-out the winning variant flips per fold and is always wrong on the held-out alert.
    labels = {f"A{i}": _label() for i in range(1, 5)}
    va = {f"A{i}": _verdict(f"A{i}", correct=i in (1, 2)) for i in range(1, 5)}
    vb = {f"A{i}": _verdict(f"A{i}", correct=i in (3, 4)) for i in range(1, 5)}

    report = cross_validate({"va": va, "vb": vb}, labels, TACTIC)

    assert report.in_sample_best_variant == "va"  # tie -> alphabetical
    assert report.in_sample_best.overall_accuracy == pytest.approx(0.5)
    assert report.out_of_fold.overall_accuracy == pytest.approx(0.0)
    assert report.selection_optimism == pytest.approx(0.5)
    assert set(report.selected_variant_by_alert.values()) == {"va", "vb"}
    assert report.out_of_fold.scored == 4


def test_loo_falls_back_when_selected_variant_errored_on_held_out() -> None:
    labels = {f"A{i}": _label() for i in range(1, 4)}
    # va wins selection everywhere but has NO verdict for A1; vb covers A1.
    va = {"A2": _verdict("A2", correct=True), "A3": _verdict("A3", correct=True)}
    vb = {f"A{i}": _verdict(f"A{i}", correct=True) for i in range(1, 4)}
    report = cross_validate({"va": va, "vb": vb}, labels, TACTIC)
    assert report.out_of_fold.scored == 3  # A1 falls back to vb, not dropped
    assert report.selected_variant_by_alert["A1"] == "vb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_crossval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'triagemcp.eval.crossval'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/triagemcp/eval/crossval.py`:

```python
"""Leave-one-out cross-validated prompt-variant selection over persisted per-alert verdicts.

Selection on a subset counts a variant's errored (missing) alerts as incorrect, so a variant is
never rewarded for skipping hard alerts. The pooled out-of-fold and the in-sample per-variant
reports use the normal scoring (errors excluded), consistent with the rest of the eval.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from triagemcp.eval.metrics import EvalReport, score
from triagemcp.models import AlertLabel, TriageOutcome, TriageResult


class CrossValReport(BaseModel):
    """Out-of-fold accuracy plus the in-sample best variant and the selection-optimism gap."""

    model_config = ConfigDict(extra="forbid")

    out_of_fold: EvalReport
    in_sample_best_variant: str
    in_sample_best: EvalReport
    selection_optimism: float
    selected_variant_by_alert: dict[str, str]


def _per_alert_overall(verdict: TriageResult | None, label: AlertLabel) -> float:
    if verdict is None:
        return 0.0
    sev = verdict.severity == label.severity
    tech = verdict.mitre_technique_id == label.mitre_technique_id
    act = verdict.recommended_action == label.recommended_action
    return (sev + tech + act) / 3


def _selection_score(
    verdicts: Mapping[str, TriageResult], labels: Mapping[str, AlertLabel], alert_ids: list[str]
) -> float:
    if not alert_ids:
        return 0.0
    return sum(_per_alert_overall(verdicts.get(aid), labels[aid]) for aid in alert_ids) / len(alert_ids)


def _outcomes_for(
    picks: Mapping[str, TriageResult | None], labels: Mapping[str, AlertLabel]
) -> list[TriageOutcome]:
    outcomes: list[TriageOutcome] = []
    for aid in labels:
        verdict = picks.get(aid)
        if verdict is None:
            outcomes.append(TriageOutcome.failure(aid, "no verdict", iterations=0, latency_ms=0.0))
        else:
            outcomes.append(TriageOutcome.success(verdict, iterations=0, latency_ms=0.0))
    return outcomes


def cross_validate(
    verdicts_by_variant: Mapping[str, Mapping[str, TriageResult]],
    labels: Mapping[str, AlertLabel],
    tactic_by_id: Mapping[str, str],
) -> CrossValReport:
    alert_ids = sorted(labels)
    variants = sorted(verdicts_by_variant)

    in_sample: dict[str, EvalReport] = {}
    for variant in variants:
        picks: dict[str, TriageResult | None] = {
            aid: verdicts_by_variant[variant].get(aid) for aid in alert_ids
        }
        in_sample[variant] = score(_outcomes_for(picks, labels), labels, tactic_by_id)
    # max() returns the first element achieving the max in iteration order; variants is sorted,
    # so ties go to the alphabetically-first variant (deterministic).
    best_variant = max(variants, key=lambda v: in_sample[v].overall_accuracy)

    selected_by_alert: dict[str, str] = {}
    of_picks: dict[str, TriageResult | None] = {}
    for held in alert_ids:
        train_ids = [aid for aid in alert_ids if aid != held]
        # Precompute scores into a dict, then sort tuples — avoids a loop-variable closure (ruff B023).
        scores = {
            variant: _selection_score(verdicts_by_variant[variant], labels, train_ids)
            for variant in variants
        }
        ranked = [name for _, name in sorted((-scores[variant], variant) for variant in variants)]
        verdict: TriageResult | None = None
        chosen = ranked[0]
        for variant in ranked:
            candidate = verdicts_by_variant[variant].get(held)
            if candidate is not None:
                verdict, chosen = candidate, variant
                break
        of_picks[held] = verdict
        selected_by_alert[held] = chosen
    out_of_fold = score(_outcomes_for(of_picks, labels), labels, tactic_by_id)

    return CrossValReport(
        out_of_fold=out_of_fold,
        in_sample_best_variant=best_variant,
        in_sample_best=in_sample[best_variant],
        selection_optimism=in_sample[best_variant].overall_accuracy - out_of_fold.overall_accuracy,
        selected_variant_by_alert=selected_by_alert,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_crossval.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/eval/crossval.py tests/test_crossval.py
git commit -m "feat: leave-one-out cross-validated variant selection"
```

---

## Task 5: verdict persistence + live collection

**Files:**
- Modify: `src/triagemcp/eval/crossval.py`
- Test: `tests/test_crossval.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_crossval.py`:

```python
def test_verdict_artifact_round_trips() -> None:
    from triagemcp.eval.crossval import VerdictArtifact

    labels = {"A1": _label()}
    artifact = VerdictArtifact(
        verdicts_by_variant={"va": {"A1": _verdict("A1", correct=True)}}, labels=labels
    )
    again = VerdictArtifact.from_json(artifact.to_json())
    assert again == artifact
    assert again.verdicts_by_variant["va"]["A1"].alert_id == "A1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_crossval.py::test_verdict_artifact_round_trips -v`
Expected: FAIL — `ImportError: cannot import name 'VerdictArtifact'`.

- [ ] **Step 3: Write minimal implementation**

In `src/triagemcp/eval/crossval.py`, add imports at the top:

```python
from typing import Self

from triagemcp.agent.prompts import build_system_prompt
from triagemcp.config import Settings
from triagemcp.datasets import load_sample_alerts
from triagemcp.eval.experiments import PROMPT_VARIANTS
from triagemcp.eval.harness import eval_reference_clock
from triagemcp.pipeline import triage_batch
from triagemcp.server import build_runtime
```

Then add the artifact model and the live collector at the end of the module:

```python
class VerdictArtifact(BaseModel):
    """Persisted per-alert verdicts per variant plus the labels, for offline re-scoring."""

    model_config = ConfigDict(extra="forbid")

    verdicts_by_variant: dict[str, dict[str, TriageResult]]
    labels: dict[str, AlertLabel]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.model_validate_json(text)


async def collect_variant_verdicts(
    settings: Settings, *, model: str, concurrency: int
) -> dict[str, dict[str, TriageResult]]:
    """Run every prompt variant over the labeled set (live) and collect per-alert verdicts."""
    labeled = load_sample_alerts()
    alerts = [item.alert for item in labeled]
    clock = eval_reference_clock(labeled)
    collected: dict[str, dict[str, TriageResult]] = {}
    for variant, options in PROMPT_VARIANTS.items():
        system_prompt = build_system_prompt(options)
        async with build_runtime(
            settings, model=model, system_prompt=system_prompt, temperature=0.0, clock=clock
        ) as triager:
            outcomes = await triage_batch(alerts, triager, concurrency=concurrency)
        collected[variant] = {o.alert_id: o.result for o in outcomes if o.result is not None}
    return collected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_crossval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triagemcp/eval/crossval.py tests/test_crossval.py
git commit -m "feat: persist per-alert verdicts and collect them per variant"
```

---

## Task 6: reporting + `crossval` CLI

**Files:**
- Modify: `src/triagemcp/report.py`
- Modify: `src/triagemcp/cli.py`
- Test: `tests/test_report.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_report.py` (extend the existing `EvalReport(...)` construction in `test_eval_report_table_shows_overall_percentage` is fine to leave; add a new test):

```python
def test_eval_report_table_shows_action_within_one_and_ece() -> None:
    report = EvalReport(
        total=1, scored=1, errors=0, severity_exact_accuracy=1.0, severity_within_one_accuracy=1.0,
        mitre_technique_accuracy=1.0, mitre_tactic_accuracy=1.0, action_accuracy=1.0,
        action_within_one_accuracy=1.0, overall_accuracy=1.0, mean_confidence=0.9, ece=0.25,
    )
    out = _render(eval_report_table(report))
    assert "Action within one" in out
    assert "ECE" in out
    assert "0.25" in out
```

Add to `tests/test_cli.py` (add `AlertLabel` to the existing
`from triagemcp.models import (...)` import):

```python
def test_crossval_from_file_renders_offline(tmp_path: Path) -> None:
    from triagemcp.datasets import load_sample_alerts
    from triagemcp.eval.crossval import VerdictArtifact

    labeled = load_sample_alerts()
    labels = {item.alert.id: item.label for item in labeled}

    def _verdict(alert_id: str, label: AlertLabel) -> TriageResult:
        return TriageResult(
            alert_id=alert_id, severity=label.severity, confidence=0.9,
            mitre_technique_id=label.mitre_technique_id, mitre_technique_name="x",
            recommended_action=label.recommended_action, rationale="r",
        )

    verdicts = {"baseline": {aid: _verdict(aid, lab) for aid, lab in labels.items()}}
    artifact = VerdictArtifact(verdicts_by_variant=verdicts, labels=labels)
    path = tmp_path / "verdicts.json"
    path.write_text(artifact.to_json(), encoding="utf-8")

    result = runner.invoke(app, ["crossval", "--from", str(path)])
    assert result.exit_code == 0, result.output
    assert "out-of-fold" in result.output.lower()


def test_crossval_without_key_or_file_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["crossval"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_report.py::test_eval_report_table_shows_action_within_one_and_ece tests/test_cli.py::test_crossval_from_file_renders_offline -v`
Expected: FAIL — table lacks the rows; `crossval` is not a command.

- [ ] **Step 3: Implement reporting**

In `src/triagemcp/report.py`, add the two rows to `eval_report_table`'s `rows` tuple — insert after the `("Action", ...)` entry and after `("Mean confidence", ...)`:

```python
        ("Action", f"{report.action_accuracy:.1%}"),
        ("Action within one", f"{report.action_within_one_accuracy:.1%}"),
        ("Overall", f"{report.overall_accuracy:.1%}"),
        ("Mean confidence", f"{report.mean_confidence:.2f}"),
        ("ECE", f"{report.ece:.2f}"),
    )
```

Then add a crossval renderer at the end of `report.py` (import is already `from triagemcp.eval.metrics import EvalReport`; add `CrossValReport`):

```python
from triagemcp.eval.crossval import CrossValReport  # add near the top imports


def crossval_report_table(report: CrossValReport) -> Table:
    """Render a CrossValReport: out-of-fold accuracy, in-sample best, and the optimism gap."""
    table = Table(title="Cross-validation (leave-one-out)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    oof = report.out_of_fold
    table.add_row("Out-of-fold overall", f"{oof.overall_accuracy:.1%}")
    table.add_row("Out-of-fold MITRE technique", f"{oof.mitre_technique_accuracy:.1%}")
    table.add_row("Out-of-fold action (within one)", f"{oof.action_within_one_accuracy:.1%}")
    table.add_row("In-sample best variant", report.in_sample_best_variant)
    table.add_row("In-sample best overall", f"{report.in_sample_best.overall_accuracy:.1%}")
    table.add_row("Selection optimism", f"{report.selection_optimism:.1%}")
    table.add_row("ECE (out-of-fold)", f"{oof.ece:.2f}")
    return table
```

Note: to avoid a circular import (`crossval` imports `metrics`, `report` imports both), keep the `from triagemcp.eval.crossval import CrossValReport` at the **top** of `report.py` with the other imports — `crossval.py` does not import `report.py`, so there is no cycle.

- [ ] **Step 4: Implement the CLI command**

In `src/triagemcp/cli.py`, add imports near the existing eval imports:

```python
from triagemcp.eval.crossval import VerdictArtifact, collect_variant_verdicts, cross_validate
from triagemcp.report import crossval_report_table
```

(Combine with the existing `from triagemcp.report import (...)` block rather than duplicating — add `crossval_report_table` to it, and add the crossval import on its own line.)

Add the command (after the `experiment` command):

```python
@app.command()
def crossval(
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Persist the per-alert verdicts here.")
    ] = None,
    from_file: Annotated[
        Path | None,
        typer.Option("--from", help="Recompute from a persisted verdicts file (no API calls)."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c", min=1)] = None,
) -> None:
    """Cross-validated variant selection + within-one action + ECE over the labeled set."""
    labeled = load_sample_alerts()
    labels = {item.alert.id: item.label for item in labeled}
    tactic_by_id = {technique.id: technique.tactic for technique in load_mitre_techniques()}

    if from_file is not None:
        artifact = VerdictArtifact.from_json(from_file.read_text(encoding="utf-8"))
    else:
        settings = _load_settings(model)
        verdicts = asyncio.run(
            collect_variant_verdicts(
                settings, model=settings.model, concurrency=concurrency or settings.concurrency
            )
        )
        artifact = VerdictArtifact(verdicts_by_variant=verdicts, labels=labels)
        if out is not None:
            out.write_text(artifact.to_json(), encoding="utf-8")
            console.print(f"Wrote verdicts to {out}")

    report = cross_validate(artifact.verdicts_by_variant, artifact.labels, tactic_by_id)
    console.print(crossval_report_table(report))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_report.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/triagemcp/report.py src/triagemcp/cli.py tests/test_report.py tests/test_cli.py
git commit -m "feat: crossval CLI command and report rendering"
```

---

## Task 7: documentation

**Files:**
- Create: `docs/experiments/measurement-rigor.md`
- Modify: `README.md`

- [ ] **Step 1: Write the experiment write-up**

Create `docs/experiments/measurement-rigor.md`:

```markdown
# Measurement rigor: cross-validation, ordinal action scoring, calibration

Phase 1 of the accuracy program makes the eval number trustworthy rather than bigger.

**Why.** The headline eval is in-sample (the prompt was tuned on the same 38 alerts), the action
metric is single-valid (undercounting defensible alternatives), and confidence is reported but its
calibration was never quantified.

**What this adds.**
- **Within-one action accuracy** — actions are ordered (close_false_positive < monitor < investigate
  < contain < escalate); being one rung off is credited, mirroring within-one severity. Principled,
  not per-alert cherry-picking.
- **ECE + reliability table** — bins verdicts by confidence and reports the expected calibration
  error against strict (all-three-exact) correctness. Quantifies the 0.93-confidence gap.
- **Leave-one-out cross-validation** — for each alert, the best prompt variant is chosen on the
  other 37 and used to predict the held-out one. The pooled out-of-fold accuracy vs the in-sample
  best-variant accuracy is the **selection-optimism gap**.

**How to run.** `triagemcp crossval --out verdicts.json` runs the variants once (the only API spend),
persists the per-alert verdicts, and prints the report. `triagemcp crossval --from verdicts.json`
recomputes everything offline for free.

**Reading it.** A small selection-optimism gap means the variant choice generalizes (with only four
variants there is little to overfit). Within-one action is reported alongside exact action; the
headline `overall_accuracy` is unchanged by this phase by design.
```

- [ ] **Step 2: Refresh the README testing/eval language**

In `README.md`, find the eval headline area and add one sentence after the `<!-- EVAL:END -->`
paragraph (the "Eval-driven iteration" block). Append:

```
**Trustworthy measurement:** `triagemcp crossval` reports leave-one-out cross-validated accuracy
(out-of-fold, not in-sample), ordinal within-one action accuracy, and a calibration error (ECE) —
all recomputable offline from one persisted run. See
[`docs/experiments/measurement-rigor.md`](docs/experiments/measurement-rigor.md).
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/experiments/measurement-rigor.md
git commit -m "docs: describe the measurement-rigor harness"
```

---

## Task 8: full verification gate

**Files:** none (verification only).

- [ ] **Step 1: Full offline suite**

Run: `\.venv\Scripts\python.exe -m pytest -q`
Expected: all pass (156 prior + ~9 new), 1 skipped (live). Fix any regression rather than weakening assertions.

- [ ] **Step 2: Type-check**

Run: `\.venv\Scripts\python.exe -m mypy`
Expected: `Success: no issues found`. The LOO ranking precomputes a `scores` dict then sorts `(-scores[v], v)` tuples — no loop-variable closure (so neither ruff B023 nor a mypy key-type issue arises); the tuple key is `(float, str)`.

- [ ] **Step 3: Lint + format**

Run: `\.venv\Scripts\python.exe -m ruff check . ; \.venv\Scripts\python.exe -m ruff format --check .`
Expected: `All checks passed!` and no format diffs. If `ruff format` rewrites a long line, run `\.venv\Scripts\python.exe -m ruff format .` and re-check.

- [ ] **Step 4: Commit any verification fixes**

```bash
git add -- src tests README.md docs
git commit -m "fix: address verification findings for the measurement harness"
```

(Only if Steps 1–3 required changes. Never `git add -A` from the repo root.)

---

## Self-review notes

- **Spec coverage:** within-one action (Task 2), ECE/calibration (Task 3), LOO CV (Task 4), verdict persistence + live collection (Task 5), CLI + reporting (Task 6), docs (Task 7), gate (Task 8). All spec sections map to a task. The variant-error edge case (selection counts errors as misses; out-of-fold falls back to the next variant) is covered by `test_loo_falls_back_when_selected_variant_errored_on_held_out`.
- **Type consistency:** `cross_validate(verdicts_by_variant, labels, tactic_by_id) -> CrossValReport` is defined in Task 4 and called identically in Task 6. `VerdictArtifact` (Task 5) is constructed/consumed identically in Task 6. `calibration(pairs, *, bins=5) -> tuple[float, list[ReliabilityBin]]` (Task 3) — `score()` calls it with `bins` defaulted. `EvalReport` new fields (`action_within_one_accuracy`, `ece`, `reliability`) are added in Tasks 2–3 and read in Tasks 4/6.
- **No placeholders:** every code step shows complete code; every run step shows the exact command and expected result.
- **Circular-import watch:** `report.py` imports `CrossValReport` from `crossval.py`; `crossval.py` imports only from `metrics`/`models`/`agent`/`pipeline`/`server` — not `report` — so no cycle. (Confirm in Task 6 Step 3.)
