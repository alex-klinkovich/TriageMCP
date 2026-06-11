"""Pydantic v2 models: alert inputs, triage verdicts, and evaluation labels.

These types are the contract enforced at every boundary of the system:

* :class:`Alert` is what gets ingested (immutable, no surprise fields).
* :class:`TriageResult` is what the agent must produce; the agent loop rejects and
  retries any LLM output that fails to validate against it.
* :class:`LabeledAlert` couples an alert with its ground-truth :class:`AlertLabel`
  for the offline evaluation harness.
* :class:`TriageOutcome` is the per-alert envelope the batch pipeline returns,
  carrying *either* a result *or* an error so one bad alert never sinks a batch.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    model_validator,
)


class Severity(StrEnum):
    """Ordered severity scale. Definition order encodes rank (low to high)."""

    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Rank from 0 (informational) to 4 (critical)."""
        return list(Severity).index(self)

    def distance(self, other: Severity) -> int:
        """Absolute number of rungs between two severities (used by the eval)."""
        return abs(self.level - other.level)


class RecommendedAction(StrEnum):
    """The action a SOC analyst should take, from benign-close to full escalation."""

    CLOSE_FALSE_POSITIVE = "close_false_positive"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    CONTAIN = "contain"
    ESCALATE = "escalate"


_MITRE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def _validate_mitre_id(value: str) -> str:
    if not _MITRE_ID_RE.match(value):
        raise ValueError(
            f"invalid MITRE ATT&CK technique id {value!r}; "
            "expected e.g. 'T1059' or a sub-technique 'T1059.001'"
        )
    return value


MitreTechniqueId = Annotated[str, AfterValidator(_validate_mitre_id)]
"""A MITRE ATT&CK technique id constrained to the canonical ``T####[.###]`` shape."""


class Observables(BaseModel):
    """Structured indicators extracted from an alert, fed to the deterministic tools."""

    model_config = ConfigDict(extra="forbid")

    ips: list[IPvAnyAddress] = Field(default_factory=list)
    file_hashes: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)
    command_lines: list[str] = Field(default_factory=list)


_ALERT_RAW_MAX_CHARS = 50_000
"""Cap on the serialized size of an alert's free-form ``raw`` payload."""


class Alert(BaseModel):
    """An immutable security alert ingested for triage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=1024)
    description: str = Field(min_length=1, max_length=20_000)
    source: str = Field(min_length=1, max_length=256, description="Originating tool, e.g. EDR.")
    severity_reported: Severity = Field(description="Severity claimed by the source tool.")
    timestamp: AwareDatetime = Field(description="Timezone-aware event time.")
    observables: Observables = Field(default_factory=Observables)
    raw: dict[str, Any] = Field(default_factory=dict, description="Source-specific extra fields.")

    @model_validator(mode="after")
    def _bound_raw_payload(self) -> Self:
        # `raw` is attacker-influenceable and is serialized straight into the model prompt;
        # cap its serialized size so one fat alert can't drive unbounded token cost (the output
        # `rationale` is already bounded — this closes the same gap on the input side).
        if len(json.dumps(self.raw, default=str)) > _ALERT_RAW_MAX_CHARS:
            raise ValueError(f"raw payload exceeds the {_ALERT_RAW_MAX_CHARS}-character cap")
        return self


class TriageResult(BaseModel):
    """The agent's schema-validated verdict for a single alert."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(min_length=1)
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    mitre_technique_id: MitreTechniqueId
    mitre_technique_name: str = Field(min_length=1)
    recommended_action: RecommendedAction
    rationale: str = Field(min_length=1, max_length=2000)
    supporting_observations: list[str] = Field(default_factory=list)


class AlertLabel(BaseModel):
    """Ground-truth annotation for a sample alert (used only by the eval harness)."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    mitre_technique_id: MitreTechniqueId
    recommended_action: RecommendedAction


class LabeledAlert(BaseModel):
    """An alert paired with its ground-truth label."""

    model_config = ConfigDict(extra="forbid")

    alert: Alert
    label: AlertLabel


class TriageOutcome(BaseModel):
    """Per-alert pipeline envelope: exactly one of ``result`` or ``error`` is set."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(min_length=1)
    result: TriageResult | None = None
    error: str | None = None
    iterations: int = Field(ge=0, description="Agent loop turns taken.")
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _exactly_one_of_result_or_error(self) -> Self:
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one of 'result' or 'error' must be set")
        return self

    @classmethod
    def success(cls, result: TriageResult, *, iterations: int, latency_ms: float) -> Self:
        """Build a successful outcome, deriving ``alert_id`` from the result."""
        return cls(
            alert_id=result.alert_id,
            result=result,
            iterations=iterations,
            latency_ms=latency_ms,
        )

    @classmethod
    def failure(cls, alert_id: str, error: str, *, iterations: int, latency_ms: float) -> Self:
        """Build a failed outcome carrying the error message."""
        return cls(
            alert_id=alert_id,
            error=error,
            iterations=iterations,
            latency_ms=latency_ms,
        )
