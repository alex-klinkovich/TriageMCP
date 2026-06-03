"""System prompt and the terminal ``submit_triage`` tool definition.

Structured output is enforced by making the model *finish* with a tool call whose
input schema **is** :class:`~triagemcp.models.TriageResult`. The loop validates that
submission and rejects/retries anything that does not conform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triagemcp.models import Alert, TriageResult
from triagemcp.tools.mitre import load_mitre_techniques

SUBMIT_TOOL_NAME = "submit_triage"

_BASE_PROMPT = """\
You are TriageMCP, an expert Tier-2 SOC analyst. You triage one security alert at a time.

Method:
1. Read the alert and its observables.
2. Investigate with the available tools before deciding. Typical moves:
   - map_to_mitre: map the alert text / command line to MITRE ATT&CK techniques.
   - lookup_ip_reputation: check any IP observable against threat intelligence.
   - query_recent_alerts: see whether an observable (IP, host, hash, user, domain) has
     been seen recently, to spot repeat activity and reduce false positives.
   - enrich_hash: check any file hash against malware intelligence.
3. Weigh the evidence. Benign or authorized activity (known scanners, change windows,
   approved vendors) should be de-escalated, not escalated.
4. When — and only when — your investigation supports a conclusion, call the
   submit_triage tool exactly once with your final verdict.

Verdict requirements:
- severity: one of informational, low, medium, high, critical.
- recommended_action: one of close_false_positive, monitor, investigate, contain, escalate.
- mitre_technique_id: the best-fit ATT&CK technique id, even for benign
  alerts (the technique the activity resembles).
- confidence: 0.0-1.0, your calibrated confidence in this verdict.
- rationale: a concise, evidence-based justification referencing what the tools returned.

Do not fabricate tool results. If a submission is rejected for schema reasons, fix it and
resubmit. Investigate efficiently; do not loop indefinitely."""


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
    lines = [f"- {t.id} {t.name} ({t.tactic})" for t in load_mitre_techniques()]
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
        parts.append("")  # few-shot examples added in a later task
    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()


def submit_tool_spec() -> dict[str, Any]:
    """The Anthropic tool definition whose input schema is ``TriageResult``."""
    return {
        "name": SUBMIT_TOOL_NAME,
        "description": (
            "Submit your final, schema-valid triage verdict for the alert. Call this exactly "
            "once, after your investigation, to end the triage."
        ),
        "input_schema": TriageResult.model_json_schema(),
    }


def alert_user_message(alert: Alert) -> dict[str, Any]:
    """The opening user turn carrying the alert to triage."""
    return {
        "role": "user",
        "content": (
            f"Triage the following alert. When you submit, set alert_id to exactly "
            f"{alert.id!r}.\n\n{alert.model_dump_json(indent=2)}"
        ),
    }
