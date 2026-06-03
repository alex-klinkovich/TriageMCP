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
        return await run_eval(labeled, triager, concurrency=concurrency, tactic_by_id=tactic_by_id)
