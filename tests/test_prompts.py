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

    assert build_system_prompt(PromptOptions()) == SYSTEM_PROMPT


def test_few_shot_option_includes_a_worked_example() -> None:
    prompt = build_system_prompt(PromptOptions(include_few_shot=True))
    assert "Worked example" in prompt
    assert "FEWSHOT-" in prompt
