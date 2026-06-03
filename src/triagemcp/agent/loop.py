"""The triage agent: an Anthropic tool-use loop with retry, timeout, and validated output.

``TriageAgent.triage`` runs the model in a bounded loop. Each turn the model may call
investigation tools (dispatched through the :class:`~triagemcp.tools.registry.ToolRegistry`)
or the terminal ``submit_triage`` tool. A submission is validated against
:class:`~triagemcp.models.TriageResult`; an invalid one is rejected with a corrective tool
result and retried in-loop. The whole loop is bounded by a per-alert timeout and an
iteration cap, and each model call is retried on transient LLM errors.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from triagemcp.agent.llm import AssistantTurn, LLMClient, ToolCall
from triagemcp.agent.prompts import (
    SUBMIT_TOOL_NAME,
    SYSTEM_PROMPT,
    alert_user_message,
    submit_tool_spec,
)
from triagemcp.agent.retry import retry_async
from triagemcp.errors import (
    AgentTimeoutError,
    MaxIterationsError,
    ModelRefusedToSubmitError,
    TransientLLMError,
)
from triagemcp.models import Alert, TriageResult
from triagemcp.tools.registry import ToolRegistry

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class AgentConfig:
    """Tunable, injected agent behaviour. No global state."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    max_iterations: int = 8
    per_alert_timeout_s: float = 60.0
    max_retries: int = 4
    retry_base_delay_s: float = 0.5
    temperature: float = 0.0


@dataclass(frozen=True)
class AgentRun:
    """A single triage's validated verdict plus loop telemetry."""

    result: TriageResult
    iterations: int


class TriageAgent:
    """Drives one alert through investigation to a schema-valid verdict."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        *,
        sleep: SleepFn | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._config = config or AgentConfig()
        self._sleep = sleep
        self._tool_specs: list[dict[str, Any]] = [*tools.anthropic_specs(), submit_tool_spec()]

    async def run(self, alert: Alert) -> AgentRun:
        """Investigate ``alert`` and return the verdict plus telemetry, or raise an AgentError."""
        try:
            async with asyncio.timeout(self._config.per_alert_timeout_s):
                return await self._run_loop(alert)
        except TimeoutError as exc:
            raise AgentTimeoutError(
                f"triage of {alert.id} exceeded {self._config.per_alert_timeout_s}s"
            ) from exc

    async def triage(self, alert: Alert) -> TriageResult:
        """Investigate ``alert`` and return only the validated verdict (the MCP contract)."""
        return (await self.run(alert)).result

    async def _run_loop(self, alert: Alert) -> AgentRun:
        messages: list[dict[str, Any]] = [alert_user_message(alert)]
        nudged = False

        for iteration in range(1, self._config.max_iterations + 1):
            turn = await self._create_with_retry(messages)
            messages.append(_assistant_message(turn))

            if not turn.tool_calls:
                if nudged:
                    raise ModelRefusedToSubmitError(
                        f"model ended its turn without submitting a verdict for {alert.id}"
                    )
                nudged = True
                messages.append(_nudge_message())
                continue

            tool_results: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                if call.name == SUBMIT_TOOL_NAME:
                    outcome = self._finalize(call, alert)
                    if isinstance(outcome, TriageResult):
                        return AgentRun(result=outcome, iterations=iteration)
                    tool_results.append(outcome)
                else:
                    result = await self._tools.dispatch(call.name, call.arguments)
                    tool_results.append(
                        _tool_result_block(call.id, result.content, is_error=result.is_error)
                    )
            messages.append({"role": "user", "content": tool_results})

        raise MaxIterationsError(
            f"agent exceeded {self._config.max_iterations} iterations triaging {alert.id}"
        )

    async def _create_with_retry(self, messages: list[dict[str, Any]]) -> AssistantTurn:
        async def operation() -> AssistantTurn:
            return await self._llm.create(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=self._tool_specs,
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

        if self._sleep is None:
            return await retry_async(
                operation,
                max_attempts=self._config.max_retries,
                base_delay=self._config.retry_base_delay_s,
                retry_on=(TransientLLMError,),
            )
        return await retry_async(
            operation,
            max_attempts=self._config.max_retries,
            base_delay=self._config.retry_base_delay_s,
            retry_on=(TransientLLMError,),
            sleep=self._sleep,
        )

    def _finalize(self, call: ToolCall, alert: Alert) -> TriageResult | dict[str, Any]:
        # Force the alert_id so the model can never mislabel which alert it answered.
        candidate = {**call.arguments, "alert_id": alert.id}
        try:
            return TriageResult.model_validate(candidate)
        except ValidationError as exc:
            violations = [
                {"field": ".".join(str(part) for part in err["loc"]), "issue": err["msg"]}
                for err in exc.errors()
            ]
            return _tool_result_block(
                call.id,
                {
                    "error": "verdict failed schema validation; correct it and resubmit",
                    "violations": violations,
                },
                is_error=True,
            )


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    content.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in turn.tool_calls
    )
    if not content:
        content.append({"type": "text", "text": "(no content)"})
    return {"role": "assistant", "content": content}


def _tool_result_block(
    tool_use_id: str, content: dict[str, Any], *, is_error: bool
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(content),
        "is_error": is_error,
    }


def _nudge_message() -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            "You have not submitted a verdict yet. When your investigation is complete, call "
            "the submit_triage tool exactly once with your final assessment."
        ),
    }
