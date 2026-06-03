"""Test-support doubles for driving the agent offline.

Shipped in the package (not just the test tree) so the unit suite, the eval tests, and
any downstream integration tests can replay deterministic model behaviour without a key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from triagemcp.agent.llm import AssistantTurn, ToolCall


class FakeLLMClient:
    """An offline :class:`~triagemcp.agent.llm.LLMClient` that replays a scripted list.

    Each script item is either an :class:`AssistantTurn` to return or an ``Exception`` to
    raise (e.g. ``TransientLLMError`` to exercise retry). ``delay`` optionally makes every
    call sleep first, to exercise the agent's per-alert timeout.
    """

    def __init__(self, script: Sequence[AssistantTurn | Exception], *, delay: float = 0.0) -> None:
        self._script: list[AssistantTurn | Exception] = list(script)
        self._delay = delay
        self.calls = 0
        self.seen_tools: list[str] = []

    async def create(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> AssistantTurn:
        self.calls += 1
        self.seen_tools = [str(tool["name"]) for tool in tools]
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._script:
            raise AssertionError("FakeLLMClient exhausted its scripted turns")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_use(
    name: str, arguments: dict[str, Any], *, call_id: str = "call-1", text: str = ""
) -> AssistantTurn:
    """An assistant turn that calls a single investigation tool."""
    return AssistantTurn(
        stop_reason="tool_use",
        text=text,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def submit(arguments: dict[str, Any], *, call_id: str = "submit-1") -> AssistantTurn:
    """An assistant turn that calls ``submit_triage`` with the given verdict args."""
    return tool_use("submit_triage", arguments, call_id=call_id)


def end_turn(text: str = "") -> AssistantTurn:
    """An assistant turn that ends without any tool call."""
    return AssistantTurn(stop_reason="end_turn", text=text, tool_calls=())
