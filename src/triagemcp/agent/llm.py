"""The LLM boundary.

The agent loop never touches the Anthropic SDK directly. It depends only on the
:class:`LLMClient` protocol and the small provider-agnostic value objects
(:class:`AssistantTurn`, :class:`ToolCall`). :class:`AnthropicLLMClient` is the one
adapter that translates SDK ``Message`` objects into those values and maps transient
SDK failures onto :class:`~triagemcp.errors.TransientLLMError` so the loop can retry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import anthropic
from anthropic.types import Message

from triagemcp.errors import TransientLLMError


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    """One assistant response, reduced to the parts the loop needs."""

    stop_reason: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class LLMClient(Protocol):
    """The minimal contract the agent depends on (real client or test fake)."""

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


_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def message_to_turn(message: Message) -> AssistantTurn:
    """Reduce a SDK ``Message`` to an :class:`AssistantTurn`."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id, name=block.name, arguments=cast("dict[str, Any]", block.input)
                )
            )
    return AssistantTurn(
        stop_reason=message.stop_reason or "end_turn",
        text="".join(text_parts),
        tool_calls=tuple(tool_calls),
    )


class AnthropicLLMClient:
    """Adapter over ``anthropic.AsyncAnthropic`` implementing :class:`LLMClient`."""

    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

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
        try:
            message = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=cast("Any", list(tools)),
                messages=cast("Any", list(messages)),
            )
        except _TRANSIENT_ERRORS as exc:
            raise TransientLLMError(str(exc)) from exc
        return message_to_turn(message)
