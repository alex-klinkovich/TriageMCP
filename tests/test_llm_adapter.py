"""Spec for the Anthropic adapter: SDK Message -> AssistantTurn and transient mapping.

These exercise the one module that touches the SDK, without any network. The reducer only
reads attributes off the Message, so lightweight duck-typed stand-ins (cast to the SDK types)
are enough and avoid fighting the SDK's strict constructors.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock

from triagemcp.agent.llm import AnthropicLLMClient, AssistantTurn, message_to_turn


def _text(text: str) -> TextBlock:
    return cast("TextBlock", SimpleNamespace(type="text", text=text))


def _tool(tool_id: str, name: str, tool_input: dict[str, Any]) -> ToolUseBlock:
    return cast(
        "ToolUseBlock", SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)
    )


def _message(*blocks: object, stop_reason: str | None = "tool_use") -> Message:
    return cast("Message", SimpleNamespace(content=list(blocks), stop_reason=stop_reason))


def test_message_to_turn_extracts_text_and_tool_calls() -> None:
    turn = message_to_turn(
        _message(
            _text("looking into it"),
            _tool("tu_1", "map_to_mitre", {"text": "x"}),
        )
    )
    assert turn.stop_reason == "tool_use"
    assert turn.text == "looking into it"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "map_to_mitre"
    assert turn.tool_calls[0].arguments == {"text": "x"}


def test_message_to_turn_defaults_missing_stop_reason() -> None:
    turn = message_to_turn(_message(_text("done"), stop_reason=None))
    assert turn.stop_reason == "end_turn"
    assert turn.tool_calls == ()


class _RaisingMessages:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def create(self, **_kwargs: object) -> Message:
        raise self._exc


class _ReturningMessages:
    def __init__(self, message: Message) -> None:
        self._message = message

    async def create(self, **_kwargs: object) -> Message:
        return self._message


class _FakeAnthropic:
    def __init__(self, messages: object) -> None:
        self.messages = messages


def _client_with(messages: object) -> AnthropicLLMClient:
    return AnthropicLLMClient(cast("anthropic.AsyncAnthropic", _FakeAnthropic(messages)))


async def test_adapter_maps_transient_sdk_error_to_transient_error() -> None:
    from triagemcp.errors import TransientLLMError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APITimeoutError(request=request)
    client = _client_with(_RaisingMessages(err))
    with pytest.raises(TransientLLMError):
        await client.create(system="s", messages=[], tools=[], model="m", max_tokens=10)


async def test_adapter_returns_assistant_turn_on_success() -> None:
    client = _client_with(_ReturningMessages(_message(_tool("t1", "submit_triage", {"x": 1}))))
    turn = await client.create(system="s", messages=[], tools=[], model="m", max_tokens=10)
    assert isinstance(turn, AssistantTurn)
    assert turn.tool_calls[0].name == "submit_triage"


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return anthropic.APIStatusError(f"status {status}", response=response, body=None)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
async def test_adapter_maps_retryable_status_codes_to_transient(status: int) -> None:
    from triagemcp.errors import TransientLLMError

    client = _client_with(_RaisingMessages(_status_error(status)))
    with pytest.raises(TransientLLMError):
        await client.create(system="s", messages=[], tools=[], model="m", max_tokens=10)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_adapter_does_not_retry_client_errors(status: int) -> None:
    from triagemcp.errors import TransientLLMError

    client = _client_with(_RaisingMessages(_status_error(status)))
    with pytest.raises(anthropic.APIStatusError):
        await client.create(system="s", messages=[], tools=[], model="m", max_tokens=10)
    # And specifically NOT remapped to the retryable type.
    client2 = _client_with(_RaisingMessages(_status_error(status)))
    with pytest.raises(Exception) as excinfo:
        await client2.create(system="s", messages=[], tools=[], model="m", max_tokens=10)
    assert not isinstance(excinfo.value, TransientLLMError)
