"""Shared tool abstraction.

Every investigation tool:

* declares a Pydantic *input model* (its JSON schema is what the LLM sees), and
* returns a :class:`ToolResult` whose ``content`` is a JSON-serialisable dict.

:class:`BaseTool` centralises input validation so a malformed call from the model
becomes a structured, non-fatal ``is_error`` result the agent can recover from,
rather than an exception that aborts the loop. The :class:`Tool` protocol is the
narrow surface the registry and agent depend on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError


class ToolResult(BaseModel):
    """Outcome of a tool invocation, shaped for an Anthropic ``tool_result`` block."""

    model_config = ConfigDict(extra="forbid")

    content: dict[str, Any]
    is_error: bool = False

    @classmethod
    def ok(cls, content: dict[str, Any]) -> ToolResult:
        """A successful result wrapping ``content``."""
        return cls(content=content, is_error=False)

    @classmethod
    def error(cls, message: str, *, details: dict[str, Any] | None = None) -> ToolResult:
        """An error result the agent can read and react to."""
        body: dict[str, Any] = {"error": message}
        if details:
            body.update(details)
        return cls(content=body, is_error=True)


@runtime_checkable
class Tool(Protocol):
    """Structural interface the registry and agent rely on."""

    name: ClassVar[str]
    description: ClassVar[str]

    def input_schema(self) -> dict[str, Any]: ...

    async def invoke(self, raw_args: Mapping[str, Any]) -> ToolResult: ...


class BaseTool[InputT: BaseModel](ABC):
    """Base class wiring validation; subclasses set ``input_model`` and implement ``run``."""

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: type[InputT]

    def input_schema(self) -> dict[str, Any]:
        """The tool's JSON schema (sent to the model as the tool's ``input_schema``)."""
        return self.input_model.model_json_schema()

    async def invoke(self, raw_args: Mapping[str, Any]) -> ToolResult:
        """Validate ``raw_args`` then run; validation failures become error results."""
        try:
            args = self.input_model.model_validate(dict(raw_args))
        except ValidationError as exc:
            violations = [
                {"field": ".".join(str(part) for part in err["loc"]), "issue": err["msg"]}
                for err in exc.errors()
            ]
            return ToolResult.error("input validation failed", details={"violations": violations})
        return await self.run(args)

    @abstractmethod
    async def run(self, args: InputT) -> ToolResult:
        """Execute against validated input. Implemented by each concrete tool."""
        raise NotImplementedError
