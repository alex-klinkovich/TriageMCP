"""FastMCP server exposing the ``triage_alert`` tool over the Model Context Protocol.

The triager is injected via a closure so the server has no global state and is trivially
testable with a fake. :func:`build_runtime` constructs the production triager (a real
Anthropic-backed agent plus an aiosqlite-backed alert history) and tears it down cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Protocol

import aiosqlite
from anthropic import AsyncAnthropic
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from triagemcp.agent.llm import AnthropicLLMClient
from triagemcp.agent.loop import AgentConfig, TriageAgent
from triagemcp.config import Settings
from triagemcp.models import Alert, TriageResult
from triagemcp.tools.registry import build_default_registry

SERVER_NAME = "TriageMCP"
INSTRUCTIONS = (
    "Agentic security-alert triage. Call triage_alert with a single alert to receive a "
    "schema-validated verdict: severity, confidence, MITRE ATT&CK technique, recommended "
    "action, and an evidence-based rationale."
)


class AlertTriager(Protocol):
    """The single capability the server needs: turn an alert into a verdict."""

    async def triage(self, alert: Alert) -> TriageResult: ...


def create_server(triager: AlertTriager) -> FastMCP:
    """Build a FastMCP server whose ``triage_alert`` tool delegates to ``triager``."""
    mcp: FastMCP = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)

    @mcp.tool(
        name="triage_alert",
        description="Investigate a security alert and return a schema-validated triage verdict.",
    )
    async def triage_alert(alert: Alert) -> TriageResult:
        return await triager.triage(alert)

    return mcp


@contextlib.asynccontextmanager
async def build_runtime(
    settings: Settings,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> AsyncIterator[TriageAgent]:
    """Construct the production triager and guarantee its resources are released."""
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    conn = await aiosqlite.connect(settings.db_path)
    try:
        registry = await build_default_registry(conn)
        config = AgentConfig(
            model=model or settings.model,
            max_tokens=settings.max_tokens,
            max_iterations=settings.max_iterations,
            per_alert_timeout_s=settings.per_alert_timeout_s,
            max_retries=settings.max_retries,
            temperature=temperature,
        )
        if system_prompt is not None:
            config = replace(config, system_prompt=system_prompt)
        agent = TriageAgent(AnthropicLLMClient(anthropic_client), registry, config)
        yield agent
    finally:
        await conn.close()
        await anthropic_client.close()


async def serve_stdio(settings: Settings | None = None) -> None:
    """Run the MCP server over stdio (the default MCP transport)."""
    resolved = settings or Settings()
    async with build_runtime(resolved) as triager:
        await create_server(triager).run_stdio_async()


def main() -> None:
    """`python -m triagemcp.server` entry point with a friendly missing-key message."""
    try:
        settings = Settings()
    except ValidationError:
        print(
            "ANTHROPIC_API_KEY is not set; cannot start the TriageMCP server.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    asyncio.run(serve_stdio(settings))


if __name__ == "__main__":
    main()
