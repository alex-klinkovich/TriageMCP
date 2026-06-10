"""``lookup_ip_reputation``: check an IP against a threat-intel source.

Two interchangeable clients sit behind the :class:`IpReputationClient` protocol:

* :class:`LocalIpReputationClient` — the offline default, backed by a bundled store,
  so the tool (and the whole test suite) need no network.
* :class:`HttpxIpReputationClient` — the production seam that calls a real HTTP
  threat-intel API; covered by respx-mocked tests, never hitting the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, TypeAdapter

from triagemcp.datasets import read_data_text
from triagemcp.tools.base import BaseTool, ToolResult

Reputation = Literal["malicious", "suspicious", "clean", "unknown"]


class IpReputationRecord(BaseModel):
    """A normalized reputation verdict for one IP address."""

    model_config = ConfigDict(extra="forbid")

    ip: str
    reputation: Reputation
    score: int = Field(ge=0, le=100, description="Risk score, 0 (clean) to 100 (malicious).")
    categories: list[str] = Field(default_factory=list)
    last_seen: str | None = None
    source: str


class IpReputationClient(Protocol):
    """Looks up reputation for an IP, returning ``None`` when nothing is known."""

    async def lookup(self, ip: str) -> IpReputationRecord | None: ...


class LocalIpReputationClient:
    """Offline client backed by an in-memory store keyed by IP string."""

    def __init__(self, store: Mapping[str, IpReputationRecord]) -> None:
        self._store = dict(store)

    async def lookup(self, ip: str) -> IpReputationRecord | None:
        return self._store.get(ip)


class HttpxIpReputationClient:
    """Production client that queries ``{base_url}/{ip}`` over HTTP."""

    def __init__(self, client: httpx.AsyncClient, *, base_url: str, timeout: float = 5.0) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        # Explicit per-request timeout so a hung threat-intel endpoint can't hold the call open
        # until the agent's per-alert timeout fires — independent of how the client was built.
        self._timeout = httpx.Timeout(timeout)

    async def lookup(self, ip: str) -> IpReputationRecord | None:
        response = await self._client.get(f"{self._base_url}/{ip}", timeout=self._timeout)
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return IpReputationRecord.model_validate(response.json())


class LookupIpReputationInput(BaseModel):
    """A single IP address to check."""

    model_config = ConfigDict(extra="forbid")

    ip: IPvAnyAddress = Field(description="IPv4 or IPv6 address to look up.")


class LookupIpReputationTool(BaseTool[LookupIpReputationInput]):
    """Resolve an IP's reputation, falling back to an explicit ``unknown`` verdict."""

    name = "lookup_ip_reputation"
    description = (
        "Look up threat-intelligence reputation for an IPv4 or IPv6 address. Returns a "
        "reputation verdict (malicious/suspicious/clean/unknown), a 0-100 risk score, and "
        "any threat categories. Unknown addresses return reputation 'unknown' with score 0."
    )
    input_model = LookupIpReputationInput

    def __init__(self, client: IpReputationClient) -> None:
        self._client = client

    async def run(self, args: LookupIpReputationInput) -> ToolResult:
        ip = str(args.ip)
        record = await self._client.lookup(ip)
        if record is None:
            record = IpReputationRecord(
                ip=ip, reputation="unknown", score=0, categories=[], source="none"
            )
        return ToolResult.ok(record.model_dump(mode="json"))


_RECORDS = TypeAdapter(list[IpReputationRecord])


def load_ip_reputation() -> dict[str, IpReputationRecord]:
    """Load the bundled offline reputation store, keyed by IP string."""
    records = _RECORDS.validate_json(read_data_text("ip_reputation.json"))
    return {record.ip: record for record in records}
