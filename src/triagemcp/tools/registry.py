"""Tool registry: holds the investigation tools, advertises their schemas, dispatches calls.

``ToolRegistry`` is the single object the agent talks to. ``build_default_registry`` wires
the four concrete tools against the bundled offline datasets and an aiosqlite-backed alert
history seeded from the sample set.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import aiosqlite

from triagemcp.datasets import load_sample_alerts
from triagemcp.tools.base import Tool, ToolResult
from triagemcp.tools.hash_enrich import EnrichHashTool, HashIntelStore, load_hash_intel
from triagemcp.tools.ip_reputation import (
    LocalIpReputationClient,
    LookupIpReputationTool,
    load_ip_reputation,
)
from triagemcp.tools.mitre import MapToMitreTool, load_mitre_techniques
from triagemcp.tools.recent_alerts import (
    AlertHistoryStore,
    HistoryEntry,
    QueryRecentAlertsTool,
)


class ToolRegistry:
    """An immutable, name-indexed collection of tools."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        by_name: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in by_name:
                raise ValueError(f"duplicate tool name: {tool.name!r}")
            by_name[tool.name] = tool
        self._by_name = by_name

    @property
    def names(self) -> list[str]:
        """Tool names in registration order."""
        return list(self._by_name)

    def anthropic_specs(self) -> list[dict[str, Any]]:
        """Tool definitions in the shape the Anthropic Messages API expects."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema(),
            }
            for tool in self._by_name.values()
        ]

    async def dispatch(self, name: str, raw_args: Mapping[str, Any]) -> ToolResult:
        """Validate-and-run the named tool, or return an error result if it is unknown."""
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult.error(
                f"unknown tool: {name!r}", details={"available_tools": list(self._by_name)}
            )
        return await tool.invoke(raw_args)


def _history_from_samples() -> list[HistoryEntry]:
    """Seed prior-sighting rows for observables that genuinely recur across multiple alerts.

    A "repeat offender" is an observable (IP, host, hash, user, domain) that appears in two or
    more *distinct* sample alerts. Each such observable gets exactly one synthetic prior-sighting
    row, timestamped just before its earliest occurrence. Observables unique to a single alert
    are left unseeded, so they read as genuinely novel. This avoids fabricating a self-sighting
    for every alert (which made every alert look like a repeat) while staying fully deterministic.

    The sighting's ``severity`` is a fixed neutral value, never the source alert's reported
    severity: in this small labeled set a recurring observable already correlates with
    non-benign/high-severity labels, so returning a severity opinion alongside the recurrence
    signal would compound that correlation into a near-answer for the severity metric.
    """
    # (value, type) -> {alert_id: timestamp} for each distinct alert that contains it
    by_observable: dict[tuple[str, str], dict[str, dt.datetime]] = {}
    for labeled in load_sample_alerts():
        alert = labeled.alert
        obs = alert.observables
        typed_values: list[tuple[str, str]] = [
            *((str(ip), "ip") for ip in obs.ips),
            *((file_hash, "hash") for file_hash in obs.file_hashes),
            *((domain, "domain") for domain in obs.domains),
            *((user, "user") for user in obs.users),
            *((host, "host") for host in obs.hosts),
        ]
        for value, observable_type in typed_values:
            by_observable.setdefault((value, observable_type), {})[alert.id] = alert.timestamp

    entries: list[HistoryEntry] = []
    for (value, observable_type), occurrences in by_observable.items():
        if len(occurrences) < 2:
            continue  # appears in only one alert -> genuinely novel, no seeded history
        earliest_ts = min(occurrences.values())
        entries.append(
            HistoryEntry(
                alert_id=f"HIST-{observable_type}-{value}",
                observable=value,
                observable_type=observable_type,
                title=f"Prior activity involving {value}",
                severity="unknown",
                seen_at=earliest_ts - dt.timedelta(hours=1),
            )
        )
    return entries


async def build_default_registry(
    conn: aiosqlite.Connection, *, clock: Callable[[], dt.datetime] | None = None
) -> ToolRegistry:
    """Wire the four tools against bundled data and an alert history seeded from samples."""
    history_store = await AlertHistoryStore.create(conn)
    await history_store.seed(_history_from_samples())
    query_tool = (
        QueryRecentAlertsTool(history_store)
        if clock is None
        else QueryRecentAlertsTool(history_store, clock=clock)
    )
    return ToolRegistry(
        [
            MapToMitreTool(load_mitre_techniques()),
            LookupIpReputationTool(LocalIpReputationClient(load_ip_reputation())),
            EnrichHashTool(HashIntelStore(load_hash_intel())),
            query_tool,
        ]
    )
