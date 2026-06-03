"""Spec for query_recent_alerts, backed by a real (in-memory) aiosqlite store."""

from __future__ import annotations

import datetime as dt

import aiosqlite

from triagemcp.tools.recent_alerts import (
    AlertHistoryStore,
    HistoryEntry,
    QueryRecentAlertsTool,
)

NOW = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=dt.UTC)


def _entries() -> list[HistoryEntry]:
    return [
        HistoryEntry(
            alert_id="H-1",
            observable="203.0.113.45",
            observable_type="ip",
            title="Failed SSH burst",
            severity="high",
            seen_at=NOW - dt.timedelta(hours=2),
        ),
        HistoryEntry(
            alert_id="H-2",
            observable="203.0.113.45",
            observable_type="ip",
            title="Old SSH scan",
            severity="medium",
            seen_at=NOW - dt.timedelta(days=30),
        ),
        HistoryEntry(
            alert_id="H-3",
            observable="FIN-WS-118",
            observable_type="host",
            title="Encoded PowerShell",
            severity="high",
            seen_at=NOW - dt.timedelta(hours=10),
        ),
    ]


async def _build_tool(conn: aiosqlite.Connection) -> QueryRecentAlertsTool:
    store = await AlertHistoryStore.create(conn)
    await store.seed(_entries())
    return QueryRecentAlertsTool(store, clock=lambda: NOW)


async def test_query_returns_only_matches_within_lookback() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        tool = await _build_tool(conn)
        result = await tool.invoke({"observable": "203.0.113.45", "lookback_hours": 168})
    assert result.is_error is False
    assert result.content["match_count"] == 1  # H-1 is recent; H-2 (30 days) is excluded
    assert result.content["alerts"][0]["alert_id"] == "H-1"


async def test_query_filters_by_observable_type() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        tool = await _build_tool(conn)
        result = await tool.invoke({"observable": "FIN-WS-118", "observable_type": "host"})
    assert result.content["match_count"] == 1
    assert result.content["alerts"][0]["alert_id"] == "H-3"


async def test_query_unknown_observable_is_empty() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        tool = await _build_tool(conn)
        result = await tool.invoke({"observable": "10.0.0.254"})
    assert result.content["match_count"] == 0
    assert result.content["alerts"] == []


async def test_query_rejects_nonpositive_lookback() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        tool = await _build_tool(conn)
        result = await tool.invoke({"observable": "203.0.113.45", "lookback_hours": 0})
    assert result.is_error is True
