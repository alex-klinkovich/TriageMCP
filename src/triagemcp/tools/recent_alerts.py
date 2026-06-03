"""``query_recent_alerts``: find historical alerts for an observable via aiosqlite.

The store keeps a small alert-history table; the tool answers "have we seen this IP /
host / hash recently?". Time filtering is pushed into SQL against an epoch column, and
the clock is injected so tests are fully deterministic (no wall-clock dependence).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Self

import aiosqlite
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from triagemcp.tools.base import BaseTool, ToolResult

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS alert_history (
    alert_id        TEXT NOT NULL,
    observable      TEXT NOT NULL,
    observable_type TEXT NOT NULL,
    title           TEXT NOT NULL,
    severity        TEXT NOT NULL,
    seen_at         TEXT NOT NULL,
    seen_at_epoch   REAL NOT NULL
)
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_history_observable ON alert_history(observable)"
_INSERT = (
    "INSERT INTO alert_history "
    "(alert_id, observable, observable_type, title, severity, seen_at, seen_at_epoch) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)
_SELECT = (
    "SELECT alert_id, observable, observable_type, title, severity, seen_at "
    "FROM alert_history WHERE observable = ? AND seen_at_epoch >= ?"
)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class HistoryEntry(BaseModel):
    """One historical alert sighting tied to an observable."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    observable: str
    observable_type: str
    title: str
    severity: str
    seen_at: AwareDatetime


class AlertHistoryStore:
    """Thin async data-access layer over an aiosqlite connection."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def create(cls, conn: aiosqlite.Connection) -> Self:
        """Create the schema (idempotent) and return a store bound to ``conn``."""
        await conn.execute(_CREATE_TABLE)
        await conn.execute(_CREATE_INDEX)
        await conn.commit()
        return cls(conn)

    async def seed(self, entries: Iterable[HistoryEntry]) -> None:
        """Insert history rows."""
        rows = [
            (
                entry.alert_id,
                entry.observable,
                entry.observable_type,
                entry.title,
                entry.severity,
                entry.seen_at.isoformat(),
                entry.seen_at.timestamp(),
            )
            for entry in entries
        ]
        await self._conn.executemany(_INSERT, rows)
        await self._conn.commit()

    async def query(
        self, observable: str, *, observable_type: str | None, since: dt.datetime
    ) -> list[HistoryEntry]:
        """Return history rows for ``observable`` at or after ``since``, newest first."""
        sql = _SELECT
        params: list[object] = [observable, since.timestamp()]
        if observable_type is not None:
            sql += " AND observable_type = ?"
            params.append(observable_type)
        sql += " ORDER BY seen_at_epoch DESC"

        cursor = await self._conn.execute(sql, params)
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()

        return [
            HistoryEntry(
                alert_id=row[0],
                observable=row[1],
                observable_type=row[2],
                title=row[3],
                severity=row[4],
                seen_at=dt.datetime.fromisoformat(row[5]),
            )
            for row in rows
        ]


class QueryRecentAlertsInput(BaseModel):
    """Look up recent history for one observable (IP, host, hash, user, domain...)."""

    model_config = ConfigDict(extra="forbid")

    observable: str = Field(min_length=1, description="The indicator value to search for.")
    observable_type: str | None = Field(
        default=None, description="Optional type filter, e.g. 'ip', 'host', 'hash'."
    )
    lookback_hours: int = Field(default=168, ge=1, le=8760, description="Window size in hours.")


class QueryRecentAlertsTool(BaseTool[QueryRecentAlertsInput]):
    """Answer 'have we seen this observable recently?' from the alert-history store."""

    name = "query_recent_alerts"
    description = (
        "Search recent alert history for a given observable (IP, host, file hash, user, or "
        "domain) within a lookback window. Returns the count and the matching prior alerts, "
        "newest first. Useful for spotting repeat offenders and corroborating an alert."
    )
    input_model = QueryRecentAlertsInput

    def __init__(
        self, store: AlertHistoryStore, *, clock: Callable[[], dt.datetime] = _utcnow
    ) -> None:
        self._store = store
        self._clock = clock

    async def run(self, args: QueryRecentAlertsInput) -> ToolResult:
        since = self._clock() - dt.timedelta(hours=args.lookback_hours)
        entries = await self._store.query(
            args.observable, observable_type=args.observable_type, since=since
        )
        return ToolResult.ok(
            {
                "observable": args.observable,
                "observable_type": args.observable_type,
                "lookback_hours": args.lookback_hours,
                "match_count": len(entries),
                "alerts": [entry.model_dump(mode="json") for entry in entries],
            }
        )
