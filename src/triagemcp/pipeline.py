"""Concurrent batch triage with per-alert error isolation.

``triage_batch`` fans alerts out across an :class:`asyncio.TaskGroup`, rate-limited by an
:class:`asyncio.Semaphore`. TaskGroup is fail-fast — one raising child cancels its siblings —
so each worker **catches its own exceptions** and records a failure :class:`TriageOutcome`.
That catch is precisely what keeps one bad alert from sinking the whole batch. Outcomes are
returned in input order regardless of completion order.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Protocol

from triagemcp.agent.loop import AgentRun
from triagemcp.models import Alert, TriageOutcome


class Triager(Protocol):
    """Anything that can triage one alert into a verdict plus telemetry."""

    async def run(self, alert: Alert) -> AgentRun: ...


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def triage_batch(
    alerts: Sequence[Alert], triager: Triager, *, concurrency: int
) -> list[TriageOutcome]:
    """Triage ``alerts`` concurrently, returning one outcome per alert in input order."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    alert_list = list(alerts)
    outcomes: list[TriageOutcome | None] = [None] * len(alert_list)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int, alert: Alert) -> None:
        async with semaphore:
            start = time.perf_counter()
            try:
                run = await triager.run(alert)
            # Broad by design: this except IS the per-alert isolation boundary. Only Exception
            # (not BaseException), so cancellation/KeyboardInterrupt still propagate.
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                outcomes[index] = TriageOutcome.failure(
                    alert.id, _format_error(exc), iterations=0, latency_ms=latency_ms
                )
            else:
                latency_ms = (time.perf_counter() - start) * 1000.0
                outcomes[index] = TriageOutcome.success(
                    run.result, iterations=run.iterations, latency_ms=latency_ms
                )

    async with asyncio.TaskGroup() as task_group:
        for index, alert in enumerate(alert_list):
            task_group.create_task(worker(index, alert))

    # Every slot is filled: the TaskGroup awaited all workers and workers never raise.
    return [outcome for outcome in outcomes if outcome is not None]
