"""Generic async retry with exponential backoff.

The ``sleep`` callable is injectable so tests run instantly and deterministically
(no real delays, no wall-clock dependence). Backoff is deterministic exponential —
jitter is intentionally omitted to keep behaviour reproducible under test.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def _default_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
    retry_on: tuple[type[Exception], ...],
    max_delay: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] = _default_sleep,
) -> T:
    """Call ``operation`` until it succeeds or ``max_attempts`` is reached.

    Retries only on exceptions in ``retry_on``; anything else propagates immediately.
    After attempt *n* (1-indexed), sleeps ``min(base_delay * 2**(n-1), max_delay)``.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    attempt = 0
    while True:
        try:
            return await operation()
        except retry_on:
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            await sleep(delay)
