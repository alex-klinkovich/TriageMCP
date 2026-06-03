"""Spec for the generic async retry helper (deterministic via injected sleep)."""

from __future__ import annotations

import pytest

from triagemcp.agent.retry import retry_async


class _Transient(Exception):
    pass


class _Fatal(Exception):
    pass


async def _noop_sleep(_delay: float) -> None:
    return None


async def test_returns_immediately_on_first_success() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await retry_async(
        op, max_attempts=3, base_delay=0.1, retry_on=(_Transient,), sleep=_noop_sleep
    )
    assert result == "ok"
    assert calls == 1


async def test_retries_then_succeeds_with_exponential_backoff() -> None:
    calls = 0
    delays: list[float] = []

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _Transient
        return "ok"

    async def rec_sleep(delay: float) -> None:
        delays.append(delay)

    result = await retry_async(
        op, max_attempts=5, base_delay=0.1, retry_on=(_Transient,), sleep=rec_sleep
    )
    assert result == "ok"
    assert calls == 3
    assert delays == [0.1, 0.2]


async def test_exhausts_attempts_and_reraises_last() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise _Transient(f"boom-{calls}")

    with pytest.raises(_Transient, match="boom-3"):
        await retry_async(
            op, max_attempts=3, base_delay=0.1, retry_on=(_Transient,), sleep=_noop_sleep
        )
    assert calls == 3


async def test_does_not_retry_unlisted_exception() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise _Fatal

    with pytest.raises(_Fatal):
        await retry_async(
            op, max_attempts=3, base_delay=0.1, retry_on=(_Transient,), sleep=_noop_sleep
        )
    assert calls == 1


async def test_caps_delay_at_max_delay() -> None:
    calls = 0
    delays: list[float] = []

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls < 5:
            raise _Transient
        return "ok"

    async def rec_sleep(delay: float) -> None:
        delays.append(delay)

    await retry_async(
        op, max_attempts=10, base_delay=1.0, max_delay=3.0, retry_on=(_Transient,), sleep=rec_sleep
    )
    assert delays == [1.0, 2.0, 3.0, 3.0]
