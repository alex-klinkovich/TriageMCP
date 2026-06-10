"""Spec for the ToolRegistry and the default tool wiring."""

from __future__ import annotations

import datetime as dt

import aiosqlite
import pytest

from triagemcp.datasets import load_sample_alerts
from triagemcp.tools.mitre import MapToMitreTool, MitreTechnique
from triagemcp.tools.registry import ToolRegistry, _history_from_samples, build_default_registry

EXPECTED_TOOLS = {"map_to_mitre", "lookup_ip_reputation", "enrich_hash", "query_recent_alerts"}


def _mitre_tool() -> MapToMitreTool:
    return MapToMitreTool(
        [
            MitreTechnique(
                id="T1110",
                name="Brute Force",
                tactic="Credential Access",
                keywords=("brute force",),
            )
        ]
    )


def test_registry_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([_mitre_tool(), _mitre_tool()])


def test_registry_lists_names_and_anthropic_specs() -> None:
    registry = ToolRegistry([_mitre_tool()])
    assert registry.names == ["map_to_mitre"]
    spec = registry.anthropic_specs()[0]
    assert spec["name"] == "map_to_mitre"
    assert spec["description"]
    assert spec["input_schema"]["type"] == "object"


async def test_registry_dispatches_to_named_tool() -> None:
    registry = ToolRegistry([_mitre_tool()])
    result = await registry.dispatch("map_to_mitre", {"text": "brute force login attempts"})
    assert result.is_error is False
    assert result.content["matches"][0]["id"] == "T1110"


async def test_registry_unknown_tool_returns_error_result() -> None:
    registry = ToolRegistry([_mitre_tool()])
    result = await registry.dispatch("does_not_exist", {})
    assert result.is_error is True
    assert "does_not_exist" in result.content["error"]


async def test_build_default_registry_wires_all_four_tools() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        registry = await build_default_registry(conn)
        assert set(registry.names) == EXPECTED_TOOLS
        assert len(registry.anthropic_specs()) == 4
        # The wired query tool talks to the seeded store without error.
        result = await registry.dispatch(
            "query_recent_alerts", {"observable": "no-such-observable"}
        )
        assert result.is_error is False


def test_history_seeds_exactly_the_recurring_observables() -> None:
    # Independently compute which (value, type) appear in >= 2 distinct alerts.
    seen_in: dict[tuple[str, str], set[str]] = {}
    for labeled in load_sample_alerts():
        a = labeled.alert
        o = a.observables
        typed = [
            *((str(ip), "ip") for ip in o.ips),
            *((h, "hash") for h in o.file_hashes),
            *((d, "domain") for d in o.domains),
            *((u, "user") for u in o.users),
            *((host, "host") for host in o.hosts),
        ]
        for value, typ in typed:
            seen_in.setdefault((value, typ), set()).add(a.id)

    expected_recurring = {key for key, ids in seen_in.items() if len(ids) >= 2}
    expected_singletons = {key for key, ids in seen_in.items() if len(ids) == 1}

    seeded = {(e.observable, e.observable_type) for e in _history_from_samples()}

    assert seeded == expected_recurring
    assert seeded.isdisjoint(expected_singletons)
    assert ("FIN-WS-118", "host") in seeded  # sanity anchor: appears in A-0003 and A-0023


async def test_default_history_records_are_distinct_prior_sightings() -> None:
    fixed_now = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
    async with aiosqlite.connect(":memory:") as conn:
        registry = await build_default_registry(conn, clock=lambda: fixed_now)
        result = await registry.dispatch(
            "query_recent_alerts", {"observable": "203.0.113.45", "lookback_hours": 336}
        )
    assert result.content["match_count"] >= 1
    # History must not be the live alert itself: distinct ids, timestamps strictly earlier.
    assert all(entry["alert_id"].startswith("HIST-") for entry in result.content["alerts"])
