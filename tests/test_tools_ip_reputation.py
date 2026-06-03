"""Spec for lookup_ip_reputation: local store default + httpx client seam (respx-mocked)."""

from __future__ import annotations

import httpx
import respx

from triagemcp.tools.ip_reputation import (
    HttpxIpReputationClient,
    IpReputationRecord,
    LocalIpReputationClient,
    LookupIpReputationTool,
)


def _store() -> dict[str, IpReputationRecord]:
    return {
        "203.0.113.45": IpReputationRecord(
            ip="203.0.113.45",
            reputation="malicious",
            score=92,
            categories=["brute-force", "scanner"],
            last_seen="2026-05-28",
            source="local-ti",
        ),
    }


async def test_lookup_known_malicious_ip() -> None:
    tool = LookupIpReputationTool(LocalIpReputationClient(_store()))
    result = await tool.invoke({"ip": "203.0.113.45"})
    assert result.is_error is False
    assert result.content["reputation"] == "malicious"
    assert result.content["score"] == 92


async def test_lookup_unknown_ip_returns_unknown_reputation() -> None:
    tool = LookupIpReputationTool(LocalIpReputationClient(_store()))
    result = await tool.invoke({"ip": "8.8.4.4"})
    assert result.is_error is False
    assert result.content["reputation"] == "unknown"
    assert result.content["score"] == 0


async def test_lookup_rejects_non_ip() -> None:
    tool = LookupIpReputationTool(LocalIpReputationClient(_store()))
    result = await tool.invoke({"ip": "not-an-ip"})
    assert result.is_error is True


async def test_httpx_client_parses_threat_intel_response() -> None:
    payload = {
        "ip": "198.51.100.77",
        "reputation": "malicious",
        "score": 88,
        "categories": ["malware-download"],
        "last_seen": "2026-05-28",
        "source": "example-ti",
    }
    with respx.mock(base_url="https://ti.example") as router:
        router.get("/reputation/198.51.100.77").mock(return_value=httpx.Response(200, json=payload))
        async with httpx.AsyncClient() as client:
            api = HttpxIpReputationClient(client, base_url="https://ti.example/reputation")
            record = await api.lookup("198.51.100.77")
    assert record is not None
    assert record.reputation == "malicious"
    assert record.score == 88


async def test_httpx_client_treats_404_as_no_record() -> None:
    with respx.mock(base_url="https://ti.example") as router:
        router.get("/reputation/9.9.9.9").mock(return_value=httpx.Response(404))
        async with httpx.AsyncClient() as client:
            api = HttpxIpReputationClient(client, base_url="https://ti.example/reputation")
            record = await api.lookup("9.9.9.9")
    assert record is None


def test_bundled_ip_reputation_store_loads() -> None:
    from triagemcp.tools.ip_reputation import load_ip_reputation

    store = load_ip_reputation()
    assert store
    assert all(isinstance(rec, IpReputationRecord) for rec in store.values())
