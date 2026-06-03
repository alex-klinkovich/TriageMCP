"""Spec for enrich_hash: hash-type detection, intel lookup, malformed-input rejection."""

from __future__ import annotations

from triagemcp.tools.hash_enrich import (
    EnrichHashTool,
    HashIntelRecord,
    HashIntelStore,
    load_hash_intel,
)

MAL = "aa11bb22cc33dd44ee55ff6677889900aabbccddeeff00112233445566778899"


def _store() -> dict[str, HashIntelRecord]:
    return {
        MAL: HashIntelRecord(
            hash=MAL,
            hash_type="sha256",
            known_malware=True,
            malware_family="Mimikatz",
            first_seen="2025-11-30",
            detections=58,
            signatures=["HackTool:Win32/Mimikatz"],
        )
    }


async def test_enrich_known_malicious_hash_is_case_insensitive() -> None:
    tool = EnrichHashTool(HashIntelStore(_store()))
    result = await tool.invoke({"file_hash": MAL.upper()})
    assert result.is_error is False
    assert result.content["known_malware"] is True
    assert result.content["malware_family"] == "Mimikatz"
    assert result.content["hash_type"] == "sha256"


async def test_enrich_unknown_hash_reports_clean() -> None:
    tool = EnrichHashTool(HashIntelStore(_store()))
    result = await tool.invoke({"file_hash": "b" * 64})
    assert result.content["known_malware"] is False
    assert result.content["detections"] == 0
    assert result.content["hash_type"] == "sha256"


async def test_enrich_detects_md5_length() -> None:
    tool = EnrichHashTool(HashIntelStore(_store()))
    result = await tool.invoke({"file_hash": "d41d8cd98f00b204e9800998ecf8427e"})
    assert result.content["hash_type"] == "md5"


async def test_enrich_detects_sha1_length() -> None:
    tool = EnrichHashTool(HashIntelStore(_store()))
    result = await tool.invoke({"file_hash": "a" * 40})
    assert result.content["hash_type"] == "sha1"


async def test_enrich_rejects_malformed_hash() -> None:
    tool = EnrichHashTool(HashIntelStore(_store()))
    for bad in ("nothex!!", "abc", "g" * 64):
        result = await tool.invoke({"file_hash": bad})
        assert result.is_error is True, f"expected rejection for {bad!r}"


def test_bundled_hash_intel_loads() -> None:
    store = load_hash_intel()
    assert store
    assert all(isinstance(record, HashIntelRecord) for record in store.values())
