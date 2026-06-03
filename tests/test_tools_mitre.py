"""Spec for the deterministic map_to_mitre tool (and the BaseTool contract)."""

from __future__ import annotations

from triagemcp.tools.mitre import MapToMitreTool, MitreTechnique


def _techniques() -> list[MitreTechnique]:
    return [
        MitreTechnique(
            id="T1059.001",
            name="PowerShell",
            tactic="Execution",
            keywords=("powershell", "encoded command", "-enc"),
        ),
        MitreTechnique(
            id="T1110",
            name="Brute Force",
            tactic="Credential Access",
            keywords=("brute force", "failed login", "password spray"),
        ),
        MitreTechnique(
            id="T1486",
            name="Data Encrypted for Impact",
            tactic="Impact",
            keywords=("ransomware", "encrypted files", "ransom note"),
        ),
    ]


async def test_map_to_mitre_ranks_best_match_first() -> None:
    tool = MapToMitreTool(_techniques())
    result = await tool.invoke({"text": "Encoded PowerShell launched with -enc on the host"})
    assert result.is_error is False
    matches = result.content["matches"]
    assert matches[0]["id"] == "T1059.001"
    assert matches[0]["score"] >= 2
    assert matches[0]["tactic"] == "Execution"


async def test_map_to_mitre_returns_empty_when_nothing_matches() -> None:
    tool = MapToMitreTool(_techniques())
    result = await tool.invoke({"text": "user updated their email signature"})
    assert result.content["matches"] == []
    assert result.content["match_count"] == 0


async def test_map_to_mitre_respects_max_results() -> None:
    tool = MapToMitreTool(_techniques())
    result = await tool.invoke({"text": "powershell brute force ransomware", "max_results": 2})
    assert len(result.content["matches"]) == 2


async def test_map_to_mitre_rejects_blank_text() -> None:
    tool = MapToMitreTool(_techniques())
    result = await tool.invoke({"text": ""})
    assert result.is_error is True
    assert "error" in result.content


async def test_map_to_mitre_advertises_object_schema() -> None:
    tool = MapToMitreTool(_techniques())
    schema = tool.input_schema()
    assert schema["type"] == "object"
    assert "text" in schema["properties"]


def test_map_to_mitre_identity() -> None:
    tool = MapToMitreTool(_techniques())
    assert tool.name == "map_to_mitre"
    assert tool.description
