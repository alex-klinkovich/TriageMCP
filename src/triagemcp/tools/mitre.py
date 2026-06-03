"""``map_to_mitre``: deterministic keyword mapping of text to MITRE ATT&CK techniques."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from triagemcp.datasets import read_data_text
from triagemcp.models import MitreTechniqueId
from triagemcp.tools.base import BaseTool, ToolResult


class MitreTechnique(BaseModel):
    """A curated ATT&CK technique with the keywords that signal it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: MitreTechniqueId
    name: str = Field(min_length=1)
    tactic: str = Field(min_length=1)
    keywords: tuple[str, ...] = Field(min_length=1)


_TECHNIQUES = TypeAdapter(list[MitreTechnique])


def load_mitre_techniques() -> list[MitreTechnique]:
    """Load the bundled curated ATT&CK technique table."""
    return _TECHNIQUES.validate_json(read_data_text("mitre_techniques.json"))


class MapToMitreInput(BaseModel):
    """Free text to map onto ATT&CK techniques."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Alert text, command line, or description to map.")
    max_results: int = Field(default=3, ge=1, le=10)


class MapToMitreTool(BaseTool[MapToMitreInput]):
    """Score each known technique by how many of its keywords appear in the text."""

    name = "map_to_mitre"
    description = (
        "Map alert text (title, description, or command line) to the most likely MITRE "
        "ATT&CK techniques using deterministic keyword matching. Returns ranked candidates "
        "with technique id, name, tactic, and a match score."
    )
    input_model = MapToMitreInput

    def __init__(self, techniques: Sequence[MitreTechnique]) -> None:
        self._techniques = tuple(techniques)

    async def run(self, args: MapToMitreInput) -> ToolResult:
        haystack = args.text.lower()
        scored: list[tuple[int, MitreTechnique]] = []
        for technique in self._techniques:
            score = sum(1 for keyword in technique.keywords if keyword.lower() in haystack)
            if score:
                scored.append((score, technique))

        # Highest score first; break ties by technique id for stable, deterministic output.
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        top = scored[: args.max_results]

        return ToolResult.ok(
            {
                "matches": [
                    {
                        "id": technique.id,
                        "name": technique.name,
                        "tactic": technique.tactic,
                        "score": score,
                    }
                    for score, technique in top
                ],
                "match_count": len(scored),
            }
        )
