"""``enrich_hash``: look up a file hash in a malware-intelligence store.

The hash type is inferred from length (md5/sha1/sha256) and the value is validated
as hex at the input boundary, so a malformed hash from the model is rejected as a
structured error instead of silently returning a bogus verdict.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter

from triagemcp.datasets import read_data_text
from triagemcp.tools.base import BaseTool, ToolResult

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_HASH_TYPE_BY_LENGTH: dict[int, str] = {32: "md5", 40: "sha1", 64: "sha256"}


def _normalize_hash(value: str) -> str:
    candidate = value.strip().lower()
    if len(candidate) not in _HASH_TYPE_BY_LENGTH or not _HEX_RE.match(candidate):
        raise ValueError(f"not a valid md5, sha1, or sha256 hash: {value!r}")
    return candidate


FileHash = Annotated[str, AfterValidator(_normalize_hash)]
"""A hex file hash, normalized to lowercase and constrained to md5/sha1/sha256 length."""


def hash_type_of(value: str) -> str:
    """Return the hash algorithm name implied by a normalized hash's length."""
    return _HASH_TYPE_BY_LENGTH[len(value)]


class HashIntelRecord(BaseModel):
    """A malware-intelligence verdict for a file hash."""

    model_config = ConfigDict(extra="forbid")

    hash: str
    hash_type: str
    known_malware: bool
    malware_family: str | None = None
    first_seen: str | None = None
    detections: int = Field(default=0, ge=0, description="Engines flagging the sample.")
    signatures: list[str] = Field(default_factory=list)


class HashIntelStore:
    """In-memory intel store keyed by lowercase hash."""

    def __init__(self, records: Mapping[str, HashIntelRecord]) -> None:
        self._records = {key.lower(): value for key, value in records.items()}

    def get(self, file_hash: str) -> HashIntelRecord | None:
        return self._records.get(file_hash)


class EnrichHashInput(BaseModel):
    """A single file hash (md5, sha1, or sha256) to enrich."""

    model_config = ConfigDict(extra="forbid")

    file_hash: FileHash = Field(description="Hex md5, sha1, or sha256 file hash.")


class EnrichHashTool(BaseTool[EnrichHashInput]):
    """Resolve malware intelligence for a hash, defaulting to a clean verdict if unknown."""

    name = "enrich_hash"
    description = (
        "Enrich a file hash (md5, sha1, or sha256) with malware intelligence: whether it is "
        "known malware, the malware family, detection count, and signatures. Unknown hashes "
        "return known_malware=false with zero detections."
    )
    input_model = EnrichHashInput

    def __init__(self, store: HashIntelStore) -> None:
        self._store = store

    async def run(self, args: EnrichHashInput) -> ToolResult:
        file_hash = args.file_hash
        record = self._store.get(file_hash)
        if record is None:
            record = HashIntelRecord(
                hash=file_hash,
                hash_type=hash_type_of(file_hash),
                known_malware=False,
                detections=0,
                signatures=[],
            )
        return ToolResult.ok(record.model_dump(mode="json"))


_RECORDS = TypeAdapter(list[HashIntelRecord])


def load_hash_intel() -> dict[str, HashIntelRecord]:
    """Load the bundled offline hash-intelligence store, keyed by lowercase hash."""
    records = _RECORDS.validate_json(read_data_text("hash_intel.json"))
    return {record.hash.lower(): record for record in records}
