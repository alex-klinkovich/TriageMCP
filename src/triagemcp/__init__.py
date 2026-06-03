"""TriageMCP: an agentic security-alert triage engine exposed over MCP."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("triagemcp")
except PackageNotFoundError:  # pragma: no cover - package not installed (raw source tree)
    __version__ = "0.0.0"

__all__ = ["__version__"]
