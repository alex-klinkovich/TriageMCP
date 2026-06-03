"""Loaders for the packaged datasets shipped under ``triagemcp/data``.

Kept deliberately tiny and dependency-light: the sample alerts and the curated
MITRE technique table are read through :func:`importlib.resources.files`, so they
work the same whether the package is installed editable or from a built wheel.
"""

from __future__ import annotations

from importlib.resources import files

from pydantic import TypeAdapter

from triagemcp.models import LabeledAlert

_LABELED_ALERTS = TypeAdapter(list[LabeledAlert])


def read_data_text(name: str) -> str:
    """Return the UTF-8 text of a packaged data file under ``triagemcp/data``."""
    return (files("triagemcp") / "data" / name).read_text(encoding="utf-8")


def load_sample_alerts() -> list[LabeledAlert]:
    """Load and validate the bundled labeled sample alerts."""
    return _LABELED_ALERTS.validate_json(read_data_text("sample_alerts.json"))
