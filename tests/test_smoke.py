"""Smoke test: the package imports and exposes a sane version string."""

from __future__ import annotations

import importlib.metadata

import triagemcp


def test_package_exposes_version() -> None:
    assert isinstance(triagemcp.__version__, str)
    assert triagemcp.__version__, "version string must be non-empty"


def test_version_matches_installed_metadata() -> None:
    assert triagemcp.__version__ == importlib.metadata.version("triagemcp")
