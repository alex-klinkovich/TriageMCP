"""Shared pytest configuration.

Adds the ``--run-live`` opt-in. Tests marked ``live`` hit the real Anthropic API and are
skipped unless ``--run-live`` is passed, keeping the default suite offline and free.
"""

from __future__ import annotations

import pytest

from triagemcp.config import Settings


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the offline suite hermetic against a developer's real ``.env``.

    Without this, a local ``.env`` holding a real ``ANTHROPIC_API_KEY`` would leak into the
    tests: the missing-key tests would stop failing, and — far worse — the CLI tests would
    sail past the key guard and hit the real API. Production still reads ``.env`` normally;
    only tests disable it, relying solely on explicitly monkeypatched env vars.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked 'live' that call the real Anthropic API (needs ANTHROPIC_API_KEY).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live test: pass --run-live and set ANTHROPIC_API_KEY")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
