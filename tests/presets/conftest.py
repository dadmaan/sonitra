"""Tag every test in this package with the ``presets`` marker."""

from __future__ import annotations

from pathlib import Path

import pytest

_PRESET_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(_PRESET_DIR):
            item.add_marker(pytest.mark.presets)