"""Checks that dataset instructions never cite a config path that was renamed away."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_datasets.py"

_CONFIG_REFERENCE = re.compile(r"config/[A-Za-z0-9_./-]*\.yaml")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def dd() -> ModuleType:
    return _load_module()


def test_next_steps_reference_configs_that_exist(dd: ModuleType) -> None:
    # One-way check: a dataset's follow-up instructions are only useful if the
    # commands they print still run, so a renamed config breaks the documented
    # workflow from a text string that no import or schema check would notice.
    # Nothing here says anything about presets that nothing references.
    instructions: dict[str, str] = {
        key: spec["next_steps"]
        for key, spec in dd.DATASETS.items()
        if isinstance(spec.get("next_steps"), str)
    }
    assert instructions, "no dataset carries next_steps instructions"

    cited: list[str] = []
    missing: list[str] = []
    for key, steps in instructions.items():
        for reference in re.findall(_CONFIG_REFERENCE, steps):
            cited.append(reference)
            if not (REPO_ROOT / reference).is_file():
                missing.append(f"{key}: {reference}")

    # Without a cited path the loop above checks nothing at all, so the rule
    # would pass silently the day the instructions stop naming configs.
    assert cited, "no next_steps instruction cites a config/ path"
    assert not missing, "next_steps cite configs that do not exist:\n" + "\n".join(
        missing
    )
