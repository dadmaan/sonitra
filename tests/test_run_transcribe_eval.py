from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_transcribe_eval.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_transcribe_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flag_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    return command[command.index(flag) + 1]


def test_render_command_omits_seed_without_limit() -> None:
    module = _load_module()
    command = module._build_render_command(
        Path("config/examples/pedalboard_baseline.yaml"), None, None, 123
    )

    assert command[: len(module._PYTHON)] == module._PYTHON
    assert command[len(module._PYTHON) : len(module._PYTHON) + 3] == [
        "-m",
        "sonitra",
        "render",
    ]
    assert _flag_value(command, "--config") == "config/examples/pedalboard_baseline.yaml"
    assert "--dataset" not in command
    assert "--limit" not in command
    assert "--seed" not in command


def test_render_command_forwards_seed_with_limit() -> None:
    module = _load_module()
    command = module._build_render_command(
        Path("config/examples/pedalboard_baseline.yaml"), "maestro-v3", 5, 123
    )

    assert _flag_value(command, "--dataset") == "maestro-v3"
    assert _flag_value(command, "--limit") == "5"
    assert _flag_value(command, "--seed") == "123"
