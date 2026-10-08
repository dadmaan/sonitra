"""Guard the copy-paste ``Examples:`` block at the top of every ``scripts/*.py``."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((REPO / "scripts").glob("*.py"))
HEADER = "Examples:"
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def _docstring(path: Path) -> str:
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)
    assert doc, f"{path.name} has no module docstring"
    return doc


def _example_lines(path: Path) -> list[str]:
    """Return the raw lines of the trailing ``Examples:`` section."""
    lines = _docstring(path).splitlines()
    assert HEADER in lines, f"{path.name}: docstring has no '{HEADER}' section"
    return lines[lines.index(HEADER) + 1 :]


def _commands(path: Path) -> list[str]:
    """Join backslash-continued lines into one command each; skip comments and blanks."""
    commands: list[str] = []
    pending = ""
    for line in _example_lines(path):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        pending = f"{pending} {text}".strip() if pending else text
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        commands.append(pending)
        pending = ""
    assert not pending, f"{path.name}: dangling line continuation"
    return commands


def _parser_options(path: Path) -> set[str]:
    """Option strings passed to any ``add_argument(...)`` call in the file."""
    options: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    options.add(str(arg.value))
    return options | {"-h", "--help"}


def test_every_script_is_covered() -> None:
    assert len(SCRIPTS) >= 9


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_examples_block_shape(path: Path) -> None:
    commands = _commands(path)
    assert 2 <= len(commands) <= 4, f"{path.name}: expected 2-4 examples, got {len(commands)}"
    for command in commands:
        assert command.startswith(f"python scripts/{path.name}"), command


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_example_flags_exist(path: Path) -> None:
    options = _parser_options(path)
    for command in _commands(path):
        for flag in _FLAG.findall(command.split(path.name, 1)[1]):
            assert flag in options, f"{path.name}: example uses unknown option {flag!r}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_help_prints_examples_verbatim(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(path.relative_to(REPO)), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    help_lines = {line.rstrip() for line in result.stdout.splitlines()}
    for line in [HEADER, *_example_lines(path)]:
        if line.strip():
            assert line.rstrip() in help_lines, f"{path.name}: --help reflowed {line!r}"
