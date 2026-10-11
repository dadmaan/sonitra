"""Unit tests for the audit hook that records reads of the shipped config tree."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import _is_guarded, _ENFORCE, _GUARDED_EVENTS, _GUARDED_ROOT


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("key: value\n", encoding="utf-8")
    return path


def test_open_below_root_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "config"
    target = _write(root / "examples" / "x.yaml")

    hit = _is_guarded("open", (target, "r", -1), root)

    assert hit == target.resolve()
    assert hit is not None and root.resolve() in hit.parents


def test_sibling_directory_next_to_root_is_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "config"
    root.mkdir()
    sibling = _write(root / ".." / "tests" / "fixtures" / "x.yaml")

    assert _is_guarded("open", (sibling, "rb"), root) is None


def test_relative_path_is_resolved_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "config"
    expected = _write(root / "examples" / "x.yaml").resolve()

    assert _is_guarded("open", ("config/examples/x.yaml", "r"), root) == expected
    assert _is_guarded("os.scandir", ("./config/benchmark",), root) == (root / "benchmark").resolve()


def test_file_descriptor_argument_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert _is_guarded("open", (3, "r", 1), tmp_path / "config") is None


@pytest.mark.parametrize("event", ["os.scandir", "os.listdir"])
def test_directory_listing_below_root_is_recorded(
    event: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "config"
    target = (root / "benchmark").resolve()

    assert _is_guarded(event, (target,), root) == target


def test_missing_path_argument_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert _is_guarded("os.listdir", (None,), tmp_path / "config") is None
    assert _is_guarded("os.listdir", (), tmp_path / "config") is None


@pytest.mark.parametrize(
    "event, args",
    [
        ("import", ("sonitra.config", None, None, None)),
        ("exec", ("x = 1", None)),
        ("socket.connect", (None, ("localhost", 80))),
        ("os.stat", (3, None)),
    ],
)
def test_other_events_are_ignored(event: str, args: tuple[object, ...], tmp_path: Path) -> None:
    assert _is_guarded(event, args, tmp_path / "config") is None


def test_directory_named_like_the_root_elsewhere_is_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "config"
    root.mkdir()
    elsewhere = _write(tmp_path / "audio" / "config" / "x.yaml")

    assert _is_guarded("open", (elsewhere, "r"), root) is None


def test_shared_string_prefix_is_not_a_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "config"
    root.mkdir()
    lookalike = _write(tmp_path / "configs" / "x.yaml")

    assert _is_guarded("open", (lookalike, "r"), root) is None


def test_root_itself_is_a_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = (tmp_path / "config").resolve()
    root.mkdir()

    assert _is_guarded("os.scandir", (str(root),), root) == root


def test_non_path_argument_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert _is_guarded("open", (object(), "r"), tmp_path / "config") is None
    assert _is_guarded("open", (1.5, "r"), tmp_path / "config") is None


def test_guarded_root_points_at_the_shipped_config_tree() -> None:
    assert _GUARDED_ROOT == Path(__file__).resolve().parents[1] / "config"
    assert _GUARDED_EVENTS == frozenset({"open", "os.scandir", "os.listdir"})


def test_the_guard_fails_reads_of_the_shipped_tree() -> None:
    # Enforce mode: a test that reads under config/ fails in teardown with the
    # paths it read, and only the preset suite is exempt. Report mode existed
    # only to baseline the migration that moved the other tests onto fixtures.
    assert _ENFORCE is True


def test_audit_channel_never_raises_for_unusual_arguments() -> None:
    # Audit hooks run inside arbitrary library code and during interpreter
    # shutdown, so raising here would break an unrelated test.
    sys.audit("open", 3, "r", 1)
    sys.audit("os.listdir", None)
    sys.audit("sonitra.test_unrelated_event", object())