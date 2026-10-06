"""Contracts for the release helpers: changelog extraction and the tag guard.

``scripts/changelog_section.py`` is invoked as a child process with the same
interpreter that runs the tests. ``scripts/ci/check_tag_version.sh`` is copied
into a throwaway tree, together with a minimal ``pyproject.toml``, so its
resolution of the repository root from its own location can be exercised
without touching the real checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "changelog_section.py"
CHECK_TAG_SOURCE = REPO / "scripts" / "ci" / "check_tag_version.sh"

requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="bash is needed to exercise check_tag_version.sh",
)


def _run_changelog(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _write_changelog(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(text, encoding="utf-8")
    return path


# -- scripts/changelog_section.py ---------------------------------------------

MIDDLE = """\
# Changelog

## [1.2.2] - 2026-01-01

### Fixed

- First.

## [1.2.3] - 2026-01-02

### Added

- Second.

## [1.2.4] - 2026-01-03

### Changed

- Third.
"""

LAST = """\
# Changelog

## [1.0.0] - 2025-01-01

- Old.

## [1.1.0] - 2025-02-01

### Added

- Newest.

[Unreleased]: https://example.invalid/compare
[1.1.0]: https://example.invalid/compare/v1.0.0...v1.1.0
"""


def test_middle_section_has_no_headings(tmp_path: Path) -> None:
    changelog = _write_changelog(tmp_path, MIDDLE)
    result = _run_changelog("1.2.3", "--changelog", str(changelog))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "### Added\n\n- Second.\n"
    assert "## [" not in result.stdout
    assert "### Fixed" not in result.stdout
    assert "### Changed" not in result.stdout


def test_last_section_excludes_footer(tmp_path: Path) -> None:
    changelog = _write_changelog(tmp_path, LAST)
    result = _run_changelog("1.1.0", "--changelog", str(changelog))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "### Added\n\n- Newest.\n"
    assert "https://" not in result.stdout
    assert "[1.1.0]:" not in result.stdout


def test_heading_with_date_suffix_matches(tmp_path: Path) -> None:
    changelog = _write_changelog(
        tmp_path, "## [1.2.3] - 2026-01-01\n\n### Added\n\n- Dated.\n"
    )
    result = _run_changelog("1.2.3", "--changelog", str(changelog))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "### Added\n\n- Dated.\n"


def test_stable_version_does_not_match_prerelease_section(tmp_path: Path) -> None:
    changelog = _write_changelog(
        tmp_path, "## [0.5.0-dev.1]\n\n### Added\n\n- Dev only.\n"
    )
    result = _run_changelog("0.5.0", "--changelog", str(changelog))
    assert result.returncode == 1
    assert f"no section '## [0.5.0]' in {changelog}" in result.stderr


def test_prerelease_version_does_not_match_stable_section(tmp_path: Path) -> None:
    changelog = _write_changelog(tmp_path, "## [0.5.0]\n\n### Added\n\n- Stable.\n")
    result = _run_changelog("0.5.0-dev.1", "--changelog", str(changelog))
    assert result.returncode == 1
    assert f"no section '## [0.5.0-dev.1]' in {changelog}" in result.stderr


def test_leading_v_is_accepted(tmp_path: Path) -> None:
    changelog = _write_changelog(tmp_path, "## [1.2.3]\n\n- Body.\n")
    result = _run_changelog("v1.2.3", "--changelog", str(changelog))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "- Body.\n"


def test_missing_section_exits_1_with_message(tmp_path: Path) -> None:
    changelog = _write_changelog(
        tmp_path, "# Changelog\n\n## [2.0.0] - 2026-01-01\n\n- Two.\n"
    )
    result = _run_changelog("1.0.0", "--changelog", str(changelog))
    assert result.returncode == 1
    assert result.stderr == f"no section '## [1.0.0]' in {changelog}\n"
    assert result.stdout == ""


def test_allow_missing_prints_nothing_and_exits_0(tmp_path: Path) -> None:
    changelog = _write_changelog(tmp_path, "## [2.0.0]\n\n- Two.\n")
    result = _run_changelog("1.0.0", "--changelog", str(changelog), "--allow-missing")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_empty_section_prints_nothing(tmp_path: Path) -> None:
    changelog = _write_changelog(
        tmp_path,
        "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n## [1.1.0] - 2026-02-01\n\n"
        "### Added\n\n- Next.\n",
    )
    result = _run_changelog("1.0.0", "--changelog", str(changelog))
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_real_changelog_0_3_0_starts_with_added() -> None:
    result = _run_changelog("0.3.0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("### Added")
    assert result.stdout.strip()


# -- scripts/ci/check_tag_version.sh ------------------------------------------


def _install_check_tag(tmp_path: Path, version: str) -> Path:
    target = tmp_path / "scripts" / "ci" / "check_tag_version.sh"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CHECK_TAG_SOURCE, target)
    target.chmod(0o755)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8"
    )
    return target


def _run_check_tag(
    target: Path, *args: str, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(target), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


@requires_bash
def test_stable_tag_matches_pyproject(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.4.0")
    result = _run_check_tag(target, "v0.4.0", cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@requires_bash
def test_stable_tag_mismatch_names_both(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.5.0.dev0")
    result = _run_check_tag(target, "v0.4.0", cwd=tmp_path)
    assert result.returncode == 1
    assert "v0.4.0" in result.stderr
    assert "0.5.0.dev0" in result.stderr


@requires_bash
def test_prerelease_tag_matches_dev_version(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.5.0.dev0")
    result = _run_check_tag(target, "v0.5.0-dev.3", cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@requires_bash
def test_prerelease_tag_base_mismatch_names_both(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.4.0")
    result = _run_check_tag(target, "v0.5.0-dev.3", cwd=tmp_path)
    assert result.returncode == 1
    assert "v0.5.0-dev.3" in result.stderr
    assert "0.4.0" in result.stderr


@requires_bash
def test_rc_tag_matches_pep440_rc_version(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.5.0rc1")
    result = _run_check_tag(target, "v0.5.0-rc.1", cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@requires_bash
def test_pyproject_is_resolved_from_script_not_cwd(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.4.0")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _run_check_tag(target, "v0.4.0", cwd=elsewhere)
    assert result.returncode == 0, result.stderr


@requires_bash
def test_missing_tag_is_a_usage_error(tmp_path: Path) -> None:
    target = _install_check_tag(tmp_path, "0.4.0")
    result = _run_check_tag(target, cwd=tmp_path)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()
