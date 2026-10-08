#!/usr/bin/env python3
"""Print one version's section of the changelog.

Finds the heading ``## [VERSION]`` -- case-sensitive and exact, so ``0.5.0``
never matches ``## [0.5.0-dev.1]`` and the reverse never matches either -- and
writes everything after it up to the next ``## [`` heading or the first
link-reference footer line (``[label]: URL``), with blank lines trimmed from
both ends. A section that is present but empty prints nothing. Used by the
release workflow to turn a tag's changelog section into the GitHub Release
notes. Stdlib only, so it runs on a bare runner.

Examples:
    # print the section for version 0.5.0-dev.4 from the repo changelog
    python scripts/changelog_section.py 0.5.0-dev.4

    # read a different changelog, and accept a leading v on the version
    python scripts/changelog_section.py v0.4.0 --changelog path/to/CHANGELOG.md

    # exit 0 with no output when the section is absent (pre-release notes)
    python scripts/changelog_section.py 0.5.0 --allow-missing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Resolved from this file, never the working directory: the release workflow
# runs the script from the repository root, but a manual run should work too.
REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHANGELOG = REPO / "CHANGELOG.md"

# A Keep a Changelog link-reference footer: "[keep-a-changelog]: https://...".
FOOTER = re.compile(r"^\[[^\]]+\]: ")


def _section(lines: list[str], version: str) -> list[str] | None:
    """The changelog body for ``version``, or ``None`` when the heading is absent."""
    # Escape the version, and require the literal closing bracket, so a heading
    # is matched exactly rather than as a prefix.
    heading = re.compile(r"^## \[" + re.escape(version) + r"\]")
    start = None
    for index, line in enumerate(lines):
        if heading.match(line):
            start = index + 1
            break
    if start is None:
        return None

    body: list[str] = []
    for line in lines[start:]:
        if line.startswith("## [") or FOOTER.match(line):
            break
        body.append(line)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return body


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="version to extract, with or without a leading v")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=DEFAULT_CHANGELOG,
        help="changelog to read (default: CHANGELOG.md at the repository root)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 with no output when the section is absent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    changelog: Path = args.changelog
    version = args.version
    if version.startswith("v"):
        version = version[1:]

    try:
        text = changelog.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {changelog}: {exc}", file=sys.stderr)
        return 1

    body = _section(text.splitlines(), version)
    if body is None:
        if args.allow_missing:
            return 0
        print(f"no section '## [{version}]' in {changelog}", file=sys.stderr)
        return 1
    if body:
        print("\n".join(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
