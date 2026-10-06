#!/usr/bin/env bash
# check_tag_version.sh: refuse a release tag that disagrees with pyproject.toml.
#
# Usage: bash scripts/ci/check_tag_version.sh TAG
#
# A stable tag vX.Y.Z requires an exact pyproject version. A pre-release tag
# vX.Y.Z-label.N requires pyproject to be a PEP 440 pre-/dev-release of the same
# X.Y.Z; the tag's N need not equal the pyproject N, because dev tags do not
# bump pyproject.toml. Post-releases are not accepted.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/ci/check_tag_version.sh TAG" >&2
  exit 2
fi

# Resolved from this script, never from git or the working directory. CI runs it
# from a checkout, but it must also work from any cwd.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYPROJECT="$ROOT/pyproject.toml"

if [ ! -f "$PYPROJECT" ]; then
  echo "check_tag_version: no pyproject.toml at $PYPROJECT" >&2
  exit 1
fi

VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$PYPROJECT" | head -n 1)"
if [ -z "$VERSION" ]; then
  echo "check_tag_version: no version = \"...\" line in $PYPROJECT" >&2
  exit 1
fi

TAG="${1#v}"

# A stable tag carries no hyphen: the versions must be identical.
if [[ "$TAG" != *-* ]]; then
  if [ "$TAG" = "$VERSION" ]; then
    echo "tag v$TAG matches pyproject version $VERSION"
    exit 0
  fi
  echo "tag v$TAG does not match pyproject version $VERSION" >&2
  exit 1
fi

# A pre-release tag is X.Y.Z-label.N. Only the base version and the kind of
# pre-release are pinned; the tag's N may differ from pyproject's.
BASE="${TAG%%-*}"
REMAINDER="${TAG#*-}"
LABEL="${REMAINDER%.*}"
TAG_N="${REMAINDER##*.}"

if ! [[ "$BASE" =~ ^[0-9]+(\.[0-9]+)*$ ]] || ! [[ "$TAG_N" =~ ^[0-9]+$ ]]; then
  echo "tag v$TAG is not of the form X.Y.Z-label.N" >&2
  exit 1
fi

# The tag label -> the PEP 440 suffix pyproject must carry.
case "$LABEL" in
  dev) SUFFIX_RE='\.dev' ;;
  rc | alpha | beta | pre | preview | a | b | c) SUFFIX_RE="$LABEL" ;;
  *)
    echo "tag v$TAG has an unsupported pre-release label '$LABEL'" >&2
    exit 1
    ;;
esac

BASE_RE="${BASE//./\\.}"
if [[ "$VERSION" =~ ^${BASE_RE}${SUFFIX_RE}[0-9]+$ ]]; then
  echo "tag v$TAG matches pyproject version $VERSION"
  exit 0
fi

echo "tag v$TAG does not match pyproject version $VERSION" >&2
exit 1
