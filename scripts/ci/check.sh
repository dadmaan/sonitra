#!/usr/bin/env bash
# check.sh: the one set of checks CI and the local pre-push gate both run.
# Usage: bash scripts/ci/check.sh [--full]   (--full runs the slow tests too)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# The lock file is the contract with uv 0.7.8; a stale one makes the run below
# resolve different versions than CI would.
if ! uv lock --check; then
  echo "uv.lock is out of date: run 'uv lock' on the host and commit it" >&2
  exit 1
fi

if [ "${1:-}" = "--full" ]; then
  uv run --no-sync pytest tests/ -q
else
  uv run --no-sync pytest tests/ -q -m "not slow"
fi
