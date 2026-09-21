#!/usr/bin/env bash
# Reproduces the lock's transkun torch fork into /workspace/.venv when the devcontainer
# is created. Safe to re-run by hand; see docs/devcontainer.md.
#
# The fork follows the image, not the host: $NV/cudnn/lib exists only when the image was
# built with GPU_EXTRAS=gpu (docker-compose.gpu.yml), which is also what installs the
# TensorFlow CUDA wheels. A GPU host running the base compose file gets the CPU fork,
# matching the container's absent GPU passthrough.
set -euo pipefail

cd "$(dirname "$0")/.."

EXTRA="transkun"
SYNC_EXTRAS=(--extra transkun --extra dev)
if [ -n "${NV:-}" ] && [ -d "$NV/cudnn/lib" ]; then
    EXTRA="transkun-gpu"
    SYNC_EXTRAS=(--extra transkun-gpu --extra xla-ptx --extra dev)
fi

# Lets F5 verify the selection without a multi-GB sync.
if [ "${1:-}" = "--print-extra" ]; then
    echo "$EXTRA"
    exit 0
fi

echo "post-create: uv sync --locked ${SYNC_EXTRAS[*]}"
# Self-heal a broken venv (e.g. a partial sync through a Windows bind mount):
# uv errors instead of recreating when .venv exists without bin/python.
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
    echo "post-create: removing broken .venv (no bin/python)"
    rm -rf .venv
fi
uv sync --locked "${SYNC_EXTRAS[@]}"

# `--no-sync`: this check must not touch the environment it just built.
uv run --no-sync python -c "import sonitra"