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

# The models bind mount is created by the container runtime when the host directory
# does not exist yet, and it creates it as root -- after which the `node` user this
# script runs as cannot write into it, and the weights download below would fail at
# create time. Hand the directory to the current user when it is not already ours.
# Idempotent: an /models we can already write to is left untouched, and a failure
# here must not fail the create (the setup step reports its own problem instead).
MODELS_DIR="${SONITRA_MODELS_DIR:-/models}"
if [ ! -w "$MODELS_DIR" ]; then
    echo "post-create: $MODELS_DIR is not writable by $(id -un); handing it to $(id -u):$(id -g)"
    sudo mkdir -p "$MODELS_DIR" \
        && sudo chown "$(id -u):$(id -g)" "$MODELS_DIR" \
        || echo "post-create: warning: could not take ownership of $MODELS_DIR; the weights download will likely fail" >&2
fi

echo "post-create: uv sync --locked ${SYNC_EXTRAS[*]}"
# Self-heal a broken venv (e.g. a partial sync through a Windows bind mount):
# uv errors instead of recreating when .venv exists without bin/python.
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
    echo "post-create: removing broken .venv (no bin/python)"
    rm -rf .venv
fi
uv sync --locked "${SYNC_EXTRAS[@]}"

# Verify the fork actually landed: uv sync exits 0 without swapping the torch
# build across a fork switch, so check which build is installed via
# torch.version.cuda (not torch.cuda.is_available(), which reflects GPU
# visibility rather than the installed build).
probe_torch_cuda() {
    uv run --no-sync python -c 'try:
    import torch
    print(torch.version.cuda or "cpu")
except ImportError:
    print("missing")'
}

TORCH_CUDA="$(probe_torch_cuda)"
if [ "$EXTRA" = "transkun-gpu" ]; then
    if [ "$TORCH_CUDA" = "cpu" ] || [ "$TORCH_CUDA" = "missing" ]; then
        if [ "$TORCH_CUDA" = "missing" ]; then
            echo "post-create: torch is missing but the GPU fork was selected; reinstalling torch/torchaudio"
        else
            echo "post-create: torch is a CPU build but the GPU fork was selected; reinstalling torch/torchaudio"
        fi
        uv sync --locked "${SYNC_EXTRAS[@]}" --reinstall-package torch --reinstall-package torchaudio
        TORCH_CUDA="$(probe_torch_cuda)"
        if [ "$TORCH_CUDA" = "cpu" ] || [ "$TORCH_CUDA" = "missing" ]; then
            echo "post-create: expected a CUDA torch build for EXTRA=transkun-gpu but found '${TORCH_CUDA}'; failing" >&2
            exit 1
        fi
    fi
else
    if [ "$TORCH_CUDA" != "cpu" ]; then
        if [ "$TORCH_CUDA" = "missing" ]; then
            echo "post-create: torch is missing but the CPU fork was selected; reinstalling torch/torchaudio"
        else
            echo "post-create: torch is a CUDA build but the CPU fork was selected; reinstalling torch/torchaudio"
        fi
        uv sync --locked "${SYNC_EXTRAS[@]}" --reinstall-package torch --reinstall-package torchaudio
        TORCH_CUDA="$(probe_torch_cuda)"
        if [ "$TORCH_CUDA" != "cpu" ]; then
            echo "post-create: expected a CPU torch build for EXTRA=transkun but found '${TORCH_CUDA}'; failing" >&2
            exit 1
        fi
    fi
fi

# `--no-sync`: this check must not touch the environment it just built.
uv run --no-sync python -c "import sonitra"

# Install the transcriber weights into the shared models mount (scripts/
# setup_hft_transformer.py). `--no-sync` for the same reason as above. Fail-soft:
# the other backends work without the weights, and the hft_transformer backend
# raises its own error naming this command when they are missing.
#
#   0  installed, or already installed -> nothing to say
#   3  torch is not in this venv -> skip quietly, only note it once
#   *  anything else -> one warning naming the command to run by hand
setup_rc=0
uv run --no-sync python scripts/setup_hft_transformer.py --models-dir "$MODELS_DIR" || setup_rc=$?
case "$setup_rc" in
    0) ;;
    3) echo "post-create: torch is not installed, so the hft_transformer setup is skipped" ;;
    *)
        echo "post-create: warning: the hft_transformer weights were not installed; run 'uv run python scripts/setup_hft_transformer.py --models-dir $MODELS_DIR' by hand" >&2
        ;;
esac