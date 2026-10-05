#!/bin/sh
# =============================================================================
# Sonitra container entrypoint
#
# Runs as root during setup, then steps down to the non-root `sonitra` user
# before executing the container CMD / override.
#
# Responsibilities:
#   1. Ensure bind-mounted directories (/app/corpus /app/output /app/config
#      /models) are writable by the sonitra user.  Docker Desktop for Windows /
#      macOS mounts host directories as root; without this step the non-root
#      user cannot create files inside them.
#   2. Resolve SONITRA_CONFIG to /app/config/source.yaml so that
#      default_config_path() in the application code finds it.
#   3. Install the transcriber weights with scripts/setup_hft_transformer.py
#      when SONITRA_SETUP_HFT is on.  Runs as the sonitra user, because the
#      weights land in the /models bind mount that user owns; a failure only
#      warns, since the other backends work without them.
#   4. Step down to the sonitra user and exec the container CMD.
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# 1. Fix bind-mount ownership
# ---------------------------------------------------------------------------
# The image is built with the `sonitra` user's UID/GID matching the host
# user (see HOST_UID/HOST_GID in docker/Dockerfile + docker-compose.yml), so
# in the common case these directories already have the right ownership and
# no chown is needed. We only recurse into a directory when its top-level
# ownership doesn't already match `sonitra` — this covers Docker Desktop
# (which mounts host dirs as root regardless of HOST_UID/HOST_GID) and
# first-run bind mounts of fresh/empty host directories, without repeatedly
# rewriting ownership of a repo directory that a Linux host user already
# owns correctly (which would otherwise lock them out again on every run if
# HOST_UID/HOST_GID were ever left unset).
SONITRA_UID="$(id -u sonitra)"
SONITRA_GID="$(id -g sonitra)"
for dir in /app/corpus /app/output /app/config /models; do
    if [ -d "$dir" ]; then
        current_uid="$(stat -c '%u' "$dir")"
        if [ "$current_uid" != "$SONITRA_UID" ]; then
            chown -R "$SONITRA_UID:$SONITRA_GID" "$dir"
        fi
    fi
done

# The /app root must also be writable for files like renders.jsonl (default
# manifest path).  This is already set during the image build but Docker
# Desktop for Windows / macOS does not persist image ownership into runtime.
chown "$SONITRA_UID:$SONITRA_GID" /app

# ---------------------------------------------------------------------------
# 2. Resolve the pipeline config symlink
# ---------------------------------------------------------------------------
CONFIG_SRC="${SONITRA_CONFIG:-/app/config/config.yaml}"
if [ -f "$CONFIG_SRC" ]; then
    ln -sf "$CONFIG_SRC" /app/config/source.yaml
fi

# ---------------------------------------------------------------------------
# 3. Install the transcriber weights
# ---------------------------------------------------------------------------
# scripts/setup_hft_transformer.py is idempotent and takes an exclusive lock on
# the models directory, so a second container (or the devcontainer) sharing the
# same /models mount only prints "already installed". It has to run as the
# sonitra user: that is who owns the /models bind mount, and the converted
# weights must be readable by the very process that later loads them.
#
# Run it from the venv explicitly, since the script imports sonitra itself.
# Exit codes are load-bearing: 0 installed, 3 torch is absent from this image
# (INSTALL_TRANSKUN=0), anything else is a failure that must not stop startup.
SETUP_RC=0
if [ "${SONITRA_SETUP_HFT:-1}" != "0" ]; then
    gosu sonitra /app/.venv/bin/python /app/scripts/setup_hft_transformer.py \
        --models-dir "${SONITRA_MODELS_DIR:-/models}" || SETUP_RC=$?
    case "$SETUP_RC" in
        0) ;;                       # installed, or already installed
        3) ;;                       # no torch in this image: quiet skip
        *)
            echo "entrypoint: warning: the hft_transformer weights were not installed; run 'python scripts/setup_hft_transformer.py --models-dir ${SONITRA_MODELS_DIR:-/models}' inside the container" >&2
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# 4. Step down and exec
# ---------------------------------------------------------------------------
exec gosu sonitra "$@"
