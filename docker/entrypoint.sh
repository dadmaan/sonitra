#!/bin/sh
# =============================================================================
# Sonitra container entrypoint
#
# Runs as root during setup, then steps down to the non-root `sonitra` user
# before executing the container CMD / override.
#
# Responsibilities:
#   1. Ensure bind-mounted directories (/app/corpus /app/output /app/config)
#      are writable by the sonitra user.  Docker Desktop for Windows / macOS
#      mounts host directories as root; without this step the non-root user
#      cannot create files inside them.
#   2. Resolve SONITRA_CONFIG to /app/config/source.yaml so that
#      default_config_path() in the application code finds it.
#   3. Step down to the sonitra user and exec the container CMD.
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
for dir in /app/corpus /app/output /app/config; do
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
# 3. Step down and exec
# ---------------------------------------------------------------------------
exec gosu sonitra "$@"
