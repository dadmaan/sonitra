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
# chown -R is safe here because /app/corpus, /app/output, and /app/config are
# either bind-mounted empty dirs or image-owned dirs with the right content.
# We run as root so the sonitra user gets write access regardless of the host
# OS (Windows, macOS, or Linux with a mismatched UID).
for dir in /app/corpus /app/output /app/config; do
    if [ -d "$dir" ]; then
        chown -R sonitra:sonitra "$dir"
    fi
done

# The /app root must also be writable for files like renders.jsonl (default
# manifest path).  This is already set during the image build but Docker
# Desktop for Windows / macOS does not persist image ownership into runtime.
chown sonitra:sonitra /app

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
