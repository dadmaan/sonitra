#!/bin/sh
# =============================================================================
# Sonitra container entrypoint
#
# Responsibilities:
#   1. Resolve SONITRA_CONFIG to /app/config.yaml, the path that
#      default_config_path() always returns for the installed package.
#      This lets users place their config under the /app/config/ volume
#      mount without patching application code.
#   2. exec the container CMD (or any override supplied to docker run /
#      docker compose run) so the process becomes PID 1 and receives signals.
# =============================================================================
set -e

CONFIG_SRC="${SONITRA_CONFIG:-/app/config/config.yaml}"
if [ -f "$CONFIG_SRC" ]; then
    ln -sf "$CONFIG_SRC" /app/config.yaml
fi

exec "$@"
