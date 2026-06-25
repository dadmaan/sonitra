# =============================================================================
# Sonitra — Production Dockerfile
# Multi-stage build: builder installs Python deps; runtime is a clean slim
# image with only the necessary system libraries and the built venv.
#
# Build context: repo root (set in docker/docker-compose.yml as context: ..)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# Installs uv and syncs the full dependency tree into /app/.venv.
# Build tools are kept here and discarded in the runtime stage.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System packages needed only at build time (compilers, headers).
# Most project dependencies ship as manylinux wheels so this layer is a
# lightweight safety net rather than a heavy compilation environment.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libsndfile1-dev \
        libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image — the single binary is all we need.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first so that the heavy `uv sync` layer is
# cached for all subsequent builds that don't change the lockfile.
# README.md is required because pyproject.toml declares it as the readme field.
COPY pyproject.toml uv.lock README.md ./

# Install all third-party dependencies (but not the project package itself)
# so this layer survives source-code-only changes.
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and complete the install (editable install of sonitra itself).
COPY src/ src/
# config/source.yaml is the reference config read by default_config_path().
COPY config/ config/
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# Minimal image: only runtime shared libraries + the built venv from builder.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system libraries required by the core Python packages:
#   libsndfile1       — audio file I/O (pedalboard, soundfile)
#   libfftw3-double3  — FFTW used by DawDreamer
#   fluidsynth        — SoundFont synthesis CLI (optional but low-cost to include)
#   libgomp1          — OpenMP threading used by TensorFlow / NumPy
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libfftw3-double3 \
        fluidsynth \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Bring uv into the runtime image — it is used by the default CMD.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root user for security hardening.
RUN groupadd --system --gid 1001 sonitra \
    && useradd --system --uid 1001 --gid 1001 \
               --no-create-home --shell /bin/false sonitra

WORKDIR /app

# Redirect uv's cache away from the (non-existent) home directory so it
# writes to /app/.cache/uv, which is owned by the sonitra user.
ENV UV_CACHE_DIR=/app/.cache/uv

# Copy the fully-built project tree from the builder stage, setting ownership
# in a single pass via --chown (avoids a slow chown -R on a large tree).
COPY --from=builder --chown=sonitra:sonitra /app /app

# Create named volume mount-points owned by the non-root user.
# These directories are intended to be bind-mounted at runtime; they exist
# in the image so the container starts cleanly even without mounts.
# /app/.cache/uv is created here so uv can write its runtime cache without
# requiring a home directory.
RUN mkdir -p /app/corpus /app/output /app/config /app/.cache/uv \
    && chown -R sonitra:sonitra /app/corpus /app/output /app/config /app/.cache/uv

# Entrypoint script lives in docker/ in the repo; build context is repo root.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER sonitra

# REST API port.
EXPOSE 8000

# Lightweight health check using the stdlib so no extra tool is required.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# Default: start the FastAPI server.
# Override this CMD for CLI usage, e.g.:
#   docker compose run sonitra uv run sonitra benchmark --corpus /app/corpus --workdir /app/output
CMD ["uv", "run", "sonitra", "serve", "--port", "8000"]
