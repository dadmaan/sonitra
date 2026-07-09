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
# Optional: pass GPU_EXTRAS=gpu to install tensorflow[and-cuda] for GPU inference.
# NOTE: GPU extras are installed via uv pip install, NOT uv sync --frozen,
# because the lockfile is generated without --extra gpu (GPU deps are platform-
# constrained to linux+x86_64 and cannot be resolved on all dev machines).
# Default is empty (CPU-only build).
ARG GPU_EXTRAS=""

FROM python:3.11-slim AS builder
# Re-declare so this stage inherits the global ARG value.
ARG GPU_EXTRAS

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
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first so that the heavy `uv sync` layer is
# cached for all subsequent builds that don't change the lockfile.
# README.md is required because pyproject.toml declares it as the readme field.
COPY pyproject.toml uv.lock README.md ./

# Install all third-party dependencies (but not the project package itself)
# so this layer survives source-code-only changes.
RUN uv sync --frozen --no-dev --no-install-project

# ────────────────────────────────────────────────────────────────────────────
# GPU extras (optional, CPU-only by default)
# ────────────────────────────────────────────────────────────────────────────
# Installs the nvidia-* CUDA runtime wheels that TensorFlow 2.15 needs for
# GPU inference. The tensorflow package itself is already in the venv from
# the uv sync above; this step adds only the CUDA libraries alongside it.
#
# These are the same 11 wheels listed in pyproject.toml's [gpu] extra.
# We re-list them here rather than using "uv sync --extra gpu" with
# --frozen because the nvidia-* packages are platform-constrained to
# linux+x86_64 and the lockfile is generated on all platforms.
#
# Pinned to the exact versions that tensorflow==2.15.0 declares as its
# and-cuda extras.
# ────────────────────────────────────────────────────────────────────────────
RUN if [ -n "$GPU_EXTRAS" ]; then \
        uv pip install \
            "nvidia-cublas-cu12==12.2.5.6" \
            "nvidia-cuda-cupti-cu12==12.2.142" \
            "nvidia-cuda-nvcc-cu12==12.2.140" \
            "nvidia-cuda-nvrtc-cu12==12.2.140" \
            "nvidia-cuda-runtime-cu12==12.2.140" \
            "nvidia-cudnn-cu12==8.9.4.25" \
            "nvidia-cufft-cu12==11.0.8.103" \
            "nvidia-curand-cu12==10.3.3.141" \
            "nvidia-cusolver-cu12==11.5.2.141" \
            "nvidia-cusparse-cu12==12.1.2.141" \
            "nvidia-nccl-cu12==2.16.5" \
            "nvidia-nvjitlink-cu12==12.2.140"; \
    fi

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
#   libgl1            — tensorflow/basic-pitch: indirect OpenGL dep on headless Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libfftw3-double3 \
        fluidsynth \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Bring uv into the runtime image — it is used by the default CMD.
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

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

# TF 2.15 [and-cuda] ships CUDA libs as pip wheels under
# site-packages/nvidia/*/lib (here inside the uv venv), which are not on the
# linker path by default (see tensorflow/tensorflow#65842). Expose them so
# TensorFlow's dlopen finds them. Harmless on CPU-only builds (dirs absent).
ENV NV=/app/.venv/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cuda_cupti/lib:$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib

# Create named volume mount-points owned by the non-root user.
# These directories are intended to be bind-mounted at runtime; they exist
# in the image so the container starts cleanly even without mounts.
# /app/.cache/uv is created here so uv can write its runtime cache without
# requiring a home directory.
RUN mkdir -p /app/corpus /app/output /app/.cache/uv \
    && chown -R sonitra:sonitra /app/corpus /app/output /app/.cache/uv \
    && chown sonitra:sonitra /app

# Entrypoint script lives in docker/ in the repo; build context is repo root.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER sonitra

# REST API port.
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

# Default: start the FastAPI server.
# Override this CMD for CLI usage, e.g.:
#   docker compose run sonitra sonitra benchmark --corpus /app/corpus --workdir /app/output
CMD ["/app/.venv/bin/sonitra", "serve", "--port", "8000"]
