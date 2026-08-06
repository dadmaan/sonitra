# =============================================================================
# Sonitra — Production Dockerfile
# Multi-stage build: `builder` installs Python deps into a single venv that
# both runtime images reuse; `runtime` is a clean CPU-only slim image;
# `runtime-gpu` layers the same venv onto an NVIDIA CUDA base image for GPU
# inference. Pick the target via `docker build --target runtime-gpu` (or the
# `target:` key in docker-compose.yml — see the `sonitra-gpu` service).
#
# Build context: repo root (set in docker/docker-compose.yml as context: ..)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# Installs uv and syncs the full dependency tree into /app/.venv. Produces
# one venv shared by both the CPU and GPU runtime stages: TensorFlow's CUDA
# libraries come from the GPU runtime stage's base image (apt-installed,
# discovered via the standard ldconfig cache) rather than from pip, so this
# stage never needs to know whether it's building for CPU or GPU.
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
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

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
# scripts/run_transcribe_eval.py — batch runner invoked directly, not part of the package.
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime (CPU)
# Minimal image: only runtime shared libraries + the built venv from builder.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# UID/GID for the non-root `sonitra` user, overridable at build time so it
# matches the host user on native Linux hosts (docker-compose.yml passes
# these through as HOST_UID/HOST_GID). On Linux, bind mounts share the host
# filesystem directly, so a mismatched container UID either can't write to
# the mounted ./corpus, ./config, ./output directories, or forces the
# entrypoint to chown them to a UID that doesn't belong to the host user —
# locking the host user out of their own repo files. Matching UIDs at build
# time avoids both. Defaults to 1000, the first non-root UID on most
# single-user Linux installs.
ARG HOST_UID=1000
ARG HOST_GID=1000

# Runtime system libraries required by the core Python packages:
#   libsndfile1       — audio file I/O (pedalboard, soundfile)
#   libfftw3-double3  — FFTW used by DawDreamer
#   fluidsynth        — SoundFont synthesis CLI (optional but low-cost to include)
#   libgomp1          — OpenMP threading used by TensorFlow / NumPy
#   libgl1            — tensorflow/basic-pitch: indirect OpenGL dep on headless Linux
#   gosu              — step-down privilege tool used by the entrypoint
#   tmux              — interactive terminal multiplexer for `docker exec` sessions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libfftw3-double3 \
        fluidsynth \
        libgomp1 \
        libgl1 \
        gosu \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# Bring uv into the runtime image — it is used by the default CMD.
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

# Non-root user for security hardening. UID/GID matches the host user by
# default (see HOST_UID/HOST_GID above) so bind-mounted repo directories
# never need their ownership rewritten at runtime.
RUN groupadd --system --gid "$HOST_GID" sonitra \
    && useradd --system --uid "$HOST_UID" --gid "$HOST_GID" \
               --no-create-home --shell /bin/false sonitra

WORKDIR /app

# Put the venv's console scripts (sonitra, uvicorn, ...) on PATH so `sonitra`
# works directly in interactive `docker exec` sessions, not just via the
# full path or `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

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
RUN mkdir -p /app/corpus /app/output /app/.cache/uv \
    && chown -R sonitra:sonitra /app/corpus /app/output /app/.cache/uv \
    && chown sonitra:sonitra /app

# Entrypoint script lives in docker/ in the repo; build context is repo root.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The entrypoint runs as root so it can fix bind-mount ownership before
# stepping down to the sonitra user via gosu.  The application itself never
# runs as root — see entrypoint.sh.
USER root

# REST API port.
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

# Default: start the FastAPI server.
# Override this CMD for CLI usage, e.g.:
#   docker compose run sonitra sonitra benchmark --corpus /app/corpus --workdir /app/output
CMD ["sonitra", "serve", "--port", "8000"]

# ---------------------------------------------------------------------------
# Stage 3 — runtime (GPU)
# Same venv as the CPU runtime, layered onto an NVIDIA CUDA base image
# instead of python:3.11-slim so cuDNN/cuBLAS/cuFFT/etc. are apt-installed
# system-wide and discovered via the standard ldconfig cache.
#
# This replaces an earlier approach that installed CUDA as pip wheels
# (nvidia-*-cu12, still available for bare-metal installs via pyproject.toml's
# [gpu] extra) directly into the venv with a hand-rolled LD_LIBRARY_PATH.
# That approach hit a reproducible TensorFlow 2.15 bug where GPU registration
# silently fails ("Could not find cuda drivers on your machine") even though
# the driver, NVIDIA Container Toolkit passthrough, and a raw dlopen/cuInit()
# of libcuda.so.1 all check out fine — see tensorflow/tensorflow#62412, which
# reproduces the identical symptom with an exactly-matched driver/CUDA
# version, and several similar reports (#62987, #70947, #63109). The common
# thread is TF 2.15's pip-wheel CUDA layout (one directory per nvidia-*-cu12
# package under site-packages/nvidia/, glued together only via
# LD_LIBRARY_PATH) rather than a specific driver mismatch. Installing CUDA
# system-wide via apt instead avoids that layout entirely.
#
# Uses the `devel` tag (full CUDA toolkit, not just `runtime`) so nvjitlink,
# cupti, and nvrtc are guaranteed present without hand-picking apt package
# names — at the cost of a larger image. Swap to a `-runtime-` tag plus an
# explicit package list later if the smaller image size is worth chasing.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS runtime-gpu

ARG HOST_UID=1000
ARG HOST_GID=1000

# python3.11 + libcudnn8 come from apt (Ubuntu's jammy universe repo and
# NVIDIA's own apt repo, respectively — the nvidia/cuda base image already
# has the latter configured). Not version-pinned: the exact Debian revision
# suffix for a cuDNN 8.9.4.25 build wasn't verifiable without a live apt
# query against this repo, and a wrong pin fails the build outright. Check
# `apt list -a libcudnn8` on the build host if TF's cuDNN version check ends
# up caring about the exact point release (the old pip-wheel pin was
# nvidia-cudnn-cu12==8.9.4.25, for reference).
#
# The venv copied from `builder` was built against python:3.11-slim's
# python3.11, and the venv's own python binary is a symlink to it rather
# than a self-contained copy — so this stage must expose python3.11 at the
# exact same /usr/local/bin/python3(.11) path for that symlink to resolve.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        libcudnn8 \
        libsndfile1 \
        libfftw3-double3 \
        fluidsynth \
        libgomp1 \
        libgl1 \
        gosu \
        tmux \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3.11 \
    && ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

RUN groupadd --system --gid "$HOST_GID" sonitra \
    && useradd --system --uid "$HOST_UID" --gid "$HOST_GID" \
               --no-create-home --shell /bin/false sonitra

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_CACHE_DIR=/app/.cache/uv

COPY --from=builder --chown=sonitra:sonitra /app /app

RUN mkdir -p /app/corpus /app/output /app/.cache/uv \
    && chown -R sonitra:sonitra /app/corpus /app/output /app/.cache/uv \
    && chown sonitra:sonitra /app

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER root

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

CMD ["sonitra", "serve", "--port", "8000"]
