# =============================================================================
# Sonitra — Production Dockerfile
# Multi-stage build: a shared `builder-base` (apt deps + uv binary + manifests)
# fans out into `builder-cpu` and `builder-gpu`, each running its own `uv sync`
# into its own /app/.venv (`--extra transkun` = PyTorch CPU build on Linux,
# `--extra transkun-gpu` = PyPI CUDA build); `runtime` is a clean CPU-only slim
# image carrying the CPU venv, `runtime-gpu` layers the CUDA venv onto an NVIDIA
# CUDA base image for GPU inference and copies its CPython from
# `python:3.11-slim-bookworm` so that venv runs on the interpreter it was built
# with (jammy's apt python3.11 is a pre-release; see the interpreter comment in
# the stage). Pick the target via
# `docker build --target runtime-gpu` (or the `target:` key in
# docker-compose.yml — see the `sonitra-gpu` service). Each runtime image holds
# exactly one venv — just no longer the *same* venv.
#
# Build context: repo root (set in docker/docker-compose.yml as context: ..)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder-base
# Shared base for both fork builders: build-time apt packages, the uv binary,
# and the dependency manifests. The heavy `uv sync` lives in `builder-cpu` /
# `builder-gpu` below (one venv per fork), so this stage stays fully shared
# and cached. TensorFlow's CUDA libraries come from the GPU runtime stage's
# base image (apt-installed, discovered via the standard ldconfig cache)
# rather than from pip, so no builder stage needs the CUDA toolkit.
# Build tools are kept here and discarded in the runtime stages.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder-base

# apt runs unattended during a build. Without a non-interactive frontend,
# any package whose postinst calls debconf will open a prompt that BuildKit
# gives no stdin to answer, and the build hangs indefinitely rather than
# failing — tzdata, pulled in transitively by r-base-core in the runtime
# stages below, asks for a geographic area and does exactly this.
#
# Declared as ARG, not ENV, so it applies only while building and never
# leaks into the shipped image, where `docker exec` + apt should still get
# the normal interactive frontend.
ARG DEBIAN_FRONTEND=noninteractive

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

# Copy dependency manifests first so that the fork builders' heavy `uv sync`
# layers are cached for all subsequent builds that don't change the lockfile.
# README.md is required because pyproject.toml declares it as the readme field.
COPY pyproject.toml uv.lock README.md ./

# ---------------------------------------------------------------------------
# Stage 2 — builder-cpu
# CPU fork: syncs the `transkun` extra, which resolves torch/torchaudio from
# the PyTorch CPU index on Linux (see [tool.uv.sources] in pyproject.toml).
# Produces the /app/.venv that the `runtime` stage copies.
# ---------------------------------------------------------------------------
FROM builder-base AS builder-cpu

# 1/0 switch for the transkun backend, threaded through from
# docker-compose.yml exactly like INSTALL_R (default 1). With 0 the venv
# builds without transkun and CPU torch at all.
ARG INSTALL_TRANSKUN=1

# Install all third-party dependencies (but not the project package itself)
# so this layer survives source-code-only changes. `--locked` (not `--frozen`)
# fails the build on a stale lockfile instead of silently shipping a forbidden
# pin — the stale pedalboard 0.9.23 reached the images exactly this way.
# The cache mount keeps uv's download cache across builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    TRANSKUN_EXTRA=""; \
    if [ "$INSTALL_TRANSKUN" = "1" ]; then TRANSKUN_EXTRA="--extra transkun"; fi; \
    uv sync --locked --no-dev --no-install-project ${TRANSKUN_EXTRA}

# Copy source and complete the install (editable install of sonitra itself).
COPY src/ src/
# config/source.yaml is the reference config read by default_config_path().
COPY config/ config/
# scripts/run_transcribe_eval.py — batch runner invoked directly, not part of the package.
COPY scripts/ scripts/
RUN --mount=type=cache,target=/root/.cache/uv \
    TRANSKUN_EXTRA=""; \
    if [ "$INSTALL_TRANSKUN" = "1" ]; then TRANSKUN_EXTRA="--extra transkun"; fi; \
    uv sync --locked --no-dev ${TRANSKUN_EXTRA}

# ---------------------------------------------------------------------------
# Stage 3 — builder-gpu
# CUDA fork: syncs the `transkun-gpu` extra, which takes PyPI torch (a CUDA 13
# build needing an R580-series or newer host driver to see the GPU). Produces
# the /app/.venv that the `runtime-gpu` stage copies. Stays on
# python:3.11-slim — torch and transkun ship as prebuilt manylinux wheels
# needing no CUDA toolkit at build time, and the GPU stage's python-symlink
# constraint (see `runtime-gpu`) is untouched by keeping the same interpreter
# path here.
# ---------------------------------------------------------------------------
FROM builder-base AS builder-gpu

# Same 1/0 switch as `builder-cpu` above.
ARG INSTALL_TRANSKUN=1

RUN --mount=type=cache,target=/root/.cache/uv \
    TRANSKUN_EXTRA=""; \
    if [ "$INSTALL_TRANSKUN" = "1" ]; then TRANSKUN_EXTRA="--extra transkun-gpu"; fi; \
    uv sync --locked --no-dev --no-install-project ${TRANSKUN_EXTRA}

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
RUN --mount=type=cache,target=/root/.cache/uv \
    TRANSKUN_EXTRA=""; \
    if [ "$INSTALL_TRANSKUN" = "1" ]; then TRANSKUN_EXTRA="--extra transkun-gpu"; fi; \
    uv sync --locked --no-dev ${TRANSKUN_EXTRA}

# ---------------------------------------------------------------------------
# Stage 4 — runtime (CPU)
# Minimal image: only runtime shared libraries + the CPU venv from
# `builder-cpu`.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Build-only non-interactive apt frontend — see the builder-base stage above.
ARG DEBIAN_FRONTEND=noninteractive

# Fix the container timezone explicitly. This gives tzdata (installed as a
# dependency of r-base-core below) an already-answered configuration to find,
# and pins container-local timestamps to Oslo wall-clock time regardless of
# the host's own timezone. glibc reads TZ directly, so overriding it at run
# time (`-e TZ=...`, or `environment:` in docker-compose.yml) is enough to
# shift a single container without rebuilding.
ENV TZ=Europe/Oslo

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
#
# /etc/localtime and /etc/timezone are seeded before the first apt call so
# tzdata's postinst finds the timezone already chosen (see ENV TZ above).
RUN ln -snf /usr/share/zoneinfo/Europe/Oslo /etc/localtime \
    && echo Europe/Oslo > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libfftw3-double3 \
        fluidsynth \
        libgomp1 \
        libgl1 \
        gosu \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# R + glmmTMB for scripts/run_mixed_effects_analysis.py, which fits the
# benchmark condition-effect model. That model is a beta GLMM with crossed
# random effects, which has no equivalent in the Python statistics stack, so
# the fit happens in R (scripts/mixed_effects_analysis.R) and Python drives it.
#
# Debian's r-cran-* packages are prebuilt and ABI-matched to r-base-core, so
# nothing compiles here. Installing glmmTMB via install.packages() instead
# would pull TMB + RcppEigen from source plus a full toolchain, and is the
# usual source of the "glmmTMB was built with TMB version X" mismatch.
#
# Gated because it adds roughly 400 MB to an image whose main job is audio
# rendering and transcription, not statistics: build with
# `--build-arg INSTALL_R=0` to leave it out. The library() check fails the
# build loudly rather than leaving an R that only breaks at analysis time.
ARG INSTALL_R=1
RUN if [ "$INSTALL_R" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            r-base-core \
            r-cran-glmmtmb \
            r-cran-jsonlite \
        && Rscript -e 'library(glmmTMB); library(jsonlite)' \
        && rm -rf /var/lib/apt/lists/*; \
    fi

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

# Where the transcriber weights are installed. /models is a bind mount at runtime,
# so nothing is baked in here — the entrypoint runs scripts/setup_hft_transformer.py
# on container start. The same host directory is mounted at this path by the
# devcontainer, hence the path outside /app.
ENV SONITRA_MODELS_DIR=/models

# Copy the fully-built project tree from the `builder-cpu` stage, setting ownership
# in a single pass via --chown (avoids a slow chown -R on a large tree).
COPY --from=builder-cpu --chown=sonitra:sonitra /app /app

# Create named volume mount-points owned by the non-root user.
# These directories are intended to be bind-mounted at runtime; they exist
# in the image so the container starts cleanly even without mounts.
# /app/.cache/uv is created here so uv can write its runtime cache without
# requiring a home directory. /models sits outside /app because the devcontainer
# mounts the same host directory at the same path (see SONITRA_MODELS_DIR above).
RUN mkdir -p /app/corpus /app/output /app/.cache/uv /models \
    && chown -R sonitra:sonitra /app/corpus /app/output /app/.cache/uv /models \
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
# Stage 5 — runtime (GPU)
# The CUDA venv from `builder-gpu`, layered onto an NVIDIA CUDA base image
# instead of python:3.11-slim so cuDNN/cuBLAS/cuFFT/etc. are apt-installed
# system-wide and discovered via the standard ldconfig cache. This stage also
# copies its CPython from python:3.11-slim-bookworm so the venv runs on the
# interpreter it was built with — jammy's apt python3.11 is a pre-release
# without sys.get_int_max_str_digits, which torch's dynamo path calls (see the
# interpreter comment below).
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

# Build-only non-interactive apt frontend + fixed timezone — see the builder-base
# and runtime stages above. This stage is where the tzdata prompt actually
# bit: jammy's r-base-core drags tzdata in unconfigured.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

ARG HOST_UID=1000
ARG HOST_GID=1000

# libcudnn8 comes from NVIDIA's own apt repo (the nvidia/cuda base image
# already has it configured). Not version-pinned: the exact Debian revision
# suffix for a cuDNN 8.9.4.25 build wasn't verifiable without a live apt
# query against this repo, and a wrong pin fails the build outright. Check
# `apt list -a libcudnn8` on the build host if TF's cuDNN version check ends
# up caring about the exact point release (the old pip-wheel pin was
# nvidia-cudnn-cu12==8.9.4.25, for reference).
#
# No python3.11 from apt. Jammy ships 3.11.0rc1, a pre-release that predates
# sys.get_int_max_str_digits (added before 3.11.0 final). torch's dynamo
# polyfill calls that attribute unguarded for any sys.version_info >= (3, 11),
# so `import torch._dynamo` dies with AttributeError — and TransKun reaches
# dynamo through torch.utils.checkpoint, which is how the backend failed on
# this image while basic_pitch, TensorFlow and torch.cuda all worked.
#
# The venv copied from `builder-gpu` was built on python:3.11-slim's 3.11.16
# and its bin/python is a symlink to /usr/local/bin/python3.11, so this stage
# must expose a final 3.11 at that exact path. Copying /usr/local from
# python:3.11-slim-bookworm supplies the same 3.11.16 built on Debian 12
# (glibc 2.36), which runs on jammy (2.35); the trixie tag's build does not —
# its libpython3.11.so.1.0 requires GLIBC_2.38.
#
# /etc/localtime and /etc/timezone are seeded before the first apt call so
# tzdata's postinst finds the timezone already chosen (see ENV TZ above).
RUN ln -snf /usr/share/zoneinfo/Europe/Oslo /etc/localtime \
    && echo Europe/Oslo > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libcudnn8 \
        libsndfile1 \
        libfftw3-double3 \
        fluidsynth \
        libgomp1 \
        libgl1 \
        gosu \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# The interpreter the copied venv expects (see above). Placed after the apt
# layer so a future change to the bookworm tag invalidates only this layer
# and the ones below it. The copy also provides /usr/local/bin/python3 and
# `python`; no symlinks are created here.
COPY --from=python:3.11-slim-bookworm /usr/local /usr/local

# R + glmmTMB for scripts/run_mixed_effects_analysis.py — same rationale as the
# CPU runtime stage above, but sourced from jammy's universe repo (already in
# use here for python3.11) rather than Debian's. Ubuntu builds r-base-core,
# r-cran-glmmtmb, and r-cran-tmb together in one archive, so they are
# ABI-matched to each other and nothing compiles.
#
# NOTE: jammy pins an older stack than the Debian-based CPU image — R 4.1.2 /
# glmmTMB 1.1.2.3 / TMB 1.7.22, against R 4.5 / glmmTMB 1.1.10 there. The model
# fits the same on both, but estimates need not agree to the last digit across
# glmmTMB versions, so do not mix results from the two images in one analysis.
# Every run records its exact versions in model_meta.json; check there first if
# two runs disagree.
ARG INSTALL_R=1
RUN if [ "$INSTALL_R" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            r-base-core \
            r-cran-glmmtmb \
            r-cran-jsonlite \
        && Rscript -e 'library(glmmTMB); library(jsonlite)' \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

RUN groupadd --system --gid "$HOST_GID" sonitra \
    && useradd --system --uid "$HOST_UID" --gid "$HOST_GID" \
               --no-create-home --shell /bin/false sonitra

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_CACHE_DIR=/app/.cache/uv

# Weights directory — same reasoning as in the CPU runtime stage above.
ENV SONITRA_MODELS_DIR=/models

# CUDA coexistence pin (TF 2.15 + torch in one process): TensorFlow dlopens the
# apt-installed libcudnn.so.8 by soname, while torch preloads its own bundled
# cuDNN 9 (nvidia-cudnn-cu13 wheels) by absolute path — so the two never share a
# cuDNN, provided no venv path shadows the apt library. This pin lists system
# library dirs only (driver mounts, CUDA toolkit, apt) and deliberately excludes
# every /app/.venv path. Verified by the Stage 19 library-path check.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.2/lib64:/usr/lib/x86_64-linux-gnu
COPY --from=builder-gpu --chown=sonitra:sonitra /app /app

# Fail the build if the copied interpreter and the venv disagree on the 3.11
# patch release — the python:3.11-slim and python:3.11-slim-bookworm tags
# move independently. uv 0.7.8 writes `version_info = 3.11.16` to pyvenv.cfg;
# `tr` strips any trailing whitespace so a format change cannot cause a
# spurious mismatch.
RUN test "$(python3.11 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" \
        = "$(sed -n 's/^version_info = //p' /app/.venv/pyvenv.cfg | tr -d '[:space:]')"

RUN mkdir -p /app/corpus /app/output /app/.cache/uv /models \
    && chown -R sonitra:sonitra /app/corpus /app/output /app/.cache/uv /models \
    && chown sonitra:sonitra /app

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER root

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

CMD ["sonitra", "serve", "--port", "8000"]
