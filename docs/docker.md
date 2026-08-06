# Docker

Run Sonitra without installing Python or system dependencies locally. Docker files live under `docker/`; all commands are run from the **repository root**.

## Prerequisites

```bash
cp env.example .env        # create the env file (edit values as needed)
mkdir -p corpus/test/midi config
```

On native Linux hosts (including remote servers), also set `HOST_UID`/`HOST_GID` in `.env` so the container's non-root user matches your host account:

```bash
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
```

> **Why this matters:** `./corpus`, `./config`, and `./output` are bind-mounted straight through to the host filesystem on Linux (unlike Docker Desktop's translation layer on macOS/Windows). Without matching `HOST_UID`/`HOST_GID`, the container's entrypoint has to `chown` those directories to a container-only UID so its non-root user can write to them — which also reassigns their ownership *on the host*, locking your own user out until you `sudo chown` them back. Setting these before the first `--build` avoids that entirely. If you change them later, rebuild with `--build` to pick up the new UID/GID.

Place your MIDI files under `./corpus/{dataset}/midi/` (e.g. `./corpus/test/midi/`). Then generate a starter config. **This step is required before the server can start**:

```bash
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra init --config /app/config/config.yaml
```

> **Why the config is required:** The `./config` volume mount replaces the bundled reference config inside the container. If `./config/config.yaml` (or the path set by `SONITRA_CONFIG`) does not exist when the server starts, it will exit immediately with a file-not-found error. Run `init` once to create it.

> **Profiles are required.** A profile (`cpu` or `gpu`) must always be passed — there is no profile-less default. This keeps the CPU and GPU containers from ever starting at once and colliding on port 8000.

## Start the API server

```bash
docker compose -f docker/docker-compose.yml --profile cpu up --build
```

The REST API is available at `http://localhost:8000`. The `/health` endpoint confirms the server is ready.

## GPU passthrough (optional)

```bash
docker compose -f docker/docker-compose.yml --profile gpu up --build
```

Enables NVIDIA device reservation and builds on an `nvidia/cuda` base image with CUDA/cuDNN installed system-wide via apt (as the `sonitra-gpu` service, tagged `sonitra:gpu`). Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host.

## Run CLI commands

```bash
# Render MIDI to audio (paths resolved from config's corpus_root + dataset)
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra render --config /app/config/config.yaml

# Full benchmark sweep (render + transcribe + evaluate per condition)
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra benchmark --config /app/config/config.yaml

# Transcribe only
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra transcribe --config /app/config/config.yaml

# Evaluate transcriptions against reference MIDI
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra evaluate --config /app/config/config.yaml
```

Swap `--profile cpu run --rm sonitra` for `--profile gpu run --rm sonitra-gpu` to run any of the above with GPU passthrough.

## Volume and environment reference

| Mount | Host path | Purpose |
|---|---|---|
| `/app/corpus` | `./corpus` | Dataset-first corpus root: `{dataset}/midi/`, `{dataset}/audio/`, `{dataset}/transcription/`, `{dataset}/eval_results/` |
| `/app/config` | `./config` | Pipeline YAML configs |
| `/app/output` | `./output` | Transcriptions, evaluation results, benchmarks |

Set `SONITRA_CONFIG` in `.env` to point at a different config path inside the container (default: `/app/config/config.yaml`).

---
[← Back to README](../README.md)
