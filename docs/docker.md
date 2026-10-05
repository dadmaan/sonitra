# Docker

You can run Sonitra with Docker, so you do not need to install Python or system tools on your machine. Docker files live under `docker/`. Run all commands from the repository root, the top folder of the project.

## Prerequisites

First make your env file and folders:

```bash
cp env.example .env        # create the env file (edit values as needed)
mkdir -p corpus/test/midi config
```

The env file holds settings Docker reads. The folders hold your MIDI scores and configs. MIDI is a digital score format.

If you use Linux directly, including a remote server, match the file owner IDs next. Add `HOST_UID` and `HOST_GID` to `.env` so files the container makes stay owned by you:

```bash
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
```

> Why this matters: `./corpus`, `./config`, and `./output` are shared directly with your machine on Linux. Docker Desktop on macOS and Windows adds a translation layer, but Linux does not. If the IDs do not match, the container must change those folders to a container-only user so it can write to them. That also changes the owner on your machine, and you lose access until you run `sudo chown` to take them back. Set these IDs before your first `--build` to avoid this. If you change them later, rebuild with `--build` so the new IDs take effect.

Put your MIDI files under `./corpus/{dataset}/midi/`, for example `./corpus/test/midi/`. Then make a starter config. You must do this before the server can start:

```bash
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra init --config /app/config/config.yaml
```

> Why you need the config: The `./config` folder on your machine replaces the sample config inside the container. If `./config/config.yaml`, or the path set by `SONITRA_CONFIG`, is missing when the server starts, the server stops at once with a file-not-found error. Run `init` once to make it.

> You must always pass a profile. Use `cpu` or `gpu`. There is no default. This stops the CPU and GPU servers from starting together and fighting over port 8000.

## Start the API server

This starts the web server for the REST API:

```bash
docker compose -f docker/docker-compose.yml --profile cpu up --build
```

The REST API is available at `http://localhost:8000`. Open the `/health` address to check that the server is ready.

## GPU passthrough (optional)

Use this if you have an NVIDIA graphics card and want faster transcription:

```bash
docker compose -f docker/docker-compose.yml --profile gpu up --build
```

It turns on NVIDIA device access. It builds on an `nvidia/cuda` base image with CUDA and cuDNN installed system-wide with apt. It runs as the `sonitra-gpu` service, tagged `sonitra:gpu`. CUDA and cuDNN are NVIDIA tools for GPU computing. You need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on your machine.

## Run CLI commands

Run one of these to use a single Sonitra step without starting the server:

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

To use the graphics card instead, replace `--profile cpu run --rm sonitra` with `--profile gpu run --rm sonitra-gpu` in any command above.

## Volume and environment reference

The table below shows which folders on your machine link to which folders in the container. A bind mount means a shared folder between the two.

| Mount | Host path | Purpose |
|---|---|---|
| `/app/corpus` | `./corpus` | Dataset-first corpus root: `{dataset}/midi/`, `{dataset}/audio/`, `{dataset}/transcription/`, `{dataset}/eval_results/` |
| `/app/config` | `./config` | Pipeline YAML configs |
| `/app/output` | `./output` | Transcriptions, evaluation results, benchmarks |
| `/models` | `./models` | Converted transcriber weights for `hft_transformer`. See [hFT-Transformer weights](#hft-transformer-weights) below |

Set `SONITRA_CONFIG` in `.env` to use a different config path inside the container. The default is `/app/config/config.yaml`.

## R (statistical analysis)

Both images include R with the `glmmTMB` and `jsonlite` packages. R is the stats tool Sonitra uses for the mixed-effects model. So `scripts/run_mixed_effects_analysis.py` works with no extra setup (see [Statistical analysis](statistical-analysis.md)). R adds about 400 MB. If you never run that analysis, you can skip it. Set `INSTALL_R=0` in `.env` to leave it out:

```bash
echo "INSTALL_R=0" >> .env
docker compose -f docker/docker-compose.yml --profile cpu build   # or --profile gpu
```

The two images carry different R versions because they build on different base systems. The CPU image (Debian) has R 4.5 with glmmTMB 1.1.10. The GPU image (Ubuntu 22.04) has R 4.1.2 with glmmTMB 1.1.2.3. Both fit the same model, but small digits may differ between versions. So do not mix results from the two images in one study. Each run saves the exact versions it used in `model_meta.json`.

## TransKun (piano transcription)

Both images include TransKun with its torch build. The build comes from the `transkun` extra in `pyproject.toml`. TransKun is the piano transcription backend you select with `type: transkun`. Torch is the machine-learning library it runs on. So TransKun works with no extra setup. If you never transcribe with TransKun, you can skip it. Set `INSTALL_TRANSKUN=0` in `.env` to leave it out:

```bash
echo "INSTALL_TRANSKUN=0" >> .env
docker compose -f docker/docker-compose.yml --profile cpu build   # or --profile gpu
```

The two images carry different torch builds. The CPU image carries CPU torch. The GPU image carries CUDA torch from PyPI. CUDA is NVIDIA's system for running code on the graphics card. The CUDA build needs an R580-series or newer host driver. Verified on driver 581.29 with an RTX 4090: `torch.cuda.is_available()` is True, and TensorFlow 2.15 and torch share one process in both import orders. For a bare-metal install, use `pip install 'sonitra[transkun]'`. The `transkun-gpu` extra is for the GPU image and the GPU devcontainer only (see [Devcontainer](devcontainer.md)). For a bare-metal install on a graphics card, still use the `[gpu]` extra path.

TransKun inference now runs on both images. The GPU image copies its Python from `python:3.11-slim-bookworm`, the same 3.11.16 the venv was built with. Ubuntu 22.04's apt Python is the pre-release 3.11.0rc1, which lacks `sys.get_int_max_str_digits`, a function torch's dynamo path calls. TensorFlow and torch still coexist and both see the GPU.

Measured sizes with defaults (`INSTALL_R=1`, `INSTALL_TRANSKUN=1`): `sonitra:latest` is 3.68 GB and `sonitra:gpu` is 15.6 GB. With both `INSTALL_R=0` and `INSTALL_TRANSKUN=0`: 2.56 GB and 10.5 GB. The slim figure drops R and TransKun together, so do not read it as TransKun alone. The hFT-Transformer weights are not in either figure: they live in the `./models` bind mount, not in the image.

## hFT-Transformer weights

Both images set `SONITRA_MODELS_DIR=/models`, and the host's `./models` folder is bind-mounted there. The mount sits outside `/app` on purpose: the devcontainer mounts the very same host directory at the very same path, so one download is shared by development and production instead of being repeated per image. Weights are never baked into an image.

The entrypoint installs them on container start by running `scripts/setup_hft_transformer.py`, which downloads the pinned release asset, verifies its size and sha256 against the registry, and converts it once into a `model.pt`. Torch must be present for that conversion, so an image built with `INSTALL_TRANSKUN=0` already skips it: the script exits `3` and the entrypoint stays quiet rather than reaching for the network.

The step is fail-soft, because the other backends work without these weights. A failure prints one warning naming the command to run by hand, and the API starts anyway. A later request for `type: hft_transformer` then errors naming that same setup command, so the cause stays visible instead of becoming a silent missing result. The script is idempotent, so a second container sharing the same `./models` host folder reports the existing install and moves on.

To skip the start-up download entirely, which is what an air-gapped host needs:

```bash
echo "SONITRA_SETUP_HFT=0" >> .env
```

Then install the weights on a machine that can reach the release, writing into that same host folder, before you start the container:

```bash
python scripts/setup_hft_transformer.py --models-dir ./models
```

---
[← Back to README](../README.md)
