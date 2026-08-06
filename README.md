# Sonitra

> Independent benchmarking for AI music transcription systems.

Sonitra is a research toolkit for benchmarking automatic music transcription (AMT) systems. It does not train models. The core loop renders symbolic scores to audio, optionally separates stems and applies audio effects, transcribes that audio back to symbolic form with one or more AMT backends, and scores the result against the ground-truth score.

```
MIDI → audio synthesis → transcription → evaluation vs. reference
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- [fluidsynth](https://www.fluidsynth.org/) CLI (optional, for SoundFont-based synthesis — see platform notes below)
- A VST3 plugin (optional, for synthesis and effects)
- Docker (optional, alternative to a local Python install — see [Docker](#docker) below)

## Installation

[uv](https://docs.astral.sh/uv/) is the recommended way to install Sonitra: it resolves dependencies from the checked-in `uv.lock`, so installs are reproducible. A pip fallback is noted at the end of this section.

### Linux

```bash
# One-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install fluidsynth if you want SoundFont synthesis (optional)
sudo apt install fluidsynth

# Install Sonitra with dev dependencies
uv sync --extra dev
```

### macOS

```bash
# One-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install fluidsynth if you want SoundFont synthesis (optional)
brew install fluid-synth

# Install Sonitra with dev dependencies
uv sync --extra dev
```

### Windows

```powershell
# One-time: install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Install fluidsynth if you want SoundFont synthesis (optional)
# Download from https://github.com/nicowillis/fluidsynth-builds
# or via msys2: pacman -S mingw-w64-x86_64-fluidsynth
# or via Chocolatey: choco install fluidsynth

# Install Sonitra with dev dependencies
uv sync --extra dev
```

> **Windows note:** DawDreamer synth backends (`dawdreamer_faust`, `dawdreamer_vst`) are not parallel-safe. Sonitra automatically enforces `max_workers=1` when a DawDreamer backend is active.

> **WSL2 note:** If your repo lives on the Windows filesystem (e.g., a devcontainer mount), `uv sync` may fail with an I/O error when installing packages with deeply nested file trees (TensorFlow, CUDA wheels). Avoid this by cloning on the Linux filesystem (e.g., `~/projects/sonitra`), or redirect the venv: `UV_PROJECT_ENVIRONMENT=~/.venvs/sonitra uv sync --extra dev`.

> **pip fallback:** `pip install -e ".[dev]"` still works if you prefer not to use uv.

### Docker

Run Sonitra without installing Python or system dependencies locally. All commands run from the repository root:

```bash
cp env.example .env
mkdir -p corpus/test/midi config

# Required once, before the server can start:
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra init --config /app/config/config.yaml

docker compose -f docker/docker-compose.yml --profile cpu up --build
```

The REST API is then available at `http://localhost:8000`. A profile (`cpu` or `gpu`) is always required. See **[docs/docker.md](docs/docker.md)** for GPU passthrough, running CLI commands via Compose, and the volume/environment reference.

### Running commands

`uv run` invokes any command inside the managed venv without activating it:

```bash
uv run sonitra --version
uv run pytest
```

Or activate the venv once and use commands directly:

```bash
source .venv/bin/activate   # Linux / macOS / WSL
.venv\Scripts\activate      # Windows (cmd / PowerShell)
sonitra --version
```

### GPU (optional — Linux x86_64 only)

GPU inference for Basic Pitch requires the NVIDIA CUDA runtime libraries alongside TensorFlow.

```bash
uv sync --extra gpu
```

This installs the 11 `nvidia-*` CUDA runtime wheels, pinned to the versions TensorFlow 2.15 declares in its `and-cuda` extras. TensorFlow itself arrives as a core dependency via Basic Pitch, so the `[gpu]` extra adds only the CUDA libraries. (Not `tensorflow[and-cuda]` directly — that meta-extra depends on `tensorrt-libs`, which is only available on NVIDIA's private PyPI index.)

Enable GPU inference by setting `device: GPU:0` in the `transcription.transcribers` section of your config (default: `cpu`). For GPU passthrough inside Docker, see [docs/docker.md](docs/docker.md).

## Datasets

```bash
python scripts/download_datasets.py maestro-v3      # ~1,276 piano MIDI files, ~57 MB
```

Files land under `corpus/{dataset}/midi/`. See **[docs/datasets.md](docs/datasets.md)** for the full list of supported datasets and script options.

## MIDI input files

Sonitra uses a dataset-first corpus layout: place your MIDI files under `corpus/{dataset}/midi/`.

```
corpus/
  maestro-v3/
    midi/
      2004/
        piece.midi
      2008/
        another.mid
```

Both `.mid` and `.midi` extensions are recognised; discovery is recursive at any depth. Set `io.corpus_root` and `io.dataset` in your config and all artifact paths (audio, transcriptions, evaluation results) are derived automatically:

```yaml
io:
  corpus_root: ./corpus
  dataset: maestro-v3   # scopes everything under corpus/maestro-v3/
  output_format: wav
```

For VST3 instrument/preset setup and the SoundFont fallback, see **[docs/plugins.md](docs/plugins.md)**.

## Quick start

```bash
# 0. Download a dataset (stdlib-only, no venv required)
python scripts/download_datasets.py maestro-v3

# 1. Write a starter config
sonitra init --config config.yaml
# Edit config.yaml: set io.corpus_root and io.dataset

# 2. Render + transcribe + evaluate all configs in one command
python scripts/run_transcribe_eval.py --dataset maestro-v3

# Smoke-test with a 4-file subset (reproducible via --seed)
python scripts/run_transcribe_eval.py --dataset maestro-v3 --limit 4 --seed 123

# --- or run each step individually ---
sonitra render     --config config/examples/pedalboard_baseline.yaml --dataset maestro-v3
sonitra transcribe --config config/examples/pedalboard_baseline.yaml
sonitra evaluate   --config config/examples/pedalboard_baseline.yaml
sonitra benchmark  --config config/examples/pedalboard_baseline.yaml
```

Full flag reference (explicit path overrides, `--workers`, `--jobs`, etc.) is in **[docs/cli.md](docs/cli.md)**.

## Configuration

`config/source.yaml` is the fully-annotated reference config documenting every parameter. Generate a minimal starter with `sonitra init --config config.yaml`. The config is a Pydantic model with `extra="forbid"` — unknown keys are hard errors.

See **[docs/configuration.md](docs/configuration.md)** for the full section reference, synth-backend/effects-chain tables, and transcription-backend options.

## Evaluation metrics

Note-level, frame-level, and expressive-performance metrics (mir_eval-compatible matching, implemented in NumPy/SciPy), plus an optional audio-level DTW metric. See **[docs/evaluation.md](docs/evaluation.md)** for the full metric family table.

## Python API and REST API

Sonitra can be driven programmatically (`run_pipeline`) or via a FastAPI server (`sonitra serve --port 8000`). See **[docs/python-api.md](docs/python-api.md)** and **[docs/rest-api.md](docs/rest-api.md)**.

## Testing

```bash
pytest                              # run all tests
pytest -m "not slow"                # skip heavy backend tests (Basic Pitch TF inference)
pytest -m "not skip_if_no_vst"      # skip tests that require a VST plugin path
pytest -m integration               # only end-to-end VST tests
```

Set `VST_PATH` or `VST3_PATH` in your environment to enable VST-dependent tests.

## Author

**Shayan Dadman** — [dadman.shayan@gmail.com](mailto:dadman.shayan@gmail.com)

## License

MIT
