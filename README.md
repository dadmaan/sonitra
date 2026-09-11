# Sonitra

> Independent benchmarking for AI music transcription systems.

Sonitra is a research toolkit for testing automatic music transcription (AMT) systems. AMT turns audio into written notes. Sonitra does not train models. It follows a simple loop. It turns score files into audio. It can split sounds and add effects. It transcribes that audio back into notes with one or more AMT tools. Then it scores the result against the true score. MIDI is the digital score format Sonitra starts from.

```
MIDI → audio synthesis → transcription → evaluation vs. reference
```

## Requirements

You need these before you start:

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- [fluidsynth](https://www.fluidsynth.org/) CLI (optional, a free tool that turns MIDI into audio using a SoundFont file, see platform notes below)
- A VST3 plugin (optional, a virtual instrument or effect for sound creation and audio effects)
- Docker (optional, a way to run Sonitra without a local Python install, see [Docker](#docker) below)

## Installation

[uv](https://docs.astral.sh/uv/) is the best way to install Sonitra. It reads the checked-in `uv.lock`, so you get the same versions each time. A pip fallback is at the end of this section.

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

> **Windows note:** DawDreamer synth backends (`dawdreamer_faust`, `dawdreamer_vst`) cannot run in parallel. DawDreamer is the tool Sonitra uses to play sounds. Faust is its built-in simple tone maker. VST means a virtual instrument plugin. Sonitra sets `max_workers=1` for you when a DawDreamer backend is active. This setting limits work to one task at a time.

> **WSL2 note:** WSL2 lets you run Linux inside Windows. If your repo lives on the Windows side (for example, a devcontainer mount), `uv sync` may fail with an I/O error. It happens when installing packages with very deep file trees (TensorFlow, CUDA wheels). Avoid this by cloning on the Linux side (for example `~/projects/sonitra`). Or point the env elsewhere: `UV_PROJECT_ENVIRONMENT=~/.venvs/sonitra uv sync --extra dev`.

> **pip fallback:** `pip install -e ".[dev]"` still works if you prefer not to use uv.

### Docker

You can run Sonitra without installing Python or system tools on your machine. Run all commands from the repo root, the top folder:

```bash
cp env.example .env
mkdir -p corpus/test/midi config

# Required once, before the server can start:
docker compose -f docker/docker-compose.yml --profile cpu run --rm sonitra \
    uv run --no-sync sonitra init --config /app/config/config.yaml

docker compose -f docker/docker-compose.yml --profile cpu up --build
```

The REST API is then at `http://localhost:8000`. A REST API lets your programs talk to Sonitra over the web. You must always pass a profile, `cpu` or `gpu`. See [docs/docker.md](docs/docker.md) for GPU use, CLI commands through Compose, and the volume and setting reference.

### Running commands

`uv run` runs any command inside the managed env with no need to turn it on first:

```bash
uv run sonitra --version
uv run pytest
```

Or turn on the env once and use commands on their own:

```bash
source .venv/bin/activate   # Linux / macOS / WSL
.venv\Scripts\activate      # Windows (cmd / PowerShell)
sonitra --version
```

### GPU (optional, Linux x86_64 only)

Use this for faster Basic Pitch transcription. Basic Pitch is Spotify's free tool that turns audio into notes. It needs NVIDIA CUDA support files plus TensorFlow.

```bash
uv sync --extra gpu
```

This installs the 11 `nvidia-*` CUDA support wheels, pinned to the versions TensorFlow 2.15 asks for in its `and-cuda` extras. TensorFlow itself comes as a core need through Basic Pitch, so the `[gpu]` extra adds only the CUDA files. Sonitra does not use `tensorflow[and-cuda]` on its own, because that group needs `tensorrt-libs`, which lives only on NVIDIA's private package index.

To use the graphics card, set `device: GPU:0` in the `transcription.transcribers` part of your config. The default is `cpu`. For GPU use inside Docker, see [docs/docker.md](docs/docker.md).

## Datasets

Use the download script to fetch test MIDI files. MIDI means digital scores:

```bash
python scripts/download_datasets.py maestro-v3-midi  # ~1,276 piano MIDI files, ~57 MB
python scripts/download_datasets.py                  # interactive picker (rich table)
python scripts/download_datasets.py --all --jobs 4   # download everything, 4 at a time
```

Files land under `corpus/{dataset}/midi/`. See [docs/datasets.md](docs/datasets.md) for the full list of sets and script options.

## MIDI input files

Put your MIDI files under `corpus/{dataset}/midi/`. This is the dataset-first layout:

```
corpus/
  maestro-v3/
    midi/
      2004/
        piece.midi
      2008/
        another.mid
```

Sonitra finds both `.mid` and `.midi` endings. It searches all subfolders at any depth. Set `io.corpus_root` and `io.dataset` in your config. Sonitra then works out all output paths for you, such as audio, transcriptions, and scores:

```yaml
io:
  corpus_root: ./corpus
  dataset: maestro-v3   # scopes everything under corpus/maestro-v3/
  output_format: wav
```

For VST3 instrument and preset setup and the SoundFont fallback, see [docs/plugins.md](docs/plugins.md). VST3 is a common plugin format. A SoundFont is a file of sampled sounds.

Bringing your own MIDI, or MIDI plus matching recordings? See [docs/custom-datasets.md](docs/custom-datasets.md) for the folder layout and naming rules.

## Quick start

Run these steps to do your first full test. Render means turn scores into audio. Transcribe means turn audio back into notes. Evaluate means score the notes:

```bash
# 0. Download a dataset (stdlib-only, no venv required)
python scripts/download_datasets.py maestro-v3-midi

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

See [docs/cli.md](docs/cli.md) for the full flag list, such as explicit path flags, `--workers`, and `--jobs`.

## Configuration

`config/source.yaml` is the full sample config. It explains every setting. Make a small starter with `sonitra init --config config.yaml`. Sonitra checks the config with Pydantic, a Python validation tool. It uses `extra="forbid"`, which means unknown keys stop the run with an error.

See [docs/configuration.md](docs/configuration.md) for the full section guide, synth and effects tables, and transcription options.

## Evaluation metrics

Sonitra scores note hits, frame hits, and expressive playing. Note means a single musical note. Frame means a short 10 ms slice of sound. It uses mir_eval-style matching, a standard music-scoring method built with NumPy and SciPy. It also has an optional audio check with DTW. DTW means dynamic time warping, a way to line up two audio clips in time and measure the gap. See [docs/evaluation.md](docs/evaluation.md) for the full table.

## Statistical analysis

`python scripts/run_mixed_effects_analysis.py --work-dir DIR` fits a mixed-effects regression to a benchmark run. This form of statistics pulls apart each test setup's effect from how hard each piece was. You need R with `glmmTMB`. R is a stats language. See [docs/statistical-analysis.md](docs/statistical-analysis.md).

## Python API and REST API

You can drive Sonitra from Python with `run_pipeline`. Or you can run a web server with `sonitra serve --port 8000`. The server uses FastAPI, a Python web tool. See [docs/python-api.md](docs/python-api.md) and [docs/rest-api.md](docs/rest-api.md).

## Testing

Run the test suite with:

```bash
pytest                              # run all tests
pytest -m "not slow"                # skip heavy backend tests (Basic Pitch TF inference)
pytest -m "not skip_if_no_vst"      # skip tests that require a VST plugin path
pytest -m integration               # only end-to-end VST tests
```

Set `VST_PATH` or `VST3_PATH` in your env to turn on tests that need a VST plugin path. VST is a virtual instrument format.

## Author

**Shayan Dadman**: [dadman.shayan@gmail.com](mailto:dadman.shayan@gmail.com)

## License

Sonitra is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE) (`AGPL-3.0-or-later`).

Third-party tools ship under their own terms, including GPLv3 parts (`pedalboard`, `dawdreamer`). GPLv3 is a free-software licence. Sets fetched by `scripts/download_datasets.py` carry their own licences. For example, MAESTRO is CC BY-NC-SA 4.0, which means free use for non-commercial sharing with credit. These sets are not covered by Sonitra's licence.
