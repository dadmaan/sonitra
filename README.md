# Sonitra

> Independent benchmarking for AI music transcription systems.

Sonitra is a research toolkit for benchmarking automatic music transcription (AMT) systems. It does not train models. The core loop renders symbolic scores to audio, optionally separates stems and applies audio effects, transcribes that audio back to symbolic form with one or more AMT backends, and scores the result against the ground-truth score.

```
MIDI → audio synthesis → transcription → evaluation vs. reference
```

## Dataset

The pipeline is currently aimed at testing with the [MAESTRO V3.0.0](https://magenta.withgoogle.com/datasets/maestro) dataset, a collection of ~200 hours of virtuosic piano performances with aligned MIDI. The corpus will grow to cover additional datasets and instruments in future phases.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager — install once, works everywhere)
- [fluidsynth](https://www.fluidsynth.org/) CLI (optional, for SoundFont-based synthesis — see platform notes below)
- A VST3 plugin (optional, for synthesis and effects)

## Installation

[uv](https://docs.astral.sh/uv/) is the recommended way to install Sonitra. It resolves dependencies from the checked-in `uv.lock` for reproducible installs and is faster than pip. If you prefer pip, see the fallback note at the end of this section.

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

> **Windows note:** DawDreamer rendering modes are not parallel-safe. Sonitra automatically enforces `max_workers=1` when a DawDreamer mode is active.

> **WSL2 note:** If your repo lives on the Windows filesystem (e.g., a devcontainer mount), `uv sync` may fail with an I/O error when installing TensorFlow's deep CUDA headers. Avoid this by cloning on the Linux filesystem (e.g., `~/projects/sonitra`), or redirect the venv: `UV_PROJECT_ENVIRONMENT=~/.venvs/sonitra uv sync --extra dev`.

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

> **pip fallback:** `pip install -e ".[dev]"` still works if you prefer not to use uv.

## Data and plugins

### MIDI input files

Place your MIDI files under `corpus/midi/`:

```
corpus/
  midi/
    piece1.mid
    piece2.mid
```

The pipeline reads from this directory by default (`io.midi_dir` in your config).

### VST3 plugin (optional)

Sonitra supports any VST3 instrument plugin for synthesis. [Vital](https://vital.audio/) is the tested, recommended free option.

1. Download Vital from [vital.audio](https://vital.audio/) and extract the archive.
2. Place the extracted plugin folder under `plugin/`:

```
plugin/
  vital/
    lib/
      vst3/
        Vital.vst3
```

3. Set `dawdreamer.plugin_path` in your config:

```yaml
dawdreamer:
  plugin_path: plugin/vital/lib/vst3/Vital.vst3
```

### Presets (optional)

VST3 preset files (e.g. `.vital` files for Vital) go under `preset/`:

```
preset/
  vital/
    MyPreset.vital
```

Set `dawdreamer.preset_path` in your config:

```yaml
dawdreamer:
  preset_path: preset/vital/MyPreset.vital
```

### SoundFont fallback (optional)

For SoundFont-based synthesis without a VST3 plugin:

```bash
# Linux
sudo apt install fluid-soundfont-gm

# macOS
brew install fluid-synth
```

Then set `dawdreamer.soundfont_path: /usr/share/sounds/sf2/default-GM.sf2` (or the path on your system).

### Core dependencies installed automatically

| Package | Role |
|---|---|
| `dawdreamer` | Faust/VST audio synthesis engine |
| `pedalboard` | Audio effects and instrument plugin API |
| `basic-pitch >= 0.4, < 0.5` | Default AMT backend (Spotify Basic Pitch) |
| `mido` | MIDI file parsing |
| `fastapi` + `uvicorn` | REST API server |
| `pydantic` + `pyyaml` | Config validation and loading |
| `numpy` + `scipy` | Evaluation metric computation |

## Docker

The fastest way to run Sonitra without installing Python or system dependencies locally. Docker files live under `docker/`; all commands are run from the **repository root**.

### Prerequisites

```bash
cp env.example .env        # create the env file (edit values as needed)
mkdir -p corpus config output
```

Place your MIDI files under `./corpus/` and a `config.yaml` under `./config/`. Generate a starter config with:

```bash
docker compose -f docker/docker-compose.yml run --rm sonitra \
    uv run sonitra init --config /app/config/config.yaml
```

### Start the API server

```bash
docker compose -f docker/docker-compose.yml up --build
```

The REST API is available at `http://localhost:8000`. The `/health` endpoint confirms the server is ready.

### Run CLI commands

```bash
# Full benchmark sweep
docker compose -f docker/docker-compose.yml run --rm sonitra \
    uv run sonitra benchmark --corpus /app/corpus --workdir /app/output

# Transcribe only
docker compose -f docker/docker-compose.yml run --rm sonitra \
    uv run sonitra transcribe --audio /app/corpus/audio --output /app/output/transcriptions

# Evaluate transcriptions against reference MIDI
docker compose -f docker/docker-compose.yml run --rm sonitra \
    uv run sonitra evaluate \
    --reference /app/corpus/midi --estimate /app/output/transcriptions/basic_pitch
```

### Volume and environment reference

| Mount | Host path | Purpose |
|---|---|---|
| `/app/corpus` | `./corpus` | MIDI input and rendered audio |
| `/app/config` | `./config` | Pipeline YAML configs |
| `/app/output` | `./output` | Transcriptions, evaluation results, benchmarks |

Set `SONITRA_CONFIG` in `.env` to point at a different config path inside the container (default: `/app/config/config.yaml`).

## Quick start

```bash
# 1. Write a starter config
sonitra init --config config.yaml

# 2. Render your MIDI corpus to audio
sonitra render --corpus corpus/midi --output corpus/audio

# 3. Transcribe the rendered audio
sonitra transcribe --audio corpus/audio --output transcriptions

# 4. Score transcriptions against reference MIDI
sonitra evaluate --reference corpus/midi --estimate transcriptions/basic_pitch

# 5. Run a full benchmark sweep (render + transcribe + evaluate per condition)
sonitra benchmark --corpus corpus/midi --workdir benchmark
```

## CLI reference

```bash
sonitra init --config config.yaml          # write a starter config.yaml
sonitra render --corpus DIR --output DIR   # synthesise audio from MIDI
sonitra transcribe --audio DIR --output DIR [--transcriber NAME]
sonitra evaluate --reference DIR --estimate DIR
sonitra benchmark --corpus DIR --workdir DIR
sonitra serve --port 8000                  # start the FastAPI server
sonitra --version
```

## Configuration

`config/source.yaml` in the repository is the fully-annotated reference config documenting every available parameter. Copy it as a starting point, or generate a minimal starter config with:

```bash
sonitra init --config config.yaml
```

The config file is validated by Pydantic — unknown keys are hard errors.

Key sections and their purpose:

| Section | Controls |
|---|---|
| `pipeline` | Rendering mode, sample rate, bit depth, channels, parallelism |
| `io` | Input/output directories, file format (`wav`, `flac`, `mp3`) |
| `dawdreamer` | Faust script path, VST3 plugin path, SoundFont path |
| `pedalboard` | Instrument plugin path, effects chain |
| `normalisation` | Peak or RMS normalisation, target dB, pre/post effects |
| `quality_gates` | Silence, clipping, and minimum duration checks |
| `transcription` | Transcriber backends, output directory, per-backend thresholds |
| `evaluation` | Metric families to enable, tolerance values |
| `benchmark` | Conditions, parameter sweeps, baseline name |
| `observability` | JSONL manifest, failed-file list, SSE event emission |

### Rendering modes

| Mode | Synth | Effects |
|---|---|---|
| `dawdreamer_only` | DawDreamer (Faust/VST) | None |
| `pedalboard_only` | Pedalboard instrument plugin | Configurable effects chain |
| `dawdreamer_synth_pedalboard_fx` | DawDreamer | Pedalboard effects chain |

Set `pipeline.rendering_mode` in your config to select a mode.

### Transcription backends

| Backend | `type` value | Notes |
|---|---|---|
| Spotify Basic Pitch | `basic_pitch` | Installed by default |
| Pre-exported MIDI | `precomputed` | Point at a directory of MIDI from external tools |
| Any CLI tool | `external_command` | Template: `"tool transcribe {input} -o {output}"` |

### Built-in audio effects

Compressor, Reverb, Limiter, Chorus, Delay, Distortion, Gain, VST3 plugin. Each is a named entry under `pedalboard.effects` with an `enabled` flag. VST3 plugins are loaded at their factory default settings; parameters cannot be set from YAML.

## Evaluation metrics

| Family | What is measured |
|---|---|
| Note-level | Onset F1 (±50 ms), onset+offset F1, onset+offset+velocity F1 |
| Frame-level | Precision/recall/F1 over 10 ms piano-roll frames |
| Expressive | Onset MAE/bias, IOI correlation, key-overlap-ratio correlation, velocity correlation, windowed pitch-class harmony similarity |
| Audio (optional) | Path-normalised DTW distance over chroma features between rendered audio and re-synthesised transcription |

Metrics are implemented in NumPy/SciPy with mir_eval-compatible bipartite matching semantics. Undefined values (e.g. correlations over too few matched notes) are reported as `NaN` and skipped in aggregation.

## Python API

```python
from sonitra.config import load_config
from sonitra.pipeline import run_pipeline
from pathlib import Path

cfg = load_config("config.yaml")
result = run_pipeline(
    sorted(Path("corpus/midi").glob("*.mid")),
    out_dir="corpus/audio",
    config=cfg,
)
print(f"Done: {result.succeeded}, Failed: {result.failed}")
```

## REST API

Start the server with `sonitra serve --port 8000` or programmatically:

```python
import uvicorn
from sonitra.api.app import create_app

uvicorn.run(create_app(), host="0.0.0.0", port=8000)
```

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /ready` | Readiness check |
| `POST /jobs` | Create a render job |
| `GET /jobs` | List all jobs |
| `GET /jobs/{id}` | Get job status |
| `DELETE /jobs/{id}` | Cancel/delete a job |
| `GET /config` | Get current config |
| `PUT /config` | Reload config at runtime |
| `GET /status/{job_id}/stream` | SSE status stream |

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
