# Sonitra

> Independent benchmarking for AI music transcription systems.

Sonitra is a research toolkit for evaluating automatic music transcription (AMT) systems across open-source models and commercial APIs. It provides a reproducible pipeline for converting symbolic scores into audio, transcribing the resulting audio back into symbolic form, and comparing the output against ground truth under controlled experimental conditions.

## Why Sonitra

Automatic music transcription tools are improving quickly, but independent, systematic benchmarks are still rare. Sonitra is designed to assess how well existing transcription systems perform without building new AMT models. The focus is on transparent evaluation, reproducibility, and comparative analysis across tools, datasets, and audio conditions.

## Core pipeline

```text
MIDI / MusicXML
    ↓
Audio synthesis
    ↓
Optional stem separation
    ↓
Transcription (audio → symbolic)
    ↓
Score comparison and analysis
```

The default workflow is:

1. Start from a symbolic score corpus such as MIDI or MusicXML.
2. Render audio from the score using synthesis tools.
3. Apply controlled transformations such as reverb, noise, or style variations.
4. Run transcription using one or more AMT systems.
5. Compare transcribed output against the original score using evaluation metrics.

## Research questions

Sonitra is intended to support questions such as:

- How do open-source and commercial AMT tools compare on the same inputs?
- Which audio conditions degrade transcription quality most strongly?
- How much does stem separation improve transcription accuracy?
- Which tools are robust to polyphony, effects, and different musical styles?
- How can AMT benchmarking be made reproducible and extensible?

## Initial tool targets

Potential tools and services include:

- Basic Pitch
- klang.io
- Moises.ai
- Songsterr (benchmark-only, limited manual use)
- Demucs or similar stem separation tools
- DawDreamer and Pedalboard for synthesis/effects processing

## Evaluation focus

Sonitra is centered on comparative evaluation rather than model training. Example independent variables include:

- reverb amount
- audio quality / noise
- instrument type
- polyphony level
- tempo
- genre or historical style proxy
- presence or absence of stem separation

## Repository structure

```
sonitra/
├── src/
│   └── sonitra/
│       ├── __init__.py           # Package root
│       ├── __main__.py           # python -m sonitra
│       ├── cli.py                # Typer CLI (render, serve, init)
│       ├── py.typed              # PEP 561 marker
│       ├── config.py             # Pydantic config schema + YAML loader
│       ├── engine.py             # DawDreamer processing engine
│       ├── renderer.py           # Faust/VST note rendering
│       ├── midi_reader.py        # MIDI file parsing
│       ├── normaliser.py         # Peak/RMS normalisation
│       ├── quality_gate.py       # Silence/clip/duration guards
│       ├── manifest.py           # renders.jsonl writer
│       ├── storage.py            # WAV/FLAC/MP3 output
│       ├── pipeline.py           # Orchestration (config + legacy paths)
│       ├── midi_writer.py        # Note dicts -> MIDI file (transcription output)
│       ├── effects/
│       │   ├── __init__.py
│       │   ├── base.py           # EffectsChain protocol
│       │   ├── builtin_effects.py# Pydantic effect config models
│       │   └── chain_builder.py  # Config -> pedalboard.Pedalboard factory
│       ├── synth/
│       │   ├── __init__.py
│       │   ├── protocol.py       # SynthesiserProtocol + make_synth factory
│       │   ├── dawdreamer_synth.py  # DawDreamer backend wrapper
│       │   └── pedalboard_synth.py  # Pedalboard instrument renderer
│       ├── separation/
│       │   ├── __init__.py
│       │   ├── protocol.py       # StemSeparatorProtocol + make_separator factory
│       │   ├── passthrough.py    # No-op separator
│       │   └── demucs_separator.py  # Demucs backend (optional dep)
│       ├── transcribe/
│       │   ├── __init__.py
│       │   ├── base.py           # TranscriptionResult + errors
│       │   ├── configs.py        # Pydantic transcriber config models
│       │   ├── protocol.py       # TranscriberProtocol + registry/factory
│       │   ├── basic_pitch.py    # Spotify Basic Pitch backend (optional dep)
│       │   ├── external_command.py  # Any CLI tool via {input}/{output} template
│       │   └── precomputed.py    # Pre-exported MIDI (commercial tools)
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── types.py          # NoteEvent + conversions
│       │   ├── protocol.py       # SymbolicMetric/AudioMetric + registries
│       │   ├── builtin_metrics.py# Config -> metric factories
│       │   ├── note_metrics.py   # Note-level P/R/F1 (mir_eval-compatible)
│       │   ├── frame_metrics.py  # Frame-level P/R/F1
│       │   ├── expressive_metrics.py # IOI/KOR/velocity/harmony (mpteval-inspired)
│       │   └── dtw_metric.py     # DTW audio similarity over chroma
│       ├── benchmark/
│       │   ├── __init__.py
│       │   ├── conditions.py     # Conditions, sweeps, config overrides
│       │   ├── runner.py         # render -> separate -> transcribe -> evaluate
│       │   └── results.py        # JSONL records, summaries, degradation tables
│       └── api/
│           ├── __init__.py
│           ├── app.py            # FastAPI application
│           ├── models.py         # API request/response models
│           ├── job_store.py      # Job state management
│           ├── worker.py         # Background rendering worker
│           └── routers/
│               ├── __init__.py
│               ├── config.py     # GET/PUT /config
│               ├── health.py     # Health check endpoints
│               ├── jobs.py       # Job CRUD
│               └── status.py     # SSE status stream
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── fixtures/
│   │   ├── config_*.yaml         # Test config variants
│   │   └── test_*.mid            # MIDI test files
│   ├── test_config.py
│   ├── test_chain_builder.py
│   ├── test_pedalboard_synth.py
│   ├── test_synth_protocol.py
│   ├── test_normaliser.py
│   ├── test_quality_gate.py
│   ├── test_manifest.py
│   ├── test_pipeline*.py
│   └── api/
│       ├── test_config.py
│       ├── test_jobs.py
│       ├── test_job_store.py
│       ├── test_worker.py
│       ├── test_health.py
│       └── test_openapi.py
├── scripts/                      # Dev/CI helper scripts
├── CHANGELOG/                    # One file per release
│   └── 0.1.0.md
├── config.yaml                   # Default configuration
├── pyproject.toml
├── README.md
├── SECURITY.md
├── CONTRIBUTING.md
├── LICENSE
└── zensical.toml                 # Documentation site config
```

## Features

- **Dual synthesis backends** — DawDreamer (Faust/VST) and Pedalboard (instrument plugins)
- **Built-in effects** — Compressor, Reverb, Limiter, Chorus, Delay, Distortion, Gain, VST3 plugins
- **Config-driven pipeline** — YAML configuration validated via Pydantic, runtime reloadable
- **Three rendering modes** — DawDreamer-only, Pedalboard-only, or hybrid (DawDreamer synth + Pedalboard FX)
- **Output formats** — WAV, FLAC, MP3 (via Pedalboard AudioFile)
- **Normalisation** — Peak or RMS, configurable pre/post effects
- **Quality gates** — Silence, clipping, and minimum duration checks
- **Manifest tracking** — Per-file JSONL log with effects chain hashing
- **FastAPI server** — Job queue, runtime config reload, SSE status stream
- **Worker constraint** — DawDreamer modes automatically serialised to single worker
- **Transcription layer** — Pluggable AMT backends: Basic Pitch, any CLI tool, or pre-exported MIDI from commercial services
- **Stem separation** — Optional Demucs pass between rendering and transcription
- **Evaluation suite** — Note-level F1 (onset / onset+offset / +velocity, mir_eval-compatible tolerances), frame-level F1, musically informed expressive metrics (timing, articulation, dynamics, harmony), DTW audio similarity
- **Benchmark orchestration** — Parameter sweeps over any config field, per-condition metrics, degradation-vs-baseline tables

## Configuration

Copy the default config and customise:

```yaml
# config.yaml — default Sonitra config, works out of the box.
# Basic Pitch is installed by default; install sonitra[demucs] to use Demucs.
pipeline:
  rendering_mode: dawdreamer_synth_pedalboard_fx  # or dawdreamer_only, pedalboard_only
  sample_rate: 44100
  bit_depth: 24
  channels: 2
  duration_padding_sec: 2.0
  overwrite: false
  resume: true
  max_workers: 1
  log_level: INFO

io:
  midi_dir: ./corpus/midi
  output_dir: ./corpus/audio
  output_format: wav   # wav, flac, or mp3
  mp3_bitrate_kbps: 192
  file_naming: "{stem}"

pedalboard:
  enabled: true
  instrument:
    plugin_path: null
    preset_path: null
    reload_plugin_per_file: false
    silence_flush_sec: 0.5
  effects:
    - type: Compressor
      threshold_db: -18.0
      ratio: 4.0
      attack_ms: 5.0
      release_ms: 100.0
      enabled: true
    - type: Reverb
      room_size: 0.4
      damping: 0.5
      wet_level: 0.15
      dry_level: 0.85
      width: 1.0
      freeze_mode: false
      enabled: true
    # Optional: add a Limiter here for pre-normalisation dynamics control. It is
    # NOT required to prevent clipping — post-effects peak normalisation below
    # already guarantees the output peak is < clip_threshold. If you add it,
    # note that it shifts later effect indices and would move the reverb_wet
    # sweep target.
    # - type: Limiter
    #   threshold_db: -1.0
    #   release_ms: 100.0
    #   enabled: true

# Post-effects peak normalisation is the clipping fix. In peak mode the signal
# is scaled so its peak == 10^(target_db/20) ≈ 0.891, then clipped to [-1, 1],
# which is guaranteed below clip_threshold: 1.0. pre_effects: false runs it
# AFTER the effects chain.
normalisation:
  enabled: true
  mode: peak
  target_db: -1.0
  pre_effects: false

quality_gates:
  silence_threshold_rms: 0.001
  min_duration_sec: 0.1
  max_duration_deviation_sec: 1.0
  clip_threshold: 1.0

# Observability is disabled in the checked-in default to avoid writing files in
# the working directory. Enable these explicitly for batch workflows.
observability:
  write_manifest: false
  manifest_path: ./renders.jsonl
  write_failed_list: false
  emit_sse_events: false

separation:
  enabled: false
  backend: passthrough     # set to "demucs" only after `pip install sonitra[demucs]`
  model: htdemucs
  device: cpu
  stem: null
  output_dir: stems

transcription:
  output_dir: transcriptions
  transcribers:
    - type: basic_pitch    # installed by default
      enabled: true
      onset_threshold: 0.5
      frame_threshold: 0.3
      minimum_note_length_ms: 127.7
      minimum_frequency_hz: null
      maximum_frequency_hz: null
    # - type: precomputed    # MIDI exported from a commercial tool
    #   name: klangio
    #   midi_dir: ./external/klangio
    # - type: external_command  # any CLI transcription tool
    #   name: my-tool
    #   command: "amt-tool transcribe {input} -o {output}"

evaluation:
  note_metrics:
    enabled: true
    onset_tolerance_sec: 0.05   # mir_eval-standard tolerances
    offset_ratio: 0.2
    offset_min_tolerance_sec: 0.05
    velocity_tolerance: 0.1
  frame_metrics:
    enabled: true
    hop_sec: 0.01
  expressive_metrics:
    enabled: true
    harmony_window_sec: 2.0
  dtw:
    enabled: false              # re-synthesises transcriptions for audio DTW

benchmark:
  results_path: benchmark_results.jsonl
  include_baseline: true
  baseline_name: baseline
  conditions: []
  sweeps:                       # each value becomes a condition
    - name: reverb_wet
      parameter: pedalboard.effects.1.wet_level
      values: [0.0, 0.3, 0.6]
```

## Usage

### CLI

`pip install sonitra` installs the bare `sonitra` command on your PATH. You can also use `python -m sonitra` if the binary is not on PATH.

```bash
# Run the pipeline
sonitra render --corpus corpus/midi --output corpus/audio

# Start the FastAPI server
sonitra serve --port 8000

# Write a starter config.yaml
sonitra init --config config.yaml

# Transcribe rendered audio with all configured transcribers
sonitra transcribe --audio corpus/audio --output transcriptions

# Score transcriptions against reference MIDI (paired by file stem)
sonitra evaluate --reference corpus/midi --estimate transcriptions/basic_pitch

# Run the full benchmark: render, transcribe, evaluate per condition
sonitra benchmark --corpus corpus/midi --workdir benchmark

# Show version
sonitra --version
```

### Python API

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

### Benchmark API

```python
from sonitra.config import load_config
from sonitra.benchmark import run_benchmark
from pathlib import Path

cfg = load_config("config.yaml")
result = run_benchmark(
    sorted(Path("corpus/midi").glob("*.mid")),
    work_dir="benchmark",
    config=cfg,
)
for row in result.summary:
    print(row["condition"], row["transcriber"], row.get("note.onset_f1"))
for row in result.degradation:   # metric deltas vs the baseline condition
    print(row["condition"], row.get("delta_note.onset_f1"))
```

### FastAPI server (programmatic)

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
| `PUT /config` | Update config at runtime |
| `GET /status/{job_id}/stream` | SSE status stream |

## Testing

```bash
pytest                          # all tests
pytest tests/ -v               # verbose
pytest -m "not skip_if_no_vst"  # skip VST-dependent tests
```

## Evaluation metrics

The metric suite follows current AMT benchmarking practice (see `research.md`
for the full literature survey):

| Family | Metrics | Basis |
|---|---|---|
| Note-level | onset F1 (±50 ms), onset+offset F1 (20%/50 ms rule), onset+offset+velocity F1 | mir_eval / Hawthorne et al. 2018 |
| Frame-level | precision/recall/F1 over 10 ms piano-roll frames | MIREX convention |
| Expressive | onset MAE/bias, IOI correlation (timing), key-overlap-ratio correlation (articulation), velocity correlation (dynamics), windowed pitch-class similarity (harmony) | mpteval / Hu et al. 2024 |
| Audio | path-normalised DTW distance between rendered audio and re-synthesised transcription, over chroma features | Aria-AMT / Bradshaw et al. 2024 |

All metrics are implemented natively (NumPy/SciPy) with mir_eval-compatible
matching semantics: bipartite matching with equal pitch, ±50 ms onsets, and
offsets within max(50 ms, 20% of reference duration). Undefined values (e.g.
correlations over too few matched notes) are reported as NaN and skipped in
aggregation.

The benchmark runner produces degradation curves in the style of Edwards et
al. 2024: define sweeps over any config parameter (reverb wet level, noise,
sample rate, ...) and read per-condition metric deltas against the clean
baseline from `summary.json`.

## Rendering modes

| Mode | Synth backend | Effects |
|---|---|---|
| `dawdreamer_only` | DawDreamer (Faust/VST) | None |
| `pedalboard_only` | Pedalboard instrument plugin | Configurable effects chain |
| `dawdreamer_synth_pedalboard_fx` | DawDreamer | Pedalboard effects chain |

## Dependencies

- Python >= 3.11
- [dawdreamer](https://github.com/DBraun/DawDreamer) — DawDreamer (Faust/VST synthesis engine)
- [pedalboard](https://github.com/spotify/pedalboard) — Pedalboard (audio effects and instrument API)
- [mido](https://github.com/mido/mido) — MIDI message parsing
- [FastAPI](https://fastapi.tiangolo.com/) — API server
- [basic-pitch](https://github.com/spotify/basic-pitch) — installed by default as the default AMT backend; adds a non-trivial transitive ML runtime (TensorFlow/ONNX/CoreML depending on platform)
- Optional: VST3 instrument and effect plugins
- Optional extras: `sonitra[demucs]` ([Demucs](https://github.com/adefossez/demucs) stem separation). The `[basicpitch]` extra remains as a backward-compatible alias.

## License

MIT

## Status

This repository is currently in early setup as part of a MishMash seed-funding project on AI transcription of music and the evaluation of state-of-the-art systems.