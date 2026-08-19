# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sonitra is a research toolkit for **benchmarking automatic music transcription (AMT) systems**. It does not train models. The core loop renders symbolic scores to audio, optionally degrades/separates that audio, transcribes it back to symbolic form with one or more AMT backends, and scores the result against the ground-truth score.

```
MIDI → synthesise audio → transcribe (audio→MIDI) → evaluate vs reference
```

## Commands

```bash
uv sync --extra dev              # preferred: editable install + test deps (lockfile)
pip install -e ".[dev]"          # fallback if uv is unavailable

uv run pytest                         # run the suite (no venv activation needed)
uv run pytest tests/test_pipeline_config.py -v  # single file
uv run pytest tests/test_pipeline_config.py::test_name  # single test
uv run pytest -m "not skip_if_no_vst"   # skip tests needing a real VST plugin path
uv run pytest -m integration            # only end-to-end VST tests
uv run pytest -m "not slow"             # skip tests that invoke heavy backends (basic-pitch)
```

Test markers (`pyproject.toml`): `skip_if_no_vst` and `integration` both require a real VST. `slow` marks tests that invoke optional/heavy backends such as basic-pitch (TensorFlow inference). VST-dependent fixtures read the plugin path from the `VST_PATH` / `VST3_PATH` env vars and `pytest.skip` when unset.

CLI entry points (Typer, also `python -m sonitra`). The recommended invocation style uses `--config` + `--dataset`:

```bash
sonitra init       --config FILE                                          # write a starter config
sonitra render     --config FILE [--dataset NAME] [--limit N] [--seed N]
sonitra transcribe --config FILE [--dataset NAME] [--transcriber NAME]
sonitra evaluate   --config FILE [--dataset NAME]
sonitra benchmark  --config FILE [--dataset NAME]
sonitra serve      --port 8000
sonitra --version
```

Explicit path overrides also work (`--corpus`, `--output`, `--audio`, `--reference`, `--estimate`).
Batch runner script: `python scripts/run_transcribe_eval.py` (see Scripts section below).

> There is **no ruff/black/mypy config in `pyproject.toml`** — `pytest` is the real quality gate.

## Architecture

### Pluggable-backend pattern (the central idiom)

Every swappable component — synthesisers, transcribers, evaluation metrics — follows the same three-part shape:

1. A **`Protocol`** (`runtime_checkable`) defining the interface — e.g. `TranscriberProtocol.transcribe()`, `SymbolicMetric.compute()`, `SynthesiserProtocol`.
2. A **registry + decorator** keyed by a config discriminator — `@register_transcriber("basic_pitch")`, `@register_symbolic_metric("note")`.
3. A **`make_*` factory** that reads config and returns instances — `make_synth(cfg)`, `make_transcriber(cfg)`, `make_symbolic_metrics(section)`.

Crucially, factories **lazily import backend modules inside the function body** so registration happens on first use and optional dependencies (e.g. dawdreamer VST) are never imported at package load time. When adding a backend: define it in its subpackage, decorate it with the registry decorator, and ensure the factory's lazy import covers the new module. Don't import backends at module top level.

### Config (`config.py`)

Single Pydantic `PipelineConfig` tree, one nested section per concern (`render_pipeline`, `io`, `dawdreamer`, `fluidsynth`, `pedalboard`, `normalisation`, `quality_gates`, `transcription`, `evaluation`, `benchmark`, `observability`, `separation`). All sections use `extra="forbid"` — unknown YAML keys are hard errors. `model_validate` re-raises validation failures as `ConfigError`. `config/source.yaml` is the fully-annotated reference documenting every parameter; `default_config_path()` in `config.py` points there. `PipelineConfig.save(path)` serialises a config instance back to YAML.

`render_pipeline.synth_backend` (enum `SynthBackend`: `fluidsynth | dawdreamer_faust | dawdreamer_vst | pedalboard_instrument`) and `render_pipeline.effects_chain` (enum `EffectsChain`: `none | pedalboard`) replace the former `rendering_mode` key. `render_pipeline.bpm` (default 120) sets the playback tempo. `render_pipeline.max_workers` controls render-step parallelism; `transcription.max_workers`, `evaluation.max_workers`, and `benchmark.max_workers` provide independent per-step parallelism.

Backend-specific validators enforce: `synth_backend=fluidsynth` requires `fluidsynth.soundfont_path`; `synth_backend=dawdreamer_vst` requires `dawdreamer.plugin_path`; `synth_backend=dawdreamer_faust` must NOT set `dawdreamer.plugin_path`.

`validate_worker_constraint()` forces `render_pipeline.max_workers=1` for DawDreamer synth backends (`dawdreamer_faust`, `dawdreamer_vst`) — DawDreamer/JUCE is not safe to run concurrently. `fluidsynth` and `pedalboard_instrument` are unaffected. Call it before parallelising work.

### Synth backends (`synth/`)

`make_synth(cfg)` in `synth/protocol.py` selects the synthesis backend from `cfg.render_pipeline.synth_backend`:

- **`PedalboardSynth`** — when `synth_backend=pedalboard_instrument`; renders via `pedalboard.instrument.plugin_path`. If `pedalboard.instrument.plugin_path` is null but `fluidsynth.soundfont_path` is set, falls back to `FluidSynth` with a logged warning.
- **`FluidSynth`** — when `synth_backend=fluidsynth`; requires `fluidsynth.soundfont_path`; invokes the `fluidsynth` CLI against the named `.sf2` SoundFont file. Lazily imported from `synth/fluid_synth.py`.
- **`DawDreamerSynth`** — when `synth_backend=dawdreamer_faust` (built-in Faust oscillator) or `synth_backend=dawdreamer_vst` (VST3 plugin, requires `dawdreamer.plugin_path`); both values map to the same implementation branch.

### Pipeline (`pipeline.py`)

`run_pipeline` has **two code paths**: the config-driven path (when `config=` is passed — this is the real one) and a legacy `engine=`-based path kept for the older API/tests. New work goes through the config path. The pipeline is fail-soft: each MIDI file is rendered in a try/except that logs a per-file record and continues; it never aborts the batch. Per-file outcomes are written to a JSONL manifest (`renders.jsonl`) plus an optional `.failed.txt`. Note that some setup steps, such as building the effects chain from config, currently run before the per-file loop and can abort the batch if they fail (e.g., a VST3 plugin cannot be loaded). Order within a file: parse MIDI → synth.render → pre-normalise → effects chain (skipped when `render_pipeline.effects_chain: none`) → post-normalise → quality gate → write.

### Benchmark (`benchmark/`)

`run_benchmark` is the top-level orchestrator. It expands **conditions** and **sweeps** (`benchmark/conditions.py`) into a list of variants, each defined by **dotted-path config overrides** (e.g. `pedalboard.effects.1.wet_level`) applied via `apply_overrides`. For every condition it re-renders the corpus, runs each enabled transcriber, and scores against the cached reference notes. Outputs: per-(condition×transcriber×file) records to JSONL (each record carries its condition's `overrides` dict), `summary.json` containing the aggregate `summary` (each row also carries its condition's `overrides`) and a `degradation` table of metric deltas vs the `baseline` condition (with `overrides` carried through unchanged, not diffed), and a `config.yaml` snapshot of the fully-resolved `PipelineConfig` the run actually used. `scripts/export_regression_table.py` flattens a run's JSONL into a per-file regression-ready CSV (metrics + overrides as columns, pedalboard effect slots labeled by type when `config.yaml` is present).

### Evaluation (`evaluation/`)

Metrics implemented natively in NumPy/SciPy with mir_eval-compatible matching semantics (no mir_eval dependency). Two protocol families: `SymbolicMetric` (note/frame/expressive — reference vs estimated notes) and `AudioMetric` (DTW — compares the rendered audio against a **re-synthesis of the transcription** using the same synth config). Results are flattened into `"<metric>.<key>"` keys. **Undefined values are `NaN`** (e.g. correlations over too few matched notes) and aggregation deliberately skips them — preserve this convention when adding metrics.

### API (`api/`)

FastAPI app (`create_app`) with a `JobStore`, a synchronous render worker (`worker.py`) running `run_pipeline` on the event loop while holding an `asyncio.Lock` (DawDreamer/JUCE global state is not safe to run concurrently), runtime config reload via `PUT /config`, and an SSE status stream. Config is loaded into `app.state.config` at startup. Routers are split by concern under `api/routers/`. `tests/api/openapi_snapshot.json` is a checked-in OpenAPI snapshot — schema changes will require regenerating it (see `tests/api/test_openapi.py`).

### Scripts (`scripts/`)

`scripts/run_transcribe_eval.py` is a batch runner that iterates over every YAML file in `config/examples/`, runs `sonitra render`, `sonitra transcribe`, then `sonitra evaluate` for each, and writes results under a dataset-first layout:

- `corpus/{dataset}/eval_results/<config>.jsonl` — per-file evaluation records for each config
- `corpus/{dataset}/eval_results/summary.jsonl` — one line per config, mean of per-file metrics
- `corpus/{dataset}/eval_results/summary.csv` — same data as `summary.jsonl` in CSV format
- `corpus/{dataset}/eval_results/all_results.csv` — flat table with one row per (config, file)

When no `--dataset` is passed the paths collapse to `corpus/eval_results/`, `corpus/audio/`, etc.

NaN values are written as `null` in JSONL and as empty cells `""` in CSV.

The script accepts `--jobs N` (default: 1) to process N configs in parallel; each config's render→transcribe→evaluate steps still run serially within the worker. Use `--skip-render` to reuse previously rendered audio.

### Config directory (`config/`)

`config/source.yaml` is the fully-annotated reference config documenting every parameter (not a runnable pipeline config). Runnable preset configs are split across two subdirectories — these are not test fixtures (those live in `tests/fixtures/`):

- `config/examples/` — 18 preset configs used by `scripts/run_transcribe_eval.py` and the roundtrip tests: `pedalboard_baseline`, `pedalboard_no_effects`, `pedalboard_no_effects_parallel`, `pedalboard_parallel`, `pedalboard_all_effects`, `pedalboard_extreme_reverb`, `pedalboard_heavy_compression`, `pedalboard_chorus_delay`, `pedalboard_distortion_gain`, `pedalboard_vital`, `dawdreamer_soundfont`, `dawdreamer_faust`, `dawdreamer_vital`, `dawdreamer_vital_pedalboard`, `dawdreamer_vital_goodies`, `dawdreamer_vital_goodies_pedalboard`, `dawdreamer_vital_delayed_flight`, `dawdreamer_vital_delayed_flight_pedalboard`.
- `config/benchmark/` — 6 parametric study configs plus a `README.md`: `reverb_sweep`, `compression_sweep`, `distortion_sweep`, `effects_combinations`, `synthesis_backends`, `benchmark_test` (smoke test).

### GPU support

The `[gpu]` optional extras (`uv sync --extra gpu`, Linux x86_64 only) install `tensorflow[and-cuda]` to enable GPU inference for Basic Pitch. Set `device: GPU:0` (or the relevant TF device string) on any `basic_pitch` transcriber entry in the `transcription.transcribers` list. The default is `device: cpu`.

Docker GPU passthrough is a profile on the single `docker/docker-compose.yml` (`--profile gpu`, service `sonitra-gpu`; `--profile cpu`/service `sonitra` for the default build — a profile is always required, there is no profile-less default). Devcontainer GPU passthrough is a separate override file, `.devcontainer/docker-compose.gpu.yml` (added to the `dockerComposeFile` array in `devcontainer.json`), since VS Code's Dev Containers tooling doesn't support Compose profiles.

## Conventions

- `from __future__ import annotations` at the top of every module; type everything.
- New tunable behaviour goes through the Pydantic config tree, not function kwargs — keep `extra="forbid"` working by updating the relevant section.
- Pipeline/benchmark/CLI loops are fail-soft: log a structured per-item record and continue rather than raising out of a batch.
- `research.md` is the literature survey backing the metric choices; cite it when changing metric definitions.
- `basic-pitch` is a **core dependency** (in `dependencies`, not just extras). The `[basicpitch]` extra remains as a backward-compatible alias only.
