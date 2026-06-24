# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sonitra is a research toolkit for **benchmarking automatic music transcription (AMT) systems**. It does not train models. The core loop renders symbolic scores to audio, optionally degrades/separates that audio, transcribes it back to symbolic form with one or more AMT backends, and scores the result against the ground-truth score.

```
MIDI → synthesise audio → (stem separation) → transcribe (audio→MIDI) → evaluate vs reference
```

## Commands

```bash
pip install -e ".[dev]"          # editable install + test deps
pip install -e ".[basicpitch]"   # optional: Basic Pitch transcriber backend
pip install -e ".[demucs]"       # optional: Demucs stem separation backend

pytest                           # run the suite
pytest tests/test_pipeline.py -v # single file
pytest tests/test_pipeline.py::test_name   # single test
pytest -m "not skip_if_no_vst"   # skip tests needing a real VST plugin path
pytest -m integration            # only end-to-end VST tests
```

Test markers (`pyproject.toml`): `skip_if_no_vst` and `integration` both require a real VST. VST-dependent fixtures read the plugin path from the `VST_PATH` / `VST3_PATH` env vars and `pytest.skip` when unset.

CLI entry points (Typer, also `python -m sonitra`):

```bash
sonitra render --corpus corpus/midi --output corpus/audio
sonitra transcribe --audio corpus/audio --output transcriptions [--transcriber NAME]
sonitra evaluate --reference corpus/midi --estimate transcriptions/basic_pitch
sonitra benchmark --corpus corpus/midi --workdir benchmark
sonitra serve --port 8000
sonitra init --config config.yaml
```

> Note: `CONTRIBUTING.md` is partly inherited from another project (ARIA) and references paths/tooling that do not exist here (`src/aria/`, `docs/`, ghsom scopes, ruff `--select D` ratchet). There is **no ruff/black/mypy config in `pyproject.toml`** — `pytest` is the real quality gate. Treat the commit-message convention in CONTRIBUTING as advisory, not its specific scope list.

## Architecture

### Pluggable-backend pattern (the central idiom)

Every swappable component — synthesisers, stem separators, transcribers, evaluation metrics — follows the same three-part shape:

1. A **`Protocol`** (`runtime_checkable`) defining the interface — e.g. `TranscriberProtocol.transcribe()`, `SymbolicMetric.compute()`, `SynthesiserProtocol`.
2. A **registry + decorator** keyed by a config discriminator — `@register_transcriber("basic_pitch")`, `@register_symbolic_metric("note")`.
3. A **`make_*` factory** that reads config and returns instances — `make_synth(cfg)`, `make_transcriber(cfg)`, `make_separator(cfg)`, `make_symbolic_metrics(section)`.

Crucially, factories **lazily import backend modules inside the function body** so registration happens on first use and optional dependencies (basic-pitch, demucs, dawdreamer VST) are never imported at package load time. When adding a backend: define it in its subpackage, decorate it with the registry decorator, and ensure the factory's lazy import covers the new module. Don't import backends at module top level.

### Config (`config.py`)

Single Pydantic `PipelineConfig` tree, one nested section per concern (`pipeline`, `io`, `dawdreamer`, `pedalboard`, `separation`, `transcription`, `evaluation`, `benchmark`, ...). All sections use `extra="forbid"` — unknown YAML keys are hard errors. `model_validate` re-raises validation failures as `ConfigError`. `RenderingMode` is the enum that selects the synthesis/effects path.

`validate_worker_constraint()` forces `max_workers=1` for any DawDreamer rendering mode (DawDreamer is not safe to run concurrently). Call it before parallelising work.

### Pipeline (`pipeline.py`)

`run_pipeline` has **two code paths**: the config-driven path (when `config=` is passed — this is the real one) and a legacy `engine=`-based path kept for the older API/tests. New work goes through the config path. The pipeline is fail-soft: each MIDI file is rendered in a try/except that logs a per-file record and continues; it never aborts the batch. Per-file outcomes are written to a JSONL manifest (`renders.jsonl`) plus an optional `.failed.txt`. Order within a file: parse MIDI → synth.render → pre-normalise → effects chain (skipped for `dawdreamer_only`) → post-normalise → quality gate → write.

### Benchmark (`benchmark/`)

`run_benchmark` is the top-level orchestrator. It expands **conditions** and **sweeps** (`benchmark/conditions.py`) into a list of variants, each defined by **dotted-path config overrides** (e.g. `pedalboard.effects.1.wet_level`) applied via `apply_overrides`. For every condition it re-renders the corpus, runs each enabled transcriber, and scores against the cached reference notes. Outputs: per-(condition×transcriber×file) records to JSONL, plus `summary.json` containing the aggregate `summary` and a `degradation` table of metric deltas vs the `baseline` condition.

### Evaluation (`evaluation/`)

Metrics implemented natively in NumPy/SciPy with mir_eval-compatible matching semantics (no mir_eval dependency). Two protocol families: `SymbolicMetric` (note/frame/expressive — reference vs estimated notes) and `AudioMetric` (DTW — compares the rendered audio against a **re-synthesis of the transcription** using the same synth config). Results are flattened into `"<metric>.<key>"` keys. **Undefined values are `NaN`** (e.g. correlations over too few matched notes) and aggregation deliberately skips them — preserve this convention when adding metrics.

### API (`api/`)

FastAPI app (`create_app`) with a `JobStore`, a `ThreadPoolExecutor` worker (`worker.py`) running `run_pipeline` per job, runtime config reload via `PUT /config`, and an SSE status stream. Config is loaded into `app.state.config` at startup. Routers are split by concern under `api/routers/`. `tests/api/openapi_snapshot.json` is a checked-in OpenAPI snapshot — schema changes will require regenerating it (see `tests/api/test_openapi.py`).

## Conventions

- `from __future__ import annotations` at the top of every module; type everything.
- New tunable behaviour goes through the Pydantic config tree, not function kwargs — keep `extra="forbid"` working by updating the relevant section.
- Pipeline/benchmark/CLI loops are fail-soft: log a structured per-item record and continue rather than raising out of a batch.
- `research.md` is the literature survey backing the metric choices; cite it when changing metric definitions.
