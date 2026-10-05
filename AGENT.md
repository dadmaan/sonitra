# AGENT.md

Guidance for coding agents working in this repository. Deeper detail lives in `ARCHITECTURE.md` and `docs/`.

## Project

- Sonitra benchmarks automatic music transcription (AMT) systems; it does not train models.
- Core loop: MIDI → synthesise audio (optional effects / stem separation) → transcribe audio → MIDI → score against the reference.
- Audio-input mode (`render_pipeline.input_type: audio`) skips synthesis and pairs real recordings with reference MIDI.
- Data is dataset-first: `corpus/<dataset>/{midi,recordings,audio,transcription,eval_results}/`.

## Commands

```bash
uv sync --extra dev --extra transkun      # install, CPU fork (GPU: --extra transkun-gpu; fallback: pip install -e ".[dev]")
uv run pytest tests/                      # always scope to tests/ (no testpaths configured)
uv run pytest tests/test_config.py::test_name
uv run pytest tests/ -m "not slow"        # skip heavy backends (basic-pitch, transkun)
uv run --no-sync pytest tests/ -m slow    # heavy backends only
```

- `uv sync` is exact and strips packages the requested set does not name (`transkun`, `torch`, `pytest` among them); re-sync with `--extra ... --extra dev` or run `.devcontainer/post-create.sh` (see `docs/devcontainer.md`). `uv run` is inexact and safe; `--no-sync` mutates nothing.
- Weights are installed by `python scripts/setup_hft_transformer.py`. Exit code `3` means torch is absent, so skip it quietly; other non-zero codes are real failures. It runs automatically in both containers.
- Switching between the `transkun` (CPU) and `transkun-gpu` forks in an existing venv needs `--reinstall-package torch --reinstall-package torchaudio`; the GPU fork's specifier has no local segment, so a plain `uv sync` can audit the environment as correct and keep the old build.
- Without `uv`: `python -m pytest tests/`.
- In-container work must not commit `uv.lock`: the containers sync `--locked`, so a lock change breaks every other environment until it is reviewed deliberately.
- Markers: `skip_if_no_vst` / `integration` need a VST (`VST_PATH` / `VST3_PATH`); `slow` runs heavy backends; `requires_r` needs R with glmmTMB.
- No linter or type-checker config; pytest is the quality gate.
- CLI (Typer, also `python -m sonitra`): `sonitra init|render|transcribe|evaluate|benchmark --config FILE [--dataset NAME]`, `sonitra serve --port 8000`.

## Architecture

- **Backends**: each swappable component is a `runtime_checkable` Protocol plus a `make_*` factory driven by config.
  - Transcribers, metrics and separators also register via decorators (e.g. `@register_transcriber("basic_pitch")`).
  - Factories import backend modules lazily, so optional dependencies never load at import time. Never import a backend at package top level.
- **Config** (`config.py`): one Pydantic `PipelineConfig` tree.
  - Every section is `extra="forbid"`; validation failures raise `ConfigError`.
  - `config/source.yaml` documents every key (reference only, not runnable).
  - By `synth_backend`: `fluidsynth` requires `fluidsynth.soundfont_path`; `dawdreamer_vst` requires `dawdreamer.plugin_path`; `dawdreamer_faust` forbids it.
  - DawDreamer/JUCE is not thread-safe: call `validate_worker_constraint()` before parallelising (forces `render_pipeline.max_workers=1`).
- **Synth** (`synth/`): `make_synth` dispatches on `render_pipeline.synth_backend`; `pedalboard_instrument` with no plugin falls back to FluidSynth when a SoundFont is set.
- **Pipeline** (`pipeline.py`): `run_pipeline(config=...)` is the real path; `engine=` is legacy. Per file: load → synth → normalise → effects → normalise → quality gate → write, logged to `renders.jsonl`.
- **File selection**: `io.where`/`io.sample` (dataset = `io.dataset`); resolved only in `sonitra.selection` (reference-keyed; audio pairs against the full reference list); never filter corpus files ad hoc.
- **Benchmark** (`benchmark/`): expands conditions/sweeps into dotted-path overrides (`pedalboard.effects.1.wet_level`), then renders, transcribes and scores each. Writes per-file JSONL, `summary.json` (summary + degradation vs. `baseline`) and the resolved `config.yaml`.
- **Evaluation** (`evaluation/`): NumPy/SciPy metrics with mir_eval-compatible matching (no mir_eval dependency). Keys are `"<metric>.<key>"`; undefined values are `NaN` and aggregation skips them.
- **API** (`api/`): FastAPI; renders are serialised by an `asyncio.Lock`. Schema changes require regenerating `tests/api/openapi_snapshot.json`.
- **Scripts** (`scripts/`): standalone dataset and analysis tools; `--help` and `docs/` hold usage.
  - `download_datasets.py` stays stdlib-only (`rich` optional); dataset revisions are pinned to commit SHAs, never `main`.
  - `check_dataset.py` pairs via `sonitra.corpus.pair_audio_to_reference`; never re-implement pairing.
  - `run_mixed_effects_analysis.py` fits in R via `Rscript`. Do not alter its base model spec.
  - `export_model_baselines.py` owns the generated block in `docs/model-cards.md`; never hand-edit between `<!-- BEGIN/END GENERATED: baselines -->`. `corpus/` is untracked, so `--check` cannot run in CI.
  - Scripts that write files must never overwrite their inputs.
- **Configs**: `config/examples/` holds presets (used by `run_transcribe_eval.py` and roundtrip tests); `config/benchmark/` holds benchmark studies; test fixtures live in `tests/fixtures/`.
- **Models dir** (downloaded weights, not bundled): `$SONITRA_MODELS_DIR` when set and non-empty, else `~/.cache/sonitra/models`. Never resolve it repo-relative, even though the containers mount a host `models/` at `/models`; the repo's own `models/` is gitignored.
- **GPU**: set `device: GPU:0` on a `basic_pitch` transcriber; Docker needs `--profile gpu` or `--profile cpu` (no default profile).

## Conventions

- `from __future__ import annotations` in every module; type everything.
- New tunables go through the config tree (keep `extra="forbid"` valid), not function kwargs.
- Batch loops are fail-soft: log a per-item record and continue.
- Note dict contract (`src/sonitra/notes.py:make_note`): `pitch` 0..127 raises `ValueError`, `velocity` clamps 1..127, `start_sec` clamps >=0, `duration_sec <= 0` drops via `None`; non-finite (NaN/+-Inf) `pitch`/`velocity`/`start_sec`/`duration_sec` always raises `ValueError` too (never dropped or clamped — a non-finite value is always a producer bug, unlike a legitimate non-positive duration); output is sorted by `(start_sec, pitch)`. Every producer (transcriber, converter, test helper) must route notes through `make_note`; check results with `tests/helpers.py::assert_notes_satisfy_contract`. `midi_writer._collect_note_events` reimplements the same clamps inline (documented bypass, no pitch validation) and carries the same finite guard. `TranscriptionResult.backend_type` is the fixed `type` discriminator used for the writer registry; `transcriber` is the user-overridable `name`.
- Cite `docs/research.md` when changing a metric definition.
- `basic-pitch` is a core dependency; the `[basicpitch]` extra is only an alias.
- **Module docstrings** must be very concise and self-explanatory:
  - A one-line summary, plus at most a few short lines on essential behaviour.
  - Understandable without external context: no references to plans, phases, tickets, upstream code or other files unless needed to use the module.
  - No usage examples, flag lists, output schemas or design history; put those in `--help`, function docstrings, comments or `docs/`.
  - Exception: `scripts/*.py` docstrings end with an `Examples:` block of 2–4 copy-paste commands (`# comment` above each, repo-root paths, raw `r"""` when a line continues with `\`). The parser prints it via `description=__doc__` + `RawDescriptionHelpFormatter`; `tests/test_script_examples.py` checks flags and `--help` output.
- **No plan-relative references**: inline comments, docstrings, config comments and docs must be self-explanatory without the plan. Never cite "Phase N", "Step N", lane/arm labels ("2a", "3e", "P5"), tickets, or "this phase" — state the reason instead (e.g. "GPU stays at batch 1: batched-GPU throughput is unmeasured", not "Phase 2 decides").
- Keep this file concise: add only essential facts an agent cannot quickly derive from the code.
