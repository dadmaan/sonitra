# Changelog

All notable changes to the Sonitra project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Core audio engine, MIDI reader, Faust/VST renderer, and multi-format storage (WAV/FLAC/MP3)
- `SynthesiserProtocol` with `DawDreamerSynth` wrapper and `make_synth` factory
- `PedalboardSynth` for MIDI-to-audio rendering via pedalboard instrument plugins
- `FluidSynth` SoundFont-based synthesiser backend (`synth/fluid_synth.py`); selected automatically when `dawdreamer.soundfont_path` is set in config
- Polyphonic Faust note processor (16 voices with gain, gate, and frequency controls), replacing the previous mono oscillator
- VST3 preset loading: `render_notes_vst` accepts an optional `preset_path`; Vital `.vital` native presets are automatically converted to `.vstpreset` containers for DawDreamer compatibility
- Effects chain builder mapping 8 built-in pedalboard types plus VST3 plugins
- Peak/RMS normaliser with configurable pre/post-effects ordering
- Quality gates for silence, clipping, and minimum duration checks
- Manifest writer with JSONL render log, failed-file list, and effects chain hash
- Config-driven pipeline with three rendering modes (`dawdreamer_only`, `pedalboard_only`, `dawdreamer_synth_pedalboard_fx`)
- Pydantic `PipelineConfig` schema with YAML loader, rendering mode enum, and worker constraint validation
- Transcription abstraction layer (`sonitra.transcribe`) with `TranscriberProtocol`, registry/factory, and three backends: Basic Pitch, external CLI commands, and precomputed MIDI directories for commercial tools
- `basic-pitch` as a core dependency; `[basicpitch]` extra remains as a backward-compatible alias only
- Evaluation suite (`sonitra.evaluation`) with mir_eval-compatible note-level P/R/F1 (onset, onset+offset, onset+offset+velocity), frame-level P/R/F1, musically informed expressive metrics (onset deviation, IOI/KOR/velocity correlations, windowed pitch-class harmony similarity), and DTW audio similarity over chroma features
- Metric registry with `SymbolicMetric`/`AudioMetric` protocols and config-driven factories
- Stem separation layer (`sonitra.separation`) with `StemSeparatorProtocol`, passthrough backend, and Demucs adapter (optional `[demucs]` extra)
- `SeparationError` for separation backend failures; raised with an install hint when Demucs is not installed
- Benchmark orchestration (`sonitra.benchmark`) running render → separate → transcribe → evaluate per experimental condition, with parameter sweeps via dotted-path config overrides, JSONL per-file records, aggregate summaries, and degradation-vs-baseline tables
- MIDI writer for persisting transcriptions (round-trip compatible with the MIDI reader)
- `read_audio` storage helper for loading rendered audio
- New config sections: `separation`, `transcription`, `evaluation`, `benchmark` (all defaulted; existing configs stay valid)
- CLI commands: `sonitra transcribe`, `sonitra evaluate`, `sonitra benchmark`, `sonitra init`
- `sonitra` console script entry point registered via `[project.scripts]`
- FastAPI server with job queue, runtime config reload, and SSE status streaming
- `slow` pytest marker for tests that invoke heavy backends such as Basic Pitch (TensorFlow inference)
- `vital_vst_path` test fixture for VST3 integration tests
- CLI test suite (`tests/test_cli.py`) covering `sonitra init` round-trips, rendering with the starter config, transcription, and console-script registration
- CLI `transcribe` command tests (via `CliRunner`) covering per-backend MIDI output, empty audio directory error, and `--transcriber` name filtering with `PrecomputedTranscriber`
- `BasicPitchTranscriber` protocol conformance, chord detection, silence handling, note sorting, and type-validation tests
- End-to-end transcription roundtrip tests (`test_transcription_roundtrip.py`) rendering via SoundFont and Vital VST3 configs, transcribing with Basic Pitch, and evaluating against reference MIDI with the full metric suite
- Regression tests verifying the default config renders without clipping, `sonitra init` emits a working starter config, and `basic_pitch` can transcribe rendered fixtures
- `config/` directory with preset configuration files for Vital VST3, SoundFont, and pedalboard effect chains
- `scripts/run_transcribe_eval.py` — batch runner that iterates over all configs in `config/`, runs `sonitra transcribe` + `sonitra evaluate`, and writes `summary.jsonl`, `summary.csv`, and `all_results.csv` to `corpus/eval_results/`
- `ARCHITECTURE.md` with Mermaid flowchart documenting the pipeline data flow
- `config/pedalboard_vital.yaml` — Pedalboard-only rendering preset using Vital VST3 instrument with no effects chain
- `test_vst3_effect_guard_rejects_instrument` — ensures instrument VST3 plugins raise `ValueError` when loaded as effects in `build_effects_chain`
- `test_run_pipeline_pedalboard_only_with_vital` — end-to-end Pedalboard + Vital VST3 render test with non-silent audio verification
- `uv.lock` for reproducible dependency resolution via `uv`

### Changed

- Default `config.yaml` now uses post-effects peak normalisation (`target_db: -1.0`) as the sole clipping guarantee; the `Limiter` is removed from the default effects chain and shown only as a commented optional effect
- `sonitra init` now writes a working starter config using `dawdreamer_only`, enabled normalisation, and a `basic_pitch` transcriber
- `BasicPitchTranscriber` docstring and error text describe `basic-pitch` as installed by default
- `DemucsSeparator` now raises `SeparationError` with an install hint pointing to `pip install sonitra[demucs]`
- API worker now renders synchronously on the API event loop while holding an `asyncio.Lock`, with a brief lock-free yield at entry so PENDING cancellation requests can land before rendering starts; avoids DawDreamer/JUCE thread/global state bleed that previously hung the full test suite
- API test client fixture now resets `app.state.config` after each test to prevent config-mutation leaks between API tests
- API renderer VST processor cache key now includes `preset_path` so different presets are correctly cached as separate processors
- `session_engine` test fixture changed from session-scoped to function-scoped to prevent DawDreamer state bleed across tests
- `test_cancel_running_job_via_api` replaced by `test_cancel_pending_job_via_api` to reflect the synchronous worker model where mid-render cancellation is not supported
- `README.md` fully rewritten: concise pipeline overview, platform-specific install steps, quick-start workflow, configuration reference, and testing guidance
- `CONTRIBUTING.md` stripped of inherited ARIA-specific content (ghsom, ruff D-ratchet, pre-commit, docs build stages); scoped to Sonitra conventions, scopes, and pytest-based quality workflow
- `CLAUDE.md` updated with synth backend routing documentation (PedalboardSynth / FluidSynth / DawDreamerSynth dispatch), scripts output layout, and config directory listing; removed inherited-project note
- `config/dawdreamer_vital_pedalboard.yaml` — compressor threshold reduced to −30 dB, ratio raised to 10:1, attack/release tightened; reverb wet level increased to 0.5, room size to 0.8, damping reduced
- `config/pedalboard_distortion_gain.yaml` — Gain effect removed, Distortion drive increased from 25 dB to 50 dB, normalisation moved to pre-effects
- `config/pedalboard_heavy_compression.yaml` — compressor threshold reduced to −50 dB, ratio raised to 20:1, normalisation moved to pre-effects

### Fixed

- Default `config.yaml` no longer clips every render
- Full `pytest` suite no longer deadlocks after API worker tests
- `sonitra` binary is available on PATH after `pip install -e .`
- API integration test now passes with the fixed default config
