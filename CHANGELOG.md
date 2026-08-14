# Changelog

All notable changes to the Sonitra project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Five new `pedalboard`-backed filter effect types available under
  `pedalboard.effects`: `HighpassFilter`, `LowpassFilter`, `HighShelfFilter`,
  `LowShelfFilter`, `PeakFilter`.
- Four new benchmark scenario studies under `config/benchmark/`: `old_recording/` (vintage 78rpm shellac,
  early reel-to-reel tape, and AM radio broadcast chains — phase-1
  bandwidth-and-dynamics ablations at two severities against a common
  baseline),
  `telephone_channel/` (voice-channel bandwidth + AGC — ITU-T G.722 wideband
  VoIP, ITU-T G.711 narrowband PSTN, land-mobile-radio intercom; 4
  conditions), `venue_acoustics/` (RT60-calibrated `Reverb` ablation —
  studio, recital hall, symphony hall, cathedral; 5 conditions), and
  `rotary_speaker/` (Leslie rotary-speaker chorale/tremolo character via
  `Chorus`; 3 conditions). Config-and-documentation only, no `src/sonitra/`
  changes.
- `docker/Dockerfile` runtime stage: `tmux` installed for interactive
  `docker exec` terminal sessions into running containers
- `docker/Dockerfile`: `HOST_UID`/`HOST_GID` build args (default `1000`)
  baked into the non-root `sonitra` user and passed through from `.env` via
  `docker-compose.yml`, so bind-mounted repo directories keep host ownership
  on native Linux; the entrypoint now only `chown`s a directory when its
  top-level ownership doesn't already match `sonitra`. `/app/.venv/bin`
  added to `PATH` so `sonitra`/`uvicorn` work directly in `docker exec`
  sessions
- `BasicPitchTranscriberConfig` gains `melodia_trick` (default `true`, HMM/
  melodia post-processing smoothing) and `multiple_pitch_bends` (default
  `false`) knobs, forwarded to `basic_pitch.inference.predict()`. Note:
  `multiple_pitch_bends: true` changes note eventing only — the `bends`
  tuples are still dropped and `midi_writer.py` writes no pitch-wheel
  messages, so glissando curves are not represented in the output MIDI
  (documented limitation).
- New `save_raw_outputs` flag (default `false`) on the `basic_pitch`
  transcriber: when enabled, the model's raw onset/contour/note probability
  maps (currently discarded by `transcribe()`) are persisted as a wide
  441-column piano-roll CSV (`<stem>.model_outputs.csv`, one row per model
  frame, `time_sec` derived from `basic_pitch.note_creation.
  model_frames_to_time`) next to each transcribed MIDI, via the new
  `write_transcription_outputs`/`write_raw_outputs` helpers in
  `midi_writer.py`
- `scripts/download_datasets.py`: interactive dataset picker backed by a rich
  table (number, key, name, size, target path, and present/missing status;
  prompt accepts comma-separated numbers, `all`, or `q`) when run without
  arguments on a real terminal; `--jobs N` for downloading up to N selected
  datasets concurrently (default 1 = serial); `--list` now renders a rich table
  when available. Falls back gracefully to the previous stdlib-only behaviour
  (plain-text `--list`, original error message) when `rich` is unavailable or
  stdin is not a TTY
- `scripts/download_datasets.py`: added the Beethoven Symphony Excerpt Dataset
  (BSED) v1.0; the single `zip_strip_prefix` extraction generalised into an
  `extract_map` of `(zip_prefix, target_subdir)` pairs so a dataset can route
  different zip subfolders to different corpus subdirs — BSED splits into
  `midi/` and `recordings/` (the latter deliberately distinct from `audio/`,
  reserved for the pipeline's own rendered output)
- `benchmark.save_audio` (default `true`) and `benchmark.resume` (default
  `false`) config knobs: `save_audio: false` deletes a condition's rendered
  audio (and separated stems) right after that condition's transcription and
  evaluation finish, bounding peak disk usage to roughly one condition
  instead of the whole sweep (results, summaries, and transcriptions are
  always kept). `resume: true` continues a stopped run by treating every
  `(condition, file, transcriber)` triple already recorded — succeeded,
  failed, or `render_failed` — as done and only computing what is missing; a
  config fingerprint is stored next to `benchmark_results.jsonl` and compared
  on resume, so a config edit that would change what a condition or record
  means raises an error instead of silently mixing results
- `observability.log_level` (validated root-logger override, takes precedence
  over `render_pipeline.log_level`) and `observability.progress` (default `true`;
  master switch for live CLI progress bars) config fields
- New `sonitra.terminal` module: rich console singleton, idempotent rich
  logging setup, effective-log-level resolution, per-file speed column, and a
  `BenchmarkProgress` protocol with `Null`/`Rich` implementations;
  `RichBenchmarkProgress` renders a header (device chips, worker pids,
  failures), a sweep bar, and one row per pool worker under a single `Live`
- Benchmark run now streams `WorkerEvent` records for each `(file,
  transcriber)` cell through a multiprocessing queue so live progress works
  in parallel mode; worker fd 1/2 are redirected to per-worker log files
  (`<work_dir>/logs/`) so TF C++ output cannot corrupt the shared terminal
- `RichBenchmarkProgress` worker rows now show the current pipeline stage
  (`render`, optional `separate`, `transcribe`) alongside
  `condition × transcriber`, in both serial and parallel mode;
  `WorkerEvent` gains a `stage` field and a new `status == "stage"` value
  for these non-cell-boundary transitions, carried by the existing
  worker-event queue with no new plumbing.
- `RichBenchmarkProgress` worker rows now split into two adjacent Progress rows
  per pool worker: a coloured header row (pid, condition, transcriber, stage,
  device — each field individually styled) with no bar, and an indented detail
  row underneath (file, progress bar, M-of-N, elapsed), fixing illegible
  line-wrapping on narrow terminals where the previous single combined row would
  overflow. Condition/transcriber names and file paths are escaped before going
  through Rich markup so `[`/`]` characters in user-authored benchmark YAML or
  filesystem paths can't raise `MarkupError` or mis-render.
- `render`, `transcribe`, `evaluate`, `benchmark`, `init`, and `serve` CLI
  output beautified with `rich`: per-file progress bars, failure tables, and
  benchmark summary/degradation tables; new global `--verbose`/`--quiet`
  flags; `pipeline.on_file_done` wired into the render progress bar; TF/
  absl/basic-pitch logging silenced so backends cannot corrupt the display.
  Adds `rich` to the core dependencies
- `--limit`/`--seed` CLI flags on `sonitra transcribe` and `sonitra
  evaluate`, mirroring the existing `render`/`benchmark` flags; `evaluate`
  samples only reference files with matching estimates so `--limit N` means
  at most N evaluated pairs, staying coherent after a limited
  render/transcribe run
- Audio-input benchmark mode: `render_pipeline.input_type: audio` reads
  source recordings directly from `{corpus_root}/{dataset}/recordings/` (new
  `CorpusPaths.recordings`), skipping synthesis entirely. New
  `sonitra.corpus` module (`discover_midi_files`, `discover_audio_files`,
  `pair_audio_to_reference` — deterministic token-prefix pairing of
  recordings to reference MIDIs) and `sonitra.source` module
  (`SourceProtocol` with `MidiSource`/`AudioSource`, `make_source` factory;
  `load()` returns the real sample rate, the source file's own rate in audio
  mode). Audio-mode benchmark cells are (recording × transcriber) and
  `evaluation.dtw` is skipped (re-synthesised audio vs a real recording is
  not meaningful); conditions/sweeps may not override
  `render_pipeline.input_type` (validated once at setup). `ManifestEntry`
  and `BenchmarkRecord` gain a `source_path` field naming the recording in
  audio mode (`midi_path` always stays the reference MIDI)
- `scripts/download_datasets.py`: new datasets — `maestro-v3-midi`/`-wav`/
  `-full` (the old `maestro-v3` key is renamed), `musicnet`, and
  `e-gmd-midi`/`e-gmd-full` (drum dataset, download-only); entries can pull
  from multiple URLs and both `.zip` and `.tar.gz`, routing members to
  `midi/`, `recordings/`, or the new `metadata/` by prefix/extension rules
- API render worker discovers audio files (`.wav`/`.flac`/`.mp3`) instead of
  only `.mid` when the active config is in audio mode

### Changed

- **BREAKING:** the `pipeline` config section is renamed to
  `render_pipeline` (`extra="forbid"` rejects the old key); the new
  `render_pipeline.input_type` field (`midi` | `audio`, default `midi`)
  selects MIDI synthesis vs direct audio input, and in audio mode the
  synth-backend field requirements are skipped since the synth is never
  constructed

- All six `config/benchmark/*.yaml` presets now set `device: GPU:0` on their
  `basic_pitch` transcriber (previously unset, defaulting to `cpu` — so
  Basic Pitch ran CPU inference even inside a working GPU container), and
  raise `transcription.max_workers` and `evaluation.max_workers` from 1 to 4.
  `render_pipeline.max_workers` and `benchmark.max_workers` stay at 1, each with an
  inline comment recording why: the former is only honoured for the
  `pedalboard_instrument` synth backend, and the latter spawns a process pool
  whose workers would each need their own GPU memory allocation.
- `README.md` condensed from ~490 to ~210 lines: the Docker quick-start now
  lives alongside Linux/macOS/Windows under Installation instead of appearing
  after "Data and plugins"; detailed reference material (Docker, VST3/preset/
  SoundFont setup, full CLI flags, configuration tables, evaluation metrics,
  Python API, REST API, datasets) moved to individual pages under `docs/`,
  linked from the corresponding condensed README section
- **BREAKING:** `docker/docker-compose.yml` and `docker/docker-compose.gpu.yml` merged
  into a single `docker/docker-compose.yml` with two Compose profiles: `sonitra`
  (`--profile cpu`) and `sonitra-gpu` (`--profile gpu`, tagged `sonitra:gpu`), sharing
  common config via a YAML anchor. A profile must now always be passed — there is no
  profile-less default — so the CPU and GPU containers can never both start at once and
  collide on port 8000. `docker/docker-compose.gpu.yml` is removed; all `README.md`
  Docker invocations updated to include `--profile cpu`/`--profile gpu`.
- `README.md`: GPU setup section rewritten to document the `nvidia-*` CUDA wheel
  approach with an explanation of why `tensorflow[and-cuda]` does not work; Docker
  CLI commands now use `--no-sync` to prevent dependency re-resolution inside the
  container; WSL2 note generalized from "TensorFlow's deep CUDA headers" to
  "packages with deeply nested file trees"
- `docker/Dockerfile` GPU extras comment updated to reference `pyproject.toml`'s
  `[gpu]` extra instead of the removed `tensorflow[and-cuda]` workaround

### Fixed

- `sonitra/terminal.py`: `on_worker_event`'s fallback branch called
  `logger.warning(...)` with no `logger` ever defined in the module —
  latent `NameError` on any unrecognized `WorkerEvent.status`. Added the
  missing module-level `logger = logging.getLogger(__name__)`.
- `docker/Dockerfile` GPU image (`sonitra-gpu` service, `runtime-gpu` target):
  rebuilt on an `nvidia/cuda:12.2.2-devel-ubuntu22.04` base with CUDA/cuDNN
  installed system-wide via apt, replacing the pip `nvidia-*-cu12` wheel
  approach. The wheel approach hit a reproducible TensorFlow 2.15 bug where
  `tf.config.list_physical_devices('GPU')` returns `[]` and TF logs "Could not
  find cuda drivers on your machine" even though the NVIDIA driver, Container
  Toolkit passthrough, and a raw `dlopen`/`cuInit()` of `libcuda.so.1` all
  succeed (see upstream tensorflow/tensorflow#62412, reproduced there with an
  exactly-matched driver/CUDA version — not a version-skew issue). `docker
  -compose.yml`'s `sonitra` (CPU) and `sonitra-gpu` services now set an
  explicit `target:` (`runtime`/`runtime-gpu`) since the Dockerfile gained a
  third stage. `pyproject.toml`'s `[gpu]` extra is unchanged and still
  available for bare-metal (non-Docker) installs.
- `docker/Dockerfile` builder stage: `scripts/` directory was never copied
  into the image, so `scripts/run_transcribe_eval.py` (the batch runner) was
  missing at runtime; now copied alongside `src/` and `config/`
- `[gpu]` optional extras: `tensorflow[and-cuda]==2.15.0` replaced with 11
  `nvidia-*` CUDA runtime wheels. The former dependency transitively required
  `tensorrt-libs==8.6.1` from NVIDIA's private PyPI (not available on the public
  index), causing `uv sync --extra gpu` / `pip install ".[gpu]"` to fail on Linux
  x86_64. The `nvidia-*` wheels are available on the standard PyPI, pinned to the
  same versions TF 2.15 declares as its `and-cuda` extras, and are equivalent
  for GPU inference.
- `docker/Dockerfile`: `/app` directory now explicitly `chown`'d to `sonitra` at
  build time, fixing the `PermissionError: [Errno 13] Permission denied:
  'renders.jsonl'` when writing the default manifest path as a non-root user
- `docker/Dockerfile` and `docker/entrypoint.sh`: entrypoint rewritten to run
  as root temporarily so it can `chown` bind-mounted directories (`/app/corpus`,
  `/app/output`, `/app/config`) to the `sonitra` user, then step down via `gosu`
  for security; this fixes permission errors on Docker Desktop for Windows and
  macOS where host directories are mounted as root inside the container
- `sonitra evaluate --config --dataset`: estimate output directory now
  resolved by the transcriber's backend type when its optional `name` field
  is unset, instead of crashing with `PosixPath / None` (mirrors how
  `transcribe` names its output directories)
- `sonitra benchmark` with `benchmark.max_workers: 1` (serial mode): backends
  that print directly to stdout (e.g. `basic-pitch`'s bare `print()` in
  `predict()`) were writing straight to the terminal and corrupting the Rich
  `Live` progress display's cursor tracking, so the display never visibly
  updated during the run and only rendered once, stale, at exit. Serial-mode
  condition processing now redirects stdout/stderr to `<work_dir>/logs/
  serial.log` while a display is active, mirroring the fd-redirection pool
  workers already got; `sonitra.terminal.get_console()` now pins its `file`
  to `sys.stdout` at construction time so the display itself keeps writing to
  the real terminal regardless of this redirection
- `RichBenchmarkProgress`'s outer `Live` now sets `transient=True`, so the
  header/progress rows are cleared on exit instead of being left behind as a
  permanent, truncated last frame (visible as stray `pid N · condition ·
  file…` lines after a parallel-mode benchmark run finished)
- `sonitra benchmark` summary/degradation tables now list conditions in the
  order they're declared in `benchmark.conditions`/`benchmark.sweeps`,
  regardless of which condition happened to finish first in parallel mode
  (`benchmark.max_workers > 1` gathers results via `as_completed`, which is
  nondeterministic); a new `order_by_condition()` helper in `benchmark/
  results.py` restores declared order after aggregation

## [0.2.0] - 2026-07-01

### Added

- `max_workers` config field in `transcription`, `evaluation`, and `benchmark` sections for controlling parallel execution granularity
- `device` config field in `BasicPitchTranscriberConfig` for selecting TensorFlow inference device (e.g. `cpu`, `GPU:0`)
- `--jobs` / `-j` CLI flag on `scripts/run_transcribe_eval.py` for parallel processing of multiple configs
- `threading.Lock` in `ManifestWriter.write()` to support safe concurrent writes from multiple threads
- `_init_thread_synth_chain()` initializer and `_render_file()` helper in `pipeline.py`, extracted from the main loop to support per-thread synth/effect-chain instances
- `_condition_worker()` subprocess entry point in `benchmark/runner.py` for running benchmark conditions in isolated processes via `ProcessPoolExecutor`
- `SynthBackend` and `EffectsChain` enums replacing the removed `RenderingMode` enum
- `FluidSynthSection` and `DawDreamerSection` Pydantic models for new config sections
- `bpm` field on `PipelineSection` for configuring tempo per pipeline run
- `PipelineConfig.save()` method for serialising config to YAML
- `tests/test_config_new_fields.py` — config schema validation tests for new field combinations
- `tests/test_fluid_synth_bpm.py` — 6 BPM timing invariance tests for FluidSynth backend
- 5 CLI init output verification tests (`synth_backend`, `effects_chain`, no `rendering_mode`, `fluidsynth` section present, no section-level `enabled`)
- `--limit` / `--seed` CLI flags on `sonitra benchmark` command for quick smoke tests on large corpora
- `config/benchmark/` directory with benchmark configuration files for parametric AMT evaluation studies: reverb sweep (11 conditions), compression sweep (13), distortion sweep (9), effects combinations (7), synthesis backends comparison (3 backends), and a quick smoke test (4 conditions)
- `parse_midi()` now returns `initial_bpm` (first `set_tempo` event) instead of the last tempo when `return_meta=True`, enabling tempo-aware rendering
- `_scale_note_timings()` helper in `pipeline.py` for scaling note timing arrays by a tempo ratio
- BPM scaling in `_render_file()`: native MIDI BPM is read and note timings are scaled by `native_bpm / cfg.pipeline.bpm`, allowing config BPM to differ from the MIDI's native tempo
- `tests/test_bpm_scaling.py` — 5 unit tests for `_scale_note_timings` and 1 integration test for `_render_file` BPM scaling
- `[gpu]` optional extras group in `pyproject.toml` — installs `tensorflow[and-cuda]==2.15.0` (linux/x86_64 only) for TensorFlow GPU auto-detection via the correct CUDA 12.2 + cuDNN 8.9 stack; install with `pip install ".[gpu]"` or `uv sync --extra gpu`
- `docker/docker-compose.gpu.yml` — Compose override enabling GPU passthrough for the production image; usage: `docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build`
- `.devcontainer/docker-compose.gpu.yml` — opt-in GPU override for the devcontainer; activate by adding this file to the `dockerComposeFile` array in `devcontainer.json`

### Changed

- `run_pipeline` now uses `ThreadPoolExecutor` for PEDALBOARD_ONLY mode to render MIDI files concurrently, with per-thread synth and effects chain instances
- CLI `transcribe` command now parallelises per-backend audio transcription via `ThreadPoolExecutor` when `transcription.max_workers > 1`
- CLI `evaluate` command now parallelises reference/estimate scoring via `ThreadPoolExecutor` when `evaluation.max_workers > 1`
- `run_benchmark` now parallelises benchmark conditions via `ProcessPoolExecutor` when `benchmark.max_workers > 1`, isolating DawDreamer/JUCE state per subprocess
- `_run_condition` and `_evaluate_one` in `benchmark/runner.py` now accept optional `writer` parameter (default `None`) for subprocess worker mode
- `scripts/run_transcribe_eval.py` main loop refactored into `_process_config()` function to enable parallel config execution
- `BasicPitchTranscriber.transcribe()` wraps the `predict()` call in `tf.device(self.device)` context manager to respect the configured device
- `RenderingMode` enum replaced by `SynthBackend` + `EffectsChain` enums across config, CLI, pipeline, and API (BREAKING: old `rendering_mode` config key is rejected by `extra="forbid"`)
- `dawdreamer.bpm` flattened to `pipeline.bpm`; `dawdreamer.soundfont_path` moved to dedicated `fluidsynth.soundfont_path` section (BREAKING: old keys rejected)
- Section-level `enabled:` removed from `dawdreamer` and `pedalboard` config sections (per-effect `enabled:` preserved)
- FluidSynth, DawDreamerSynth, and PedalboardSynth now accept `bpm` parameter; `make_synth` passes `bpm=cfg.pipeline.bpm` to all three backends
- `PedalboardSynth.render()` raises `ValueError` with actionable message instead of returning silent `np.zeros` when no VST instrument plugin is configured
- `make_synth()` in `protocol.py` automatically falls back to `FluidSynth` when `synth_backend=pedalboard_instrument` has `plugin_path: null` but `fluidsynth.soundfont_path` is configured, with a logged warning
- `sonitra init` now writes starter config using `SynthBackend.DAWDREAMER_FAUST` and `EffectsChain.NONE` with explicit `fluidsynth`/`dawdreamer` sections
- All preset configs and test fixtures updated from `rendering_mode`/`soundfont_path`/`bpm` to the new config structure
- Config directory restructured: flat `config/*.yaml` presets moved to `config/examples/`; benchmark-specific configs organized under `config/benchmark/` with dedicated parametric study files (reverb sweep, compression sweep, distortion sweep, effects combinations, synthesis backends)
- `sonitra benchmark` output directory now scoped by config stem: `benchmark/<config_stem>/` instead of `benchmark/`
- `scripts/run_transcribe_eval.py` config discovery path updated from `config/` to `config/examples/`
- `config/source.yaml` bpm comment updated to note MIDI-derived default behaviour when the `bpm` field is commented out
- `.devcontainer/` now installs `[dev,gpu]` extras by default (packages only; GPU device passthrough remains opt-in via the separate `docker-compose.gpu.yml` override) and sets `NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}` + `NVIDIA_DRIVER_CAPABILITIES: compute,utility` in the base compose file
- `docker/Dockerfile` accepts a `GPU_EXTRAS` build ARG (default `""`) to opt into `tensorflow[and-cuda]` CUDA packages without changing the base image; GPU deps are installed before `COPY src/` for Docker layer cache efficiency

### Fixed

- Devcontainer GPU passthrough now actually activates: `docker-compose.gpu.yml` added to the `dockerComposeFile` array in `.devcontainer/devcontainer.json` (previously the override was documented but never referenced, so every rebuild was CPU-only)
- GPU compose overrides (`.devcontainer/` and `docker/`) switched from the legacy `runtime: nvidia` to the `deploy.resources.reservations.devices` form, which is the syntax Docker Desktop / WSL2 honours (the nvidia runtime is not registered there)
- `LD_LIBRARY_PATH` now points at the `tensorflow[and-cuda]` CUDA wheel directories (`site-packages/nvidia/*/lib`) in both `.devcontainer/Dockerfile` and `docker/Dockerfile`, fixing TensorFlow GPU dlopen discovery (tensorflow/tensorflow#65842); the previous GPU builds installed the CUDA wheels but TensorFlow could not locate them
- GPU build failure in `.devcontainer/Dockerfile` (and latent same failure in `docker/Dockerfile`): the previous `pip install -e ".[gpu]"` / `uv pip install "tensorflow[and-cuda]==2.15.0"` forms both fail because `tensorflow[and-cuda]==2.15.0` transitively requires `tensorrt-libs==8.6.1`, which is only available on NVIDIA's private PyPI index (`pypi.nvidia.com`) and not on public PyPI; fixed by installing the 11 `nvidia-*` CUDA runtime wheels directly (pinned to the same versions that TF 2.15 declares as its `and-cuda` extras), which are all present on the standard PyPI index and are equivalent for GPU inference (TensorRT is not required for basic-pitch)

## [0.1.0] - 2026-06-26

### Added

- `env.example` with documented `SONITRA_CONFIG` and `LOG_LEVEL` environment variables for container setup
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
- `config/source.yaml` — fully-annotated reference config documenting every available parameter with commentary, replacing the deleted root-level `config.yaml`
- Docker Compose setup under `docker/` with multi-stage `Dockerfile`, `docker-compose.yml`, and `entrypoint.sh` for containerised operation; exposes the REST API on port 8000, supports CLI one-off commands, and bundles system deps (libsndfile, fluidsynth, FFTW)
- `ROADMAP.md` tracking planned features: stem separation, additional datasets/instruments, and additional AMT backends
- `CorpusPaths` frozen dataclass and `resolve_corpus_paths()` helper for deriving dataset-scoped midi/audio/transcription/eval_results subdirectory paths from `corpus_root` and optional `dataset`
- `--dataset` / `-d` CLI flag on `render`, `transcribe`, `evaluate`, and `benchmark` commands; all four commands derive default corpus paths from config when `--dataset` is set
- `_apply_dataset()` helper for injecting `dataset` into config before pipeline execution
- `--skip-render` flag to `scripts/run_transcribe_eval.py` for resuming from transcription on already-rendered audio
- `config/source.yaml` now documents the optional `dataset` field
- Dataset support test suite (`tests/test_dataset_support.py`) covering `IOSection` field acceptance, `resolve_corpus_paths` path derivation with/without dataset/config_name, YAML round-trip, and backward-compatibility error on removed keys
- `_discover_midi_files()` helper for recursive `.mid`/`.midi` discovery via `rglob` with case-insensitive extension matching
- `_apply_subset()` helper for reproducible random subset sampling with configurable seed
- `--limit` / `--seed` CLI flags on `sonitra render` for quick smoke tests on large corpora
- `--config NAME [NAME ...]` filter on `scripts/run_transcribe_eval.py` to run only named preset configs instead of all configs under `config/`
- `--limit` / `--seed` passthrough from `scripts/run_transcribe_eval.py` to the render subprocess
- `scripts/download_datasets.py` — self-contained stdlib-only dataset download script supporting MAESTRO V3.0.0 MIDI, with `--list`, `--all`, and `--output-dir` options; idempotent (skips already-present datasets)
- Corpus discovery and subset test suite (`tests/test_corpus_discovery.py`) covering `_discover_midi_files` (flat/recursive/case-insensitive/non-MIDI filtering/directory-as-file), `_apply_subset` (deterministic bounded sampling), and CLI smoke tests for nested directory rendering and `--limit` enforcement

### Changed

- `IOSection`: `midi_dir` and `output_dir` fields replaced by single `corpus_root` (default: `"corpus"`) with optional `dataset` field (BREAKING: old keys raise `ConfigError` at load time via `extra="forbid"`)
- All preset configs and test fixtures updated from `midi_dir`/`output_dir` to `corpus_root`; most presets gain `dataset: test` for dataset-first output layout
- `manifest_path` in all preset configs simplified to `renders.jsonl` (now relative to dataset-scoped output directories)
- `sonitra init` template now emits `corpus_root` instead of `midi_dir`/`output_dir`
- `scripts/run_transcribe_eval.py`: added render step (step 1) before transcribe; skips `source.yaml`; reorganised path resolution with `_resolve_dirs()` for dataset-scoped layout; step numbering updated
- `CLAUDE.md`: CLI examples updated for dataset-first paths; script runner documented with dataset-first layout
- CLI `render`, `evaluate`, and `benchmark` commands now use recursive MIDI discovery via `rglob`, supporting both `.mid` and `.midi` extensions at any subdirectory depth (previously only flat `*.mid` globbing)
- `scripts/run_transcribe_eval.py`: added `--limit`, `--seed`, and `--config NAME [NAME ...]` flags; `--dataset` now forwarded to the render subprocess (previously ignored)

- `RendererEngine`: dawdreamer import moved from module level into `__init__` to avoid loading the native shared library (which requires `libGL`) at import time when DawDreamer is not configured
- Dockerfile: copy `config/` from repo into the build so the reference config is available in the runtime image; add `libgl1` runtime dependency; set `UV_CACHE_DIR` to `/app/.cache/uv`; use `--chown` in `COPY --from=builder` to set ownership in a single pass; scope `chown` to specific directory mounts instead of the entire app tree; create `/app/.cache/uv` directory for uv's runtime cache
- Docker quick-start guide in `README.md`: `mkdir` creates `corpus/midi` structure; new note explaining config pre-init requirement before server start; added `sonitra render` CLI example; benchmark command path updated to `/app/corpus/midi`; volume mount table clarifies `midi/` subdirectory
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
- Root `config.yaml` replaced by `config/source.yaml` annotated reference; `default_config_path()` in `config.py` now points to `config/source.yaml`
- `README.md` fully rewritten: uv-based install instructions with platform-specific commands, Docker quick-start section, data/plugins setup guide (MIDI corpus layout, VST3 plugin placement, presets, SoundFont), author attribution; Demucs references removed
- `CLAUDE.md` commands updated from pip to uv (`uv sync --extra dev`, `uv run pytest`); Demucs/stem separation references trimmed; `config/source.yaml` documented as the annotated reference
- Project description changed from "MIDI-to-audio batch rendering pipeline" to "Automatic music transcription (AMT) benchmark"
- Project author updated from `copilot-swe-agent[bot]` to **Shayan Dadman**
- `LICENSE` copyright year updated to 2026 and author name corrected
- `README.md` updated with dataset download documentation, `--limit`/`--seed` CLI examples, recursive MIDI discovery notes, and `--config` batch runner usage
- `ROADMAP.md` updated with `scripts/download_datasets.py` reference and planned real-audio transcription mode

### Fixed

- Default `config.yaml` no longer clips every render
- Full `pytest` suite no longer deadlocks after API worker tests
- `sonitra` binary is available on PATH after `pip install -e .`
- API integration test now passes with the fixed default config
- Docker entrypoint symlink: `SONITRA_CONFIG` is now linked to `/app/config/source.yaml` instead of the non-existent `/app/config.yaml`, matching the internal config resolution path used by `default_config_path()`
- Stem collision in nested corpora: `render`, `transcribe`, and `evaluate` now preserve
  the relative subpath from the corpus root in all output paths. Two files in different
  subdirectories with the same stem (e.g. `violin/opus.mid` and `piano/opus.mid`) no
  longer overwrite each other on disk or silently cross-pair during evaluation.
  (`corpus_root` is threaded through `run_pipeline` and `run_benchmark`; defaults to
  `None` so all existing flat-corpus workflows are unaffected.)

[Unreleased]: https://github.com/dadmaan/sonitra/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dadmaan/sonitra/releases/tag/v0.2.0
[0.1.0]: https://github.com/dadmaan/sonitra/releases/tag/v0.1.0
