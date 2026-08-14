# Roadmap

This file tracks planned work that is not yet implemented.

## Stem separation

**Status:** Not yet implemented.

Demucs-based stem separation is planned as an optional preprocessing step between audio synthesis and transcription. When enabled, it would isolate the target instrument stem before feeding audio to the transcription backend — useful for benchmarking AMT performance under realistic polyphonic mix conditions.

Planned design:
- `SeparatorProtocol` with a `separate(audio, sample_rate) -> audio` interface
- `@register_separator("demucs")` decorator following the existing pluggable-backend pattern
- `make_separator(cfg)` factory with lazy import of `demucs` so the optional dependency is never loaded at package import time
- `separation` section in `PipelineConfig` controlling model name, device (`cpu`/`cuda`), and enable flag
- `demucs` optional extra in `pyproject.toml` (already declared, not yet wired to a backend)
- Docker profile `with-demucs` for the image variant that includes the extra (~1 GB overhead)

The pipeline diagram will update to:
```
MIDI → audio synthesis → stem separation → transcription → evaluation vs. reference
```

## Additional datasets and instruments

**Status:** Piano, orchestral, chamber/orchestral, and drum corpora available; further
expansion not yet started.

`scripts/download_datasets.py` currently supports MAESTRO V3.0.0 (piano — `-midi`,
`-wav`, `-full` variants), BSED (orchestral — Beethoven symphony excerpts, MIDI + real
recordings), MusicNet (multi-instrument chamber/orchestral, MIDI + audio + per-note
label CSVs), and the Expanded Groove MIDI Dataset (drums — `-midi`/`-full`; download-only,
since transcription/evaluation here target pitched instruments, not drum-hit
classification). It also offers an interactive dataset picker (rich table) and parallel
downloads via `--jobs N`. Each entry can source from multiple URLs/archive formats
(zip and tar.gz) and route members to `midi/`, `recordings/`, or `metadata/` by
extension/prefix rules. MAPS was evaluated but dropped — it's gated behind a
registration form with no scriptable direct-download URL, incompatible with this
script's unattended-download model. Planned expansion covers more instruments,
multi-instrument datasets (e.g. Slakh2100), and automated download support for them.

## Additional transcription backends

**Status:** Not yet started.

Beyond Basic Pitch and the generic `external_command` backend, native integrations with other AMT systems (e.g. MT3, Omnizart) are under consideration.

## User-facing documentation site (Zensical)

**Status:** Not yet started.

Planned pages for a Zensical-generated documentation site (`zensical.toml` config exists):

- **Home (`index.md`)** — landing page with quick-start links
- **Tutorial (`tutorial.md`)** — step-by-step walkthrough of a full benchmark run
- **Project Structure (`project_structure.md`)** — directory layout and key file roles
- **CLI Reference (`cli_reference.md`)** — all commands, flags, and examples
- **API Reference (`api_reference.md`)** — Python API documentation for library use
- **Design Decisions (`design_decisions.md`)** — architectural rationale and trade-offs
- **Troubleshooting (`troubleshooting.md`)** — common issues and solutions

The nav section in `zensical.toml` is commented out until pages are written.

## Real-audio transcription mode

**Status:** Not yet started.

Sonitra currently transcribes only synthesised audio rendered from MIDI. A planned mode
would feed a dataset's original recordings directly into the transcription step, bypassing
the render pipeline — enabling benchmarking on real instrument audio, the canonical AMT
evaluation scenario in the literature. For MAESTRO V3.0.0 this means using the paired
`.flac` recordings rather than re-synthesising from MIDI.

Planned design:
- Pairing logic reads the dataset metadata CSV (e.g. `maestro-v3.0.0.csv`) to map each
  reference MIDI to its original `.flac`/`.wav` recording.
- The render step is skipped; transcription output is scored against the reference MIDI
  exactly as in the current workflow.
- Requires the full dataset audio download (`maestro-v3-full`/`-wav` via
  `scripts/download_datasets.py`, ~120 GB); the MIDI-only variant (`maestro-v3-midi`,
  ~57 MB) is sufficient for the synthesise-then-transcribe workflow.
- Distinct from the current pipeline, where rendering fidelity is itself part of what
  is being benchmarked.

BSED (see "Additional datasets and instruments" above) is a simpler first target for
this mode than MAESTRO: `scripts/download_datasets.py bsed` already fetches real
recordings into `corpus/bsed/recordings/` alongside `corpus/bsed/midi/`, and the two
share a common `BSED-<NN>_...` filename prefix — pairing a score to its recordings is
a filename match, no metadata CSV lookup required.

## Note-level performance-alignment annotations (BSED)

**Status:** Not yet started.

BSED ships manually-verified note-level alignments between each score and its real
recordings under `03_NoteAnnotations/Note-Level-Alignment/` (CSV + `.npz`), plus a
less-precise `Sequence-Alignment/` variant — neither is currently fetched by
`scripts/download_datasets.py`. Each maps a score note to its onset/offset in a
specific real recording, a more precise reference than the nominal MIDI-score timing
the evaluation pipeline uses today, since real performances deviate from the score
(rubato, systematic delays, etc.).

Planned use: once real-audio transcription mode (above) exists, evaluation could
optionally use the per-recording aligned annotation as ground truth instead of raw
score MIDI, giving a tighter accuracy measurement for orchestral transcription. This
would require extending the `bsed` entry's `extract_map` in
`scripts/download_datasets.py` with an `annotations/` target, and adding a reference
loader under `evaluation/` that consumes the CSV/npz alignment format instead of
parsing MIDI directly.

## Parallel benchmark conditions on GPU

**Status:** Blocked on TensorFlow GPU memory configuration; `benchmark.max_workers`
is pinned to 1 in every `config/benchmark/*.yaml` preset as a result.

`benchmark.max_workers` is the only knob that parallelises `sonitra benchmark`, and it
drives a `ProcessPoolExecutor` over conditions (`benchmark/runner.py`), not a thread
pool. Each worker is therefore a separate process with its own TensorFlow runtime.
Nothing in `src/` calls `tf.config.experimental.set_memory_growth`, so TensorFlow
falls back to its default of pre-allocating nearly all device memory on init: the
first process was observed claiming ~22 GB of a 24 GB RTX 3090, leaving a second
worker nothing. Combining `transcription.transcribers[].device: GPU:0` with
`benchmark.max_workers > 1` is expected to OOM, so the two are mutually exclusive
today.

`transcription.max_workers` and `evaluation.max_workers` are unaffected — both are
thread pools sharing one process (and so one GPU context), and both are already raised
above 1 in the benchmark presets. They are honoured by `sonitra transcribe` and
`sonitra evaluate` respectively, not by `sonitra benchmark`.

Planned design:
- Enable memory growth before the model loads in `transcribe/basic_pitch.py`, so each
  process caps at what it uses (the ICASSP 2022 model is small):
  ```python
  for gpu in tf.config.list_physical_devices("GPU"):
      tf.config.experimental.set_memory_growth(gpu, True)
  ```
  This must run before any GPU-touching TF call in the process, and raises
  `RuntimeError` if the device is already initialised — so it belongs at the top of
  the lazy-import block, guarded, not at module import time.
- Consider making it configurable rather than unconditional (e.g. a
  `gpu_memory_growth` flag on the transcription section), since memory growth trades
  some allocator efficiency for the ability to share the device.
- Once in place, raise `benchmark.max_workers` in the presets and measure — the win
  is bounded by how much of a condition's wall time is GPU inference versus
  rendering, which is serial for the `fluidsynth`/`dawdreamer_*` backends these
  presets use (see `pipeline.max_workers` note below).
- A cap on concurrent workers may still be needed: N processes each loading a
  TensorFlow runtime carries a host-RAM cost independent of GPU memory.

Related: `pipeline.max_workers` is only honoured for the `pedalboard_instrument`
synth backend (`pipeline.py`); `fluidsynth` and `dawdreamer_*` render serially
regardless, and `validate_worker_constraint()` force-resets the value to 1 for
DawDreamer because JUCE is not concurrency-safe. Parallel rendering for the
FluidSynth CLI backend is a separate opportunity.

## Vintage recording chains: additive noise and wow/flutter

**Status:** Not yet started. Full plan drafted at
`.local/notes/dev/20260810_exp_old_recording_config/PLAN_PHASE2.md`, pending
approval and several open decisions before implementation.

`config/benchmark/old_recording/vintage_scenarios.yaml` currently models three
vintage recording chains (78rpm shellac, early reel-to-reel tape, AM radio
broadcast) as a **stationary, subtractive/multiplicative** signal path —
bandwidth-limiting filters, presence/resonance coloration, saturation, and
dynamics, all built from `pedalboard` plugins. It explicitly does not model
additive noise (surface noise, tape hiss, mains hum) or wow/flutter (transport
speed instability), both plausibly larger drivers of AMT transcription failure
than anything the current config models — see that config's README for the
full framing.

`pedalboard.Plugin` is a pybind11-bound native class and `pedalboard.Pedalboard()`
is a homogeneous native container, so neither additive noise nor time-varying
effects can be added as new `EffectConfig`/`chain_builder` branches; they need
an entirely separate numpy/scipy-based processing pathway.

Planned design:
- New `src/sonitra/effects/vintage_effects.py` (config classes: `NoiseFloorConfig`,
  `CrackleConfig`, `HumConfig`, `WowFlutterConfig`, as a separate discriminated
  union from `EffectConfig`) and `vintage_chain.py` (`apply_vintage_effects`
  dispatch over plain numpy arrays).
- New `VintageEffectsSection` on `PipelineConfig` with `pre_chain`/`post_chain`
  lists and a `seed` field; addressable by the existing dotted-path benchmark
  override mechanism with no runner changes needed.
- Two new insertion points in `pipeline.py`'s shared `_render_file`: wow/flutter
  (variable-rate resampling) runs before the pedalboard effects chain; hiss/
  crackle/hum run after it and before post-normalisation. Both lists default
  to empty, a guaranteed no-op for every existing config.
- Deterministic per-file seeding (SHA256 of `f"{seed}:{midi_path.stem}"` →
  `np.random.default_rng`) so reproducibility and `benchmark.resume` stay sound.
- A new companion scenario file, `vintage_scenarios_full.yaml` (7 conditions:
  baseline + 3 eras × 2 severities, reusing the existing chains' overrides plus
  the new noise/wow effects), keeping the current file's 7-condition contract
  stable.
- Wow/flutter calibration against DIN 45507 (a weighted-RMS measurement, not a
  raw sinusoid depth) is its own sub-task, mirroring the filter-cascade
  calibration already done for the current bandwidth-limiting config.

Several parameters (crackle rate/amplitude, AM radio noise floor and hum
level, tape wow/flutter rates) have no citation yet and are flagged as open
research in the plan doc rather than given placeholder values — a possible
`REFERENCE_PHASE2.md` grounding pass would resolve these before scenario
numbers are finalized.
