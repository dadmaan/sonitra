# Configuration

`config/source.yaml` in the repository is the fully-annotated reference config documenting every available parameter. Copy it as a starting point, or generate a minimal starter config with:

```bash
sonitra init --config config.yaml
```

The config file is validated by Pydantic — unknown keys are hard errors.

Key sections and their purpose:

| Section | Controls |
|---|---|
| `render_pipeline` | Synth backend (`synth_backend`), effects chain (`effects_chain`), BPM, sample rate, bit depth, channels, parallelism (`max_workers`) |
| `io` | `corpus_root` (base path), `dataset` (scopes all paths under `corpus_root/{dataset}/`), output format (`wav`, `flac`, `mp3`), file naming template |
| `dawdreamer` | Faust script path, VST3 plugin path, preset path — required when `synth_backend: dawdreamer_vst`; `plugin_path` must NOT be set for `synth_backend: dawdreamer_faust` |
| `fluidsynth` | `soundfont_path` — path to the `.sf2` SoundFont file; required when `synth_backend: fluidsynth` |
| `pedalboard` | Pedalboard effects chain (`pedalboard.effects`); `pedalboard.instrument` sub-section configures the VST3 instrument plugin for the `pedalboard_instrument` backend |
| `normalisation` | Peak or RMS normalisation, target dB, pre/post effects |
| `quality_gates` | Silence, clipping, and minimum duration checks |
| `transcription` | Transcriber backends, output directory, per-backend thresholds |
| `evaluation` | Metric families to enable, tolerance values |
| `benchmark` | Conditions, parameter sweeps, baseline name |
| `observability` | JSONL manifest, failed-file list, SSE event emission |

## Synthesis backend and effects chain

**`render_pipeline.synth_backend`**

| Value | Synth engine | Requires |
|---|---|---|
| `dawdreamer_faust` | DawDreamer + built-in Faust oscillator | — |
| `dawdreamer_vst` | DawDreamer + VST3 instrument | `dawdreamer.plugin_path` |
| `fluidsynth` | FluidSynth CLI + SoundFont | `fluidsynth.soundfont_path` |
| `pedalboard_instrument` | Pedalboard VST3 instrument | `pedalboard.instrument.plugin_path` |

**`render_pipeline.effects_chain`**

| Value | Behaviour |
|---|---|
| `none` | No effects processing after synthesis |
| `pedalboard` | Apply the `pedalboard.effects` chain after synthesis |

## Transcription backends

| Backend | `type` value | Notes |
|---|---|---|
| Spotify Basic Pitch | `basic_pitch` | Installed by default; supports `device` field (default: `cpu`; set to `GPU:0` for GPU inference — requires `[gpu]` extras on Linux x86_64) |
| Pre-exported MIDI | `precomputed` | Point at a directory of MIDI from external tools |
| Any CLI tool | `external_command` | Template: `"tool transcribe {input} -o {output}"` |

## Built-in audio effects

Compressor, Reverb, Limiter, Chorus, Delay, Distortion, Gain, VST3 plugin, HighpassFilter, LowpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter. Each is a named entry under `pedalboard.effects` with an `enabled` flag. VST3 plugins are loaded at their factory default settings; parameters cannot be set from YAML.

## Conditions and sweeps

`expand_conditions` produces the baseline first, then one condition per `benchmark.conditions` entry, then one condition per sweep value. Sweeps are one-factor-at-a-time: each axis is expanded against the baseline independently and is never crossed with another sweep or with an explicit condition, so three 2-value sweeps give 1 + 2 + 2 + 2 = 7 conditions, not a 2×2×2 factorial. Each sweep value also carries exactly one override — `{sweep.parameter: value}` — so a sweep cannot set two keys together, such as a Reverb's `wet_level` and `dry_level`, or an effect's `enabled` flag alongside its `drive_db`.

Use sweeps for sensitivity checks that vary a single knob with everything else at baseline. For a full factorial, or any level that needs more than one override applied at once, write an explicit `benchmark.conditions` entry with the combined dotted-path overrides.

### Benchmark output

`sonitra benchmark` writes three things to `work_dir`: the per-(condition×transcriber×file) results JSONL (each record carries its condition's `overrides` dict), `summary.json` (the aggregate `summary` and `degradation` tables, each row also carrying its condition's `overrides` — not diffed in `degradation`, just passed through), and `config.yaml`, a snapshot of the fully-resolved `PipelineConfig` the run actually used (written fresh on every run, including resumes). `scripts/export_regression_table.py` flattens the results JSONL into a per-file CSV for regression analysis — metrics and overrides as columns, pedalboard effect slots labeled by type when `config.yaml` is present — and can optionally left-join a dataset's metadata CSV by filename (see `docs/datasets.md`).

## Parallelism (max_workers)

Four sections define a `max_workers` parameter. Only two of them affect `sonitra benchmark`:

| Key | Used by `sonitra benchmark`? | What it parallelises |
|---|---|---|
| `benchmark.max_workers` | Yes | Conditions (process-level, one condition per subprocess) |
| `render_pipeline.max_workers` | Yes | Per-file rendering inside each condition |
| `transcription.max_workers` | No | Audio files per transcriber in standalone `sonitra transcribe` |
| `evaluation.max_workers` | No | Reference/estimate pairs in standalone `sonitra evaluate` |

### benchmark.max_workers

Parallel benchmark conditions. When > 1, conditions run in a `ProcessPoolExecutor`. The full render → transcribe → evaluate chain for each condition runs in its own subprocess, which also provides JUCE isolation. Worker output is redirected to `work_dir/logs/worker-<pid>.log`. When 1 (default), conditions run serially in the parent process. Each subprocess loads its own transcriber/model instances, so raising this value increases memory usage proportionally (e.g. one TensorFlow copy per worker for Basic Pitch).

### render_pipeline.max_workers

Parallel per-file rendering inside each condition. Only effective when `synth_backend: pedalboard_instrument`; all other backends render serially. For `dawdreamer_faust` / `dawdreamer_vst` it is forced to 1 because DawDreamer/JUCE is not thread-safe. Effective render concurrency is `benchmark.max_workers × render_pipeline.max_workers` (for `pedalboard_instrument`). Conditions and sweeps can override it per-condition via dotted-path config overrides.

### transcription.max_workers and evaluation.max_workers

These are read only by the standalone `sonitra transcribe` and `sonitra evaluate` commands respectively. Inside `sonitra benchmark`, transcription and evaluation run serially per file within each condition worker. These two knobs have no effect on a benchmark run.

### Resume

All four `max_workers` keys are excluded from the benchmark config fingerprint, so changing them between runs does not invalidate `benchmark.resume`.

---
[← Back to README](../README.md)
