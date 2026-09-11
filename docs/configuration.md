# Configuration

`config/source.yaml` in the repo is the full reference config. It lists and explains every setting you can use. Copy it to start, or make a small starter file with:

```bash
sonitra init --config config.yaml
```

Sonitra checks your config with Pydantic, a Python tool that validates settings. If you add a key Sonitra does not know, it stops with an error.

These are the main sections and what each one does:

| Section | Controls |
|---|---|
| `render_pipeline` | Synth backend (`synth_backend`), effects chain (`effects_chain`), BPM, sample rate, bit depth, channels, parallelism (`max_workers`), input source (`input_type`: `midi` by default, or `audio` to use your own recordings instead of rendering, see [Using your own dataset](custom-datasets.md)) |
| `io` | `corpus_root` (base path), `dataset` (scopes all paths under `corpus_root/{dataset}/`), output format (`wav`, `flac`, `mp3`), file naming template |
| `dawdreamer` | Faust script path, VST3 plugin path, preset path — required when `synth_backend: dawdreamer_vst`; `plugin_path` must NOT be set for `synth_backend: dawdreamer_faust` |
| `fluidsynth` | `soundfont_path` — path to the `.sf2` SoundFont file; required when `synth_backend: fluidsynth` — plus optional `program` (0–127 GM program; `null` inherits the source MIDI's program when unambiguous) |
| `pedalboard` | Pedalboard effects chain (`pedalboard.effects`); `pedalboard.instrument` sub-section configures the VST3 instrument plugin for the `pedalboard_instrument` backend |
| `normalisation` | Peak or RMS normalisation, target dB, pre/post effects |
| `quality_gates` | Silence, clipping, and minimum duration checks |
| `transcription` | Transcriber backends, output directory, per-backend thresholds |
| `evaluation` | Metric families to enable, tolerance values |
| `benchmark` | Conditions, parameter sweeps, baseline name |
| `observability` | JSONL manifest, failed-file list, SSE event emission |

## Synthesis backend and effects chain

This section picks how Sonitra turns MIDI scores into sound, and whether it adds effects after. MIDI is a digital score format. A synth backend is the tool that plays the score. An effects chain is an optional set of audio effects applied after.

`render_pipeline.synth_backend`
| Value | Synth engine | Requires |
|---|---|---|
| `dawdreamer_faust` | DawDreamer + built-in Faust oscillator | — |
| `dawdreamer_vst` | DawDreamer + VST3 instrument | `dawdreamer.plugin_path` |
| `fluidsynth` | FluidSynth CLI + SoundFont | `fluidsynth.soundfont_path` (plus optional `fluidsynth.program` to force a GM program, otherwise inherited from the source MIDI when unambiguous) |
| `pedalboard_instrument` | Pedalboard VST3 instrument | `pedalboard.instrument.plugin_path` |

`render_pipeline.effects_chain`
| Value | Behaviour |
|---|---|
| `none` | No effects processing after synthesis |
| `pedalboard` | Apply the `pedalboard.effects` chain after synthesis |

## Transcription backends

A transcriber turns audio back into notes. AMT means automatic music transcription, turning sound into a score. Basic Pitch is Spotify's free transcription tool and the default. Set `device` to `GPU:0` to run it on your graphics card. You need the `[gpu]` install on Linux x86_64 for that.

| Backend | `type` value | Notes |
|---|---|---|
| Spotify Basic Pitch | `basic_pitch` | Installed by default; supports `device` field (default: `cpu`; set to `GPU:0` for GPU inference — requires `[gpu]` extras on Linux x86_64) |
| Pre-exported MIDI | `precomputed` | Point at a directory of MIDI from external tools |
| Any CLI tool | `external_command` | Template: `"tool transcribe {input} -o {output}"` |

## Built-in audio effects

You can add these effects under `pedalboard.effects`. Each effect has an `enabled` flag to turn it on or off. Pedalboard is the audio-effects library Sonitra uses.

Compressor, Reverb, Limiter, Chorus, Delay, Distortion, Gain, VST3 plugin, HighpassFilter, LowpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter. VST3 plugins start with their factory default sound. You cannot set their controls from YAML.

## Conditions and sweeps

A condition is one named test setup in a benchmark. A sweep changes one setting across several values to see how it affects results. `expand_conditions` builds the list in this order. It starts with the baseline. Then it adds one condition per `benchmark.conditions` entry. Then it adds one condition per sweep value. Sweeps change one setting at a time. They never combine with another sweep or with an explicit condition. So three sweeps with 2 values each give 1 + 2 + 2 + 2 = 7 conditions, not a full 2×2×2 mix. Each sweep value sets exactly one key, for example `{sweep.parameter: value}`. So a sweep cannot set two keys at once, such as a Reverb's `wet_level` and `dry_level`.

Use sweeps when you want to test one control while all else stays the same. For a full mix of settings, or any test that needs more than one change at once, write an explicit `benchmark.conditions` entry with the combined dotted-path changes.

### Benchmark output

`sonitra benchmark` saves three items in `work_dir`. It saves the per-file results in JSONL format. Each row covers one mix of condition, transcriber, and file, and it carries that condition's `overrides` list. It saves `summary.json` with the `summary` and `degradation` tables. Each row there also carries its `overrides`. It saves `config.yaml`, a copy of the exact `PipelineConfig` it used. Sonitra rewrites this copy on every run, even when you resume. `scripts/export_regression_table.py` flattens the results file into a per-file CSV for further analysis. It turns metrics and overrides into columns. It labels pedalboard effect slots by type when `config.yaml` is present. It can also join a dataset metadata CSV by filename (see `docs/datasets.md`).

## Parallelism (max_workers)

Four sections have a `max_workers` setting. It sets how many tasks run at once. Only two of them affect `sonitra benchmark`:

| Key | Used by `sonitra benchmark`? | What it parallelises |
|---|---|---|
| `benchmark.max_workers` | Yes | Conditions (process-level, one condition per subprocess) |
| `render_pipeline.max_workers` | Yes | Per-file rendering inside each condition |
| `transcription.max_workers` | No | Audio files per transcriber in standalone `sonitra transcribe` |
| `evaluation.max_workers` | No | Reference/estimate pairs in standalone `sonitra evaluate` |

### benchmark.max_workers

This sets how many benchmark conditions run at once. When it is more than 1, each condition runs in its own subprocess through a `ProcessPoolExecutor`. A subprocess is a separate worker process. The full render, transcribe, and evaluate chain for each condition runs in its own process. This also keeps JUCE, the audio code behind DawDreamer, separate per process. Each worker writes to `work_dir/logs/worker-<pid>.log`. When it is 1, which is the default, conditions run one after another in the main process. Each subprocess loads its own copy of the transcription model. So higher values use more memory. For example, you get one TensorFlow copy per worker with Basic Pitch.

### render_pipeline.max_workers

This sets how many files render at once inside each condition. It only works when `synth_backend` is `pedalboard_instrument`. All other backends render one file at a time. For `dawdreamer_faust` and `dawdreamer_vst` Sonitra forces it to 1, because DawDreamer and JUCE cannot safely share threads. Total render tasks at once equal `benchmark.max_workers × render_pipeline.max_workers` for `pedalboard_instrument`. You can change it per condition with a dotted-path override.

### transcription.max_workers and evaluation.max_workers

Sonitra reads these only for the single-step commands `sonitra transcribe` and `sonitra evaluate`. Inside `sonitra benchmark`, transcription and evaluation run one file at a time within each condition worker. These two settings change nothing in a benchmark run.

### Resume

Sonitra leaves all four `max_workers` keys out of the benchmark fingerprint. The fingerprint tracks whether your config changed. So you can change worker counts between runs and still resume.

---
[← Back to README](../README.md)
