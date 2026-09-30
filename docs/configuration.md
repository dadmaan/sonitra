# Configuration

`config/source.yaml` in the repo is the full reference config. It lists and explains every setting you can use. Copy it to start, or make a small starter file with:

```bash
sonitra init --config config.yaml
```

Sonitra checks your config with Pydantic, a Python tool that validates settings. If you add a key Sonitra does not know, it stops with an error.

These are the main sections and what each one does:

| Section | Controls |
|---|---|
| `render_pipeline` | Synth backend (`synth_backend`), effects chain (`effects_chain`), host BPM (tempo-synced plugins / FluidSynth tick grid; notes follow each MIDI file's own tempo map), sample rate, bit depth, channels, parallelism (`max_workers`), input source (`input_type`: `midi` by default, or `audio` to use your own recordings instead of rendering, see [Using your own dataset](custom-datasets.md)) |
| `io` | `corpus_root` (base path), `dataset` (scopes all paths under `corpus_root/{dataset}/`), output format (`wav`, `flac`, `mp3`), file naming template, file selection (`metadata_csv`, `join_column`, `where`, `sample`; see [File selection](#file-selection)) |
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

A transcriber turns audio back into notes. AMT means automatic music transcription, turning sound into a score. Basic Pitch is Spotify's free transcription tool and the default. Set `device` to `cuda` (or `GPU:0` — both spellings work) to run it on your graphics card. Sonitra translates `cuda` to the TensorFlow name `GPU:0` internally. You need the `[gpu]` install on Linux x86_64 for that.

| Backend | `type` value | Notes |
|---|---|---|
| Spotify Basic Pitch | `basic_pitch` | Installed by default; supports unified `device` values `cpu`, `cuda`, `cuda:N`, `GPU:N` (default: `cpu`; `cuda` is translated to TensorFlow's `GPU:0` internally — requires `[gpu]` extras on Linux x86_64) |
| TransKun | `transkun` | Piano only; requires `pip install 'sonitra[transkun]'` (bundled 56 MB checkpoint, PyTorch). Supports `device` values `cpu`, `cuda`, `cuda:1`, `mps` and `GPU:0` (translated to `cuda:0` internally). See `config/source.yaml` commented block and `config/benchmark/transkun_baseline.yaml` |
| Pre-exported MIDI | `precomputed` | Point at a directory of MIDI from external tools |
| Any CLI tool | `external_command` | Template: `"tool transcribe {input} -o {output}"` |

Separation `device` accepts the same unified strings (`cpu`, `cuda`, `cuda:N`, `GPU:N`); `GPU:0` is translated to torch `cuda:0` at the separator boundary.

Numeric reproducibility is controlled per run by two `transcription` keys. Both stay in the benchmark fingerprint, so changing them invalidates resume. `numeric_mode` (`off` default, `warn`, `strict` — `strict` recommended for published runs) selects deterministic framework algorithms and disables TF32 precision shortcuts (TF32 is deterministic but less accurate than float32, about 10 versus 23 mantissa bits). `warn` falls back with a warning where a kernel has no deterministic implementation; `strict` raises instead. `gpu_memory_growth` (default `false`) lets TensorFlow allocate graphics memory as needed instead of grabbing it all at start; cuDNN algorithm choice can depend on workspace size, so it is treated as result-affecting. Both settings are process-global and reach benchmark workers through the environment; the effective `numeric_mode` is recorded per row in `transcriber_metadata`.

## Built-in audio effects

You can add these effects under `pedalboard.effects`. Each effect has an `enabled` flag to turn it on or off. Pedalboard is the audio-effects library Sonitra uses.

Compressor, Reverb, Limiter, Chorus, Delay, Distortion, Gain, VST3 plugin, HighpassFilter, LowpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter. VST3 plugins start with their factory default sound. You cannot set their controls from YAML.

## File selection

By default a run uses every file in its dataset. You can restrict it with four flat keys under `io`; `io.dataset` names the dataset they apply to.

- `io.metadata_csv`: the filename of a metadata CSV, with no folder part. Sonitra looks for it under `<io.corpus_root>/<dataset>/metadata/`, so `metadata_csv: maestro-v3.0.0.csv` finds `corpus/maestro-v3/metadata/maestro-v3.0.0.csv`. The exporter scripts use the opposite rule: their `--metadata-csv` accepts a path from the repository root, and the same path here points below the metadata folder, so Sonitra reports it as missing. Required with `where`, rejected without it.
- `io.join_column`: the metadata column matched against the reference MIDI file name (folders and the file ending are ignored). The default is `midi_filename`; GAPS uses `midi_path`.
- `io.where`: the exact-match filter. A file is kept when every listed column matches (AND), and a column matches when its row value is any entry in that column's list (OR). Sonitra compares values as text and turns whole numbers written in YAML into text. It sorts lists and removes repeats, so their order never matters. An unknown column or value is an error. `where` requires `metadata_csv` and `dataset`.
- `io.sample`: a random subset taken after filtering, in the form `{n: N, seed: S}`. `n` must be at least 1, and `n` greater than or equal to the filtered size keeps all files. The seed defaults to 0, so the same command picks the same files every time.

Sonitra excludes files on disk that have no metadata row and prints a warning naming a few examples. It pairs audio files to the full reference list and keeps them only when their reference is selected. `render` and `transcribe` pair only when `where` is set, so an unfiltered `--audio <dir>` still works; `benchmark` in audio mode always pairs recordings to references, filtered or not.

The CLI flags `--limit` and `--seed` write into `io.sample` (`--seed` alone needs an existing sample). A benchmark run records the values in its `config.yaml`, and they are part of the resume fingerprint (the config record that decides whether a stopped benchmark run can continue). Every selection error (a missing metadata CSV, an unknown column or value, or a filter that matches nothing) prints an error and exits with code 1 before any work starts. `--dataset` cannot re-point a filter: if the config filters one dataset and `--dataset` names a different one, Sonitra exits with an error.

Benchmark conditions and sweeps cannot override `io.dataset`, `io.metadata_csv`, `io.join_column`, `io.where`, `io.sample`, or `benchmark.benchmark_dir`, because the run reads those keys once before it starts.

For the MAESTRO test split:

```yaml
io:
  dataset: maestro-v3
  metadata_csv: maestro-v3.0.0.csv
  join_column: midi_filename
  where:
    split: [test]
  sample: null
```

## Conditions and sweeps

A condition is one named test setup in a benchmark. A sweep changes one setting across several values to see how it affects results. `expand_conditions` builds the list in this order. It starts with the baseline. Then it adds one condition per `benchmark.conditions` entry. Then it adds one condition per sweep value. Sweeps change one setting at a time. They never combine with another sweep or with an explicit condition. So three sweeps with 2 values each give 1 + 2 + 2 + 2 = 7 conditions, not a full 2×2×2 mix. Each sweep value sets exactly one key, for example `{sweep.parameter: value}`. So a sweep cannot set two keys at once, such as a Reverb's `wet_level` and `dry_level`.

Use sweeps when you want to test one control while all else stays the same. For a full mix of settings, or any test that needs more than one change at once, write an explicit `benchmark.conditions` entry with the combined dotted-path changes.

### Benchmark output

`sonitra benchmark` saves three items in `work_dir`. It saves the per-file results in JSONL format. Each row covers one mix of condition, transcriber, and file, and it carries that condition's `overrides` list. It saves `summary.json` with the `summary` and `degradation` tables. Each row there also carries its `overrides`. It saves `config.yaml`, a copy of the exact `PipelineConfig` it used. Sonitra rewrites this copy on every run, even when you resume. `scripts/export_regression_table.py` flattens the results file into a per-file CSV for further analysis. It turns metrics and overrides into columns. It labels pedalboard effect slots by type when `config.yaml` is present. It can also join a dataset metadata CSV by filename (see `docs/datasets.md`).

`benchmark.benchmark_dir` sets the full run folder. Sonitra uses it exactly as given, adds no config name to it, and keeps it out of the resume fingerprint, so you can move the folder and resume it from the new path. The run folder is the first match in this order: `--workdir`, `benchmark.benchmark_dir`, `<io.corpus_root>/<dataset>/benchmark/<config stem>` when a dataset is set (from the YAML or `--dataset`), and `./benchmark/<config stem>` otherwise. A `benchmark_dir` set in the config plus a different `--dataset` and no `--workdir` is an error. With `resume: false`, starting a run deletes the existing results in the folder, so two configs sharing one fixed folder delete each other's results. `scripts/export_model_baselines.py` does not pick up runs outside `corpus/*/benchmark/`.

`summary.json` always has a top-level `selection` block. It says whether a selection is configured (`configured`) and, for a configured selection, records the dataset, the metadata CSV path and its sha256 (a hash of the file's contents), the join column, the filter values, the sample, the type of files counted (reference MIDI files or recordings), the counts, per-value counts for each filter column, and a sha256 of the selected file paths. A run with no selection records `configured: false`, the file type, the number of files found and the number selected, and that path hash. On resume, Sonitra refuses to continue when the metadata CSV's sha256 differs from the one recorded when the run folder was started.

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
