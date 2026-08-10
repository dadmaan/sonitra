# Benchmark Configs

This directory contains benchmark configuration files for Sonitra's AMT evaluation
pipeline. Each file is a complete `PipelineConfig` YAML with a populated `benchmark:`
section that drives the `sonitra benchmark` command.

---

## What are benchmark configs?

A benchmark config is a full `PipelineConfig` file — it specifies synthesis backend,
effects chain, transcription settings, and evaluation metrics — extended with a
`benchmark:` section that declares a set of experimental *conditions*.

Each condition is a named set of dotted-path config overrides applied on top of the
base config. Running `sonitra benchmark` renders the corpus once per condition,
transcribes every audio file with each enabled transcriber, scores the transcriptions
against the source MIDI, and writes a per-record JSONL file plus a `summary.json`
with aggregate metrics and a degradation table (metric deltas vs the baseline
condition).

---

## How they differ from top-level `config/examples/*.yaml` presets

| Preset (`config/examples/*.yaml`) | Benchmark config (`config/benchmark/*.yaml`) |
|---|---|
| One fixed pipeline run | Many experimental conditions from one file |
| Run via `sonitra render` / `sonitra transcribe` / `sonitra evaluate` | Run via `sonitra benchmark` |
| Produces audio + transcriptions for one configuration | Produces a `summary.json` with condition-by-condition metric comparison |

---

## Grounded scenario studies vs. parameter sweeps

This directory now holds two different kinds of study:

- **Abstract single-axis sweeps** — `reverb_sweep.yaml`, `compression_sweep.yaml`,
  `distortion_sweep.yaml`, `effects_combinations.yaml`. Each varies one (or a small
  combination of) `pedalboard` parameter(s) across an arbitrary, evenly-spaced grid of
  values. They answer "how does degrading this parameter change transcription quality,"
  not "what does a real-world condition sound like."
- **Grounded real-world scenario studies** — `old_recording/`, `telephone_channel/`,
  `venue_acoustics/`, `rotary_speaker/`. Each lives in its own subdirectory alongside a
  scenario-specific YAML config and a `README.md` carrying citations, calibration
  tables (measured, not assumed), and an explicit confound disclosure. These configs
  model a specific real-world signal path or acoustic space — a vintage recording
  chain, a voice-channel bandwidth, a reverberant venue, a Leslie rotary speaker — as
  closely as this pipeline's effect registry allows.

**Read the scenario's own `README.md` before drawing any conclusion from a run of a
grounded scenario config.** Each one documents what is and isn't modelled, the
calibration methodology behind its numeric parameters, and confounds (e.g. how
`normalisation.pre_effects` and post-effects peak normalisation interact with the
effect chain — verify this against your own run's actual output levels rather than
assuming the README's stated intent, since `pre_effects` is a mutually exclusive
pre-XOR-post switch, not an additive one) that apply to every result it produces.

---

## Quick start

**Smoke test** (fast, 4 conditions, 2 files):

```bash
sonitra benchmark \
  --config config/benchmark/benchmark_test.yaml \
  --dataset test \
  --limit 2
```

**Full reverb study** (11 conditions, full corpus):

```bash
sonitra benchmark \
  --config config/benchmark/reverb_sweep.yaml \
  --dataset <your-dataset-name>
```

**All files in sequence** (results go to `corpus/<dataset>/benchmark/`):

```bash
for cfg in config/benchmark/reverb_sweep.yaml \
            config/benchmark/compression_sweep.yaml \
            config/benchmark/distortion_sweep.yaml \
            config/benchmark/effects_combinations.yaml \
            config/benchmark/old_recording/vintage_scenarios.yaml \
            config/benchmark/telephone_channel/telephone_scenarios.yaml \
            config/benchmark/venue_acoustics/venue_scenarios.yaml \
            config/benchmark/rotary_speaker/rotary_scenarios.yaml; do
  sonitra benchmark --config "$cfg" --dataset my_study
done
```

---

## Anatomy of the `benchmark:` block

```yaml
benchmark:
  results_path: benchmark_results.jsonl   # per-record JSONL output filename
  include_baseline: true                  # prepend a no-override baseline condition
  baseline_name: baseline                 # name for the baseline condition
  max_workers: 1                          # conditions run sequentially (see worker notes)
  save_audio: true                        # false = delete a condition's audio/stems once it's transcribed+evaluated
  resume: false                           # true = continue a stopped run, skipping already-recorded work

  conditions:
    - name: no_reverb                     # human-readable condition name
      overrides:
        pedalboard.effects.1.enabled: false   # dotted path into the config tree

  sweeps:
    - parameter: pedalboard.effects.1.wet_level   # config key to vary
      values: [0.0, 0.3, 0.6, 0.9]               # one condition per value
      name: wet_level                             # optional axis label (default: last segment)
```

### Dotted-path overrides

Paths address nested sections and indexed list elements:

- `pipeline.sample_rate` — top-level section field
- `pedalboard.effects.1.wet_level` — second element of the `effects` list, `wet_level` key
- `pedalboard.effects.0.enabled` — boolean toggle on the first effect

Paths must resolve to existing keys. Unknown keys raise a `KeyError` at validation time.

### Sweep condition naming

Each sweep value becomes a condition named `<axis>=<value>`. The axis is `sweep.name`
when set, otherwise the last dotted segment of `sweep.parameter`. Examples:

- `parameter: pedalboard.effects.1.wet_level` → axis `wet_level` → `wet_level=0.3`
- `parameter: pedalboard.effects.0.threshold_db`, `name: threshold` → `threshold=-18.0`

---

## Worker notes

- `pipeline.max_workers` controls parallelism within a single condition's render
  pipeline. **Must stay at `1` for DawDreamer modes** (DawDreamer/JUCE global state
  is not thread-safe). FluidSynth and Pedalboard modes can use higher values, but all
  configs in this directory default to `1` for safety.
- `benchmark.max_workers` controls how many conditions run in parallel subprocesses.
  Each subprocess gets its own JUCE instance. Can be increased to speed up large
  sweeps on multi-core machines. Defaults to `1` in all configs here.

---

## Expected outputs

After `sonitra benchmark --config <file> --dataset <name>`:

```
corpus/<name>/benchmark/
  benchmark_results.jsonl   # one JSON record per (condition × transcriber × file)
  summary.json              # aggregate means + degradation-vs-baseline table
  audio/<condition>/        # rendered WAV per condition
  stems/<condition>/        # separated stems per condition (only if separation.enabled)
  transcriptions/<condition>/<transcriber>/   # MIDI transcriptions per condition
```

`benchmark_results.jsonl`, `summary.json`, and `transcriptions/<condition>/` are always
kept. `audio/<condition>/` and `stems/<condition>/` are only kept when
`benchmark.save_audio: true` (the default); with `save_audio: false`, each condition's
audio/stems are deleted right after that condition's transcription and evaluation
finish, before the next condition starts — this bounds peak disk usage to roughly one
condition's worth of rendered audio instead of the whole sweep, which matters for large
corpora or configs with many conditions.

`summary.json` structure:

```json
{
  "summary": [
    {"condition": "baseline", "note.f1": 0.82, ...},
    {"condition": "wet_level=0.3", "note.f1": 0.79, ...}
  ],
  "degradation": [
    {"condition": "wet_level=0.3", "note.f1": -0.03, ...}
  ]
}
```

`NaN` values appear when a metric is undefined (e.g. correlation over too few matched
notes). They are preserved as `null` in JSON and skipped during aggregation.

---

## Resuming a stopped run

Set `benchmark.resume: true` to continue a run that was interrupted (crash, `Ctrl-C`,
or ran out of disk) instead of starting over. On the next invocation with the same
`work_dir`, Sonitra reads `benchmark_results.jsonl`, treats every
`(condition, file, transcriber)` triple already recorded — `succeeded`, `failed`, or
`render_failed` — as done, and only computes what's missing. A condition whose records
are all already present is skipped entirely, with no re-render.

This works fine together with `save_audio: false`: a condition that finished before the
interruption already had its audio cleaned up, and resume skips it without needing that
audio back. A condition that was only partially finished still has its audio on disk
(cleanup only runs after a condition fully completes), so the corpus for it is
re-rendered for the remaining work, same as a normal run.

Resume checks that the config hasn't changed since the run it's continuing — a
fingerprint is stored next to `benchmark_results.jsonl` and compared on every resume,
so a config edit that would change what a condition or record means (transcribers,
conditions/sweeps, synth/effects settings, evaluation parameters) raises an error
instead of silently mixing results computed under two different configs. Leave
`resume: false` (the default) to always start clean; with `resume: true`, note that
`benchmark.max_workers`, `benchmark.save_audio`, and the various `max_workers` knobs
are excluded from the fingerprint since they don't affect result semantics.

---

## Config-to-study mapping

| Config file | Acoustic factor | Conditions |
|---|---|---|
| `benchmark_test.yaml` | Smoke test (reverb) | 4 |
| `reverb_sweep.yaml` | Reverberation (wet level, room size) | 11 |
| `compression_sweep.yaml` | Dynamic-range compression (ratio, threshold) | 13 |
| `distortion_sweep.yaml` | Signal distortion (drive) | 9 |
| `effects_combinations.yaml` | Combinations of effects | 7 |
| `synthesis_backends.yaml` | Synthesis engine (FluidSynth, Faust, Vital) | 3 |
| `old_recording/vintage_scenarios.yaml` | Vintage recording chains (bandwidth + dynamics) | 7 |
| `telephone_channel/telephone_scenarios.yaml` | Voice-channel bandwidth + AGC (VoIP wideband, PSTN narrowband, intercom) | 4 |
| `venue_acoustics/venue_scenarios.yaml` | Room acoustics (RT60-calibrated: studio, recital hall, symphony hall, cathedral) | 5 |
| `rotary_speaker/rotary_scenarios.yaml` | Leslie rotary speaker character (chorale/tremolo) | 3 |

`old_recording/vintage_scenarios.yaml` is a **phase-1 bandwidth-and-dynamics
ablation** for three vintage recording chains (78rpm shellac, early tape, AM
radio), not a full vintage-audio simulation — surface noise, hiss, hum, and
wow/flutter are deferred to a later phase. See
`old_recording/README.md` for the full grounding, measured calibration
tables, and interpretation constraints before drawing conclusions from it.

`telephone_channel/telephone_scenarios.yaml`, `venue_acoustics/venue_scenarios.yaml`,
and `rotary_speaker/rotary_scenarios.yaml` follow the same grounded-scenario
methodology as `old_recording` — each has its own `README.md` with citations,
a measured calibration table (filter cascade -3 dB points, RT60-calibrated
`Reverb` parameters, or Leslie rotation-rate figures, respectively), and a
confound disclosure. `rotary_speaker` in particular carries a larger
"stylized approximation" caveat than the other three — see its README's
Section 1 before treating a result from it as evidence about real Leslie
processing.

---

## Adding a new benchmark config

1. Copy one of the existing files as a starting point.
2. Edit the `pedalboard.effects` chain to match the effect you want to vary.
3. Add conditions and/or sweeps to the `benchmark:` block.
4. Validate it loads cleanly:
   ```python
   from sonitra.config import load_config
   from sonitra.benchmark.conditions import expand_conditions
   cfg = load_config("config/benchmark/my_new_benchmark.yaml")
   print(expand_conditions(cfg.benchmark))
   ```
5. Run the smoke test to confirm end-to-end execution:
   ```bash
   sonitra benchmark --config config/benchmark/my_new_benchmark.yaml --dataset test --limit 2
   ```
