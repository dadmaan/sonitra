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

## How they differ from top-level `config/*.yaml` presets

| Preset (`config/*.yaml`) | Benchmark config (`config/benchmark/*.yaml`) |
|---|---|
| One fixed pipeline run | Many experimental conditions from one file |
| Run via `sonitra render` / `sonitra transcribe` / `sonitra evaluate` | Run via `sonitra benchmark` |
| Produces audio + transcriptions for one configuration | Produces a `summary.json` with condition-by-condition metric comparison |

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
            config/benchmark/effects_combinations.yaml; do
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
  transcriptions/<condition>/<transcriber>/   # MIDI transcriptions per condition
```

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

## Config-to-study mapping

| Config file | Acoustic factor | Conditions |
|---|---|---|
| `benchmark_test.yaml` | Smoke test (reverb) | 4 |
| `reverb_sweep.yaml` | Reverberation (wet level, room size) | 11 |
| `compression_sweep.yaml` | Dynamic-range compression (ratio, threshold) | 13 |
| `distortion_sweep.yaml` | Signal distortion (drive) | 9 |
| `effects_combinations.yaml` | Combinations of effects | 7 |
| `synthesis_backends.yaml` | Synthesis engine (FluidSynth, Faust, Vital) | 3 |

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
