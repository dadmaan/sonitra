# CLI reference

Use these commands to run Sonitra from your terminal. `render`, `transcribe`, `evaluate` and `benchmark` each load a config file, and `--config` defaults to `config.yaml` for all four, `evaluate` included. Pass `--config FILE` to use another path.

```bash
sonitra init     --config FILE                                                              # write a starter config.yaml
sonitra render   --config FILE [--corpus DIR] [--output DIR] [--dataset NAME] [--workers N] [--limit N] [--seed N]
sonitra transcribe --config FILE [--audio DIR] [--output DIR] [--dataset NAME] [--transcriber NAME] [--limit N] [--seed N]
sonitra evaluate --config FILE [--reference DIR] [--estimate DIR] [--dataset NAME] [--output FILE] [--limit N] [--seed N]
sonitra benchmark --config FILE [--corpus DIR] [--workdir DIR] [--dataset NAME] [--limit N] [--seed N]
sonitra serve    --port 8000                                                                # start the FastAPI server
sonitra --version
```

`--limit N` picks N files at random so you can do a quick trial run. `--seed` sets the random starting point. By default the seed is 0, so repeated runs pick the same files. `--limit` and `--seed` write into `io.sample` (`--seed` alone needs an existing sample), and a benchmark run records the values in its `config.yaml` and includes them in the resume fingerprint (the config check that decides whether a stopped run can continue), so a `--limit` run started before this change cannot be resumed. `--limit 0` or a negative value is an error instead of a silent empty run. You can use both flags with `render`, `transcribe`, `evaluate`, and `benchmark`. The batch script (`scripts/run_transcribe_eval.py`) passes them to the render step.

All four run commands print one selection line before they start, for example `selection: maestro-v3 split=test -> 177/1276 files`. The two numbers are the files selected and the files found. A run with no dataset and no selection prints `selection: none -> N/N files`. A selection that cannot be worked out (a missing metadata CSV, an unknown column or value, or a filter that matches nothing) prints an error and exits with code 1 before any work starts.

The batch script can also run only some preset configs. Use `--config NAME [NAME …]` to name them. It will run all configs under `config/examples/` if you skip this flag. Use `--jobs N` to run N configs at the same time. The default is 1. Each config still runs its own render, transcribe, and evaluate steps one after another.

If you set `--dataset` on the command line, it replaces `io.dataset` from your config file. The exception is a config with a `where` filter: a `--dataset` that names a different dataset stops with an error, instead of applying the filter to another corpus. If you skip `--corpus`, `--audio`, `--reference`, or `--estimate`, Sonitra reads `io.corpus_root` and `io.dataset` from your config to find the paths. `benchmark`, `transcribe`, and `evaluate` take those paths from the dataset whenever one is set, whether it comes from `--dataset` or the YAML `io.dataset`.

`sonitra benchmark` saves three items in `work_dir` (see [Configuration → Benchmark output](configuration.md#benchmark-output)). It saves the results file in JSONL format, a `summary.json` file, and a `config.yaml` copy of the exact config it used. If you want to rebuild the output for a run you already did, for example after you updated Sonitra or your config, run the same command again with the same `--workdir`. This fills in `config.yaml` and per-row `overrides` again.

`--workdir` names the full run folder for that command. The config key `benchmark.benchmark_dir` sets the folder for that config, and `--workdir` wins when both are set. Without either, the folder is `corpus/{dataset}/benchmark/{config stem}` when a dataset is set, and `benchmark/{config stem}` otherwise.

```bash
sonitra benchmark --config config/benchmark/old_recording/vintage_scenarios.yaml \
  --dataset maestro-v3 --workdir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT
```

`scripts/export_regression_table.py --work-dir DIR [--metadata-csv FILE --metadata-join-column NAME] [--split VALUE ...] [--split-column NAME]` turns a benchmark results file into a per-file CSV table. You can use that CSV for further analysis. You can also join it with dataset metadata (see `docs/datasets.md`). `--split` (repeatable) keeps only rows whose metadata split column matches one of the given values, for example `--split test`. It requires `--metadata-csv`, since the split labels come from there; rows with no metadata match are dropped and reported on stderr. `--split-column` names that column (default `split`, as in MAESTRO and GAPS). The output file then defaults to `regression_table_split-<values>.csv`, so it never overwrites the unfiltered table.

`scripts/run_mixed_effects_analysis.py --work-dir DIR [--input FILE] [--output-dir DIR] [--ref-level NAME] [--rscript PATH] [--covariate COL] [--covariate-transform NAME] [--covariate-divisor N] [--interact-with-condition] [--dry-run]` fits a beta mixed-effects model to that table. A mixed-effects model is a form of statistics that separates the effect of each test condition from the natural difficulty of each piece. Results go in `regression_analysis/` next to the table. You need R with `glmmTMB` installed (see [Statistical analysis](statistical-analysis.md)).

`scripts/export_model_baselines.py [--corpus-root DIR] [--doc FILE] [--runs PREFIXES] [--check | --stdout]` rewrites the measured-baselines table in `docs/model-cards.md` from benchmark runs under `corpus/` (default `--corpus-root`). `--runs` picks which run-name prefixes count, comma-separated, default `piano_only,guitar_only`; a run with several transcribers gives one row each. `--check` verifies the table is current without writing; `--stdout` prints the table instead of writing it. The two are mutually exclusive.

`scripts/estimate_tuning.py --recordings DIR --references DIR --output FILE [--metadata FILE] [--join-column NAME] [--sample-rate HZ] [--resolution N] [--chunk-seconds SECONDS]` measures how far each recording sits from A440 in cents (a cent is a hundredth of a semitone) and writes two tables. `--recordings` is the directory of recordings to measure and `--references` the reference MIDI directory that groups them. `--output` names the per-reference CSV, one row per paired reference, and is the file to join onto dataset metadata: it carries the join column, `reference_stem`, `n_recordings`, `own_tuning_cents` (the median over that reference's measured recordings) and `own_tuning_spread_cents`. A per-recording CSV named from the same stem (`<stem>.recordings.csv`) is written beside it for inspection, with `recording`, `reference_stem`, `own_tuning_cents`, `duration_sec`, `status` and `error`. `--metadata` supplies the exact join-column value for each reference, and `--join-column` (default `midi_filename`) names its column; that value's stem is matched against the reference file stem. `--sample-rate` (default `22050`) is the rate every recording is decoded at, `--resolution` (default `0.01`, one cent) sets how finely tuning is measured, and `--chunk-seconds` (default `60.0`) analyses fixed blocks and pools their pitches, where `0` analyses each file in one piece. Every run also writes `<output>.provenance.json`, a record of where its numbers came from, beside the two tables. The script refuses to write an output that is, or lies inside, one of its input directories. The two commands run in order: measure first, then join the per-reference file onto the metadata with `scripts/enrich_metadata.py`: `python scripts/enrich_metadata.py --metadata corpus/guitarset/metadata/metadata.csv --annotations corpus/guitarset/metadata/guitarset-own-tuning.csv --on id=id --add own_tuning_cents=own_tuning_cents --output corpus/guitarset/metadata/metadata-with-tuning.csv`.

---
[← Back to README](../README.md)
