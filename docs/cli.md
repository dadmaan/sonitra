# CLI reference

Use these commands to run Sonitra from your terminal. Each command needs a config file. You pass it with `--config FILE`.

```bash
sonitra init     --config FILE                                                              # write a starter config.yaml
sonitra render   --config FILE [--corpus DIR] [--output DIR] [--dataset NAME] [--workers N] [--limit N] [--seed N]
sonitra transcribe --config FILE [--audio DIR] [--output DIR] [--dataset NAME] [--transcriber NAME] [--limit N] [--seed N]
sonitra evaluate --config FILE [--reference DIR] [--estimate DIR] [--dataset NAME] [--output FILE] [--limit N] [--seed N]
sonitra benchmark --config FILE [--corpus DIR] [--workdir DIR] [--dataset NAME] [--limit N] [--seed N]
sonitra serve    --port 8000                                                                # start the FastAPI server
sonitra --version
```

`--limit N` picks N files at random so you can do a quick trial run. `--seed` sets the random starting point, so you get the same files each time. The default seed is 123. You can use both flags with `render`, `transcribe`, `evaluate`, and `benchmark`. The batch script (`scripts/run_transcribe_eval.py`) passes them to the render step.

The batch script can also run only some preset configs. Use `--config NAME [NAME …]` to name them. It will run all configs under `config/examples/` if you skip this flag. Use `--jobs N` to run N configs at the same time. The default is 1. Each config still runs its own render, transcribe, and evaluate steps one after another.

If you set `--dataset` on the command line, it replaces `io.dataset` from your config file. If you skip `--corpus`, `--audio`, `--reference`, or `--estimate`, Sonitra reads `io.corpus_root` and `io.dataset` from your config to find the paths.

`sonitra benchmark` saves three items in `work_dir` (see [Configuration → Benchmark output](configuration.md#benchmark-output)). It saves the results file in JSONL format, a `summary.json` file, and a `config.yaml` copy of the exact config it used. If you want to rebuild the output for a run you already did, for example after you updated Sonitra or your config, run the same command again with the same `--workdir`. This fills in `config.yaml` and per-row `overrides` again.

```bash
sonitra benchmark --config config/benchmark/old_recording/vintage_scenarios.yaml \
  --dataset maestro-v3 --workdir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT
```

`scripts/export_regression_table.py --work-dir DIR [--metadata-csv FILE --metadata-join-column NAME] [--split VALUE ...] [--split-column NAME]` turns a benchmark results file into a per-file CSV table. You can use that CSV for further analysis. You can also join it with dataset metadata (see `docs/datasets.md`). `--split` (repeatable) keeps only rows whose metadata split column matches one of the given values, for example `--split test`. It requires `--metadata-csv`, since the split labels come from there; rows with no metadata match are dropped and reported on stderr. `--split-column` names that column (default `split`, as in MAESTRO and GAPS). The output file then defaults to `regression_table_split-<values>.csv`, so it never overwrites the unfiltered table.

`scripts/run_mixed_effects_analysis.py --work-dir DIR [--input FILE] [--output-dir DIR] [--ref-level NAME] [--rscript PATH] [--covariate COL] [--covariate-transform NAME] [--covariate-divisor N] [--interact-with-condition] [--dry-run]` fits a beta mixed-effects model to that table. A mixed-effects model is a form of statistics that separates the effect of each test condition from the natural difficulty of each piece. Results go in `regression_analysis/` next to the table. You need R with `glmmTMB` installed (see [Statistical analysis](statistical-analysis.md)).

`scripts/export_model_baselines.py [--corpus-root DIR] [--doc FILE] [--runs PREFIXES] [--check | --stdout]` rewrites the measured-baselines table in `docs/model-cards.md` from benchmark runs under `corpus/` (default `--corpus-root`). `--runs` picks which run-name prefixes count, comma-separated, default `piano_only,guitar_only`. `--check` verifies the table is current without writing; `--stdout` prints the table instead of writing it. The two are mutually exclusive.

---
[← Back to README](../README.md)
