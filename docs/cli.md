# CLI reference

```bash
sonitra init     --config FILE                                                              # write a starter config.yaml
sonitra render   --config FILE [--corpus DIR] [--output DIR] [--dataset NAME] [--workers N] [--limit N] [--seed N]
sonitra transcribe --config FILE [--audio DIR] [--output DIR] [--dataset NAME] [--transcriber NAME] [--limit N] [--seed N]
sonitra evaluate --config FILE [--reference DIR] [--estimate DIR] [--dataset NAME] [--output FILE] [--limit N] [--seed N]
sonitra benchmark --config FILE [--corpus DIR] [--workdir DIR] [--dataset NAME] [--limit N] [--seed N]
sonitra serve    --port 8000                                                                # start the FastAPI server
sonitra --version
```

`--limit N` selects a reproducible random subset of N files (useful for smoke testing). `--seed` controls the random draw (default: 123). The flags are available on all four commands (`render`, `transcribe`, `evaluate`, `benchmark`); the batch runner (`scripts/run_transcribe_eval.py`) forwards them to the render step.

The batch runner additionally accepts `--config NAME [NAME …]` to run only the named preset configs instead of all configs under `config/examples/`, and `--jobs N` (default: 1) to process N configs in parallel (each config's render→transcribe→evaluate steps still run serially within the worker).

When `--dataset` is set on the CLI it overrides `io.dataset` from the config file. When `--corpus`/`--audio`/`--reference`/`--estimate` are omitted, the paths are resolved from `io.corpus_root` and `io.dataset` in the config.

`sonitra benchmark` writes its results JSONL, `summary.json`, and a `config.yaml` snapshot of the resolved config it ran with to `work_dir` (see [Configuration → Benchmark output](configuration.md#benchmark-output)). To re-generate an existing benchmark's output in place (e.g. after a `sonitra`/config upgrade, so `config.yaml` and per-row `overrides` get (re)populated), re-run the same command with the same `--workdir`:

```bash
sonitra benchmark --config config/benchmark/old_recording/vintage_scenarios.yaml \
  --dataset maestro-v3 --workdir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT
```

`scripts/export_regression_table.py --work-dir DIR [--metadata-csv FILE --metadata-join-column NAME]` turns a benchmark's results JSONL into a per-file regression-ready CSV, optionally joined with a downloaded dataset's metadata (see `docs/datasets.md`).

---
[← Back to README](../README.md)
