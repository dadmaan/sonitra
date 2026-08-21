# Statistical analysis

`scripts/run_mixed_effects_analysis.py` fits a mixed-effects regression to a benchmark run.

A plain per-condition mean confounds the condition's effect with whatever mix of pieces happened to be in the corpus. This model separates them.

| | |
|---|---|
| Response | `note.onset_f1` |
| Fixed effects | `condition`, `duration`, `year` |
| Random intercepts | `song`, `composer` |
| Family | Beta, logit link (F1 is a proportion, bounded to `(0, 1)`) |

```
note.onset_f1 ~ condition + duration + year + (1 | song) + (1 | composer)
```

The fit runs in **R** (`glmmTMB`), driven by the Python script. A beta GLMM with crossed random effects has no equivalent in the Python statistics stack, so the model lives in `scripts/mixed_effects_analysis.R`.

## Requirements

R with the `glmmTMB` and `jsonlite` packages.

```bash
# Debian / Ubuntu — prebuilt, compiles nothing
sudo apt-get install -y r-base-core r-cran-glmmtmb r-cran-jsonlite

# conda / micromamba
micromamba install -c conda-forge r-base r-glmmtmb r-jsonlite "r-tmb=1.9.19"

# macOS
brew install r && Rscript -e 'install.packages(c("glmmTMB","jsonlite"), repos="https://cloud.r-project.org")'
```

> **Pin `TMB` to the version `glmmTMB` was built against.** A mismatch produces a `glmmTMB was built with TMB package version X` warning and can segfault mid-fit. Distribution packages (`r-cran-glmmtmb`) are already matched; conda-forge needs the explicit `r-tmb` pin shown above.

The script finds `Rscript` on `PATH`, or via `$SONITRA_RSCRIPT`, or via `--rscript`. If R or a package is missing it exits with the install commands rather than a stack trace.

The Docker images install R by default — see [docker.md](docker.md).

## Preparing the input

The model needs composer, duration, and year, which live in the dataset's metadata rather than in the benchmark results. So export the regression table **with** `--metadata-csv`:

```bash
# 1. Run the benchmark
sonitra benchmark --config config/benchmark/old_recording/vintage_scenarios.yaml \
  --dataset maestro-v3 --workdir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT

# 2. Export the regression table, joined with dataset metadata
python scripts/export_regression_table.py \
  --work-dir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT \
  --metadata-csv corpus/maestro-v3/metadata/maestro-v3.0.0.csv \
  --metadata-join-column midi_filename \
  --output corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT/regression_table_with_metadata.csv
```

Without `--metadata-csv` the table has no `meta.*` columns and the script stops with an explanation. See [datasets.md](datasets.md#joining-dataset-metadata-into-a-benchmark-export) for the join.

## Running it

```bash
# Point at the benchmark directory (finds regression_table_with_metadata.csv)
python scripts/run_mixed_effects_analysis.py \
  --work-dir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT

# Or name the CSV directly
python scripts/run_mixed_effects_analysis.py --input path/to/table.csv
```

| Flag | Purpose |
|---|---|
| `--work-dir DIR` | Benchmark directory containing `regression_table_with_metadata.csv` |
| `--input FILE` | The regression table CSV, named directly |
| `--output-dir DIR` | Override the output location (default: `regression_analysis/` beside the input) |
| `--ref-level NAME` | Condition every other condition is compared against (default: `baseline`) |
| `--rscript PATH` | Rscript interpreter to use (default: `$SONITRA_RSCRIPT`, then `PATH`) |
| `--dry-run` | Report the design and stop, without fitting |

`--dry-run` is the cheap way to confirm the table is what you think it is before committing to a fit:

## Output

Results land in `regression_analysis/`, beside the input CSV:

| File | Contents |
|---|---|
| `model_summary.txt` | R's `summary()` verbatim, plus composer random intercepts sorted ascending |
| `fixed_effects.csv` | One row per term: `estimate`, `std_error`, `statistic`, `p_value` |
| `random_effects_composer.csv` | Per-composer intercepts, sorted — which composers this system finds hard |
| `random_effects_song.csv` | Per-song intercepts |
| `model_meta.json` | Convergence status, AIC/BIC/logLik, variance components, R and package versions, input SHA-256, exact command |
| `fit.R` | Byte-identical copy of the R script that produced these numbers |

Between `model_meta.json` and `fit.R`, a result is reproducible without the surrounding repository state: you can tell exactly which input, which model, and which package versions produced any given number.

## Things the script warns about

It inspects the table before fitting and flags what the model would otherwise absorb silently:

- **More than one transcriber.** The model has no transcriber term, so rows from different systems collapse into a single intercept. Filter the table per transcriber first.
- **Rows with `status != "succeeded"`.** Fitted alongside the rest unless you remove them.
- **Missing values** in a model column — R drops those rows; the count is reported.
- **`note.onset_f1` at exactly 0 or 1.** The beta family is defined on the *open* interval, so glmmTMB will fail. Happens with sparse or degenerate transcriptions.


---
[← Back to README](../README.md)
