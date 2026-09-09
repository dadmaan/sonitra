# Statistical analysis

`scripts/run_mixed_effects_analysis.py` fits a mixed-effects regression to a benchmark run.

A plain per-condition mean confounds the condition's effect with whatever mix of pieces happened to be in the corpus. This model separates them.

| | |
|---|---|
| Response | `note.onset_f1` |
| Fixed effects | `condition`, `duration`, `performance_year` |
| Random intercepts | `song`, `composer` |
| Family | Beta, logit link (F1 is a proportion, bounded to `(0, 1)`) |

```
note.onset_f1 ~ condition + duration + performance_year + (1 | song) + (1 | composer)
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

The model needs composer, duration, and performance year (`performance_year`, from `meta.year`), which live in the dataset's metadata rather than in the benchmark results. So export the regression table **with** `--metadata-csv`:

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

### Composition year (optional enrichment)

`meta.year` is MAESTRO's competition year (2004–2018). It indexes the recording batch, not the music. `misc/MAESTRO_comp_year.txt` carries an AI-compiled composition (finalization) year per work. Use it when the question is about the age of the music rather than the recording session. It spans 1612–2006 across 60 composers. 27 of 60 composers have works in more than one composition year; for the remaining 33 it is a composer-level constant, so the main effect is estimated largely between composers — effective N is nearer 60 than 8932.

Enrich the metadata first, then export against the enriched file. The export flags are unchanged:

```bash
python scripts/enrich_metadata.py \
  --metadata corpus/maestro-v3/metadata/maestro-v3.0.0.csv \
  --annotations misc/MAESTRO_comp_year.txt --annotations-delimiter '|' \
  --on canonical_composer=Composer --on canonical_title=Piece \
  --add Year=composition_year \
  --output corpus/maestro-v3/metadata/maestro-v3.0.0-with-composition-year.csv

python scripts/export_regression_table.py \
  --work-dir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI \
  --metadata-csv corpus/maestro-v3/metadata/maestro-v3.0.0-with-composition-year.csv \
  --metadata-join-column midi_filename \
  --output corpus/maestro-v3/benchmark/vintage_scenarios_MIDI/regression_table_with_metadata_comp_year.csv
```

The regression table gains `meta.composition_year`. The script writes `<output>.provenance.json` (input SHA-256s, argv, coverage) alongside the enriched CSV. Annotation quoting is disabled by default (the comp-year file has unbalanced quotes); `--require-full-coverage` exits non-zero on any unmatched row.

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
| `--covariate COL` | CSV column to add as a fixed effect, e.g. `meta.composition_year` (generic; composition year is the first use) |
| `--covariate-transform NAME` | `none` \| `center` \| `center-scale` (default: `center-scale`) |
| `--covariate-divisor N` | Fixed divisor for `center-scale` (default: `100`) |
| `--interact-with-condition` | Also fit `condition * <covariate>` on top of the main-effect model |
| `--dry-run` | Report the design and stop, without fitting |

`--dry-run` is the cheap way to confirm the table is what you think it is before committing to a fit:

## Composition year as a model term

The covariate interface is generic and dataset-agnostic — composition year is the first use. The CSV column stays `meta.composition_year`; the R-side variable strips a leading `meta.`, so `meta.composition_year` becomes `composition_year` in the formula. Competition year stays in the model as `performance_year` (from `meta.year`): the two are not collinear (r = -0.149).

```bash
python scripts/run_mixed_effects_analysis.py \
  --input corpus/maestro-v3/benchmark/vintage_scenarios_MIDI/regression_table_with_metadata_comp_year.csv \
  --covariate meta.composition_year --interact-with-condition
```

| Flag | Default | Meaning |
|---|---|---|
| `--covariate NAME` | none | CSV column to add, e.g. `meta.composition_year` |
| `--covariate-transform` | `center-scale` | `none` \| `center` \| `center-scale` |
| `--covariate-divisor N` | `100` | Fixed divisor for `center-scale` |
| `--interact-with-condition` | off | Also fit `condition * <covariate>` |

`center-scale` is a fixed divisor, not an SD. The covariate is centred on its fitted-sample mean and divided by `--covariate-divisor` (default 100, so a composition-year estimate reads per century). A fixed divisor keeps estimates comparable across reruns and filtered subsets, where an SD would drift.

The script fits a nested set — `base` -> `<covariate>` -> `<covariate>_x_condition` (the last only with `--interact-with-condition`) — on a single complete-case subset over the largest model's variables, so logLik/AIC/LRT are comparable and the recorded centre matches the data actually fitted. Every model is fitted with raised optimizer limits (`iter.max`/`eval.max` 10000); the comparison is a direct LRT chain, not a parsed `anova()` frame.

On MAESTRO (8932 rows), once converged the `condition x composition_year` interaction is the result: LRT vs base `Chisq 215.30, 6 df, p < 2e-16`. The main effect alone is marginal (`beta = -0.00082/yr, p = 0.025`; base -> main `Chisq ~= 4.68, 1 df, p ~= 0.031`; main -> interaction `Chisq` in the low hundreds, `6 df`, `p < 2e-16`). Convergence matters here: the interaction model does not converge under glmmTMB's default control (AIC `-59344.33`) and reaches `conv: 0` with the raised limits plus `/100` scaling (AIC `-59410.55`, 66 units apart).

With a covariate the output gains two things; without one it is unchanged. Flat top-level artifacts always describe the largest model fitted:

| File | Contents |
|---|---|
| `model_comparison.csv` | One row per model: `label, formula, n_obs, df, log_likelihood, aic, bic, lrt_chisq, lrt_df, lrt_p, converged, convergence_message, positive_definite_hessian` (LRT columns blank on the first row) |
| `models/<label>/` | Same four files per model (`model_summary.txt`, `fixed_effects.csv`, `random_effects_{song,composer}.csv`), including the primary |
| `model_meta.json` | Keeps every existing top-level key for the primary model and adds `primary_model`, `models` (per-model blocks), `covariate` (`column, variable, transform, center, divisor`), `n_complete`, and `n_dropped_covariate` |

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
