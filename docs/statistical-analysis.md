# Statistical analysis

`scripts/run_mixed_effects_analysis.py` fits a mixed-effects regression to a benchmark run. A mixed-effects model is a form of statistics that splits results into fixed parts you test and random parts you adjust for.

A plain per-condition mean mixes two things. It mixes how good a test setup is with how hard its pieces were. This model pulls them apart.

| | |
|---|---|
| Response | `note.onset_f1` |
| Fixed effects | `condition`, `duration`, `performance_year` |
| Random intercepts | `song`, `composer` |
| Family | Beta, logit link (F1 is a proportion, bounded to `(0, 1)`) |

Here response means the score you predict, in this case note-onset F1. F1 is a score from 0 to 1 that blends missed notes and extra notes. Fixed effects are the factors you test, such as test setup, piece length, and year. Random intercepts are baselines the model gives to each song and composer to allow for easy and hard pieces. Beta with a logit link is the model type Sonitra uses because F1 is a share between 0 and 1.

```
note.onset_f1 ~ condition + duration + performance_year + (1 | song) + (1 | composer)
```

Read the line above as: predict note-onset F1 from condition, duration, and performance year, plus a separate baseline for each song and each composer.

The fit runs in R (`glmmTMB`), started by the Python script. A beta GLMM with crossed random effects has no match in Python's stats tools. GLMM means generalised linear mixed model. Crossed means songs and composers vary on their own. So the model lives in `scripts/mixed_effects_analysis.R`.

## Requirements

You need R with the `glmmTMB` and `jsonlite` packages. R is a language for statistics. `glmmTMB` fits the model. `jsonlite` reads JSON files.

```bash
# Debian / Ubuntu — prebuilt, compiles nothing
sudo apt-get install -y r-base-core r-cran-glmmtmb r-cran-jsonlite

# conda / micromamba
micromamba install -c conda-forge r-base r-glmmtmb r-jsonlite "r-tmb=1.9.19"

# macOS
brew install r && Rscript -e 'install.packages(c("glmmTMB","jsonlite"), repos="https://cloud.r-project.org")'
```

> Keep `TMB` matched to `glmmTMB`. TMB is the math engine behind `glmmTMB`. If the versions differ, you see a `glmmTMB was built with TMB package version X` warning and the fit can crash. Linux system packages (`r-cran-glmmtmb`) already match. Conda-forge needs the explicit `r-tmb` pin shown above.

The script finds `Rscript` on your `PATH`, which is the list of folders your system searches for programs. It also checks `$SONITRA_RSCRIPT` and `--rscript`, in that order. If R or a package is missing, it prints the install commands instead of a long error.

The Docker images include R by default (see [docker.md](docker.md)).

## Preparing the input

The model needs composer, duration, and performance year (`performance_year`, from `meta.year`). These live in the dataset notes, not in the benchmark results. So export the table with `--metadata-csv`:

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

Without `--metadata-csv` the table has no `meta.*` columns and the script stops with an explanation. See [datasets.md](datasets.md#joining-dataset-metadata-into-a-benchmark-export) for how the join works.

### Composition year (optional enrichment)

`meta.year` is MAESTRO's contest year, from 2004 to 2018. It marks the recording batch, not when the music was written. `misc/MAESTRO_comp_year.txt` holds an AI-compiled year when each work was finished. Use it when your question is about the age of the music, not the recording date. It spans 1612-2006 across 60 composers. 27 of 60 composers have works in more than one composition year. For the other 33 it never changes within a composer, so the main result mostly compares composers. Your true sample is closer to 60 than 8932.

Enrich the metadata first, then export against the enriched file. The export flags stay the same:

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

The regression table gains `meta.composition_year`. The script writes `<output>.provenance.json` next to the enriched CSV. That file logs input checksums, the exact command, and coverage. The annotation file uses `|` as a separator. Quoting is off by default because the comp-year file has unbalanced quotes. `--require-full-coverage` stops with an error if any row finds no match.

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

`--dry-run` is the cheap way to check the table before a long fit. It reports the design and stops.

## Composition year as a model term

The covariate flags work for any dataset (composition year is just the first use). A covariate is an extra column you add to the model. The CSV column stays `meta.composition_year`. In R it becomes `composition_year` because Sonitra strips a leading `meta.`. Contest year stays in the model as `performance_year` (from `meta.year`). The two years hardly track each other (r = -0.149), so you can use both.

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

`center-scale` uses a fixed divisor, not a standard deviation. The covariate is centred on its fitted-sample mean and divided by `--covariate-divisor`. The default is 100, so a composition-year result reads per century. A fixed divisor keeps results comparable across reruns and filtered sets. A standard deviation would shift each time.

The script fits a nested set (`base` -> `<covariate>` -> `<covariate>_x_condition`, the last only with `--interact-with-condition`) on one shared subset with full data for the largest model. Shared means logLik, AIC, and LRT stay comparable. LRT means likelihood-ratio test, a check of whether a larger model fits clearly better. AIC means Akaike information criterion, a score where lower is better. The saved centre matches the rows actually fitted. Every model runs with raised optimizer limits (`iter.max` and `eval.max` set to 10000). The comparison is a direct LRT chain, not a parsed `anova()` table.

On MAESTRO (8932 rows), once converged the `condition x composition_year` interaction is the result: LRT vs base `Chisq 215.30, 6 df, p < 2e-16`. Chisq is the test score, df is degrees of freedom (here the number of extra terms), and p is the chance the gap is noise. The main effect alone is small (`beta = -0.00082/yr, p = 0.025`; base -> main `Chisq ~= 4.68, 1 df, p ~= 0.031`; main -> interaction `Chisq` in the low hundreds, `6 df`, `p < 2e-16`). Convergence matters here. The interaction model does not converge with glmmTMB's defaults (AIC `-59344.33`). It reaches `conv: 0`, which means success, with the raised limits plus `/100` scaling (AIC `-59410.55`, 66 units apart).

Without a covariate the output is unchanged. With one it gains two items. Flat top-level files always describe the largest model fitted:

| File | Contents |
|---|---|
| `model_comparison.csv` | One row per model: `label, formula, n_obs, df, log_likelihood, aic, bic, lrt_chisq, lrt_df, lrt_p, converged, convergence_message, positive_definite_hessian` (LRT columns blank on the first row) |
| `models/<label>/` | Same four files per model (`model_summary.txt`, `fixed_effects.csv`, `random_effects_{song,composer}.csv`), including the primary |
| `model_meta.json` | Keeps every existing top-level key for the primary model and adds `primary_model`, `models` (per-model blocks), `covariate` (`column, variable, transform, center, divisor`), `n_complete`, and `n_dropped_covariate` |

## Output

Results go in `regression_analysis/`, next to the input CSV:

| File | Contents |
|---|---|
| `model_summary.txt` | R's `summary()` verbatim, plus composer random intercepts sorted ascending |
| `fixed_effects.csv` | One row per term: `estimate`, `std_error`, `statistic`, `p_value` |
| `random_effects_composer.csv` | Per-composer intercepts, sorted — which composers this system finds hard |
| `random_effects_song.csv` | Per-song intercepts |
| `model_meta.json` | Convergence status, AIC/BIC/logLik, variance components, R and package versions, input SHA-256, exact command |
| `fit.R` | Byte-identical copy of the R script that produced these numbers |

Between `model_meta.json` and `fit.R`, you can trace any number back to its source. You see the exact input, model, and package versions with no need for the rest of the repo.

## Things the script warns about

The script checks the table before fitting and flags what the model would else hide:

- More than one transcriber. A transcriber is a tool that turns audio into notes. The model has no transcriber term, so rows from different tools blend into one baseline. Filter the table to one transcriber first.
- Rows with `status != "succeeded"`. Sonitra fits them with the rest unless you remove them.
- Missing values in a model column. R drops those rows. The script reports the count.
- `note.onset_f1` at exactly 0 or 1. The beta family only allows values strictly between 0 and 1, so glmmTMB will fail. This happens with thin or broken transcriptions.


---
[← Back to README](../README.md)
