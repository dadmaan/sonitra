#!/usr/bin/env Rscript
#
# Fit the SONITRA condition-effect mixed model and write its results as files.
#
#   note.onset_f1 ~ condition + duration + year + (1 | song) + (1 | composer)
#   family = beta_family(link = "logit")
#
# The model specification is deliberately verbatim from
# misc/SONITRA-mixed-effects-regresion-model.R -- this script adds file output
# and provenance around that fit, it does not change the statistics. Known
# limitations of the specification are catalogued in
# .local/notes/analysis/20260821_R_regression_model/comments.md.
#
# Usage:
#   Rscript mixed_effects_analysis.R --input TABLE.csv --output-dir DIR \
#       [--ref-level baseline]
#
# Normally invoked through scripts/run_mixed_effects_analysis.py, which
# validates the input and reports results; this script is standalone and can be
# run by hand or by a collaborator without Python.
#
# Requires: glmmTMB, jsonlite.

options(warn = 1)

# ---------------------------------------------------------------------------
# arguments (base R only, to keep the dependency list at glmmTMB + jsonlite)
# ---------------------------------------------------------------------------

parse_args <- function(argv) {
  defaults <- list(input = NA_character_, `output-dir` = NA_character_, `ref-level` = "baseline")
  index <- 1
  while (index <= length(argv)) {
    key <- sub("^--", "", argv[[index]])
    if (identical(key, argv[[index]])) {
      stop("unexpected positional argument: ", argv[[index]], call. = FALSE)
    }
    if (!key %in% names(defaults)) {
      stop("unknown argument: --", key, call. = FALSE)
    }
    if (index + 1 > length(argv)) {
      stop("--", key, " requires a value", call. = FALSE)
    }
    defaults[[key]] <- argv[[index + 1]]
    index <- index + 2
  }
  for (required in c("input", "output-dir")) {
    if (is.na(defaults[[required]])) {
      stop("--", required, " is required", call. = FALSE)
    }
  }
  defaults
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
input_path <- args$input
output_dir <- args$`output-dir`
ref_level <- args$`ref-level`

if (!file.exists(input_path)) {
  stop("input file not found: ", input_path, call. = FALSE)
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(glmmTMB))
suppressPackageStartupMessages(library(jsonlite))

# ---------------------------------------------------------------------------
# data preparation -- verbatim from the reference script
# ---------------------------------------------------------------------------

benchmark <- read.csv(input_path)
n_rows_read <- nrow(benchmark)

if (!ref_level %in% unique(benchmark$condition)) {
  stop(
    "--ref-level '", ref_level, "' is not among the conditions in the input: ",
    paste(sort(unique(benchmark$condition)), collapse = ", "),
    call. = FALSE
  )
}

benchmark$song <- as.factor(benchmark$song)
benchmark$condition <- as.factor(benchmark$condition)
benchmark$condition <- relevel(benchmark$condition, ref = ref_level)
benchmark$composer <- as.factor(benchmark$meta.canonical_composer)
benchmark$duration <- as.numeric(benchmark$meta.duration)
benchmark$year <- as.numeric(benchmark$meta.year)

model_formula <- note.onset_f1 ~ condition + duration + year + (1 | song) + (1 | composer)

model <- glmmTMB(
  model_formula,
  data = benchmark,
  family = beta_family(link = "logit")
)

model_summary <- summary(model)

# ---------------------------------------------------------------------------
# human-readable report -- mirrors what the reference script prints
# ---------------------------------------------------------------------------

report <- c(
  capture.output(print(model_summary)),
  "",
  "Composer random intercepts (ascending):",
  capture.output({
    composer_ranef <- ranef(model)$cond$composer
    print(composer_ranef[order(composer_ranef[, 1]), , drop = FALSE])
  })
)
writeLines(report, file.path(output_dir, "model_summary.txt"))

# ---------------------------------------------------------------------------
# machine-readable tables
# ---------------------------------------------------------------------------

coefficients <- model_summary$coefficients$cond
fixed_effects <- data.frame(
  term = rownames(coefficients),
  estimate = coefficients[, "Estimate"],
  std_error = coefficients[, "Std. Error"],
  statistic = coefficients[, "z value"],
  p_value = coefficients[, "Pr(>|z|)"],
  row.names = NULL,
  stringsAsFactors = FALSE
)
write.csv(fixed_effects, file.path(output_dir, "fixed_effects.csv"), row.names = FALSE)

write_ranef <- function(group) {
  values <- ranef(model)$cond[[group]]
  ordered <- values[order(values[, 1]), , drop = FALSE]
  frame <- data.frame(
    level = rownames(ordered),
    intercept = ordered[, 1],
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  write.csv(frame, file.path(output_dir, paste0("random_effects_", group, ".csv")), row.names = FALSE)
  nrow(frame)
}

n_composers <- write_ranef("composer")
n_songs <- write_ranef("song")

# ---------------------------------------------------------------------------
# provenance / diagnostics
# ---------------------------------------------------------------------------

variance_components <- lapply(VarCorr(model)$cond, function(component) {
  list(variance = unname(component[1, 1]), sd = unname(attr(component, "stddev")[[1]]))
})

meta <- list(
  # deparse() line-wraps long formulas; collapse the padding back out.
  formula = gsub("\\s+", " ", paste(deparse(model_formula), collapse = " ")),
  family = model$modelInfo$family$family,
  link = model$modelInfo$family$link,
  ref_level = ref_level,
  n_rows_read = n_rows_read,
  n_obs = as.integer(nobs(model)),
  n_dropped = as.integer(n_rows_read - nobs(model)),
  n_songs = n_songs,
  n_composers = n_composers,
  conditions = sort(levels(benchmark$condition)),
  log_likelihood = as.numeric(logLik(model)),
  df = as.integer(attr(logLik(model), "df")),
  aic = unname(AIC(model)),
  bic = unname(BIC(model)),
  dispersion = unname(sigma(model)),
  variance_components = variance_components,
  converged = isTRUE(model$fit$convergence == 0),
  convergence_code = as.integer(model$fit$convergence),
  convergence_message = if (is.null(model$fit$message)) NA_character_ else model$fit$message,
  positive_definite_hessian = isTRUE(model$sdr$pdHess),
  r_version = R.version.string,
  packages = list(
    glmmTMB = as.character(packageVersion("glmmTMB")),
    TMB = as.character(packageVersion("TMB")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  fitted_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)

write_json(
  meta,
  file.path(output_dir, "model_meta.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = NA,
  na = "null"
)

cat("wrote model artifacts to", output_dir, "\n")
