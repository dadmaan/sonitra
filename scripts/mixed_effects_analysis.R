#!/usr/bin/env Rscript
#
# Fit the SONITRA condition-effect mixed model and write its results as files.
#
#   note.onset_f1 ~ condition + duration + performance_year + (1 | song) + (1 | composer)
#   family = beta_family(link = "logit")
#
# The base model specification is deliberately verbatim from
# misc/SONITRA-mixed-effects-regresion-model.R apart from the rename of the
# competition-year variable (`year` -> `performance_year`, still from
# `meta.year`) -- this script adds file output and provenance around that fit,
# it does not change the statistics. Known limitations of the specification
# are catalogued in .local/notes/analysis/20260821_R_regression_model/comments.md.
#
# Optionally, a generic numeric covariate (e.g. `meta.composition_year`) can be
# added as a nested model set: base -> <covariate> -> <covariate>_x_condition
# (the last only with --interact-with-condition). Flat top-level artifacts
# always describe the largest model fitted.
#
# Usage:
#   Rscript mixed_effects_analysis.R --input TABLE.csv --output-dir DIR \
#       [--ref-level baseline] [--covariate NAME] \
#       [--covariate-transform none|center|center-scale] \
#       [--covariate-divisor N] [--interact-with-condition]
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
  defaults <- list(
    input = NA_character_,
    `output-dir` = NA_character_,
    `ref-level` = "baseline",
    covariate = NA_character_,
    `covariate-transform` = "center-scale",
    `covariate-divisor` = "100",
    `interact-with-condition` = FALSE
  )
  boolean_flags <- c("interact-with-condition")
  index <- 1
  while (index <= length(argv)) {
    key <- sub("^--", "", argv[[index]])
    if (identical(key, argv[[index]])) {
      stop("unexpected positional argument: ", argv[[index]], call. = FALSE)
    }
    if (!key %in% names(defaults)) {
      stop("unknown argument: --", key, call. = FALSE)
    }
    if (key %in% boolean_flags) {
      defaults[[key]] <- TRUE
      index <- index + 1
      next
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
covariate_column <- args$covariate
covariate_transform <- args$`covariate-transform`
covariate_divisor_raw <- args$`covariate-divisor`
interact_with_condition <- isTRUE(args$`interact-with-condition`)

if (!covariate_transform %in% c("none", "center", "center-scale")) {
  stop(
    "--covariate-transform must be one of none|center|center-scale, got '",
    covariate_transform, "'",
    call. = FALSE
  )
}

covariate_divisor <- suppressWarnings(as.numeric(covariate_divisor_raw))
if (length(covariate_divisor) != 1 || is.na(covariate_divisor) ||
    !is.finite(covariate_divisor) || covariate_divisor == 0) {
  stop(
    "--covariate-divisor must be a finite non-zero number, got '",
    covariate_divisor_raw, "'",
    call. = FALSE
  )
}

has_covariate <- !is.na(covariate_column) && nzchar(covariate_column)

if (!has_covariate && interact_with_condition) {
  stop("--interact-with-condition requires --covariate", call. = FALSE)
}

if (!file.exists(input_path)) {
  stop("input file not found: ", input_path, call. = FALSE)
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(glmmTMB))
suppressPackageStartupMessages(library(jsonlite))

# ---------------------------------------------------------------------------
# data preparation -- verbatim from the reference script apart from the
# competition-year rename (year -> performance_year)
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
benchmark$performance_year <- as.numeric(benchmark$meta.year)

# ---------------------------------------------------------------------------
# generic covariate: variable naming + validation
# ---------------------------------------------------------------------------

response_var <- "note.onset_f1"
reserved_names <- c(
  "condition", "duration", "performance_year", "song", "composer",
  "note.onset_f1", "note_onset_f1"
)

covariate_variable <- NA_character_
covariate_center <- NA_real_
if (has_covariate) {
  covariate_variable <- sub("^meta\\.", "", covariate_column)
  if (!grepl("^[A-Za-z][A-Za-z0-9_]*$", covariate_variable)) {
    stop(
      "invalid covariate variable name '", covariate_variable,
      "' derived from column '", covariate_column,
      "': must match ^[A-Za-z][A-Za-z0-9_]*$",
      call. = FALSE
    )
  }
  if (covariate_variable %in% reserved_names) {
    stop(
      "covariate variable name '", covariate_variable,
      "' collides with a base model term",
      call. = FALSE
    )
  }
  if (!covariate_column %in% names(benchmark)) {
    stop(
      "covariate column '", covariate_column, "' not found in input",
      call. = FALSE
    )
  }
  # Build the covariate still untransformed; the transform is applied after
  # the shared complete-case subset below so every model fits the same rows.
  benchmark[[covariate_variable]] <- as.numeric(benchmark[[covariate_column]])
}

# ---------------------------------------------------------------------------
# formulas built with reformulate() from validated tokens
# ---------------------------------------------------------------------------

format_formula <- function(f) {
  # deparse() line-wraps long formulas; collapse the padding back out.
  gsub("\\s+", " ", paste(deparse(f), collapse = " "))
}

base_formula <- reformulate(
  c("condition", "duration", "performance_year", "(1 | song)", "(1 | composer)"),
  response = response_var
)

model_labels <- c("base")
model_formulas <- list(base_formula)

if (has_covariate) {
  additive_formula <- reformulate(
    c("condition", "duration", "performance_year", covariate_variable,
      "(1 | song)", "(1 | composer)"),
    response = response_var
  )
  model_labels <- c(model_labels, covariate_variable)
  model_formulas <- c(model_formulas, list(additive_formula))
  if (interact_with_condition) {
    interaction_term <- paste("condition", covariate_variable, sep = " * ")
    interaction_formula <- reformulate(
      c(interaction_term, "duration", "performance_year",
        "(1 | song)", "(1 | composer)"),
      response = response_var
    )
    model_labels <- c(model_labels, paste0(covariate_variable, "_x_condition"))
    model_formulas <- c(model_formulas, list(interaction_formula))
  }
}

largest_formula <- model_formulas[[length(model_formulas)]]

# ---------------------------------------------------------------------------
# shared complete-case subset, then centre/scale on the surviving rows
# ---------------------------------------------------------------------------

vars_base <- all.vars(base_formula)
n_base_complete <- sum(complete.cases(benchmark[, vars_base, drop = FALSE]))

vars <- all.vars(largest_formula)
benchmark <- benchmark[complete.cases(benchmark[, vars, drop = FALSE]), , drop = FALSE]
n_complete <- nrow(benchmark)
n_dropped <- as.integer(n_rows_read - n_complete)

if (has_covariate) {
  x <- benchmark[[covariate_variable]]
  covariate_center <- mean(x)
  if (covariate_transform == "none") {
    # no-op
  } else if (covariate_transform == "center") {
    benchmark[[covariate_variable]] <- x - covariate_center
  } else if (covariate_transform == "center-scale") {
    benchmark[[covariate_variable]] <- (x - covariate_center) / covariate_divisor
  }
  n_dropped_covariate <- as.integer(n_base_complete - n_complete)
} else {
  n_dropped_covariate <- 0L
}

# ---------------------------------------------------------------------------
# fit every model with raised optimizer limits
# ---------------------------------------------------------------------------

glmm_control <- glmmTMBControl(optCtrl = list(iter.max = 10000, eval.max = 10000))

fitted_models <- lapply(model_formulas, function(f) {
  glmmTMB(f, data = benchmark, family = beta_family(link = "logit"), control = glmm_control)
})

# ---------------------------------------------------------------------------
# per-model file writer shared by flat + per-model outputs
# ---------------------------------------------------------------------------

write_model_files <- function(model, dest_dir) {
  dir.create(dest_dir, showWarnings = FALSE, recursive = TRUE)
  model_summary <- summary(model)

  report <- c(
    capture.output(print(model_summary)),
    "",
    "Composer random intercepts (ascending):",
    capture.output({
      composer_ranef <- ranef(model)$cond$composer
      print(composer_ranef[order(composer_ranef[, 1]), , drop = FALSE])
    })
  )
  writeLines(report, file.path(dest_dir, "model_summary.txt"))

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
  write.csv(fixed_effects, file.path(dest_dir, "fixed_effects.csv"), row.names = FALSE)

  write_ranef <- function(group) {
    values <- ranef(model)$cond[[group]]
    ordered <- values[order(values[, 1]), , drop = FALSE]
    frame <- data.frame(
      level = rownames(ordered),
      intercept = ordered[, 1],
      row.names = NULL,
      stringsAsFactors = FALSE
    )
    write.csv(frame, file.path(dest_dir, paste0("random_effects_", group, ".csv")), row.names = FALSE)
    nrow(frame)
  }

  n_composers <- write_ranef("composer")
  n_songs <- write_ranef("song")
  list(n_songs = n_songs, n_composers = n_composers, model_summary = model_summary)
}

model_stat <- function(model) {
  ll <- as.numeric(logLik(model))
  list(
    n_obs = as.integer(nobs(model)),
    df = as.integer(attr(logLik(model), "df")),
    log_likelihood = ll,
    aic = unname(AIC(model)),
    bic = unname(BIC(model)),
    converged = isTRUE(model$fit$convergence == 0),
    convergence_code = as.integer(model$fit$convergence),
    convergence_message = if (is.null(model$fit$message)) NA_character_ else model$fit$message,
    positive_definite_hessian = isTRUE(model$sdr$pdHess),
    dispersion = unname(sigma(model)),
    variance_components = lapply(VarCorr(model)$cond, function(component) {
      list(variance = unname(component[1, 1]), sd = unname(attr(component, "stddev")[[1]]))
    })
  )
}

stats_list <- lapply(fitted_models, model_stat)
formula_strings <- vapply(model_formulas, format_formula, character(1))

# LRT chain computed directly, not by parsing anova() output.
lrt_chisq <- rep(NA_real_, length(fitted_models))
lrt_df <- rep(NA_integer_, length(fitted_models))
lrt_p <- rep(NA_real_, length(fitted_models))
if (length(fitted_models) >= 2) {
  for (i in seq(2, length(fitted_models))) {
    chisq <- 2 * (stats_list[[i]]$log_likelihood - stats_list[[i - 1]]$log_likelihood)
    ddf <- stats_list[[i]]$df - stats_list[[i - 1]]$df
    lrt_chisq[i] <- chisq
    lrt_df[i] <- ddf
    lrt_p[i] <- pchisq(chisq, ddf, lower.tail = FALSE)
  }
}

# ---------------------------------------------------------------------------
# outputs: flat files describe the largest (primary) model
# ---------------------------------------------------------------------------

primary_index <- length(fitted_models)
primary_model <- fitted_models[[primary_index]]
primary_stats <- stats_list[[primary_index]]
primary_counts <- write_model_files(primary_model, output_dir)

if (has_covariate) {
  for (i in seq_along(fitted_models)) {
    write_model_files(
      fitted_models[[i]],
      file.path(output_dir, "models", model_labels[[i]])
    )
  }

  comparison <- data.frame(
    label = model_labels,
    formula = formula_strings,
    n_obs = vapply(stats_list, function(s) s$n_obs, integer(1)),
    df = vapply(stats_list, function(s) s$df, integer(1)),
    log_likelihood = vapply(stats_list, function(s) s$log_likelihood, numeric(1)),
    aic = vapply(stats_list, function(s) s$aic, numeric(1)),
    bic = vapply(stats_list, function(s) s$bic, numeric(1)),
    lrt_chisq = lrt_chisq,
    lrt_df = lrt_df,
    lrt_p = lrt_p,
    converged = vapply(stats_list, function(s) s$converged, logical(1)),
    convergence_message = vapply(
      stats_list,
      function(s) if (is.na(s$convergence_message)) NA_character_ else s$convergence_message,
      character(1)
    ),
    positive_definite_hessian = vapply(
      stats_list, function(s) s$positive_definite_hessian, logical(1)
    ),
    stringsAsFactors = FALSE
  )
  write.csv(comparison, file.path(output_dir, "model_comparison.csv"),
    row.names = FALSE, na = "")
}

# ---------------------------------------------------------------------------
# provenance / diagnostics (top-level keys describe the primary model)
# ---------------------------------------------------------------------------

meta <- list(
  formula = formula_strings[[primary_index]],
  family = primary_model$modelInfo$family$family,
  link = primary_model$modelInfo$family$link,
  ref_level = ref_level,
  n_rows_read = n_rows_read,
  n_obs = primary_stats$n_obs,
  n_dropped = as.integer(n_rows_read - primary_stats$n_obs),
  n_songs = primary_counts$n_songs,
  n_composers = primary_counts$n_composers,
  conditions = sort(levels(benchmark$condition)),
  log_likelihood = primary_stats$log_likelihood,
  df = primary_stats$df,
  aic = primary_stats$aic,
  bic = primary_stats$bic,
  dispersion = primary_stats$dispersion,
  variance_components = primary_stats$variance_components,
  converged = primary_stats$converged,
  convergence_code = primary_stats$convergence_code,
  convergence_message = primary_stats$convergence_message,
  positive_definite_hessian = primary_stats$positive_definite_hessian,
  r_version = R.version.string,
  packages = list(
    glmmTMB = as.character(packageVersion("glmmTMB")),
    TMB = as.character(packageVersion("TMB")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  fitted_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)

if (has_covariate) {
  # write_json with na="null" turns NA into null; use NA placeholders so the
  # first model's LRT entries serialize as null.
  per_model <- lapply(seq_along(fitted_models), function(i) {
    list(
      label = model_labels[[i]],
      formula = formula_strings[[i]],
      n_obs = stats_list[[i]]$n_obs,
      df = stats_list[[i]]$df,
      log_likelihood = stats_list[[i]]$log_likelihood,
      aic = stats_list[[i]]$aic,
      bic = stats_list[[i]]$bic,
      lrt_chisq = lrt_chisq[i],
      lrt_df = lrt_df[i],
      lrt_p = lrt_p[i],
      converged = stats_list[[i]]$converged,
      convergence_code = stats_list[[i]]$convergence_code,
      convergence_message = stats_list[[i]]$convergence_message,
      positive_definite_hessian = stats_list[[i]]$positive_definite_hessian
    )
  })
  meta$primary_model <- model_labels[[primary_index]]
  meta$models <- per_model
  meta$covariate <- list(
    column = covariate_column,
    variable = covariate_variable,
    transform = covariate_transform,
    center = covariate_center,
    divisor = covariate_divisor
  )
  meta$n_complete <- as.integer(n_complete)
  meta$n_dropped_covariate <- n_dropped_covariate
}

write_json(
  meta,
  file.path(output_dir, "model_meta.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = NA,
  na = "null"
)

cat("wrote model artifacts to", output_dir, "\n")
