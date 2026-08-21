"""Fit the SONITRA condition-effect mixed model on a benchmark regression table.

Runs the beta mixed model from ``misc/SONITRA-mixed-effects-regresion-model.R``::

    note.onset_f1 ~ condition + duration + year + (1 | song) + (1 | composer)
    family = beta_family(link = "logit")

against a table produced by ``scripts/export_regression_table.py`` (with
``--metadata-csv``, since the model needs the composer/duration/year columns),
and writes the results into a ``regression_analysis/`` directory **next to the
input CSV**.

The fit itself happens in R via ``scripts/mixed_effects_analysis.R``,
which this script invokes as a subprocess. This script owns input validation,
provenance, and reporting; the R script owns the statistics.

Usage:
    uv run python scripts/run_mixed_effects_analysis.py \\
        --input corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT/regression_table_with_metadata.csv

    # or point at the benchmark directory and let it find the table
    uv run python scripts/run_mixed_effects_analysis.py \\
        --work-dir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT

R must be on PATH (or given via --rscript / $SONITRA_RSCRIPT) with the glmmTMB
and jsonlite packages installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
R_SCRIPT = SCRIPT_DIR / "mixed_effects_analysis.R"

DEFAULT_TABLE_NAME = "regression_table_with_metadata.csv"
OUTPUT_DIR_NAME = "regression_analysis"

RESPONSE_COLUMN = "note.onset_f1"

#: Columns the R model reads. Absent ones are a hard error -- glmmTMB's own
#: message for a missing column is far less obvious than saying so up front.
REQUIRED_COLUMNS = (
    "condition",
    "song",
    RESPONSE_COLUMN,
    "meta.canonical_composer",
    "meta.duration",
    "meta.year",
)

#: Advisory only: used for warnings, never required.
OPTIONAL_COLUMNS = ("transcriber", "status")

R_PACKAGES = ("glmmTMB", "jsonlite")

INSTALL_HINT = """\
R with the glmmTMB and jsonlite packages is required to fit this model.

  Debian/Ubuntu (prebuilt, no compilation):
      sudo apt-get install -y r-base-core r-cran-glmmtmb r-cran-jsonlite

  conda/micromamba:
      micromamba install -c conda-forge r-base r-glmmtmb r-jsonlite "r-tmb=1.9.19"

  from R itself:
      install.packages(c("glmmTMB", "jsonlite"))

Then re-run, or point at the interpreter explicitly with --rscript /path/to/Rscript
(or set $SONITRA_RSCRIPT).\
"""


@dataclass(frozen=True)
class InputSummary:
    """What the input table contains, before R ever sees it."""

    n_rows: int
    n_songs: int
    n_composers: int
    conditions: list[str] = field(default_factory=list)
    transcribers: list[str] = field(default_factory=list)
    #: Rows with a blank/NaN value in any column the model reads -- R drops these.
    n_incomplete: int = 0
    #: Rows whose ``status`` is anything other than "succeeded".
    n_not_succeeded: int = 0
    #: Responses at exactly 0 or 1, which beta_family cannot represent.
    n_boundary: int = 0


def missing_columns(header: list[str]) -> list[str]:
    """Required model columns absent from *header*, in declaration order."""
    present = set(header)
    return [column for column in REQUIRED_COLUMNS if column not in present]


def _is_blank(value: str | None) -> bool:
    if value is None:
        return True
    text = value.strip()
    return text == "" or text.lower() in {"na", "nan", "null"}


def summarise_input(csv_path: Path) -> InputSummary:
    """Read *csv_path* once and describe the design it encodes."""
    songs: set[str] = set()
    composers: set[str] = set()
    conditions: set[str] = set()
    transcribers: set[str] = set()
    n_rows = 0
    n_incomplete = 0
    n_not_succeeded = 0
    n_boundary = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            songs.add(row.get("song", ""))
            composers.add(row.get("meta.canonical_composer", ""))
            conditions.add(row.get("condition", ""))
            transcriber = row.get("transcriber")
            if not _is_blank(transcriber):
                transcribers.add(transcriber.strip())  # type: ignore[union-attr]

            status = row.get("status")
            if status is not None and not _is_blank(status) and status.strip() != "succeeded":
                n_not_succeeded += 1

            if any(_is_blank(row.get(column)) for column in REQUIRED_COLUMNS):
                n_incomplete += 1

            response = row.get(RESPONSE_COLUMN)
            if not _is_blank(response):
                try:
                    value = float(response)  # type: ignore[arg-type]
                except ValueError:
                    n_incomplete += 1
                else:
                    if not math.isnan(value) and value in (0.0, 1.0):
                        n_boundary += 1

    return InputSummary(
        n_rows=n_rows,
        n_songs=len(songs),
        n_composers=len(composers),
        conditions=sorted(conditions),
        transcribers=sorted(transcribers),
        n_incomplete=n_incomplete,
        n_not_succeeded=n_not_succeeded,
        n_boundary=n_boundary,
    )


def default_output_dir(csv_path: Path) -> Path:
    """Results land beside the data that produced them."""
    return csv_path.parent / OUTPUT_DIR_NAME


def find_rscript(explicit: str | None, env: dict[str, str] | None = None) -> str | None:
    """Locate an Rscript interpreter: explicit path, then env var, then PATH."""
    env = os.environ.copy() if env is None else env
    for candidate in (explicit, env.get("SONITRA_RSCRIPT")):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(candidate)
        return resolved  # explicit request that doesn't resolve is an error, not a fallback
    return shutil.which("Rscript")


def missing_r_packages(rscript: str) -> list[str]:
    """R packages from :data:`R_PACKAGES` that *rscript* cannot load."""
    probe = ";".join(
        f'if (!requireNamespace("{name}", quietly = TRUE)) cat("{name}\\n")' for name in R_PACKAGES
    )
    try:
        result = subprocess.run(
            [rscript, "--vanilla", "-e", probe],
            capture_output=True,
            text=True,
            timeout=120,
            env=_r_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return list(R_PACKAGES)
    return [line.strip() for line in result.stdout.splitlines() if line.strip() in R_PACKAGES]


def available_locales() -> set[str]:
    try:
        result = subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip().lower() for line in result.stdout.splitlines() if line.strip()}


def r_locale_env(env: dict[str, str], locales: set[str]) -> dict[str, str]:
    """Pin R's locale so numeric output is stable *and* composer names survive.

    R formats decimals per LC_NUMERIC, so it has to be C-like or the CSVs come
    out with comma decimal separators. Plain ``C`` would do that, but it also
    sets LC_CTYPE to ASCII, which makes R escape every non-ASCII byte -- turning
    "Frederic Chopin" (with accents) into ``Fr\\303\\251d\\303\\251ric Chopin``.
    ``C.UTF-8`` gives C numerics with UTF-8 text; where it is unavailable, pin
    LC_NUMERIC alone and leave the character type as the user has it.
    """
    env = env.copy()
    # glibc spells it "C.utf8" in `locale -a` but accepts "C.UTF-8" in setlocale.
    normalised = {name.replace("-", "").lower() for name in locales}
    if "c.utf8" in normalised:
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        return env
    env.pop("LC_ALL", None)
    env["LC_NUMERIC"] = "C"
    return env


def _r_env() -> dict[str, str]:
    return r_locale_env(os.environ.copy(), available_locales())


def build_command(
    rscript: str,
    r_script: Path,
    csv_path: Path,
    output_dir: Path,
    ref_level: str,
) -> list[str]:
    return [
        rscript,
        "--vanilla",
        str(r_script),
        "--input",
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--ref-level",
        ref_level,
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(args: argparse.Namespace) -> Path | None:
    if args.input is not None:
        return args.input
    if args.work_dir is not None:
        return args.work_dir / DEFAULT_TABLE_NAME
    return None


def _warn_about(summary: InputSummary) -> None:
    """Surface design facts that would otherwise be silently absorbed by the fit."""
    if len(summary.transcribers) > 1:
        print(
            f"warning: {len(summary.transcribers)} transcribers present "
            f"({', '.join(summary.transcribers)}) but the model has no transcriber term -- "
            "their rows are pooled into one intercept. Filter the table per transcriber "
            "before trusting these estimates.",
            file=sys.stderr,
        )
    if summary.n_not_succeeded:
        print(
            f"warning: {summary.n_not_succeeded} row(s) have status != 'succeeded' and are "
            "fitted alongside the rest -- filter them out beforehand if that is not intended.",
            file=sys.stderr,
        )
    if summary.n_incomplete:
        print(
            f"note: {summary.n_incomplete} row(s) have a missing value in a model column; "
            "R will drop them.",
            file=sys.stderr,
        )
    if summary.n_boundary:
        print(
            f"error: {summary.n_boundary} row(s) have {RESPONSE_COLUMN} at exactly 0 or 1, "
            "which beta_family cannot represent -- glmmTMB will fail. See "
            ".local/notes/analysis/20260821_R_regression_model/comments.md (issue 4).",
            file=sys.stderr,
        )


def _augment_meta(meta_path: Path, csv_path: Path, command: list[str]) -> None:
    """Fold the provenance only Python knows into the R-written metadata."""
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return
    meta["input_csv"] = str(csv_path)
    meta["input_sha256"] = sha256_of(csv_path)
    meta["command"] = command
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.1f}"


def _report(output_dir: Path) -> None:
    """Echo the headline numbers so the terminal is useful without opening files."""
    meta_path = output_dir / "model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(
            f"\nfitted {meta['family']}/{meta['link']} model on {meta['n_obs']} observations "
            f"({meta['n_songs']} songs, {meta['n_composers']} composers)"
        )
        if meta.get("n_dropped"):
            print(f"  {meta['n_dropped']} row(s) dropped for missing values")
        # A failed fit leaves these null rather than absent, so check the value.
        print(f"  AIC {_fmt(meta.get('aic'))}   logLik {_fmt(meta.get('log_likelihood'))}")
        if not meta.get("converged") or not meta.get("positive_definite_hessian"):
            print(
                "  warning: the fit did not converge cleanly -- "
                "treat these estimates as unreliable",
                file=sys.stderr,
            )

    summary_path = output_dir / "model_summary.txt"
    if summary_path.exists():
        print()
        print(summary_path.read_text().rstrip())

    print(f"\nartifacts written to {output_dir}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_argument_group("input (give one)")
    source.add_argument(
        "--input", type=Path, default=None, help="Path to the regression table CSV."
    )
    source.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=f"Benchmark output directory containing {DEFAULT_TABLE_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where to write results (default: {OUTPUT_DIR_NAME}/ next to the input CSV).",
    )
    parser.add_argument(
        "--ref-level",
        default="baseline",
        help="Condition used as the model's reference level (default: baseline).",
    )
    parser.add_argument(
        "--rscript",
        default=None,
        help="Rscript interpreter to use (default: $SONITRA_RSCRIPT, then PATH).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the input and report the design without fitting anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    csv_path = _resolve_input(args)
    if csv_path is None:
        print("error: give either --input or --work-dir", file=sys.stderr)
        return 1
    if not csv_path.exists():
        print(f"error: input table not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    absent = missing_columns(header)
    if absent:
        print(
            f"error: {csv_path} is missing model column(s): {', '.join(absent)}.\n"
            "Regenerate it with scripts/export_regression_table.py --metadata-csv ... "
            "so the meta.* columns are present.",
            file=sys.stderr,
        )
        return 1

    summary = summarise_input(csv_path)
    print(
        f"{csv_path}: {summary.n_rows} rows, {summary.n_songs} songs, "
        f"{summary.n_composers} composers, {len(summary.conditions)} conditions "
        f"({', '.join(summary.conditions)})"
    )
    _warn_about(summary)

    if args.ref_level not in summary.conditions:
        print(
            f"error: --ref-level '{args.ref_level}' is not one of the conditions in the table.",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir or default_output_dir(csv_path)

    if args.dry_run:
        print(f"dry run: would fit {RESPONSE_COLUMN} and write to {output_dir}")
        return 0

    rscript = find_rscript(args.rscript)
    if rscript is None:
        print(f"error: Rscript not found.\n\n{INSTALL_HINT}", file=sys.stderr)
        return 1
    absent_packages = missing_r_packages(rscript)
    if absent_packages:
        print(
            f"error: {rscript} is missing R package(s): {', '.join(absent_packages)}.\n\n"
            f"{INSTALL_HINT}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(rscript, R_SCRIPT, csv_path, output_dir, args.ref_level)

    print(f"fitting with {rscript} (this can take a few minutes on a large table)...")
    result = subprocess.run(command, env=_r_env())
    if result.returncode != 0:
        print(
            f"error: the R fit failed (exit {result.returncode}); see the output above.",
            file=sys.stderr,
        )
        return result.returncode

    # Ship the exact script that produced these numbers, for provenance.
    shutil.copyfile(R_SCRIPT, output_dir / "fit.R")
    _augment_meta(output_dir / "model_meta.json", csv_path, command)
    _report(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
