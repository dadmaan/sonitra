r"""Flatten a ``sonitra benchmark`` run into a regression-ready CSV.

One row per (condition, transcriber, file), with every metric and config
override as its own column; a metadata CSV can optionally be joined on, and
rows can be restricted to the dataset splits (e.g. ``test``) it labels.

Examples:
    # flatten a run to <work-dir>/regression_table.csv
    python scripts/export_regression_table.py \
        --work-dir corpus/maestro-v3/benchmark/<run>

    # join dataset metadata (adds meta.* columns)
    python scripts/export_regression_table.py \
        --work-dir corpus/maestro-v3/benchmark/<run> \
        --metadata-csv corpus/maestro-v3/metadata/maestro-v3.0.0.csv \
        --output corpus/maestro-v3/benchmark/<run>/regression_table_with_metadata.csv

    # keep only the test split (writes regression_table_split-test.csv)
    python scripts/export_regression_table.py \
        --work-dir corpus/maestro-v3/benchmark/<run> \
        --metadata-csv corpus/maestro-v3/metadata/maestro-v3.0.0.csv --split test
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sonitra.benchmark.results import BenchmarkRecord, load_records  # noqa: E402
from sonitra.corpus import match_token_prefix  # noqa: E402

_EFFECT_PATH_RE = re.compile(r"^pedalboard\.effects\.(\d+)\.(.+)$")

_IDENTITY_COLUMNS = ["condition", "transcriber", "song", "recording", "midi_path", "source_path", "status"]


def load_effect_types(config_path: Path) -> dict[int, str]:
    """Map pedalboard effect slot index -> effect type, from a saved config.yaml.

    Returns {} if *config_path* doesn't exist (older benchmark runs, from
    before ``run_benchmark`` started snapshotting the resolved config).
    """
    if not config_path.exists():
        return {}
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    effects = (data.get("pedalboard") or {}).get("effects") or []
    return {index: effect["type"] for index, effect in enumerate(effects) if "type" in effect}


def rename_override_key(key: str, effect_types: dict[int, str]) -> str:
    """Turn a dotted-path override key into a self-describing CSV column name.

    ``pedalboard.effects.<N>.<param>`` becomes
    ``override.pedalboard.effects.<N>_<Type>.<param>`` when the effect's type
    is known (from ``load_effect_types``), so e.g. a highpass cutoff is
    labeled instead of an opaque slot index -- different condition families
    (shellac/tape/am) use different slots for conceptually different effects.
    Any other path (or an unknown slot) passes through as ``override.<path>``.
    """
    match = _EFFECT_PATH_RE.match(key)
    if match:
        index, rest = int(match.group(1)), match.group(2)
        effect_type = effect_types.get(index)
        if effect_type:
            return f"override.pedalboard.effects.{index}_{effect_type}.{rest}"
    return f"override.{key}"


def load_metadata_join(csv_path: Path, join_column: str) -> dict[str, dict[str, str]]:
    """Index a metadata CSV by the basename (no extension) of *join_column*.

    Dataset-agnostic by design: any dataset's metadata CSV that names a MIDI
    file in one column can be joined this way, matching the same
    single-extension-stripping ``build_rows`` already uses for ``song``
    (``Path(midi_path).stem``) -- no assumption about composer/work/etc.
    vocabulary, since that doesn't hold across datasets (see CLAUDE.md-level
    discussion: MusicNet has movement/ensemble, MAESTRO doesn't).

    Returns {} if *csv_path* doesn't exist. Raises ValueError if
    *join_column* isn't an actual column of the CSV. On a duplicate join key
    the first row wins and a warning is printed to stderr -- a silent
    overwrite would corrupt a downstream regression in a hard-to-notice way.
    """
    if not csv_path.exists():
        return {}
    index: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if join_column not in (reader.fieldnames or []):
            raise ValueError(
                f"--metadata-join-column '{join_column}' not found in {csv_path} "
                f"(columns: {reader.fieldnames})"
            )
        for row in reader:
            key = Path(row[join_column]).stem
            if key in index:
                print(
                    f"warning: duplicate metadata join key '{key}' in {csv_path} "
                    "-- keeping first row",
                    file=sys.stderr,
                )
                continue
            index[key] = row
    return index


def _resolve_song_metadata_keys(
    songs: set[str],
    metadata: dict[str, dict[str, str]],
    mode: str,
) -> dict[str, str | None]:
    """Resolve each song stem to the metadata key that should be joined.

    In ``exact`` mode this is a direct dict lookup.  In ``token-prefix``
    mode an exact match is tried first; if that misses, a unique
    token-prefix match is attempted through :func:`match_token_prefix`,
    cached per song.  Ambiguous and unmatched songs map to ``None`` and
    are counted in the existing warning.

    Args:
        songs: Distinct ``song`` stems (``Path(midi_path).stem``).
        metadata: Index returned by :func:`load_metadata_join`.
        mode: ``"exact"`` or ``"token-prefix"``.

    Returns:
        Mapping ``song -> metadata_key`` (or ``None`` when no unique
        match exists).  The returned key is guaranteed to be present in
        *metadata* when not ``None``.
    """
    result: dict[str, str | None] = {}
    if not metadata:
        return {s: None for s in songs}
    if mode == "exact":
        for song in songs:
            result[song] = song if song in metadata else None
        return result
    # token-prefix mode
    candidates: dict[str, list[str]] = {key: key.split("_") for key in metadata}
    for song in songs:
        if song in metadata:
            result[song] = song
            continue
        query_tokens = song.split("_")
        match, _, _ = match_token_prefix(query_tokens, candidates)
        result[song] = match if match is not None else None
    return result


def build_rows(
    records: list[BenchmarkRecord],
    effect_types: dict[int, str],
    metadata: dict[str, dict[str, str]] | None = None,
    metadata_match: str = "exact",
    song_to_metadata_key: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Lazy prefix state when caller did not supply a pre-resolved map.
    _prefix_candidates: dict[str, list[str]] | None = None
    _prefix_cache: dict[str, str | None] = {}
    if metadata and metadata_match == "token-prefix" and song_to_metadata_key is None:
        _prefix_candidates = {key: key.split("_") for key in metadata}

    for record in records:
        row: dict[str, Any] = {
            "condition": record.condition,
            "transcriber": record.transcriber,
            "song": Path(record.midi_path).stem,
            "midi_path": record.midi_path,
            "status": record.status,
        }
        if record.source_path is not None:
            row["recording"] = Path(record.source_path).stem
            row["source_path"] = record.source_path
        for metric_name, value in record.metrics.items():
            row[metric_name] = "" if isinstance(value, float) and math.isnan(value) else value
        for override_key, override_value in record.overrides.items():
            row[rename_override_key(override_key, effect_types)] = override_value
        if metadata:
            meta_row: dict[str, str] | None = None
            song = row["song"]
            if metadata_match == "token-prefix":
                if song_to_metadata_key is not None:
                    meta_key = song_to_metadata_key.get(song)
                    meta_row = metadata.get(meta_key) if meta_key is not None else None
                else:
                    # Cached per-song lookup.
                    if song not in _prefix_cache:
                        if song in metadata:
                            _prefix_cache[song] = song
                        else:
                            assert _prefix_candidates is not None
                            query_tokens = song.split("_")
                            match, _, _ = match_token_prefix(query_tokens, _prefix_candidates)
                            _prefix_cache[song] = match if match is not None else None
                    meta_key = _prefix_cache[song]
                    meta_row = metadata.get(meta_key) if meta_key is not None else None
            else:
                if song_to_metadata_key is not None:
                    meta_key = song_to_metadata_key.get(song)
                    meta_row = metadata.get(meta_key) if meta_key is not None else None
                else:
                    meta_row = metadata.get(song)
            if meta_row is not None:
                for column, value in meta_row.items():
                    row[f"meta.{column}"] = value
        rows.append(row)
    return rows


def filter_rows_by_split(
    rows: list[dict[str, Any]], split_column: str, splits: set[str]
) -> list[dict[str, Any]]:
    """Keep rows whose joined ``meta.<split_column>`` is one of *splits*.

    Rows without a metadata match carry no ``meta.`` keys, so their split is
    unknown and they are dropped rather than guessed.
    """
    key = f"meta.{split_column}"
    return [row for row in rows if row.get(key) in splits]


def split_output_name(splits: set[str]) -> str:
    """Default output filename for a split export, e.g. ``regression_table_split-test.csv``."""
    return f"regression_table_split-{'+'.join(sorted(splits))}.csv"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    present: set[str] = set()
    for row in rows:
        present.update(row.keys())

    fieldnames = [column for column in _IDENTITY_COLUMNS if column in present]
    seen: set[str] = set(fieldnames)
    metric_columns: list[str] = []
    override_columns: list[str] = []
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            (override_columns if key.startswith("override.") else metric_columns).append(key)
    fieldnames.extend(metric_columns)
    fieldnames.extend(override_columns)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--work-dir", required=True, type=Path,
        help="A sonitra benchmark output directory (contains the results JSONL "
        "and, optionally, config.yaml).",
    )
    parser.add_argument(
        "--results-file", default="benchmark_results.jsonl",
        help="Results JSONL filename within --work-dir (default: benchmark_results.jsonl).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <work-dir>/regression_table.csv).",
    )
    parser.add_argument(
        "--metadata-csv", type=Path, default=None,
        help="Optional metadata CSV to left-join onto each row, matching the basename of "
        "--metadata-join-column against 'song'. Every other column is added as meta.<column>.",
    )
    parser.add_argument(
        "--metadata-join-column", default="midi_filename",
        help="Column in --metadata-csv naming each file to join on (default: midi_filename, "
        "MAESTRO's column name -- just a default, override for other datasets). "
        "Ignored if --metadata-csv is not given.",
    )
    parser.add_argument(
        "--metadata-match",
        choices=["exact", "token-prefix"],
        default="exact",
        help="How to match song stems to metadata keys (default: exact). "
        "Use token-prefix for MusicNet score MIDI where the reference stem "
        "includes composer/work tokens beyond the numeric id "
        "(e.g. 1727_schubert_op114_2 -> 1727).  In token-prefix mode an "
        "exact match is tried first, then a unique token-prefix match via "
        "the same logic as audio-to-MIDI pairing (see sonitra.corpus). "
        "For aligned references use --metadata-csv metadata/musicnet.csv "
        "with the default; for score references use "
        "--metadata-csv metadata/musicnet_metadata.csv --metadata-join-column id "
        "--metadata-match token-prefix.",
    )
    parser.add_argument(
        "--split", action="append", default=None, metavar="VALUE",
        help="Keep only rows whose metadata --split-column equals VALUE (repeatable, "
        "e.g. --split test). Requires --metadata-csv. Rows with no metadata match are "
        "dropped because their split is unknown. Default output becomes "
        "<work-dir>/regression_table_split-<values>.csv, so the unfiltered table is "
        "never overwritten.",
    )
    parser.add_argument(
        "--split-column", default="split",
        help="Metadata column holding split labels (default: split, as in MAESTRO and GAPS).",
    )
    args = parser.parse_args(argv)
    if args.split and args.metadata_csv is None:
        parser.error("--split requires --metadata-csv (the split labels come from the metadata)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    work_dir: Path = args.work_dir
    results_path = work_dir / args.results_file
    if not results_path.exists():
        print(f"error: results file not found: {results_path}", file=sys.stderr)
        return 1

    config_path = work_dir / "config.yaml"
    effect_types = load_effect_types(config_path)
    if not effect_types:
        print(
            f"note: no pedalboard.effects type info found at {config_path} "
            "-- override columns will use raw dotted paths.",
            file=sys.stderr,
        )

    metadata: dict[str, dict[str, str]] = {}
    if args.metadata_csv is not None:
        metadata = load_metadata_join(args.metadata_csv, args.metadata_join_column)

    splits: set[str] = set(args.split or [])
    if splits:
        if not metadata:
            print(
                f"error: --split needs split labels, but metadata CSV {args.metadata_csv} "
                "is missing or has no rows",
                file=sys.stderr,
            )
            return 1
        columns = next(iter(metadata.values())).keys()
        if args.split_column not in columns:
            print(
                f"error: split column '{args.split_column}' not found in {args.metadata_csv} "
                f"(columns: {', '.join(columns)}); set --split-column",
                file=sys.stderr,
            )
            return 1
        available = {row[args.split_column] for row in metadata.values()}
        unknown = sorted(splits - available)
        if unknown:
            print(
                f"error: split value(s) {', '.join(repr(v) for v in unknown)} not found in "
                f"column '{args.split_column}' (available: "
                f"{', '.join(sorted(v for v in available if v))})",
                file=sys.stderr,
            )
            return 1

    records = load_records(results_path)

    # Single resolved song -> metadata key map. Both build_rows
    # and the unmatched count use this map; otherwise prefix matches would
    # still be counted as unmatched.
    song_to_metadata_key: dict[str, str | None] | None = None
    if args.metadata_csv is not None:
        distinct_for_map = {Path(r.midi_path).stem for r in records}
        song_to_metadata_key = _resolve_song_metadata_keys(
            distinct_for_map, metadata, args.metadata_match
        )

    rows = build_rows(
        records,
        effect_types,
        metadata,
        metadata_match=args.metadata_match,
        song_to_metadata_key=song_to_metadata_key,
    )
    if splits:
        total = len(rows)
        unmatched_rows = [row for row in rows if f"meta.{args.split_column}" not in row]
        rows = filter_rows_by_split(rows, args.split_column, splits)
        label = "+".join(sorted(splits))
        print(f"kept {len(rows)}/{total} rows ({args.split_column}: {label})", file=sys.stderr)
        if unmatched_rows:
            n_songs = len({row["song"] for row in unmatched_rows})
            print(
                f"note: {len(unmatched_rows)} rows ({n_songs} songs) have no metadata match, "
                "so their split is unknown -- excluded",
                file=sys.stderr,
            )
        if not rows:
            print(
                f"error: no rows belong to {args.split_column} {label}; nothing written",
                file=sys.stderr,
            )
            return 1

    default_name = split_output_name(splits) if splits else "regression_table.csv"
    output_path = args.output or (work_dir / default_name)
    write_csv(rows, output_path)

    n_conditions = len({row["condition"] for row in rows})
    print(f"wrote {len(rows)} rows across {n_conditions} conditions to {output_path}")

    if args.metadata_csv is not None:
        # Count over every record, not just rows a --split filter kept.
        distinct_songs = {Path(record.midi_path).stem for record in records}
        if song_to_metadata_key is not None:
            unmatched = sorted(song for song in distinct_songs if song_to_metadata_key.get(song) is None)
        else:
            unmatched = sorted(song for song in distinct_songs if song not in metadata)
        if unmatched:
            sample = ", ".join(unmatched[:5])
            msg = (
                f"warning: {len(unmatched)}/{len(distinct_songs)} songs had no metadata "
                f"match (e.g. {sample})"
            )
            if args.metadata_match == "exact":
                msg += " -- try --metadata-match token-prefix for MusicNet score MIDI"
            print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
