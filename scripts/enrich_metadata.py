"""Left-join a delimited annotation table onto a dataset metadata CSV.

Rows match on a composite key; unmatched rows get blank cells. Annotation
quoting is disabled by default so stray quotes cannot merge rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: How many distinct unmatched keys to keep for the coverage report / stderr sample.
_UNMATCHED_SAMPLE_SIZE = 5


@dataclass
class CoverageReport:
    """How many base rows found an annotation match."""

    n_matched: int
    n_total: int
    n_unmatched_keys: int
    n_unmatched_rows: int
    unmatched_sample: List[Tuple[str, ...]] = field(default_factory=list)


def parse_mapping(spec: str) -> Tuple[str, str]:
    """Parse a ``LEFT=RIGHT`` mapping spec, or a bare ``RIGHT`` identity spec.

    Args:
        spec: Mapping spec such as ``"canonical_composer=Composer"`` or ``"Year"``.

    Returns:
        ``(left, right)`` pair; a bare name maps to itself.

    Raises:
        ValueError: If the spec is empty, has an empty side, or contains more
            than one ``=``.
    """
    parts = spec.split("=")
    if len(parts) == 1:
        name = parts[0].strip()
        if not name:
            raise ValueError(f"malformed mapping spec: {spec!r}")
        return (name, name)
    if len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        if not left or not right:
            raise ValueError(f"malformed mapping spec: {spec!r}")
        return (left, right)
    raise ValueError(f"malformed mapping spec: {spec!r}")


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_key(parts: List[str], case_insensitive: bool) -> Tuple[str, ...]:
    """Build a join key: per-field strip, tuple (never a joined string)."""
    stripped = [part.strip() for part in parts]
    if case_insensitive:
        stripped = [part.casefold() for part in stripped]
    return tuple(stripped)


def load_annotations(
    path: Path,
    delimiter: str = ",",
    comment_prefix: str = "#",
    quotechar: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Load an annotation table, skipping leading comment lines.

    Args:
        path: Annotation file path.
        delimiter: Field delimiter (default ``','``).
        comment_prefix: *All* leading lines starting with this prefix are
            skipped; the first remaining line is the header.
        quotechar: When None (default) quoting is disabled -- lines are split
            manually on *delimiter* with only trailing ``\\r``/``\\n`` stripped,
            so unbalanced/embedded quotes survive verbatim (QUOTE_NONE
            semantics, mirroring R's ``quote=""``). Otherwise the ``csv`` module
            parses with this quotechar (RFC4180).

    Returns:
        ``(header, rows)`` with rows as column-name -> value dicts.

    Raises:
        ValueError: If a data line has a different field count than the header
            (manual-split mode only; the csv module handles ragged rows itself).
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_lines = handle.read().splitlines(keepends=True)

    data_lines: List[str] = []
    seen_header = False
    for raw in raw_lines:
        line = raw.rstrip("\r\n")
        if not seen_header:
            if comment_prefix and line.startswith(comment_prefix):
                continue
            if line.strip() == "":
                continue
            seen_header = True
        else:
            if line.strip() == "":
                continue
        data_lines.append(line)

    if not data_lines:
        return ([], [])

    if quotechar is not None:
        reader = csv.reader(data_lines, delimiter=delimiter, quotechar=quotechar)
        header = [name.strip() for name in next(reader)]
        rows: List[Dict[str, str]] = []
        for record in reader:
            if not record:
                continue
            rows.append(
                {
                    column: (value if value is not None else "")
                    for column, value in zip(header, record)
                }
            )
            for column in header[len(record):]:
                rows[-1][column] = ""
        return (header, rows)

    header = [name.strip() for name in data_lines[0].split(delimiter)]
    rows = []
    for lineno, line in enumerate(data_lines[1:], start=2):
        parts = line.split(delimiter)
        if len(parts) != len(header):
            raise ValueError(
                f"{path}:{lineno}: expected {len(header)} fields, "
                f"got {len(parts)}: {line[:80]!r}..."
            )
        rows.append(dict(zip(header, parts)))
    return (header, rows)


def build_annotation_index(
    header: List[str],
    rows: List[Dict[str, str]],
    annotation_key_cols: List[str],
    case_insensitive: bool = False,
) -> Tuple[Dict[Tuple[str, ...], Dict[str, str]], Dict[str, int]]:
    """Index annotation rows by their (stripped, tuple) join key.

    First row wins on duplicate keys, mirroring R's named-vector lookup; identical
    vs conflicting duplicates are counted separately and warned about distinctly
    so a real conflict is never silent.

    Args:
        header: Annotation column names.
        rows: Annotation rows from :func:`load_annotations`.
        annotation_key_cols: Annotation-side key columns, in join order.
        case_insensitive: Casefold key parts before comparing.

    Returns:
        ``(index, duplicate_counts)`` where ``duplicate_counts`` has
        ``"identical"`` and ``"conflicting"`` keys.

    Raises:
        ValueError: If a key column is absent from the annotation file, naming
            the column.
    """
    for column in annotation_key_cols:
        if column not in header:
            raise ValueError(
                f"annotation key column '{column}' not found "
                f"(columns: {header})"
            )
    non_key_cols = [column for column in header if column not in annotation_key_cols]
    index: Dict[Tuple[str, ...], Dict[str, str]] = {}
    duplicate_counts = {"identical": 0, "conflicting": 0}
    for row in rows:
        key = _normalize_key(
            [(row.get(column) or "") for column in annotation_key_cols],
            case_insensitive,
        )
        if key in index:
            existing = index[key]
            identical = all(
                (existing.get(column) or "").strip() == (row.get(column) or "").strip()
                for column in non_key_cols
            )
            kind = "identical" if identical else "conflicting"
            duplicate_counts[kind] += 1
            print(
                f"warning: duplicate annotation key {key} with {kind} values "
                "-- keeping first row",
                file=sys.stderr,
            )
            continue
        index[key] = row
    return (index, duplicate_counts)


def enrich_rows(
    base_header: List[str],
    base_rows: List[Dict[str, str]],
    key_pairs: List[Tuple[str, str]],
    add_pairs: List[Tuple[str, str]],
    index: Dict[Tuple[str, ...], Dict[str, str]],
    case_insensitive: bool = False,
) -> Tuple[List[str], List[Dict[str, str]], CoverageReport]:
    """Left-join annotation values onto base rows.

    Args:
        base_header: Base metadata column names.
        base_rows: Base metadata rows.
        key_pairs: ``(base_column, annotation_column)`` join keys, order
            significant.
        add_pairs: ``(annotation_column, output_column)`` columns to carry over.
        index: Annotation index from :func:`build_annotation_index`.
        case_insensitive: Casefold key parts before comparing (must match the
            flag used to build *index*).

    Returns:
        ``(output_header, output_rows, coverage)``; unmatched rows get blank
        (``""``, never ``"NA"``) added cells.

    Raises:
        ValueError: If a base key column is absent, or an output column name
            already exists in the base metadata -- naming the column.
    """
    for base_col, _ in key_pairs:
        if base_col not in base_header:
            raise ValueError(
                f"base metadata key column '{base_col}' not found "
                f"(columns: {base_header})"
            )
    for _, output_col in add_pairs:
        if output_col in base_header:
            raise ValueError(
                f"output column '{output_col}' already exists in base metadata "
                f"(columns: {base_header})"
            )

    output_header = list(base_header) + [output_col for _, output_col in add_pairs]
    output_rows: List[Dict[str, str]] = []
    n_matched = 0
    unmatched_keys: List[Tuple[str, ...]] = []
    seen_unmatched = set()
    for row in base_rows:
        key = _normalize_key(
            [(row.get(base_col) or "") for base_col, _ in key_pairs],
            case_insensitive,
        )
        match = index.get(key)
        out_row = dict(row)
        if match is None:
            for _, output_col in add_pairs:
                out_row[output_col] = ""
            if key not in seen_unmatched:
                seen_unmatched.add(key)
                if len(unmatched_keys) < _UNMATCHED_SAMPLE_SIZE:
                    unmatched_keys.append(key)
        else:
            n_matched += 1
            for annotation_col, output_col in add_pairs:
                out_row[output_col] = match.get(annotation_col) or ""
        output_rows.append(out_row)

    coverage = CoverageReport(
        n_matched=n_matched,
        n_total=len(base_rows),
        n_unmatched_keys=len(seen_unmatched),
        n_unmatched_rows=len(base_rows) - n_matched,
        unmatched_sample=unmatched_keys,
    )
    return (output_header, output_rows, coverage)


def write_csv(header: List[str], rows: List[Dict[str, str]], path: Path) -> None:
    """Write rows as RFC4180 CSV (quoting comma/quote-bearing fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_provenance(path: Path, payload: Dict[str, object]) -> None:
    """Write a JSON provenance sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", required=True, type=Path,
        help="Base dataset metadata CSV (RFC4180).",
    )
    parser.add_argument(
        "--annotations", required=True, type=Path,
        help="Auxiliary annotation table to left-join onto --metadata.",
    )
    parser.add_argument(
        "--on", required=True, action="append", default=[], dest="on",
        help="Repeatable composite join key as LEFT=RIGHT (base column = "
        "annotation column); order is significant.",
    )
    parser.add_argument(
        "--add", action="append", default=[], dest="add",
        help="Repeatable annotation column to carry over as RIGHT[=NEWNAME]; "
        "default: every non-key annotation column under its own name.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Enriched CSV output path (must differ from --metadata).",
    )
    parser.add_argument(
        "--annotations-delimiter", default=",",
        help="Annotation field delimiter (default: ',').",
    )
    parser.add_argument(
        "--annotations-comment-prefix", default="#",
        help="Annotation leading comment-line prefix (default: '#'; all "
        "leading comment lines are skipped).",
    )
    parser.add_argument(
        "--annotations-quotechar", default=None,
        help="Annotation quotechar (default: quoting DISABLED, QUOTE_NONE "
        "semantics). Pass e.g. '\"' for RFC4180 annotation files.",
    )
    parser.add_argument(
        "--case-insensitive", action="store_true",
        help="Casefold join-key parts on both sides before comparing.",
    )
    parser.add_argument(
        "--require-full-coverage", action="store_true",
        help="Exit non-zero if any base row has no annotation match.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    if len(args.annotations_delimiter) != 1:
        print(
            f"error: --annotations-delimiter must be a single character, "
            f"got {args.annotations_delimiter!r}",
            file=sys.stderr,
        )
        return 1
    if args.annotations_quotechar is not None and len(args.annotations_quotechar) != 1:
        print(
            f"error: --annotations-quotechar must be a single character, "
            f"got {args.annotations_quotechar!r}",
            file=sys.stderr,
        )
        return 1

    try:
        key_pairs = [parse_mapping(spec) for spec in args.on]
        add_specs = [parse_mapping(spec) for spec in args.add]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metadata_path: Path = args.metadata
    output_path: Path = args.output
    if output_path.resolve() == metadata_path.resolve():
        print(
            f"error: refusing to write output over --metadata input ({metadata_path}); "
            "choose a different --output",
            file=sys.stderr,
        )
        return 1

    if not metadata_path.exists():
        print(f"error: metadata file not found: {metadata_path}", file=sys.stderr)
        return 1
    annotations_path: Path = args.annotations
    if not annotations_path.exists():
        print(f"error: annotations file not found: {annotations_path}", file=sys.stderr)
        return 1

    try:
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            base_header = list(reader.fieldnames or [])
            base_rows = [dict(row) for row in reader]
    except OSError as exc:
        print(f"error: cannot read metadata {metadata_path}: {exc}", file=sys.stderr)
        return 1
    if not base_header:
        print(f"error: metadata {metadata_path} has no header row", file=sys.stderr)
        return 1

    try:
        annotation_header, annotation_rows = load_annotations(
            annotations_path,
            delimiter=args.annotations_delimiter,
            comment_prefix=args.annotations_comment_prefix,
            quotechar=args.annotations_quotechar,
        )
    except (OSError, ValueError) as exc:
        print(f"error: cannot load annotations {annotations_path}: {exc}", file=sys.stderr)
        return 1

    annotation_key_cols = [right for _, right in key_pairs]
    if args.add:
        add_pairs = add_specs
        for annotation_col, _ in add_pairs:
            if annotation_col not in annotation_header:
                print(
                    f"error: annotation column '{annotation_col}' not found "
                    f"(columns: {annotation_header})",
                    file=sys.stderr,
                )
                return 1
    else:
        add_pairs = [
            (column, column)
            for column in annotation_header
            if column not in annotation_key_cols
        ]

    try:
        index, duplicate_counts = build_annotation_index(
            annotation_header,
            annotation_rows,
            annotation_key_cols,
            case_insensitive=args.case_insensitive,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        output_header, output_rows, coverage = enrich_rows(
            base_header,
            base_rows,
            key_pairs,
            add_pairs,
            index,
            case_insensitive=args.case_insensitive,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        write_csv(output_header, output_rows, output_path)
    except OSError as exc:
        print(f"error: cannot write output {output_path}: {exc}", file=sys.stderr)
        return 1

    provenance = {
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_of(metadata_path),
        "annotations": str(annotations_path),
        "annotations_sha256": sha256_of(annotations_path),
        "argv": effective_argv,
        "on": list(args.on),
        "add": list(args.add),
        "annotations_delimiter": args.annotations_delimiter,
        "annotations_quotechar": args.annotations_quotechar,
        "case_insensitive": args.case_insensitive,
        "base_key_columns": [left for left, _ in key_pairs],
        "annotation_key_columns": annotation_key_cols,
        "added_columns": [new for _, new in add_pairs],
        "n_matched": coverage.n_matched,
        "n_total": coverage.n_total,
        "n_unmatched_keys": coverage.n_unmatched_keys,
        "n_unmatched_rows": coverage.n_unmatched_rows,
        "duplicate_keys_identical": duplicate_counts["identical"],
        "duplicate_keys_conflicting": duplicate_counts["conflicting"],
        "unmatched_sample": ["|".join(key) for key in coverage.unmatched_sample],
    }
    provenance_path = Path(str(output_path) + ".provenance.json")
    try:
        write_provenance(provenance_path, provenance)
    except OSError as exc:
        print(
            f"error: cannot write provenance {provenance_path}: {exc}", file=sys.stderr
        )
        return 1

    print(
        f"matched {coverage.n_matched}/{coverage.n_total} rows "
        f"({coverage.n_unmatched_keys} distinct keys unmatched)",
        file=sys.stderr,
    )
    if coverage.n_unmatched_keys:
        sample = ", ".join("|".join(key) for key in coverage.unmatched_sample)
        print(
            f"warning: {coverage.n_unmatched_rows}/{coverage.n_total} rows had no "
            f"annotation match ({coverage.n_unmatched_keys} distinct keys "
            f"unmatched; e.g. {sample})",
            file=sys.stderr,
        )

    print(f"wrote {len(output_rows)} rows to {output_path}")
    if args.require_full_coverage and coverage.n_matched < coverage.n_total:
        print(
            f"error: --require-full-coverage: only {coverage.n_matched}/"
            f"{coverage.n_total} rows matched",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
