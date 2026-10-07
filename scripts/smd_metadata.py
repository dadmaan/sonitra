"""Write SMD metadata CSVs that link each real performance to its synthesised version.

Reads the MIDI file names of the SMD piano and SMD-synth datasets and writes one CSV per
dataset with a shared performance id, the fields in the file name and note counts, plus a
JSON provenance sidecar. Either dataset may be missing.

Examples:
    # count files and matches without writing anything
    python scripts/smd_metadata.py --dry-run

    # write metadata/<dataset>.csv for both SMD datasets under corpus/
    python scripts/smd_metadata.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sonitra.corpus import discover_midi_files  # noqa: E402
from sonitra.midi_reader import parse_midi  # noqa: E402

#: SMD performance stem: ``Composer_Work_NNN_YYYYMMDD-SMD[-synth]``.
_STEM_RE = re.compile(
    r"^(?P<composer>[A-Za-z]+)_(?P<work>[A-Za-z0-9-]+)_(?P<performer>\d{3})_"
    r"(?P<date>\d{8})-SMD(?P<synth>-synth)?$"
)

#: Documented composer-spelling normalisation (upstream splits Rachmaninoff).
_COMPOSER_ALIASES = {"Rachmaninov": "Rachmaninoff"}

#: Columns of the output metadata CSV, in this exact order.
_METADATA_COLUMNS = [
    "midi_filename",
    "performance_id",
    "variant",
    "composer",
    "work",
    "performer_id",
    "recording_date",
    "smd_piano_stem",
    "smd_synth_stem",
    "n_notes",
    "duration_sec",
]

_BLANK_FIELDS = {
    "composer": "",
    "work": "",
    "performer_id": "",
    "recording_date": "",
}


def _performance_id(stem: str) -> str:
    """Stem with a trailing ``-SMD-synth`` (first) or ``-SMD`` removed."""
    for suffix in ("-SMD-synth", "-SMD"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _parse_stem_fields(stem: str) -> Optional[Dict[str, str]]:
    """Filename fields for one SMD stem, or ``None`` when it does not match."""
    match = _STEM_RE.match(stem)
    if match is None:
        return None
    composer = match.group("composer")
    date = match.group("date")
    return {
        "composer": _COMPOSER_ALIASES.get(composer, composer),
        "work": match.group("work"),
        "performer_id": match.group("performer"),
        "recording_date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
    }


def _scan_dataset(
    variant: str, dataset: str, midi_dir: Path
) -> Tuple[List[Dict[str, Any]], List[str], Optional[str]]:
    """Collect one dataset's MIDI records.

    Returns ``(records, unparsed_stems, duplicate_error)``. A missing or empty
    ``midi_dir`` means the dataset is absent: a warning is printed and an empty
    record list returned. ``duplicate_error`` is a message naming both paths
    when two files share a ``performance_id``; callers must abort before
    writing anything.
    """
    unparsed: List[str] = []
    if not midi_dir.is_dir():
        print(
            f"warning: {variant} dataset '{dataset}' MIDI directory not found: "
            f"{midi_dir}",
            file=sys.stderr,
        )
        return [], unparsed, None
    paths = discover_midi_files(midi_dir)
    if not paths:
        print(
            f"warning: no MIDI files found for {variant} dataset '{dataset}' "
            f"in {midi_dir}",
            file=sys.stderr,
        )
        return [], unparsed, None

    records: List[Dict[str, Any]] = []
    seen: Dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        performance_id = _performance_id(stem)
        if performance_id in seen:
            return (
                [],
                unparsed,
                f"duplicate performance_id '{performance_id}' in {variant} "
                f"dataset '{dataset}': {seen[performance_id]} and {path}",
            )
        seen[performance_id] = path
        fields = _parse_stem_fields(stem)
        if fields is None:
            print(
                f"warning: cannot parse SMD filename fields from '{stem}' "
                f"({path}); fields left blank",
                file=sys.stderr,
            )
            fields = dict(_BLANK_FIELDS)
            if stem not in unparsed:
                unparsed.append(stem)
        records.append(
            {
                "dataset": variant,
                "filename": stem,
                "path": path,
                "performance_id": performance_id,
                "variant": variant,
                **fields,
                "n_notes": "",
                "duration_sec": "",
                "status": "ok",
                "error": "",
            }
        )

    for record in records:
        try:
            notes = parse_midi(record["path"])
        except Exception as exc:  # noqa: BLE001 - any unreadable MIDI is a per-file error
            record["status"] = "error"
            record["error"] = str(exc)
            print(
                f"error: cannot parse {record['path']}: {exc}",
                file=sys.stderr,
            )
            continue
        record["n_notes"] = len(notes)
        record["duration_sec"] = round(
            max(
                (
                    float(note["start_sec"]) + float(note["duration_sec"])
                    for note in notes
                ),
                default=0.0,
            ),
            3,
        )
    return records, unparsed, None


def _csv_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CSV rows for one dataset, sorted by ``midi_filename``."""
    rows: List[Dict[str, Any]] = []
    for record in sorted(records, key=lambda record: record["filename"]):
        row = {column: record.get(column, "") for column in _METADATA_COLUMNS}
        row["midi_filename"] = record["filename"]
        rows.append(row)
    return rows


def _write_outputs(
    csv_path: Path, rows: List[Dict[str, Any]], provenance_text: str
) -> bool:
    """Write one CSV and its provenance sidecar; False on OSError."""
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=_METADATA_COLUMNS, restval=""
            )
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"error: cannot write metadata {csv_path}: {exc}", file=sys.stderr)
        return False
    provenance_path = Path(str(csv_path) + ".provenance.json")
    try:
        provenance_path.write_text(provenance_text, encoding="utf-8")
    except OSError as exc:
        print(
            f"error: cannot write provenance {provenance_path}: {exc}",
            file=sys.stderr,
        )
        return False
    return True


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-root", default=Path("corpus"), type=Path,
        help="Corpus root directory (default: corpus).",
    )
    parser.add_argument(
        "--piano-dataset", default="smd-piano-v2",
        help="Piano dataset subdirectory under --corpus-root "
        "(default: smd-piano-v2).",
    )
    parser.add_argument(
        "--synth-dataset", default="smd-synth-v1",
        help="Synthesised dataset subdirectory under --corpus-root "
        "(default: smd-synth-v1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and count without writing any CSV or provenance.",
    )
    args = parser.parse_args(argv)
    if args.piano_dataset == args.synth_dataset:
        parser.error("--piano-dataset and --synth-dataset must differ")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    corpus_root: Path = args.corpus_root

    piano_dir = corpus_root / args.piano_dataset / "midi"
    synth_dir = corpus_root / args.synth_dataset / "midi"

    piano_records, piano_unparsed, duplicate = _scan_dataset(
        "piano", args.piano_dataset, piano_dir
    )
    if duplicate is not None:
        print(f"error: {duplicate}", file=sys.stderr)
        return 1
    synth_records, synth_unparsed, duplicate = _scan_dataset(
        "synth", args.synth_dataset, synth_dir
    )
    if duplicate is not None:
        print(f"error: {duplicate}", file=sys.stderr)
        return 1

    if not piano_records and not synth_records:
        print(
            f"error: no SMD MIDI files found for '{args.piano_dataset}' or "
            f"'{args.synth_dataset}' under {corpus_root}",
            file=sys.stderr,
        )
        return 1

    piano_by_id = {record["performance_id"]: record for record in piano_records}
    synth_by_id = {record["performance_id"]: record for record in synth_records}

    matched = set(piano_by_id) & set(synth_by_id)
    piano_only = sorted(set(piano_by_id) - set(synth_by_id))
    synth_only = sorted(set(synth_by_id) - set(piano_by_id))
    for performance_id in piano_only:
        print(
            f"warning: piano-only performance_id '{performance_id}' has no "
            f"synth counterpart",
            file=sys.stderr,
        )
    for performance_id in synth_only:
        print(
            f"warning: synth-only performance_id '{performance_id}' has no "
            f"piano counterpart",
            file=sys.stderr,
        )

    for record in piano_records:
        counterpart = synth_by_id.get(record["performance_id"])
        record["smd_piano_stem"] = record["filename"]
        record["smd_synth_stem"] = (
            counterpart["filename"] if counterpart is not None else ""
        )
    for record in synth_records:
        counterpart = piano_by_id.get(record["performance_id"])
        record["smd_piano_stem"] = (
            counterpart["filename"] if counterpart is not None else ""
        )
        record["smd_synth_stem"] = record["filename"]

    all_records = piano_records + synth_records
    n_errors = sum(1 for record in all_records if record["status"] == "error")
    unparsed = sorted(set(piano_unparsed) | set(synth_unparsed))

    piano_csv = (
        corpus_root / args.piano_dataset / "metadata" / f"{args.piano_dataset}.csv"
    )
    synth_csv = (
        corpus_root / args.synth_dataset / "metadata" / f"{args.synth_dataset}.csv"
    )

    provenance = {
        "corpus_root": str(corpus_root),
        "datasets": {
            "piano": {
                "dataset": args.piano_dataset,
                "midi_dir": str(piano_dir),
                "output_metadata": str(piano_csv) if piano_records else None,
                "n_files": len(piano_records),
            },
            "synth": {
                "dataset": args.synth_dataset,
                "midi_dir": str(synth_dir),
                "output_metadata": str(synth_csv) if synth_records else None,
                "n_files": len(synth_records),
            },
        },
        "argv": effective_argv,
        "dry_run": bool(args.dry_run),
        "composer_aliases": dict(_COMPOSER_ALIASES),
        "matched": len(matched),
        "piano_only": piano_only,
        "synth_only": synth_only,
        "unparsed": unparsed,
        "n_errors": n_errors,
        "files": [
            {
                "dataset": record["dataset"],
                "filename": record["filename"],
                "performance_id": record["performance_id"],
                "n_notes": record["n_notes"],
                "duration_sec": record["duration_sec"],
                "status": record["status"],
                "error": record["error"],
            }
            for record in sorted(
                all_records,
                key=lambda record: (record["dataset"], record["filename"]),
            )
        ],
    }

    if not args.dry_run:
        outputs = (
            (piano_csv, piano_records),
            (synth_csv, synth_records),
        )
        provenance_text = json.dumps(provenance, indent=2) + "\n"
        for csv_path, records in outputs:
            if not records:
                continue
            if not _write_outputs(csv_path, _csv_rows(records), provenance_text):
                return 1
            print(f"wrote {len(records)} rows to {csv_path}")

    counts = (
        f"{len(piano_records)} piano + {len(synth_records)} synth MIDI files, "
        f"{len(matched)} matched, {len(piano_only)} piano-only, "
        f"{len(synth_only)} synth-only, {n_errors} errors"
    )
    if args.dry_run:
        print(f"dry run: {counts}")
    else:
        print(counts)

    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
