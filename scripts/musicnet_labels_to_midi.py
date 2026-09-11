"""Convert MusicNet label CSVs to sample-aligned reference MIDI and a metadata CSV.

Writes one MIDI file per recording (1 tick = 1 sample) plus a metadata CSV with
a JSON provenance sidecar. An existing score-MIDI tree in the output directory
is only replaced when ``--replace-score-midi`` is given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sonitra.midi_writer import write_multi_program_midi  # noqa: E402
from sonitra.unisons import _UNISON_ONSET_TOLERANCE_SEC, count_unisons, dedupe_unisons  # noqa: E402

#: Tempo written into every output MIDI.
_TEMPO_BPM = 120.0

#: Default sample rate (MusicNet recordings are 44.1 kHz).
_DEFAULT_SAMPLE_RATE = 44100

#: Default velocity (labels have no dynamics).
_DEFAULT_VELOCITY = 100

#: Columns for the output metadata CSV.
_METADATA_COLUMNS = [
    "midi_filename",
    "id",
    "split",
    "composer",
    "composition",
    "movement",
    "ensemble",
    "source",
    "transcriber",
    "catalog_name",
    "seconds",
    "score_midi_filename",
    "n_notes",
    "n_programs",
    "programs",
    "labels_end_sec",
    "unisons_detected",
    "same_channel_overlaps",
]

#: Validated from musicnet_metadata.csv header; these are copied verbatim.
_SOURCE_METADATA_COLUMNS = [
    "composer",
    "composition",
    "movement",
    "ensemble",
    "source",
    "transcriber",
    "catalog_name",
    "seconds",
]

_SKIP_REASONS = ("invalid_value", "non_positive_duration", "out_of_range_pitch", "invalid_instrument")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return path.is_relative_to(parent)
    except AttributeError:  # Python < 3.9 fallback
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_same_channel_overlaps(notes: List[Dict[str, Any]]) -> int:
    """Count overlapping intervals per (program, pitch).

    Notes are dicts with ``pitch``, ``program``, ``onset``, ``offset``.
    """
    from collections import defaultdict

    groups: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for n in notes:
        groups[(n["program"], n["pitch"])].append((n["onset"], n["offset"]))
    overlaps = 0
    for intervals in groups.values():
        intervals.sort(key=lambda x: x[0])
        if not intervals:
            continue
        current_end = intervals[0][1]
        for onset, offset in intervals[1:]:
            if onset < current_end - 1e-9:  # overlapping (allow tiny epsilon)
                overlaps += 1
                current_end = max(current_end, offset)
            else:
                current_end = offset
    return overlaps


def _parse_label_csv(
    csv_path: Path,
    sample_rate: int,
    velocity: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], float]:
    """Parse one MusicNet label CSV.

    Returns ``(notes, skipped, max_end_sec)`` where notes have ``pitch``,
    ``program``, ``onset``, ``offset``, ``velocity``.
    """
    skipped: Dict[str, int] = {reason: 0 for reason in _SKIP_REASONS}
    notes: List[Dict[str, Any]] = []
    max_end_sec = 0.0
    # Read CSV; may raise OSError or have malformed header
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing header in {csv_path.name}")
        required = {"start_time", "end_time", "instrument", "note"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"missing required columns in {csv_path.name}: {reader.fieldnames}")
        for row in reader:
            try:
                # Extract required fields
                start_time_raw = row.get("start_time")
                end_time_raw = row.get("end_time")
                instrument_raw = row.get("instrument")
                note_raw = row.get("note")
                if start_time_raw is None or end_time_raw is None or instrument_raw is None or note_raw is None:
                    skipped["invalid_value"] += 1
                    continue
                start_time = int(float(start_time_raw)) if isinstance(start_time_raw, str) and "." in str(start_time_raw) else int(start_time_raw)
                # Actually CSV stores ints as strings, may be int-like floats? Use float then int?
                # Safer: float then int
                try:
                    start_sample = int(float(start_time_raw))
                    end_sample = int(float(end_time_raw))
                    instrument = int(float(instrument_raw))
                    pitch = int(float(note_raw))
                except (ValueError, TypeError):
                    skipped["invalid_value"] += 1
                    continue
                # Validate instrument
                if not (1 <= instrument <= 128):
                    skipped["invalid_instrument"] += 1
                    continue
                program = instrument - 1
                if not (0 <= pitch <= 127):
                    skipped["out_of_range_pitch"] += 1
                    continue
                # Compute timing
                start_sec = start_sample / sample_rate
                end_sec = end_sample / sample_rate
                duration_sec = end_sec - start_sec
                if duration_sec <= 0:
                    skipped["non_positive_duration"] += 1
                    continue
                # Update max end
                if end_sec > max_end_sec:
                    max_end_sec = end_sec
                notes.append(
                    {
                        "pitch": pitch,
                        "program": program,
                        "onset": start_sec,
                        "offset": end_sec,
                        "velocity": velocity,
                    }
                )
            except (ValueError, TypeError, AttributeError):
                skipped["invalid_value"] += 1
                continue
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes, skipped, max_end_sec


def _load_source_metadata(path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Load musicnet_metadata.csv, keyed by id string.

    Returns ``(by_id, columns)`` where columns are the header fields.
    """
    by_id: Dict[str, Dict[str, str]] = {}
    columns: List[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return by_id, columns
        columns = list(reader.fieldnames)
        # Determine id column: try 'id' then first column
        id_col = "id" if "id" in columns else columns[0]
        for row in reader:
            key = row.get(id_col, "")
            if key is None:
                continue
            key = str(key).strip()
            if not key:
                continue
            by_id[key] = {k: (v if v is not None else "") for k, v in row.items()}
    return by_id, columns


def _find_score_midi_map(score_midi_dir: Path) -> Dict[str, str]:
    """Map id -> score_midi_filename (relative path under score_midi_dir).

    Searches recursively for *.mid/*.midi, mapping ``<id>`` to the first file
    whose stem is ``<id>`` or starts with ``<id>_``.
    """
    mapping: Dict[str, str] = {}
    if not score_midi_dir.is_dir():
        return mapping
    for midi_path in list(score_midi_dir.rglob("*.mid")) + list(score_midi_dir.rglob("*.midi")):
        stem = midi_path.stem
        # Extract id as prefix before '_' or whole stem
        if "_" in stem:
            candidate_id = stem.split("_", 1)[0]
        else:
            candidate_id = stem
        # Only map if candidate looks numeric? But keep generic
        if candidate_id not in mapping:
            # Store relative posix path
            rel = midi_path.relative_to(score_midi_dir).as_posix()
            mapping[candidate_id] = rel
        else:
            # If multiple, keep first; could be duplicate but ignore
            pass
    return mapping


def _has_non_top_level_midi(midi_dir: Path) -> List[Path]:
    """Return list of MIDI files under midi_dir that are not top-level <id>.mid."""
    if not midi_dir.is_dir():
        return []
    non_top = []
    for p in list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi")):
        rel = p.relative_to(midi_dir)
        # Top-level means one part and stem equals filename without extension
        # and parent is '.' (i.e., direct child)
        if len(rel.parts) != 1:
            non_top.append(p)
        else:
            # Check stem? It should be numeric id but we just check top-level?
            # For musicnet, top-level should be <id>.mid, but we treat any single-level as top-level ok
            # However score tree has nested paths, so any nested is bad
            pass
    return non_top


def _handle_replace_score_midi(
    output_midi: Path,
    score_midi: Path,
) -> Optional[str]:
    """Handle --replace-score-midi: move/delete score files in output_midi.

    Returns error message string on failure, or None on success.
    """
    non_top = _has_non_top_level_midi(output_midi)
    if not non_top:
        return None
    for src in non_top:
        rel = src.relative_to(output_midi)
        dest = score_midi / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            # Compare bytes
            try:
                if src.read_bytes() == dest.read_bytes():
                    # Identical: delete src
                    src.unlink()
                else:
                    return f"refusing to delete {src} whose copy at {dest} differs (would lose data)"
            except OSError as exc:
                return f"cannot compare {src} and {dest}: {exc}"
        else:
            # Move
            try:
                # Use replace to move atomically; ensure parent exists
                src.replace(dest)
            except OSError as exc:
                return f"cannot move {src} to {dest}: {exc}"
    # Remove empty directories under output_midi
    # Walk bottom-up
    for dirpath in sorted(output_midi.rglob("*"), reverse=True):
        if dirpath.is_dir():
            try:
                dirpath.rmdir()
            except OSError:
                pass
    return None


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        default=Path("corpus"),
        type=Path,
        help="Corpus root directory (default: corpus).",
    )
    parser.add_argument(
        "--dataset",
        default="musicnet",
        help="Dataset subdirectory under --corpus-root (default: musicnet).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=Path,
        help="Label CSV input directory (default: <corpus-root>/<dataset>/annotations/labels).",
    )
    parser.add_argument(
        "--source-metadata",
        default=None,
        type=Path,
        help="Source metadata CSV (default: <corpus-root>/<dataset>/metadata/musicnet_metadata.csv).",
    )
    parser.add_argument(
        "--score-midi",
        default=None,
        type=Path,
        help="Score MIDI input directory (default: <corpus-root>/<dataset>/annotations/score_midi).",
    )
    parser.add_argument(
        "--output-midi",
        default=None,
        type=Path,
        help="MIDI output directory (default: <corpus-root>/<dataset>/midi).",
    )
    parser.add_argument(
        "--output-metadata",
        default=None,
        type=Path,
        help="Metadata CSV output path (default: <corpus-root>/<dataset>/metadata/musicnet.csv).",
    )
    parser.add_argument(
        "--sample-rate",
        default=_DEFAULT_SAMPLE_RATE,
        type=int,
        help="Sample rate for timing conversion (default: 44100). Must be even and <=65534.",
    )
    parser.add_argument(
        "--velocity",
        default=_DEFAULT_VELOCITY,
        type=int,
        help="Constant MIDI velocity for every note (default: 100).",
    )
    parser.add_argument(
        "--no-program",
        action="store_true",
        help="Write no program_change at all (note-only files). Overrides program mapping.",
    )
    parser.add_argument(
        "--dedupe-unisons",
        action="store_true",
        help="Remove unison duplicates (keep earliest onset per pitch group within tolerance).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing .mid files (default: skip them).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count without writing any MIDI, CSV, or provenance.",
    )
    parser.add_argument(
        "--replace-score-midi",
        action="store_true",
        help="Move/delete score MIDI tree in --output-midi into --score-midi before converting.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    # Resolve defaults
    labels_dir: Path = args.labels or (args.corpus_root / args.dataset / "annotations" / "labels")
    source_metadata_path: Path = args.source_metadata or (
        args.corpus_root / args.dataset / "metadata" / "musicnet_metadata.csv"
    )
    score_midi_dir: Path = args.score_midi or (args.corpus_root / args.dataset / "annotations" / "score_midi")
    midi_dir: Path = args.output_midi or (args.corpus_root / args.dataset / "midi")
    metadata_path: Path = args.output_metadata or (
        args.corpus_root / args.dataset / "metadata" / "musicnet.csv"
    )

    # Validate sample rate
    if args.sample_rate % 2 != 0 or args.sample_rate > 65534 or args.sample_rate < 1:
        print(
            f"error: --sample-rate must be even and <=65534, got {args.sample_rate}",
            file=sys.stderr,
        )
        return 1
    if not 1 <= args.velocity <= 127:
        print(f"error: --velocity must be in 1..127, got {args.velocity}", file=sys.stderr)
        return 1

    # At 120 BPM this makes 1 tick = 1 sample, so label sample indices are exact.
    ticks_per_beat = args.sample_rate // 2

    # Never-clobber guards
    try:
        resolved_labels = labels_dir.resolve()
        resolved_score = score_midi_dir.resolve()
        resolved_midi = midi_dir.resolve()
        resolved_metadata = metadata_path.resolve()
        resolved_source = source_metadata_path.resolve()
    except Exception:
        # If resolve fails due to missing, fallback to absolute
        resolved_labels = labels_dir.absolute()
        resolved_score = score_midi_dir.absolute()
        resolved_midi = midi_dir.absolute()
        resolved_metadata = metadata_path.absolute()
        resolved_source = source_metadata_path.absolute()

    if resolved_midi == resolved_labels or _is_within(resolved_midi, resolved_labels):
        print(
            f"error: refusing to write --output-midi over --labels input ({labels_dir}); choose a different --output-midi",
            file=sys.stderr,
        )
        return 1
    if resolved_metadata == resolved_labels or _is_within(resolved_metadata, resolved_labels):
        print(
            f"error: refusing to write --output-metadata inside --labels input ({labels_dir}); choose a different --output-metadata",
            file=sys.stderr,
        )
        return 1
    if resolved_midi == resolved_score or _is_within(resolved_midi, resolved_score):
        print(
            f"error: refusing to write --output-midi over --score-midi input ({score_midi_dir}); choose a different --output-midi",
            file=sys.stderr,
        )
        return 1
    if resolved_metadata == resolved_score or _is_within(resolved_metadata, resolved_score):
        print(
            f"error: refusing to write --output-metadata inside --score-midi input ({score_midi_dir}); choose a different --output-metadata",
            file=sys.stderr,
        )
        return 1
    if resolved_metadata == resolved_source:
        print(
            f"error: refusing to write --output-metadata over --source-metadata file ({source_metadata_path}); choose a different --output-metadata",
            file=sys.stderr,
        )
        return 1

    # Validate input dirs/files existence (except labels which may be empty? Check)
    if not labels_dir.is_dir():
        print(f"error: labels directory not found: {labels_dir}", file=sys.stderr)
        return 1
    if not source_metadata_path.is_file():
        print(f"error: source metadata not found: {source_metadata_path}", file=sys.stderr)
        return 1

    # Handle takeover of midi/ via --replace-score-midi or refusal
    if not args.dry_run:
        non_top = _has_non_top_level_midi(midi_dir)
        if non_top:
            if args.replace_score_midi:
                err = _handle_replace_score_midi(midi_dir, score_midi_dir)
                if err is not None:
                    print(f"error: {err}", file=sys.stderr)
                    return 1
            else:
                print(
                    f"error: --output-midi {midi_dir} contains score MIDI tree (e.g. {non_top[0].relative_to(midi_dir)}); "
                    f"pass --replace-score-midi to move it to --score-midi ({score_midi_dir}) or remove it",
                    file=sys.stderr,
                )
                return 1

    # Load source metadata
    try:
        source_by_id, source_columns = _load_source_metadata(source_metadata_path)
    except (OSError, csv.Error) as exc:
        print(f"error: cannot read source metadata {source_metadata_path}: {exc}", file=sys.stderr)
        return 1

    # Find label CSVs
    all_csvs = list(labels_dir.rglob("*.csv"))
    # Filter whose parent is *_labels
    label_csvs = [p for p in all_csvs if p.parent.name.endswith("_labels")]
    # If none, warn but continue (like guitarset)
    if not label_csvs:
        print(f"warning: no label CSVs found under {labels_dir} (looking for '*_labels/*.csv')", file=sys.stderr)

    label_csvs = sorted(label_csvs)

    # Prepare score midi map
    score_map = _find_score_midi_map(score_midi_dir)

    # For dry-run, don't create output dirs
    if not args.dry_run:
        midi_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    n_errors = 0
    total_skipped: Dict[str, int] = {reason: 0 for reason in _SKIP_REASONS}
    total_unisons = 0
    total_unisons_removed = 0
    total_overlaps = 0
    total_notes = 0
    midi_written = 0
    # For provenance sha256s
    input_sha256s: Dict[str, str] = {}
    try:
        input_sha256s[str(source_metadata_path)] = _sha256(source_metadata_path)
    except OSError:
        input_sha256s[str(source_metadata_path)] = ""

    # Also compute sha for each label file lazily per file

    # Whether to write programs
    write_programs = not args.no_program

    for csv_path in label_csvs:
        # Derive id and split
        stem = csv_path.stem
        # id is stem? For musicnet, file like 2104.csv -> id 2104
        file_id = stem
        parent_name = csv_path.parent.name
        if parent_name.endswith("_labels"):
            split = parent_name[: -len("_labels")]
        else:
            split = ""
        # Check for existing midi and overwrite handling
        midi_path = midi_dir / f"{file_id}.mid"
        if midi_path.exists() and not args.overwrite and not args.dry_run:
            # Describe without writing
            try:
                notes, skipped, max_end = _parse_label_csv(csv_path, args.sample_rate, args.velocity)
            except (OSError, ValueError, csv.Error) as exc:
                n_errors += 1
                print(f"error: {file_id}: cannot parse ({exc})", file=sys.stderr)
                records.append(
                    {
                        "filename": file_id,
                        "id": file_id,
                        "split": split,
                        "n_notes": 0,
                        "skipped": {reason: 0 for reason in _SKIP_REASONS},
                        "unisons_detected": 0,
                        "unisons_removed": 0,
                        "same_channel_overlaps": 0,
                        "programs": [],
                        "labels_end_sec": 0.0,
                        "score_midi_filename": score_map.get(file_id, ""),
                        "source_fields": source_by_id.get(file_id, {}),
                        "status": "error",
                        "error": str(exc),
                        "csv_path": str(csv_path),
                    }
                )
                continue
            # Count unisons etc but don't write
            unisons_detected = count_unisons([{"pitch": n["pitch"], "onset": n["onset"]} for n in notes])
            unisons_removed = 0
            if args.dedupe_unisons and unisons_detected:
                # For counting, we need to dedupe using same helper but notes have program/onset etc.
                # Use dedupe_unisons which expects notes with pitch/onset
                deduped, removed = dedupe_unisons(
                    [{"pitch": n["pitch"], "onset": n["onset"], "program": n["program"], "offset": n["offset"]} for n in notes]
                )
                # But dedupe_unisons sorts by onset/pitch; we need to keep original structure?
                # For provenance we just count removed
                unisons_removed = removed
                # For skipped counts we keep original skipped
            overlaps = _count_same_channel_overlaps(notes)
            print(
                f"{file_id}: skipped (exists, pass --overwrite to rewrite)",
                file=sys.stderr,
            )
            records.append(
                {
                    "filename": file_id,
                    "id": file_id,
                    "split": split,
                    "n_notes": len(notes) - unisons_removed,
                    "skipped": dict(skipped),
                    "unisons_detected": unisons_detected,
                    "unisons_removed": unisons_removed,
                    "same_channel_overlaps": overlaps,
                    "programs": sorted({n["program"] for n in notes}),
                    "labels_end_sec": max_end,
                    "score_midi_filename": score_map.get(file_id, ""),
                    "source_fields": source_by_id.get(file_id, {}),
                    "status": "skipped",
                    "error": "",
                    "csv_path": str(csv_path),
                }
            )
            for reason in _SKIP_REASONS:
                total_skipped[reason] += skipped.get(reason, 0)
            total_unisons += unisons_detected
            total_unisons_removed += unisons_removed
            total_overlaps += overlaps
            total_notes += len(notes) - unisons_removed
            # Do not count as midi_written
            continue

        if args.dry_run:
            try:
                notes, skipped, max_end = _parse_label_csv(csv_path, args.sample_rate, args.velocity)
            except (OSError, ValueError, csv.Error) as exc:
                n_errors += 1
                print(f"error: {file_id}: cannot parse ({exc})", file=sys.stderr)
                records.append(
                    {
                        "filename": file_id,
                        "id": file_id,
                        "split": split,
                        "n_notes": 0,
                        "skipped": {reason: 0 for reason in _SKIP_REASONS},
                        "unisons_detected": 0,
                        "unisons_removed": 0,
                        "same_channel_overlaps": 0,
                        "programs": [],
                        "labels_end_sec": 0.0,
                        "score_midi_filename": score_map.get(file_id, ""),
                        "source_fields": source_by_id.get(file_id, {}),
                        "status": "error",
                        "error": str(exc),
                        "csv_path": str(csv_path),
                    }
                )
                continue
            unisons_detected = count_unisons([{"pitch": n["pitch"], "onset": n["onset"]} for n in notes])
            unisons_removed = 0
            if args.dedupe_unisons and unisons_detected:
                _, unisons_removed = dedupe_unisons(
                    [{"pitch": n["pitch"], "onset": n["onset"], "program": n["program"], "offset": n["offset"]} for n in notes]
                )
            overlaps = _count_same_channel_overlaps(notes)
            # Update totals
            for reason in _SKIP_REASONS:
                total_skipped[reason] += skipped.get(reason, 0)
            total_unisons += unisons_detected
            total_unisons_removed += unisons_removed
            total_overlaps += overlaps
            total_notes += len(notes) - unisons_removed
            print(
                f"{file_id}: {len(notes) - unisons_removed} notes, {sum(skipped.values())} skipped, "
                f"{unisons_detected} unisons, {overlaps} overlaps",
                file=sys.stderr,
            )
            records.append(
                {
                    "filename": file_id,
                    "id": file_id,
                    "split": split,
                    "n_notes": len(notes) - unisons_removed,
                    "skipped": dict(skipped),
                    "unisons_detected": unisons_detected,
                    "unisons_removed": unisons_removed,
                    "same_channel_overlaps": overlaps,
                    "programs": sorted({n["program"] for n in notes}),
                    "labels_end_sec": max_end,
                    "score_midi_filename": score_map.get(file_id, ""),
                    "source_fields": source_by_id.get(file_id, {}),
                    "status": "ok",
                    "error": "",
                    "csv_path": str(csv_path),
                }
            )
            # Dry-run does not write
            # Capture sha for provenance
            try:
                input_sha256s[str(csv_path)] = _sha256(csv_path)
            except OSError:
                input_sha256s[str(csv_path)] = ""
            continue

        # Normal conversion
        try:
            notes, skipped, max_end = _parse_label_csv(csv_path, args.sample_rate, args.velocity)
        except (OSError, ValueError, csv.Error) as exc:
            n_errors += 1
            print(f"error: {file_id}: cannot parse ({exc})", file=sys.stderr)
            records.append(
                {
                    "filename": file_id,
                    "id": file_id,
                    "split": split,
                    "n_notes": 0,
                    "skipped": {reason: 0 for reason in _SKIP_REASONS},
                    "unisons_detected": 0,
                    "unisons_removed": 0,
                    "same_channel_overlaps": 0,
                    "programs": [],
                    "labels_end_sec": 0.0,
                    "score_midi_filename": score_map.get(file_id, ""),
                    "source_fields": source_by_id.get(file_id, {}),
                    "status": "error",
                    "error": str(exc),
                    "csv_path": str(csv_path),
                }
            )
            continue

        # Compute unisons
        unison_notes = [{"pitch": n["pitch"], "onset": n["onset"]} for n in notes]
        unisons_detected = count_unisons(unison_notes)
        unisons_removed = 0
        notes_for_midi = notes
        if args.dedupe_unisons and unisons_detected:
            # Need to dedupe keeping earliest; dedupe_unisons expects notes with pitch/onset
            # We'll create list with program/offset preserved
            deduped, removed = dedupe_unisons(
                [{"pitch": n["pitch"], "onset": n["onset"], "program": n["program"], "offset": n["offset"], "velocity": n["velocity"]} for n in notes]
            )
            # deduped is list of those dicts, but we need to map back to notes structure
            # deduped dicts have pitch/onset/program/offset etc; convert to notes_for_midi
            notes_for_midi = [
                {
                    "pitch": d["pitch"],
                    "program": d["program"],
                    "onset": d["onset"],
                    "offset": d["offset"],
                    "velocity": d["velocity"],
                }
                for d in deduped
            ]
            unisons_removed = removed

        overlaps = _count_same_channel_overlaps(notes)

        # Prepare midi notes for writer: pitch, velocity, start_sec, duration_sec, program
        midi_notes = [
            {
                "pitch": n["pitch"],
                "velocity": n["velocity"],
                "start_sec": n["onset"],
                "duration_sec": n["offset"] - n["onset"],
                "program": n["program"],
            }
            for n in notes_for_midi
        ]

        # Capture sha before write
        try:
            input_sha256s[str(csv_path)] = _sha256(csv_path)
        except OSError:
            input_sha256s[str(csv_path)] = ""

        try:
            write_multi_program_midi(
                midi_notes,
                midi_path,
                ticks_per_beat=ticks_per_beat,
                tempo_bpm=_TEMPO_BPM,
                write_programs=write_programs,
            )
        except (OSError, ValueError) as exc:
            n_errors += 1
            print(f"error: {file_id}: failed ({exc})", file=sys.stderr)
            records.append(
                {
                    "filename": file_id,
                    "id": file_id,
                    "split": split,
                    "n_notes": len(notes_for_midi),
                    "skipped": dict(skipped),
                    "unisons_detected": unisons_detected,
                    "unisons_removed": unisons_removed,
                    "same_channel_overlaps": overlaps,
                    "programs": sorted({n["program"] for n in notes}),
                    "labels_end_sec": max_end,
                    "score_midi_filename": score_map.get(file_id, ""),
                    "source_fields": source_by_id.get(file_id, {}),
                    "status": "error",
                    "error": str(exc),
                    "csv_path": str(csv_path),
                }
            )
            continue

        print(
            f"{file_id}: {len(notes_for_midi)} notes, {sum(skipped.values())} skipped, "
            f"{unisons_detected} unisons, {overlaps} overlaps",
            file=sys.stderr,
        )
        # Update totals
        for reason in _SKIP_REASONS:
            total_skipped[reason] += skipped.get(reason, 0)
        total_unisons += unisons_detected
        total_unisons_removed += unisons_removed
        total_overlaps += overlaps
        total_notes += len(notes_for_midi)
        midi_written += 1
        records.append(
            {
                "filename": file_id,
                "id": file_id,
                "split": split,
                "n_notes": len(notes_for_midi),
                "skipped": dict(skipped),
                "unisons_detected": unisons_detected,
                "unisons_removed": unisons_removed,
                "same_channel_overlaps": overlaps,
                "programs": sorted({n["program"] for n in notes}),
                "labels_end_sec": max_end,
                "score_midi_filename": score_map.get(file_id, ""),
                "source_fields": source_by_id.get(file_id, {}),
                "status": "ok",
                "error": "",
                "csv_path": str(csv_path),
            }
        )

    # After loop, handle case where some files were skipped due to existing but we still need to count their sha? Already done?

    # Summary totals for provenance: totals across all records that are not error?
    # For dry-run, midi_written is 0 as above

    rows: List[Dict[str, str]] = []
    for rec in records:
        if rec["status"] == "error":
            continue
        src_fields = rec["source_fields"]
        row: Dict[str, str] = {
            "midi_filename": rec["filename"],
            "id": rec["id"],
            "split": rec["split"],
            "score_midi_filename": rec["score_midi_filename"],
            "n_notes": str(rec["n_notes"]),
            "n_programs": str(len(rec["programs"])),
            "programs": ",".join(str(p) for p in rec["programs"]),
            "labels_end_sec": str(rec["labels_end_sec"]),
            "unisons_detected": str(rec["unisons_detected"]),
            "same_channel_overlaps": str(rec["same_channel_overlaps"]),
        }
        # Add every source metadata column, default empty if missing
        for col in _SOURCE_METADATA_COLUMNS:
            row[col] = src_fields.get(col, "") if src_fields else ""
        # Also include any extra columns from source that are not in predefined list? Use source_columns
        # But per spec, include every musicnet_metadata.csv column; we already handle the 8 known ones.
        # If source has extra like 'id', we already have id; but handle generically?
        for col in source_columns:
            if col not in row:
                row[col] = src_fields.get(col, "") if src_fields else ""
        rows.append(row)

    # Sort rows by midi_filename for determinism
    rows = sorted(rows, key=lambda r: r["midi_filename"])

    if not args.dry_run:
        # Write metadata CSV
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_METADATA_COLUMNS, restval="")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            print(f"error: cannot write metadata {metadata_path}: {exc}", file=sys.stderr)
            return 1

        # Provenance
        provenance = {
            "labels": str(labels_dir),
            "output_midi": str(midi_dir),
            "output_metadata": str(metadata_path),
            "source_metadata": str(source_metadata_path),
            "score_midi": str(score_midi_dir),
            "argv": effective_argv,
            "sample_rate": args.sample_rate,
            "ticks_per_beat": ticks_per_beat,
            "tempo_bpm": _TEMPO_BPM,
            "velocity": args.velocity,
            "write_programs": write_programs,
            "dedupe_unisons": args.dedupe_unisons,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "replace_score_midi": args.replace_score_midi,
            "unison_onset_tolerance_sec": _UNISON_ONSET_TOLERANCE_SEC,
            "labels_read": len(label_csvs),
            "midi_written": midi_written,
            "n_notes": total_notes,
            "skipped": total_skipped,
            "unisons_detected": total_unisons,
            "unisons_removed": total_unisons_removed,
            "same_channel_overlaps": total_overlaps,
            "n_errors": n_errors,
            "inputs_sha256": input_sha256s,
            "files": [
                {
                    "filename": rec["filename"],
                    "id": rec["id"],
                    "split": rec["split"],
                    "n_notes": rec["n_notes"],
                    "skipped": rec["skipped"],
                    "unisons_detected": rec["unisons_detected"],
                    "unisons_removed": rec["unisons_removed"],
                    "same_channel_overlaps": rec["same_channel_overlaps"],
                    "programs": rec["programs"],
                    "labels_end_sec": rec["labels_end_sec"],
                    "score_midi_filename": rec["score_midi_filename"],
                    "status": rec["status"],
                    "error": rec["error"],
                }
                for rec in records
            ],
        }
        provenance_path = Path(str(metadata_path) + ".provenance.json")
        try:
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write provenance {provenance_path}: {exc}", file=sys.stderr)
            return 1

    if args.dry_run:
        print(
            f"dry run: would write {sum(1 for r in records if r['status'] != 'error')} MIDI files "
            f"({total_notes} notes, {sum(total_skipped.values())} skipped) from {len(label_csvs)} label CSVs in {labels_dir}"
        )
    else:
        print(
            f"wrote {midi_written} MIDI files ({total_notes} notes, {sum(total_skipped.values())} skipped) "
            f"from {len(label_csvs)} label CSVs to {midi_dir}; metadata: {len(rows)} rows to {metadata_path}"
        )
    if n_errors:
        print(f"warning: {n_errors}/{len(label_csvs)} files failed (see provenance)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
