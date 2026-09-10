"""Convert GuitarSet JAMS annotations to MIDI reference files + metadata CSV.

Reads ``corpus/guitarset/annotations/*.jams`` (GuitarSet ground truth ships as
JAMS, not MIDI) and writes one ``*.mid`` per excerpt (unsuffixed stems) plus a
metadata CSV and a JSON provenance sidecar.

Conversion semantics (ported from ``marl/GuitarSet`` ``interpreter.py``):

- The six per-string ``note_midi`` blocks are merged into one note list. Blocks
  are selected by ``annotation_metadata.data_source == "0".."5"`` (strings),
  never by position -- they are interleaved with ``pitch_contour`` blocks in
  real files. Unknown namespaces are ignored.
- Both JAMS ``data`` layouts are tolerated *per namespace*: list-of-
  observations (``[{time, duration, value, confidence}]``, what ``note_midi``
  uses) and dict-of-arrays (``{time: [...], duration: [...], ...}``, what
  ``pitch_contour`` uses). Missing/empty data, null confidences, and empty
  per-string blocks are tolerated.
- ``pitch = round(float(value))``; ``onset = time``; ``offset = time +
  duration``.
- Constant ``--velocity`` (default 100): GuitarSet has no dynamics, so
  ``interpreter.py``'s ``100 + np.random.choice(range(-5, 5))`` fabricated
  noise is deliberately *not* reproduced.
- Pitch is guarded to 0-127 and non-positive durations are skipped, both
  counted in the provenance. ``sonitra.midi_writer.write_midi`` clamps
  velocity but not pitch, so an out-of-range value would raise from ``mido``.
  Real guitar range is ~40-88, so this guard is defensive only.
- Unison detection: reference notes sharing a pitch within the onset tolerance
  (two strings sounding the same note) are unavoidable false negatives under
  1-to-1 bipartite matching. They are counted and *kept* by default;
  ``--dedupe-unisons`` is opt-in (keeps the earliest onset per pitch group)
  because deleting reference notes inflates recall and silently changes the
  ground truth.
- Tempo is written as 120 BPM; timings are absolute seconds and
  ``parse_midi`` reads them back unscaled.
- A GM ``program_change`` (default 24, Acoustic Guitar (nylon) -- GuitarSet
  was recorded on a nylon-string guitar) is written on channel 0 before the
  first note, so General MIDI players do not fall back to their default of
  program 0, Acoustic Grand Piano. Purely cosmetic: ``parse_midi`` ignores
  program, so no metric sees it. ``--no-program`` restores the bare
  note-only files.

Filenames encode ``{player}_{style}{progression}-{tempo}-{key}_{comp|solo}``
(e.g. ``00_BN1-129-Eb_comp``); style codes map to full names as in mirdata
(``BN`` -> ``Bossa Nova``, ``SS`` -> ``Singer-Songwriter``) with unknown codes
kept raw.

Usage:
    python scripts/guitarset_jams_to_midi.py --dry-run
    python scripts/guitarset_jams_to_midi.py --velocity 100
    python scripts/guitarset_jams_to_midi.py --program 27 --overwrite
    python scripts/guitarset_jams_to_midi.py --dedupe-unisons --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sonitra.midi_writer import write_midi  # noqa: E402

#: Onset tolerance for unison detection (two strings, same pitch). Matches the
#: onset tolerance used by the note metrics, so a counted unison is exactly a
#: pair the evaluator cannot match 1-to-1.
_UNISON_ONSET_TOLERANCE_SEC = 0.05

#: Tempo written into every output MIDI. Timings are absolute seconds;
#: ``parse_midi`` reads them back unscaled, so this only labels the file.
_TEMPO_BPM = 120.0

#: GM program written into every output MIDI: 24 is Acoustic Guitar (nylon),
#: matching GuitarSet's nylon-string guitar. Playback timbre only -- the
#: evaluator reads notes, never program.
_DEFAULT_GM_PROGRAM = 24

#: Style-code mapping mirroring mirdata's GuitarSet ``_STYLE_DICT``.
_STYLE_DICT = {
    "BN": "Bossa Nova",
    "SS": "Singer-Songwriter",
    "Jazz": "Jazz",
    "Rock": "Rock",
    "Funk": "Funk",
}

#: Valid per-string ``data_source`` values (strings, never ints/positions).
_STRING_SOURCES = frozenset({"0", "1", "2", "3", "4", "5"})

#: ``{player}_{style}{progression}-{tempo}-{key}_{comp|solo}``.
_FILENAME_RE = re.compile(
    r"^(?P<player>[^_]+)_"
    r"(?P<styleprog>[^-]+)-"
    r"(?P<tempo>[^-]+)-"
    r"(?P<key>[^_]+)_"
    r"(?P<mode>comp|solo)$"
)

#: Multi-letter style code + trailing progression digit(s), e.g. ``BN1``.
_STYLE_PROG_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

_METADATA_COLUMNS = [
    "midi_filename",
    "player_id",
    "style",
    "progression",
    "tempo_bpm",
    "key",
    "mode",
    "duration",
    "tempo_measured",
    "key_mode",
    "n_notes",
    "n_strings_used",
]

_SKIP_REASONS = ("out_of_range_pitch", "non_positive_duration", "invalid_value")


def parse_filename(stem: str) -> Dict[str, str]:
    """Parse a GuitarSet excerpt stem into its metadata fields.

    Args:
        stem: JAMS/MIDI stem such as ``"00_BN1-129-Eb_comp"``.

    Returns:
        Dict with ``player_id``, ``style`` (full name via ``_STYLE_DICT``,
        raw code when unknown), ``progression``, ``tempo_bpm``, ``key``
        (``#`` preserved, e.g. ``"C#"``) and ``mode`` (``"comp"``/``"solo"``).

    Raises:
        ValueError: If *stem* does not match the expected pattern.
    """
    match = _FILENAME_RE.match(stem)
    if match is None:
        raise ValueError(
            f"filename does not match "
            f"'{{player}}_{{style}}{{prog}}-{{tempo}}-{{key}}_{{comp|solo}}': {stem!r}"
        )
    styleprog = match.group("styleprog")
    style_match = _STYLE_PROG_RE.match(styleprog)
    if style_match is None:
        code, progression = styleprog, ""
    else:
        code, progression = style_match.group(1), style_match.group(2)
    return {
        "player_id": match.group("player"),
        "style": _STYLE_DICT.get(code, code),
        "progression": progression,
        "tempo_bpm": match.group("tempo"),
        "key": match.group("key"),
        "mode": match.group("mode"),
    }


def _iter_observation_dicts(data: Any) -> Iterator[Dict[str, Any]]:
    """Normalize either JAMS ``data`` layout to observation dicts.

    Tolerates list-of-observations (``note_midi``) and dict-of-arrays
    (``pitch_contour``). Missing, empty, or unrecognized ``data`` yields no
    observations. Confidence values (including nulls) are passed through
    untouched -- callers ignore them.
    """
    if data is None:
        return
    if isinstance(data, list):
        for obs in data:
            if isinstance(obs, dict):
                yield obs
        return
    if isinstance(data, dict):
        times = data.get("time")
        durations = data.get("duration")
        values = data.get("value")
        confidences = data.get("confidence")
        if times is None or durations is None or values is None:
            return
        if not all(
            isinstance(seq, (list, tuple)) for seq in (times, durations, values)
        ):
            return
        if not isinstance(confidences, (list, tuple)):
            confidences = []
        count = min(len(times), len(durations), len(values))
        for index in range(count):
            yield {
                "time": times[index],
                "duration": durations[index],
                "value": values[index],
                "confidence": confidences[index] if index < len(confidences) else None,
            }
        return
    # Unknown layout: tolerated, ignored.


def extract_notes(
    doc: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """Merge the six per-string ``note_midi`` blocks into one note list.

    Args:
        doc: Parsed JAMS document.

    Returns:
        ``(notes, skipped, strings_used)`` where each note has ``pitch``
        (rounded int), ``onset``/``offset`` (seconds) and ``data_source``;
        ``skipped`` counts rejections by reason; ``strings_used`` lists the
        sorted data sources contributing at least one kept note.
    """
    notes: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {reason: 0 for reason in _SKIP_REASONS}
    strings: set = set()
    annotations = doc.get("annotations") or []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("namespace") != "note_midi":
            continue
        metadata = annotation.get("annotation_metadata") or {}
        source = metadata.get("data_source")
        source_str = str(source) if source is not None else ""
        if source_str not in _STRING_SOURCES:
            continue
        for obs in _iter_observation_dicts(annotation.get("data")):
            try:
                value = float(obs.get("value"))  # type: ignore[arg-type]
                onset = float(obs.get("time"))  # type: ignore[arg-type]
                duration = float(obs.get("duration"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                skipped["invalid_value"] += 1
                continue
            if not (
                math.isfinite(value)
                and math.isfinite(onset)
                and math.isfinite(duration)
            ):
                skipped["invalid_value"] += 1
                continue
            if duration <= 0.0:
                skipped["non_positive_duration"] += 1
                continue
            pitch = int(round(value))
            if pitch < 0 or pitch > 127:
                skipped["out_of_range_pitch"] += 1
                continue
            notes.append(
                {
                    "pitch": pitch,
                    "onset": onset,
                    "offset": onset + duration,
                    "data_source": source_str,
                }
            )
            strings.add(source_str)
    notes.sort(key=lambda note: (note["onset"], note["pitch"]))
    return (notes, skipped, sorted(strings))


def count_unisons(
    notes: List[Dict[str, Any]],
    tolerance: float = _UNISON_ONSET_TOLERANCE_SEC,
) -> int:
    """Count notes sharing a pitch within *tolerance* of a group onset.

    Notes are grouped per pitch by ascending onset; the first note opens a
    group and every later note within *tolerance* of the group onset is one
    detected unison. A note further than *tolerance* away opens a new group.
    """
    by_pitch: Dict[int, List[float]] = {}
    for note in notes:
        by_pitch.setdefault(note["pitch"], []).append(note["onset"])
    detected = 0
    for onsets in by_pitch.values():
        onsets.sort()
        group_start: Optional[float] = None
        for onset in onsets:
            if group_start is None or onset - group_start > tolerance:
                group_start = onset
            else:
                detected += 1
    return detected


def dedupe_unisons(
    notes: List[Dict[str, Any]],
    tolerance: float = _UNISON_ONSET_TOLERANCE_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop unison duplicates, keeping the earliest onset per pitch group.

    Grouping is identical to :func:`count_unisons`; the return value is
    ``(kept_notes, n_removed)`` with kept notes re-sorted by onset.
    """
    by_pitch: Dict[int, List[Dict[str, Any]]] = {}
    for note in notes:
        by_pitch.setdefault(note["pitch"], []).append(note)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for group in by_pitch.values():
        group.sort(key=lambda note: note["onset"])
        group_start: Optional[float] = None
        for note in group:
            if group_start is None or note["onset"] - group_start > tolerance:
                group_start = note["onset"]
                kept.append(note)
            else:
                removed += 1
    kept.sort(key=lambda note: (note["onset"], note["pitch"]))
    return (kept, removed)


def _first_annotation_value(doc: Dict[str, Any], namespace: str) -> str:
    """Return the first observed ``value`` for *namespace*, or ``""``."""
    for annotation in doc.get("annotations") or []:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("namespace") != namespace:
            continue
        for obs in _iter_observation_dicts(annotation.get("data")):
            value = obs.get("value")
            if value is None:
                continue
            return str(value)
    return ""


def _duration_of(doc: Dict[str, Any]) -> str:
    """Excerpt duration from ``file_metadata.duration``, or ``""``."""
    metadata = doc.get("file_metadata") or {}
    duration = metadata.get("duration")
    if duration is None:
        return ""
    return str(duration)


def convert_file(
    jams_path: Path,
    midi_path: Path,
    velocity: int,
    dedupe: bool,
    program: Optional[int] = _DEFAULT_GM_PROGRAM,
) -> Dict[str, Any]:
    """Convert one JAMS file to MIDI, returning its provenance record.

    Raises:
        OSError: If the JAMS file cannot be read or the MIDI cannot be written.
        ValueError: If the JAMS content is not valid JSON / not an object.
    """
    stem = jams_path.stem
    try:
        doc = json.loads(jams_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {jams_path.name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"JAMS top level must be an object: {jams_path.name}")

    notes, skipped, strings_used = extract_notes(doc)
    unisons_detected = count_unisons(notes)
    unisons_removed = 0
    if dedupe and unisons_detected:
        notes, unisons_removed = dedupe_unisons(notes)

    midi_notes = [
        {
            "pitch": note["pitch"],
            "velocity": velocity,
            "start_sec": note["onset"],
            "duration_sec": note["offset"] - note["onset"],
        }
        for note in notes
    ]
    write_midi(midi_notes, midi_path, tempo_bpm=_TEMPO_BPM, program=program)

    try:
        file_fields = parse_filename(stem)
    except ValueError as exc:
        print(f"warning: {exc} -- metadata fields left blank", file=sys.stderr)
        file_fields = {
            "player_id": "",
            "style": "",
            "progression": "",
            "tempo_bpm": "",
            "key": "",
            "mode": "",
        }

    duration = _duration_of(doc)
    return {
        "filename": stem,
        "n_notes": len(notes),
        "skipped": dict(skipped),
        "unisons_detected": unisons_detected,
        "unisons_removed": unisons_removed,
        "strings_used": strings_used,
        "duration": duration,
        "tempo_measured": _first_annotation_value(doc, "tempo"),
        "key_mode": _first_annotation_value(doc, "key_mode"),
        "file_fields": file_fields,
        "status": "ok",
        "error": "",
    }


def _describe_file(jams_path: Path, dedupe: bool) -> Dict[str, Any]:
    """Parse one JAMS file into a provenance record without writing MIDI.

    Returns a record dict with ``status == "ok"`` (callers may override, e.g.
    to ``"skipped"``); raises ``OSError``/``ValueError`` on unreadable input.
    """
    stem = jams_path.stem
    doc = json.loads(jams_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"JAMS top level must be an object: {jams_path.name}")
    notes, skipped, strings_used = extract_notes(doc)
    unisons_detected = count_unisons(notes)
    unisons_removed = 0
    if dedupe and unisons_detected:
        notes, unisons_removed = dedupe_unisons(notes)
    return {
        "filename": stem,
        "n_notes": len(notes),
        "skipped": dict(skipped),
        "unisons_detected": unisons_detected,
        "unisons_removed": unisons_removed,
        "strings_used": strings_used,
        "duration": _duration_of(doc),
        "tempo_measured": _first_annotation_value(doc, "tempo"),
        "key_mode": _first_annotation_value(doc, "key_mode"),
        "file_fields": _safe_file_fields(stem),
        "status": "ok",
        "error": "",
    }


def _error_record(stem: str, exc: Exception) -> Dict[str, Any]:
    """Blank provenance record for a file that could not be processed."""
    return {
        "filename": stem,
        "n_notes": 0,
        "skipped": {reason: 0 for reason in _SKIP_REASONS},
        "unisons_detected": 0,
        "unisons_removed": 0,
        "strings_used": [],
        "duration": "",
        "tempo_measured": "",
        "key_mode": "",
        "file_fields": _safe_file_fields(stem),
        "status": "error",
        "error": str(exc),
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return path.is_relative_to(parent)
    except AttributeError:  # Python < 3.9 fallback
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root", default=Path("corpus"), type=Path,
        help="Corpus root directory (default: corpus).",
    )
    parser.add_argument(
        "--dataset", default="guitarset",
        help="Dataset subdirectory under --corpus-root (default: guitarset).",
    )
    parser.add_argument(
        "--annotations", default=None, type=Path,
        help="JAMS input directory "
        "(default: <corpus-root>/<dataset>/annotations).",
    )
    parser.add_argument(
        "--output-midi", default=None, type=Path,
        help="MIDI output directory (default: <corpus-root>/<dataset>/midi).",
    )
    parser.add_argument(
        "--output-metadata", default=None, type=Path,
        help="Metadata CSV output path "
        "(default: <corpus-root>/<dataset>/metadata/guitarset.csv).",
    )
    parser.add_argument(
        "--velocity", default=100, type=int,
        help="Constant MIDI velocity for every note (GuitarSet has no dynamics; "
        "default: 100).",
    )
    parser.add_argument(
        "--program", default=_DEFAULT_GM_PROGRAM, type=int,
        help=f"GM program (0-127) written as a program_change on channel 0 so "
        f"players do not default to piano (default: {_DEFAULT_GM_PROGRAM}, "
        f"Acoustic Guitar (nylon)). Playback timbre only; no metric reads it.",
    )
    parser.add_argument(
        "--no-program", action="store_true",
        help="Write no program_change at all (note-only files, as before "
        "--program existed). Overrides --program.",
    )
    parser.add_argument(
        "--dedupe-unisons", action="store_true",
        help="Remove unison duplicates (keep earliest onset per pitch group "
        "within tolerance). Off by default: deleting reference notes inflates "
        "recall and silently changes the ground truth.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rewrite existing .mid files (default: skip them).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and count without writing any MIDI, CSV, or provenance.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    annotations_dir: Path = (
        args.annotations or (args.corpus_root / args.dataset / "annotations")
    )
    midi_dir: Path = args.output_midi or (args.corpus_root / args.dataset / "midi")
    metadata_path: Path = (
        args.output_metadata
        or (args.corpus_root / args.dataset / "metadata" / "guitarset.csv")
    )

    resolved_annotations = annotations_dir.resolve()
    resolved_midi = midi_dir.resolve()
    resolved_metadata = metadata_path.resolve()
    if resolved_midi == resolved_annotations or _is_within(
        resolved_midi, resolved_annotations
    ):
        print(
            f"error: refusing to write --output-midi over --annotations input "
            f"({annotations_dir}); choose a different --output-midi",
            file=sys.stderr,
        )
        return 1
    if resolved_metadata == resolved_annotations or _is_within(
        resolved_metadata, resolved_annotations
    ):
        print(
            f"error: refusing to write --output-metadata inside --annotations input "
            f"({annotations_dir}); choose a different --output-metadata",
            file=sys.stderr,
        )
        return 1

    if not 1 <= args.velocity <= 127:
        print(
            f"error: --velocity must be in 1..127, got {args.velocity}",
            file=sys.stderr,
        )
        return 1

    if not 0 <= args.program <= 127:
        print(
            f"error: --program must be in 0..127, got {args.program}",
            file=sys.stderr,
        )
        return 1
    program: Optional[int] = None if args.no_program else args.program

    if not annotations_dir.is_dir():
        print(
            f"error: annotations directory not found: {annotations_dir}",
            file=sys.stderr,
        )
        return 1

    jams_files = sorted(annotations_dir.glob("*.jams"))
    if not jams_files:
        print(
            f"warning: no .jams files found in {annotations_dir}",
            file=sys.stderr,
        )

    if not args.dry_run:
        midi_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    n_errors = 0
    for jams_path in jams_files:
        stem = jams_path.stem
        midi_path = midi_dir / f"{stem}.mid"
        if midi_path.exists() and not args.overwrite and not args.dry_run:
            try:
                record = _describe_file(jams_path, args.dedupe_unisons)
            except (OSError, ValueError) as exc:
                n_errors += 1
                print(f"error: {stem}: cannot parse ({exc})", file=sys.stderr)
                records.append(_error_record(stem, exc))
                continue
            print(
                f"{stem}: skipped (exists, pass --overwrite to rewrite)",
                file=sys.stderr,
            )
            record["status"] = "skipped"
            records.append(record)
            continue
        if args.dry_run:
            try:
                record = _describe_file(jams_path, args.dedupe_unisons)
            except (OSError, ValueError) as exc:
                n_errors += 1
                print(f"error: {stem}: cannot parse ({exc})", file=sys.stderr)
                records.append(_error_record(stem, exc))
                continue
            print(
                f"{stem}: {record['n_notes']} notes, "
                f"{sum(record['skipped'].values())} skipped, "
                f"{record['unisons_detected']} unisons "
                f"({len(record['strings_used'])} strings)",
                file=sys.stderr,
            )
            records.append(record)
            continue
        try:
            record = convert_file(
                jams_path, midi_path, args.velocity, args.dedupe_unisons, program
            )
        except (OSError, ValueError) as exc:
            n_errors += 1
            print(f"error: {stem}: failed ({exc})", file=sys.stderr)
            records.append(_error_record(stem, exc))
            continue
        print(
            f"{stem}: {record['n_notes']} notes, "
            f"{sum(record['skipped'].values())} skipped, "
            f"{record['unisons_detected']} unisons "
            f"({len(record['strings_used'])} strings)",
            file=sys.stderr,
        )
        records.append(record)

    total_skipped: Dict[str, int] = {reason: 0 for reason in _SKIP_REASONS}
    for record in records:
        for reason in _SKIP_REASONS:
            total_skipped[reason] += record["skipped"].get(reason, 0)
    if args.dry_run:
        midi_written = 0
    else:
        midi_written = sum(1 for record in records if record["status"] == "ok")
    total_notes = sum(record["n_notes"] for record in records if record["status"] != "error")
    total_unisons = sum(record["unisons_detected"] for record in records)
    total_removed = sum(record["unisons_removed"] for record in records)

    rows: List[Dict[str, str]] = []
    for record in records:
        if record["status"] == "error":
            continue
        fields = record["file_fields"]
        rows.append(
            {
                "midi_filename": record["filename"],
                "player_id": fields["player_id"],
                "style": fields["style"],
                "progression": fields["progression"],
                "tempo_bpm": fields["tempo_bpm"],
                "key": fields["key"],
                "mode": fields["mode"],
                "duration": record["duration"],
                "tempo_measured": record["tempo_measured"],
                "key_mode": record["key_mode"],
                "n_notes": str(record["n_notes"]),
                "n_strings_used": str(len(record["strings_used"])),
            }
        )

    if not args.dry_run:
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_METADATA_COLUMNS, restval="")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            print(
                f"error: cannot write metadata {metadata_path}: {exc}",
                file=sys.stderr,
            )
            return 1

        provenance = {
            "annotations": str(annotations_dir),
            "output_midi": str(midi_dir),
            "output_metadata": str(metadata_path),
            "argv": effective_argv,
            "velocity": args.velocity,
            "program": program,
            "dedupe_unisons": args.dedupe_unisons,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "unison_onset_tolerance_sec": _UNISON_ONSET_TOLERANCE_SEC,
            "tempo_bpm": _TEMPO_BPM,
            "jams_read": len(jams_files),
            "midi_written": midi_written,
            "n_notes": total_notes,
            "skipped": total_skipped,
            "unisons_detected": total_unisons,
            "unisons_removed": total_removed,
            "n_errors": n_errors,
            "files": [
                {
                    "filename": record["filename"],
                    "n_notes": record["n_notes"],
                    "skipped": record["skipped"],
                    "unisons_detected": record["unisons_detected"],
                    "unisons_removed": record["unisons_removed"],
                    "strings_used": record["strings_used"],
                    "duration": record["duration"],
                    "status": record["status"],
                    "error": record["error"],
                }
                for record in records
            ],
        }
        provenance_path = Path(str(metadata_path) + ".provenance.json")
        try:
            provenance_path.write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(
                f"error: cannot write provenance {provenance_path}: {exc}",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print(
            f"dry run: would write {sum(1 for r in records if r['status'] != 'error')} "
            f"MIDI files ({total_notes} notes, "
            f"{sum(total_skipped.values())} skipped) "
            f"from {len(jams_files)} JAMS in {annotations_dir}"
        )
    else:
        print(
            f"wrote {midi_written} MIDI files ({total_notes} notes, "
            f"{sum(total_skipped.values())} skipped) from {len(jams_files)} JAMS "
            f"to {midi_dir}; metadata: {len(rows)} rows to {metadata_path}"
        )
    if n_errors:
        print(
            f"warning: {n_errors}/{len(jams_files)} files failed (see provenance)",
            file=sys.stderr,
        )
        return 1
    return 0


def _safe_file_fields(stem: str) -> Dict[str, str]:
    """Filename fields, blanked with a warning when the stem is unexpected."""
    try:
        return parse_filename(stem)
    except ValueError as exc:
        print(f"warning: {exc} -- metadata fields left blank", file=sys.stderr)
        return {
            "player_id": "",
            "style": "",
            "progression": "",
            "tempo_bpm": "",
            "key": "",
            "mode": "",
        }


if __name__ == "__main__":
    sys.exit(main())
