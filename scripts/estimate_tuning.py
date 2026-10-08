r"""Measure each recording's own tuning offset from A440, in cents.

Reports a per-recording table and a per-reference median, and never corrects
audio. One whole-file number cannot see register-dependent tuning, such as a
piano's stretched octaves or a guitar's fret intonation.

Examples:
    # MAESTRO: one row per reference, keyed by the metadata's own file column
    python scripts/estimate_tuning.py \
        --recordings corpus/maestro-v3/recordings \
        --references corpus/maestro-v3/midi \
        --metadata corpus/maestro-v3/metadata/maestro-v3.0.0.csv \
        --join-column midi_filename \
        --output corpus/maestro-v3/metadata/maestro-v3.0.0-own-tuning.csv

    # GuitarSet: _mic and _mix pair to one reference, so both land on one row
    python scripts/estimate_tuning.py \
        --recordings corpus/guitarset/recordings \
        --references corpus/guitarset/midi \
        --metadata corpus/guitarset/metadata/metadata.csv --join-column id \
        --output corpus/guitarset/metadata/guitarset-own-tuning.csv

    # attach the result to the metadata CSV afterwards
    #   python scripts/enrich_metadata.py \
    #       --metadata corpus/guitarset/metadata/metadata.csv \
    #       --annotations corpus/guitarset/metadata/guitarset-own-tuning.csv \
    #       --on id=id --add own_tuning_cents=own_tuning_cents \
    #       --output corpus/guitarset/metadata/metadata-with-tuning.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np

from sonitra.corpus import discover_audio_files, discover_midi_files, pair_audio_to_reference

logger = logging.getLogger(__name__)

#: librosa reports tuning in bins, not cents, so a raw estimate converts with
#: ``raw * 1200 / _BINS_PER_OCTAVE``; at 12 bins per octave a bin is 100 cents.
#: Its histogram is built over one bin centred on zero, so a recording more than
#: half a semitone from A440 folds onto the opposite edge instead of reading its
#: true distance.
_BINS_PER_OCTAVE = 12

#: Cell substitutions applied to every string written to either CSV. After them
#: no cell can hold a comma, a double quote, a CR or an LF, so nothing needs
#: quoting and a hostile file name cannot shift the columns beside it.
_CELL_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (",", ";"),
    ('"', "'"),
    ("\r", " "),
    ("\n", " "),
)

_REFERENCE_COLUMNS: tuple[str, ...] = (
    "reference_stem",
    "n_recordings",
    "own_tuning_cents",
    "own_tuning_spread_cents",
)
_RECORDING_COLUMNS: tuple[str, ...] = (
    "recording",
    "reference_stem",
    "own_tuning_cents",
    "duration_sec",
    "status",
    "error",
)

#: Analysis window of the per-block pitch tracker. Large enough to resolve the
#: low partial of a bass note at 22.05 kHz, small enough that its frame-by-bin
#: magnitude array stays cheap.
_BLOCK_N_FFT = 2048


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list without the program name; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--recordings", required=True, type=Path,
        help="Directory of recordings to measure (searched recursively).",
    )
    parser.add_argument(
        "--references", required=True, type=Path,
        help="Directory of reference MIDI files, used to group recordings.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Per-reference CSV, one row per paired reference (never an input).",
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Dataset metadata CSV supplying the join column's exact value.",
    )
    parser.add_argument(
        "--join-column", default="midi_filename",
        help="Metadata column naming each recording's reference (default: "
        "midi_filename); its stem is matched against the reference file stem.",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=22050,
        help="Analysis sample rate every recording is decoded at (default: 22050).",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.01,
        help="Tuning resolution as a fraction of a bin, so 0.01 is one cent "
        "(default: 0.01).",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=60.0,
        help="Analyse fixed blocks of this many seconds and pool their pitches, "
        "which keeps peak memory proportional to the block rather than to the "
        "whole recording; 0 analyses each file in one piece (default: 60.0).",
    )
    return parser.parse_args(argv)


def _sanitize(value: str) -> str:
    """Return *value* with every character that could break a CSV cell replaced.

    Args:
        value: Raw text destined for a cell.

    Returns:
        The text with commas, double quotes, CRs and LFs replaced.
    """
    for bad, good in _CELL_SUBSTITUTIONS:
        value = value.replace(bad, good)
    return value


def _bins_to_cents(raw: float) -> float:
    """Convert a librosa tuning estimate from bins to cents.

    Args:
        raw: Estimate as returned by ``librosa.estimate_tuning`` or
            ``librosa.pitch_tuning``, both of which report bins.

    Returns:
        The same estimate in cents.
    """
    return raw * 1200.0 / _BINS_PER_OCTAVE


def _whole_file_cents(
    y: np.ndarray, sample_rate: float, resolution: float
) -> float:
    """Estimate the tuning of one signal in a single piece.

    Args:
        y: Mono samples.
        sample_rate: Sample rate of *y* in Hz.
        resolution: Tuning resolution as a fraction of a bin.

    Returns:
        The estimated offset from A440 in cents.
    """
    raw = librosa.estimate_tuning(
        y=y, sr=sample_rate, resolution=resolution, bins_per_octave=_BINS_PER_OCTAVE
    )
    return _bins_to_cents(float(raw))


def _block_frequencies(block: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return the frequencies of the strongest partials in one block.

    A block's own median magnitude is its threshold: without it a block's
    weakest partials would weigh as much as its strongest and drag the pooled
    estimate off the tuning the recording actually has.

    Args:
        block: Mono samples of one full block.
        sample_rate: Sample rate of *block* in Hz.

    Returns:
        Tracked frequencies in Hz above the block's magnitude threshold.
    """
    pitch, magnitude = librosa.piptrack(y=block, sr=sample_rate, n_fft=_BLOCK_N_FFT)
    voiced = pitch > 0
    if not bool(voiced.any()):
        return np.empty(0, dtype=float)
    threshold = float(np.median(magnitude[voiced]))
    return np.asarray(pitch[(magnitude >= threshold) & voiced], dtype=float).ravel()


def _chunked_cents(
    y: np.ndarray, sample_rate: float, resolution: float, chunk_seconds: float
) -> float:
    """Estimate the tuning of one signal from pooled per-block pitches.

    Only whole blocks count: a recording long enough for several blocks would
    otherwise end in a partial block whose pitches bias the pool.

    Args:
        y: Mono samples.
        sample_rate: Sample rate of *y* in Hz.
        resolution: Tuning resolution as a fraction of a bin.
        chunk_seconds: Block length in seconds.

    Returns:
        The estimated offset from A440 in cents.

    Raises:
        ValueError: No block carries a partial above its own threshold.
    """
    block_samples = int(chunk_seconds * sample_rate)
    pooled: list[np.ndarray] = []
    for start in range(0, len(y) - block_samples + 1, block_samples):
        pooled.append(_block_frequencies(y[start : start + block_samples], sample_rate))
    frequencies = np.concatenate(pooled) if pooled else np.empty(0, dtype=float)
    if frequencies.size == 0:
        raise ValueError(
            f"no pitched content: no block yielded a partial above its own median "
            f"magnitude over {len(pooled)} block(s)"
        )
    raw = librosa.pitch_tuning(
        frequencies, resolution=resolution, bins_per_octave=_BINS_PER_OCTAVE
    )
    return _bins_to_cents(float(raw))


def _measure(
    path: Path, sample_rate: float, resolution: float, chunk_seconds: float
) -> tuple[float, float, bool]:
    """Measure one recording's own tuning offset from A440.

    The whole file is always covered: truncating to the opening seconds would
    miss a detuned passage, and a 45-minute file decoded at 22.05 kHz reaches
    3.75 GB of peak memory in one piece, so anything longer than one block is
    pooled block by block.

    Args:
        path: Recording to measure.
        sample_rate: Sample rate to decode at in Hz.
        resolution: Tuning resolution as a fraction of a bin.
        chunk_seconds: Block length in seconds; ``0`` disables block pooling.

    Returns:
        A tuple of the offset in cents, the duration in seconds, and whether
        the pooled block path ran.

    Raises:
        ValueError: The recording carries no pitched content, or the estimate
            came back non-finite, so no honest number can be reported.
    """
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0 or float(np.max(np.abs(y))) == 0.0:
        raise ValueError("no pitched content: the recording is silent")
    duration_sec = len(y) / sample_rate
    block_samples = int(chunk_seconds * sample_rate) if chunk_seconds > 0 else 0
    if block_samples and len(y) // block_samples >= 2:
        cents = _chunked_cents(y, sample_rate, resolution, chunk_seconds)
        chunked = True
    else:
        cents = _whole_file_cents(y, sample_rate, resolution)
        chunked = False
    if not np.isfinite(cents):
        raise ValueError(f"tuning estimate came back as {cents!r}")
    return cents, duration_sec, chunked


def _read_join_index(metadata: Path, join_column: str) -> dict[str, str]:
    """Index a metadata CSV by the stem of its join column.

    Args:
        metadata: Metadata CSV to read.
        join_column: Column holding each reference's file name.

    Returns:
        Mapping from reference file stem to that row's exact join value, so a
        value with directories survives into the output untouched.

    Raises:
        OSError: The file is missing or unreadable.
        ValueError: The column is absent, or the file is not a CSV.
    """
    index: dict[str, str] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or join_column not in reader.fieldnames:
            raise ValueError(
                f"--join-column {join_column!r} is not a column of {metadata} "
                f"(columns: {reader.fieldnames})"
            )
        for row in reader:
            value = row.get(join_column) or ""
            stem = Path(value.strip()).stem
            if not stem:
                continue
            if stem in index:
                logger.warning(
                    "Metadata %s holds more than one row for %r; keeping the first",
                    metadata,
                    stem,
                )
                continue
            index[stem] = value
    return index


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of *path*, read in chunks.

    Args:
        path: File to digest.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_of(distribution: str) -> str:
    """Return the installed version of *distribution*, or ``unknown``.

    Args:
        distribution: Distribution name to look up.

    Returns:
        The version string, or ``unknown`` when the package is not installed
        as a distribution (a source checkout, say).
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _is_inside(path: Path, root: Path) -> bool:
    """Return whether *path* is *root* or lies under it.

    Args:
        path: Resolved candidate path.
        root: Resolved directory.

    Returns:
        ``True`` when the path is inside the directory tree.
    """
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _output_clash(
    outputs: list[Path],
    recordings: Path,
    references: Path,
    metadata: Path | None,
    inputs: set[Path],
) -> str | None:
    """Return why an output path is unusable, or ``None`` when all are fine.

    Args:
        outputs: Paths the run would write.
        recordings: Input recordings directory.
        references: Input references directory.
        metadata: Input metadata file, if any.
        inputs: Resolved discovered audio and MIDI files.

    Returns:
        The reason to refuse, naming the offending path, or ``None``.
    """
    roots = (("--recordings", recordings.resolve()), ("--references", references.resolve()))
    for output in outputs:
        target = output.resolve()
        if metadata is not None and target == metadata.resolve():
            return f"{output} is the --metadata input"
        if target in inputs:
            return f"{output} is a discovered input file"
        for label, root in roots:
            if _is_inside(target, root):
                return f"{output} lies inside the {label} directory tree"
    return None


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    """Write one CSV, creating the parent directory.

    Args:
        path: Destination file.
        header: Column names in output order.
        rows: Rows as mappings, already sanitised.

    Raises:
        OSError: The file could not be created or written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _relative_recording(path: Path, recordings: Path) -> str:
    """Return *path* relative to the recordings root, with POSIX separators.

    Args:
        path: Discovered recording.
        recordings: Input recordings directory.

    Returns:
        The recording's name inside the dataset, or its own path when it sits
        outside the root the search was given.
    """
    try:
        return path.relative_to(recordings).as_posix()
    except ValueError:
        return path.as_posix()


def _measure_rows(
    audio_paths: list[Path],
    recordings: Path,
    pairing_stems: dict[Path, str],
    sample_rate: float,
    resolution: float,
    chunk_seconds: float,
) -> tuple[list[dict[str, str]], dict[Path, float], int, bool]:
    """Measure every recording, keeping going past the ones that fail.

    Unpaired recordings are measured too: the per-recording table exists for
    inspection, and an unmeasured row there would be indistinguishable from a
    broken one.

    Args:
        audio_paths: Discovered recordings, in discovery order.
        recordings: Input recordings directory.
        pairing_stems: Recording path to its paired reference stem ("" when
            unpaired).
        sample_rate: Sample rate to decode at in Hz.
        resolution: Tuning resolution as a fraction of a bin.
        chunk_seconds: Block length in seconds; ``0`` disables block pooling.

    Returns:
        The per-recording rows, the measured cents by recording, the number of
        failures, and whether the pooled block path ran for any recording.
    """
    rows: list[dict[str, str]] = []
    measured: dict[Path, float] = {}
    n_failed = 0
    chunked = False
    for path in audio_paths:
        stem = pairing_stems.get(path, "")
        row = {
            "recording": _sanitize(_relative_recording(path, recordings)),
            "reference_stem": _sanitize(stem),
            "own_tuning_cents": "",
            "duration_sec": "",
            "status": "ok",
            "error": "",
        }
        try:
            cents, duration_sec, used_chunks = _measure(
                path, sample_rate, resolution, chunk_seconds
            )
        except Exception as exc:  # one broken file must not sink the batch
            n_failed += 1
            row["status"] = "failed"
            row["error"] = _sanitize(f"{type(exc).__name__}: {exc}")
            logger.warning("Could not measure %s: %s", path, row["error"])
        else:
            measured[path] = cents
            row["own_tuning_cents"] = f"{cents:.1f}"
            row["duration_sec"] = f"{duration_sec:.1f}"
            chunked = chunked or used_chunks
            logger.info("%s: %+.1f cents from A440", row["recording"], cents)
        rows.append(row)
    rows.sort(key=lambda row: row["recording"])
    return rows, measured, n_failed, chunked


def _reference_rows(
    references: list[Path],
    measured_by_reference: dict[Path, list[float]],
    join_index: dict[str, str],
    join_column: str | None,
) -> list[dict[str, str]]:
    """Summarise the recordings measured for each paired reference.

    Args:
        references: Paired reference MIDI files, one row each.
        measured_by_reference: Measured cents per reference file.
        join_index: Reference stem to the metadata row's exact join value.
        join_column: Metadata column being joined on, or ``None`` without one.

    Returns:
        One row per reference, with the median and spread over the recordings
        that measured for it, and empty statistics for the ones that did not.
    """
    rows: list[dict[str, str]] = []
    for reference in references:
        values = measured_by_reference.get(reference, [])
        row: dict[str, str] = {}
        if join_column is not None:
            row[join_column] = _sanitize(join_index.get(reference.stem, ""))
        row["reference_stem"] = _sanitize(reference.stem)
        row["n_recordings"] = str(len(values))
        if values:
            row["own_tuning_cents"] = f"{float(np.median(values)):.1f}"
            row["own_tuning_spread_cents"] = f"{max(values) - min(values):.1f}"
        else:
            row["own_tuning_cents"] = ""
            row["own_tuning_spread_cents"] = ""
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Measure every recording's own tuning offset and write the tables.

    Args:
        argv: Argument list without the program name; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` once both tables and the provenance sidecar are written, which
        includes a run where individual recordings failed, and non-zero for a
        usage problem, a missing or empty input, an output that would overwrite
        an input, or an unwritable output.
    """
    args = _parse_args(argv)
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    recordings: Path = args.recordings
    references: Path = args.references
    output: Path = args.output
    metadata: Path | None = args.metadata
    recordings_output = output.with_name(f"{output.stem}.recordings.csv")

    for label, directory in (("--recordings", recordings), ("--references", references)):
        if not directory.is_dir():
            print(f"error: {label} directory not found: {directory}", file=sys.stderr)
            return 1

    audio_paths = discover_audio_files(recordings)
    midi_paths = discover_midi_files(references)

    join_index: dict[str, str] = {}
    if metadata is not None:
        try:
            join_index = _read_join_index(metadata, args.join_column)
        except (OSError, ValueError, csv.Error) as exc:
            print(f"error: cannot read --metadata {metadata}: {exc}", file=sys.stderr)
            return 1

    clash = _output_clash(
        [output, recordings_output],
        recordings,
        references,
        metadata,
        {path.resolve() for path in (*audio_paths, *midi_paths)},
    )
    if clash is not None:
        print(f"error: refusing to write output: {clash}", file=sys.stderr)
        return 1

    if not audio_paths:
        print(
            f"error: no recordings found under {recordings}; nothing to measure",
            file=sys.stderr,
        )
        return 1

    pairing = pair_audio_to_reference(audio_paths, midi_paths)
    paired_references = sorted(
        {reference for reference in pairing.mapping.values()}, key=lambda path: path.stem
    )
    pairing_stems = {
        path: reference.stem for path, reference in pairing.mapping.items()
    }

    rows, measured, n_failed, chunked = _measure_rows(
        audio_paths,
        recordings,
        pairing_stems,
        float(args.sample_rate),
        float(args.resolution),
        float(args.chunk_seconds),
    )

    measured_by_reference: dict[Path, list[float]] = {}
    for path, reference in pairing.mapping.items():
        if path in measured:
            measured_by_reference.setdefault(reference, []).append(measured[path])

    join_column = args.join_column if metadata is not None else None
    reference_header = ([join_column] if join_column else []) + list(_REFERENCE_COLUMNS)
    try:
        _write_csv(output, reference_header, _reference_rows(
            paired_references, measured_by_reference, join_index, join_column
        ))
        _write_csv(recordings_output, list(_RECORDING_COLUMNS), rows)
    except OSError as exc:
        print(f"error: cannot write output {output}: {exc}", file=sys.stderr)
        return 1

    n_multiple = sum(
        1
        for reference in paired_references
        if len(measured_by_reference.get(reference, [])) >= 2
    )
    provenance: dict[str, object] = {
        "argv": effective_argv,
        "recordings": str(recordings),
        "references": str(references),
    }
    if metadata is not None:
        provenance["metadata"] = str(metadata)
        provenance["metadata_sha256"] = _sha256_of(metadata)
    provenance.update(
        {
            "output": str(output),
            "recordings_output": str(recordings_output),
            "versions": {
                "librosa": librosa.__version__,
                "sonitra": _version_of("sonitra"),
            },
            "estimator": {
                "sample_rate": float(args.sample_rate),
                "resolution": float(args.resolution),
                "bins_per_octave": _BINS_PER_OCTAVE,
                "chunk_seconds": float(args.chunk_seconds),
                "chunked": chunked,
            },
            "counts": {
                "n_recordings": len(audio_paths),
                "n_measured": len(measured),
                "n_failed": n_failed,
                "n_unpaired": len(pairing.unpaired_audio),
                "n_references": len(paired_references),
                "n_references_with_multiple_recordings": n_multiple,
            },
        }
    )
    provenance_path = Path(f"{output}.provenance.json")
    try:
        provenance_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"error: cannot write provenance {provenance_path}: {exc}", file=sys.stderr)
        return 1

    print(
        f"measured {len(measured)}/{len(audio_paths)} recordings across "
        f"{len(paired_references)} references; {n_failed} failed, "
        f"{len(pairing.unpaired_audio)} unpaired -> {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
