"""Check a dataset folder before a benchmark run, and fix recording names safely.

Looks at ``<corpus-root>/<dataset>/`` the way ``sonitra benchmark`` will and
reports what would go wrong: recordings that pair with no MIDI file or with
more than one (using ``sonitra.corpus.pair_audio_to_reference`` itself, so the
report can never disagree with the benchmark), MIDI names that can never pair,
MIDI files whose first tempo differs from the render tempo (they render at the
wrong speed in MIDI-input mode), several instrument programs, drum-channel
notes, empty or unreadable MIDI, files Sonitra ignores, recordings left in
``audio/``, and symlinked folders the file search skips.

Input type defaults to ``audio`` when ``recordings/`` holds audio and ``midi``
otherwise; ``--input-type`` overrides it. Exit code 1 means at least one error
(something that makes files drop out of the run or score wrong); warnings and
notes alone exit 0.

Recordings that miss their MIDI file only by letter case or separators
(spaces, hyphens, dots) get a *safe rename*: one that the pairing code confirms
lands on exactly one MIDI file and collides with nothing. Nothing is renamed
unless you ask: ``--plan FILE`` writes the renames to a CSV for review, and
``--apply FILE`` checks every row first, renames only if all rows are valid,
and writes ``<plan>.undo.csv`` (itself a plan, so ``--apply`` on it reverts).
Only recordings are ever renamed, never MIDI files, because metadata joins key
on MIDI file names. Near misses get a "did you mean" hint instead.

Usage:
    python scripts/check_dataset.py --dataset my-dataset
    python scripts/check_dataset.py --dataset my-dataset --plan renames.csv
    python scripts/check_dataset.py --dataset my-dataset --apply renames.csv
    python scripts/check_dataset.py --dataset my-dataset --input-type midi --bpm 120
"""

from __future__ import annotations

import argparse
import csv
import difflib
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mido

from sonitra.corpus import PairingResult, discover_audio_files, discover_midi_files, pair_audio_to_reference
from sonitra.midi_reader import parse_midi

#: Findings shown per group before "... and N more" (``--verbose`` shows all).
_SHOWN_PER_GROUP = 10
#: Relative tempo difference below which a file counts as matching the render tempo.
_TEMPO_TOLERANCE = 1e-3
_DRUM_CHANNEL = 9
_SEPARATORS = re.compile(r"[\s_.\-]+")
_PLAN_COLUMNS = ("recording", "new_name", "pairs_with", "reason")

#: Display order and (singular, plural) headline for each finding code;
#: ``{n}`` fills in the count.
_GROUPS: Dict[str, Tuple[str, str]] = {
    "no-midi": ("No MIDI files in midi/", "No MIDI files in midi/"),
    "no-recordings": ("No recordings in recordings/", "No recordings in recordings/"),
    "unreadable": ("1 MIDI file could not be read", "{n} MIDI files could not be read"),
    "no-notes": ("1 MIDI file has no notes", "{n} MIDI files have no notes"),
    "unpaired": ("1 recording matches no MIDI file", "{n} recordings match no MIDI file"),
    "ambiguous": ("1 recording matches more than one MIDI file", "{n} recordings match more than one MIDI file"),
    "duplicate-name": (
        "1 MIDI file shares its name with another MIDI file",
        "{n} MIDI files share a name with another MIDI file",
    ),
    "name-clash": (
        "1 MIDI file can never pair, because another MIDI name is its name plus _ and more",
        "{n} MIDI files can never pair, because another MIDI name is theirs plus _ and more",
    ),
    "tempo": (
        "1 MIDI file does not start at the render tempo, so it renders at the wrong speed",
        "{n} MIDI files do not start at the render tempo, so they render at the wrong speed",
    ),
    "programs": ("1 MIDI file uses several instrument programs", "{n} MIDI files use several instrument programs"),
    "drums": (
        "1 MIDI file has drum-channel notes, which are scored as pitched notes",
        "{n} MIDI files have drum-channel notes, which are scored as pitched notes",
    ),
    "audio-dir": (
        "1 audio file sits in audio/, which Sonitra never reads recordings from",
        "{n} audio files sit in audio/, which Sonitra never reads recordings from",
    ),
    "ignored": (
        "1 file in recordings/ has an ending Sonitra does not read (it reads .wav, .flac, .mp3)",
        "{n} files in recordings/ have an ending Sonitra does not read (it reads .wav, .flac, .mp3)",
    ),
    "symlink": (
        "1 symlinked folder inside midi/ or recordings/ is skipped by the file search",
        "{n} symlinked folders inside midi/ or recordings/ are skipped by the file search",
    ),
    "unused-midi": (
        "1 MIDI file has no recording, so the benchmark leaves it out",
        "{n} MIDI files have no recording, so the benchmark leaves them out",
    ),
}
_LEVELS = ("error", "warning", "note")


class PlanError(Exception):
    """A rename plan could not be written or applied; nothing was changed."""


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    path: Path
    detail: str = ""


@dataclass(frozen=True)
class Rename:
    """One safe rename: *recording* (relative to recordings/) becomes *new_name*."""

    recording: Path
    new_name: str
    pairs_with: Path
    reason: str


@dataclass
class Report:
    dataset_dir: Path
    input_type: str
    input_type_reason: str
    bpm: float
    midi_files: List[Path]
    recordings: List[Path]
    pairing: Optional[PairingResult]
    findings: List[Finding] = field(default_factory=list)
    renames: List[Rename] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if any(finding.level == "error" for finding in self.findings) else 0


# --- checking ----------------------------------------------------------------------


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _fmt_bpm(bpm: float) -> str:
    return f"{round(bpm, 2):g}"


def _norm_tokens(stem: str) -> Tuple[str, ...]:
    return tuple(token.casefold() for token in _SEPARATORS.split(stem) if token)


def _check_midi_contents(midi_files: Sequence[Path], input_type: str, bpm: float) -> List[Finding]:
    findings: List[Finding] = []
    for path in midi_files:
        try:
            meta = parse_midi(path, return_meta=True)
            drum_notes = sum(
                1
                for track in mido.MidiFile(path).tracks
                for message in track
                if message.type == "note_on" and message.velocity > 0 and message.channel == _DRUM_CHANNEL
            )
        except Exception as exc:  # noqa: BLE001 - any parse failure is reported, not raised
            findings.append(Finding("error", "unreadable", path, f"{type(exc).__name__}: {exc}"[:120]))
            continue
        if not meta["notes"]:
            findings.append(Finding("error", "no-notes", path))
        if input_type == "midi":
            native = float(meta["bpm"])
            if native > 0 and abs(native / bpm - 1.0) > _TEMPO_TOLERANCE:
                findings.append(
                    Finding(
                        "error",
                        "tempo",
                        path,
                        f"first tempo {_fmt_bpm(native)} BPM, so it renders at {bpm / native:.2f}x "
                        f"the speed of its reference (render tempo {_fmt_bpm(bpm)} BPM)",
                    )
                )
            if len(meta["programs"]) > 1:
                programs = ", ".join(str(p) for p in meta["programs"])
                findings.append(
                    Finding(
                        "warning",
                        "programs",
                        path,
                        f"programs {programs}; FluidSynth plays the SoundFont default unless "
                        "fluidsynth.program is set",
                    )
                )
        if drum_notes:
            findings.append(Finding("warning", "drums", path, f"{drum_notes} drum-channel notes"))
    return findings


def _check_midi_names(midi_files: Sequence[Path], dataset_dir: Path, input_type: str) -> List[Finding]:
    findings: List[Finding] = []
    by_stem: Dict[str, List[Path]] = {}
    for path in midi_files:
        by_stem.setdefault(path.stem, []).append(path)
    level = "error" if input_type == "audio" else "warning"
    for stem, paths in by_stem.items():
        if len(paths) > 1:
            others = ", ".join(_rel(p, dataset_dir) for p in paths)
            for path in paths:
                findings.append(Finding(level, "duplicate-name", path, f"same name: {others}"))
    if input_type == "audio":
        # Same "_"-token, case-sensitive comparison the pairing code uses.
        by_tokens: Dict[Tuple[str, ...], List[Path]] = {}
        for path in midi_files:
            by_tokens.setdefault(tuple(path.stem.split("_")), []).append(path)
        flagged: Dict[Path, Path] = {}
        for tokens, paths in by_tokens.items():
            for k in range(1, len(tokens)):
                for shorter in by_tokens.get(tokens[:k], []):
                    flagged.setdefault(shorter, paths[0])
        for shorter, longer in sorted(flagged.items()):
            findings.append(
                Finding("error", "name-clash", shorter, f"{_rel(longer, dataset_dir)} starts with this name plus _")
            )
    return findings


def _symlinked_dirs(top: Path) -> List[Path]:
    if not top.is_dir():
        return []
    found: List[Path] = []
    for dirpath, dirnames, _ in os.walk(top, followlinks=False):
        for name in dirnames:
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                found.append(candidate)
    return sorted(found)


def _propose_renames(
    recordings: Sequence[Path],
    midi_files: Sequence[Path],
    pairing: PairingResult,
    recordings_dir: Path,
    midi_dir: Path,
) -> Tuple[Dict[Path, Rename], Dict[Path, List[Path]]]:
    """Safe renames for unmatched recordings, plus near-miss suggestions for the rest."""
    index: Dict[Tuple[str, ...], List[Path]] = {}
    for midi in midi_files:
        index.setdefault(_norm_tokens(midi.stem), []).append(midi)

    proposals: Dict[Path, Tuple[str, Path]] = {}
    for audio in pairing.unpaired_audio:
        if audio in pairing.ambiguous:
            continue  # caused by MIDI names; renaming the recording cannot fix it
        original = [token for token in _SEPARATORS.split(audio.stem) if token]
        folded = tuple(token.casefold() for token in original)
        candidates = {midi for k in range(1, len(folded) + 1) for midi in index.get(folded[:k], [])}
        if len(candidates) != 1:
            continue
        (midi,) = candidates
        rest = original[len(_norm_tokens(midi.stem)):]
        new_name = midi.stem + ("_" + "_".join(rest) if rest else "") + audio.suffix
        if new_name != audio.name:
            proposals[audio] = (new_name, midi)

    targets: Dict[str, List[Path]] = {}
    for audio, (new_name, _) in proposals.items():
        targets.setdefault(str(audio.with_name(new_name)).casefold(), []).append(audio)
    for audio, (new_name, _) in list(proposals.items()):
        target = audio.with_name(new_name)
        clash = len(targets[str(target).casefold()]) > 1
        occupied = target.exists() and not _same_file_different_case(audio, target)
        if clash or occupied:
            del proposals[audio]

    # Confirm with the real pairing code that each renamed file lands where intended.
    simulated = {audio.with_name(new_name): audio for audio, (new_name, _) in proposals.items()}
    result = pair_audio_to_reference(
        [audio.with_name(proposals[audio][0]) if audio in proposals else audio for audio in recordings],
        midi_files,
    )
    renames: Dict[Path, Rename] = {}
    for new_path, audio in simulated.items():
        new_name, midi = proposals[audio]
        if result.mapping.get(new_path) == midi:
            renames[audio] = Rename(
                Path(_rel(audio, recordings_dir)),
                new_name,
                Path(_rel(midi, midi_dir)),
                "matches this MIDI file once letter case and separators are ignored",
            )

    stems: Dict[str, List[Path]] = {}
    for midi in midi_files:
        stems.setdefault(midi.stem.casefold(), []).append(midi)
    suggestions: Dict[Path, List[Path]] = {}
    for audio in pairing.unpaired_audio:
        if audio in renames or audio in pairing.ambiguous:
            continue
        close = difflib.get_close_matches(audio.stem.casefold(), list(stems), n=3, cutoff=0.6)
        if close:
            suggestions[audio] = [midi for stem in close for midi in stems[stem]]
    return renames, suggestions


def check_dataset(dataset_dir: Path, *, input_type: Optional[str] = None, bpm: float = 120.0) -> Report:
    """Check *dataset_dir* (``<corpus-root>/<dataset>``) the way ``sonitra benchmark`` reads it."""
    midi_dir = dataset_dir / "midi"
    recordings_dir = dataset_dir / "recordings"
    audio_dir = dataset_dir / "audio"
    midi_files = discover_midi_files(midi_dir) if midi_dir.is_dir() else []
    recordings = discover_audio_files(recordings_dir) if recordings_dir.is_dir() else []

    if input_type is None:
        input_type = "audio" if recordings else "midi"
        reason = "recordings/ has audio" if recordings else "recordings/ has no audio"
    else:
        reason = "set by --input-type"

    findings: List[Finding] = []
    if not midi_files:
        findings.append(Finding("error", "no-midi", midi_dir))

    pairing: Optional[PairingResult] = None
    renames: List[Rename] = []
    if input_type == "audio":
        if not recordings:
            findings.append(Finding("error", "no-recordings", recordings_dir))
        pairing = pair_audio_to_reference(recordings, midi_files)
        by_audio, suggestions = _propose_renames(recordings, midi_files, pairing, recordings_dir, midi_dir)
        renames = [by_audio[audio] for audio in sorted(by_audio)]
        for audio in pairing.unpaired_audio:
            if audio in pairing.ambiguous:
                names = ", ".join(_rel(midi, dataset_dir) for midi in pairing.ambiguous[audio])
                findings.append(Finding("error", "ambiguous", audio, f"candidates: {names}"))
            elif audio in by_audio:
                rename = by_audio[audio]
                findings.append(
                    Finding(
                        "error",
                        "unpaired",
                        audio,
                        f"safe rename to {rename.new_name} (pairs with midi/{rename.pairs_with.as_posix()})",
                    )
                )
            elif audio in suggestions:
                names = " or ".join(_rel(midi, dataset_dir) for midi in suggestions[audio])
                findings.append(Finding("error", "unpaired", audio, f"did you mean {names}?"))
            else:
                findings.append(Finding("error", "unpaired", audio))
        for midi in pairing.unpaired_midi:
            findings.append(Finding("note", "unused-midi", midi))

        readable = set(recordings)
        for path in sorted(recordings_dir.rglob("*")) if recordings_dir.is_dir() else []:
            if path.is_file() and not path.name.startswith(".") and path not in readable:
                findings.append(Finding("warning", "ignored", path))

    findings.extend(_check_midi_names(midi_files, dataset_dir, input_type))
    findings.extend(_check_midi_contents(midi_files, input_type, bpm))

    if audio_dir.is_dir():
        # Top level only: Sonitra's own renders live in audio/<config>/ subfolders.
        for path in discover_audio_files(audio_dir):
            if path.parent == audio_dir:
                findings.append(Finding("warning", "audio-dir", path, "move it to recordings/"))
    for top in (midi_dir, recordings_dir):
        for link in _symlinked_dirs(top):
            findings.append(Finding("warning", "symlink", link, "link the whole midi/ or recordings/ folder instead"))

    return Report(
        dataset_dir=dataset_dir,
        input_type=input_type,
        input_type_reason=reason,
        bpm=bpm,
        midi_files=midi_files,
        recordings=recordings,
        pairing=pairing,
        findings=findings,
        renames=renames,
    )


# --- reporting ---------------------------------------------------------------------


def format_report(report: Report, *, verbose: bool = False) -> str:
    mode = "audio-input" if report.input_type == "audio" else "MIDI-input"
    lines = [f"Checking {report.dataset_dir} in {mode} mode ({report.input_type_reason})."]
    counts = f"midi/: {len(report.midi_files)} MIDI file{'s' if len(report.midi_files) != 1 else ''}"
    if report.input_type == "audio":
        n = len(report.recordings)
        counts += f". recordings/: {n} recording{'s' if n != 1 else ''}"
    else:
        counts += f". Render tempo: {_fmt_bpm(report.bpm)} BPM (--bpm changes it)"
    lines.append(counts + ".")
    if report.pairing is not None and report.recordings:
        paired, total = len(report.pairing.mapping), len(report.recordings)
        noun, verb = ("recording", "pairs") if total == 1 else ("recordings", "pair")
        lines.append(f"Pairing: {paired} of {total} {noun} {verb} with a MIDI file.")

    groups: Dict[str, List[Finding]] = {}
    for finding in report.findings:
        groups.setdefault(finding.code, []).append(finding)
    for level in _LEVELS:
        for code, (singular, plural) in _GROUPS.items():
            members = [f for f in groups.get(code, []) if f.level == level]
            if not members:
                continue
            lines.append("")
            n = len(members)
            lines.append(f"{level.upper():<8} {(singular if n == 1 else plural).format(n=n)}")
            if code in {"no-midi", "no-recordings"}:
                continue
            shown = members if verbose else members[:_SHOWN_PER_GROUP]
            for finding in shown:
                entry = _rel(finding.path, report.dataset_dir)
                lines.append(f"           {entry}" + (f": {finding.detail}" if finding.detail else ""))
            if len(shown) < n:
                lines.append(f"           ... and {n - len(shown)} more (--verbose lists all)")

    tally = {level: sum(1 for f in report.findings if f.level == level) for level in _LEVELS}
    lines.append("")
    if report.findings:
        lines.append(
            f"Result: {tally['error']} error{'s' if tally['error'] != 1 else ''}, "
            f"{tally['warning']} warning{'s' if tally['warning'] != 1 else ''}, "
            f"{tally['note']} note{'s' if tally['note'] != 1 else ''}."
        )
    else:
        lines.append("Result: no problems found.")
    return "\n".join(lines)


# --- plans -------------------------------------------------------------------------


def write_plan(renames: Sequence[Rename], path: Path) -> None:
    """Write *renames* as a plan CSV; never overwrites an existing file."""
    if path.exists():
        raise PlanError(f"{path} already exists; pick another name or delete it first")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PLAN_COLUMNS)
        writer.writeheader()
        for rename in renames:
            writer.writerow(
                {
                    "recording": rename.recording.as_posix(),
                    "new_name": rename.new_name,
                    "pairs_with": rename.pairs_with.as_posix(),
                    "reason": rename.reason,
                }
            )


def read_plan(path: Path) -> List[Rename]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(_PLAN_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise PlanError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        return [
            Rename(Path(row["recording"]), row["new_name"], Path(row["pairs_with"]), row["reason"])
            for row in reader
        ]


def _same_file_different_case(source: Path, target: Path) -> bool:
    """True when *target* is *source* seen through a case-insensitive file system."""
    if source.name.casefold() != target.name.casefold():
        return False
    try:
        return source.samefile(target)
    except OSError:
        return False


def _undo_path(plan_path: Path) -> Path:
    return plan_path.with_name(plan_path.stem + ".undo.csv")


def apply_plan(plan_path: Path, recordings_dir: Path) -> Path:
    """Rename recordings as *plan_path* says, after checking every row; return the undo plan."""
    renames = read_plan(plan_path)
    undo_path = _undo_path(plan_path)
    root = recordings_dir.resolve()
    problems: List[str] = []
    if undo_path.exists():
        problems.append(f"the undo file {undo_path} already exists; move it away first")

    moves: List[Tuple[Path, Path, Rename]] = []
    seen_sources: Dict[Path, int] = {}
    seen_targets: Dict[str, int] = {}
    for number, rename in enumerate(renames, start=2):  # row 1 is the header
        label = f"row {number} ({rename.recording.as_posix()})"
        source = (recordings_dir / rename.recording).resolve()
        if root not in source.parents:
            problems.append(f"{label}: points outside the recordings folder {recordings_dir}")
            continue
        if not source.is_file():
            problems.append(f"{label}: the recording does not exist")
            continue
        if Path(rename.new_name).name != rename.new_name or rename.new_name in {"", ".", ".."}:
            problems.append(f"{label}: new_name must be a plain file name, not {rename.new_name!r}")
            continue
        if Path(rename.new_name).suffix.casefold() != source.suffix.casefold():
            problems.append(f"{label}: new_name {rename.new_name} changes the file ending")
            continue
        target = source.with_name(rename.new_name)
        if target.exists() and not _same_file_different_case(source, target):
            problems.append(f"{label}: {rename.new_name} already exists")
            continue
        seen_sources[source] = seen_sources.get(source, 0) + 1
        seen_targets[str(target).casefold()] = seen_targets.get(str(target).casefold(), 0) + 1
        moves.append((source, target, rename))
    for source, target, rename in moves:
        if seen_sources[source] > 1:
            problems.append(f"{rename.recording.as_posix()} is renamed more than once")
        if seen_targets[str(target).casefold()] > 1:
            problems.append(f"{rename.new_name} is used as a new name more than once")
    if problems:
        raise PlanError("\n".join(dict.fromkeys(problems)))

    undo: List[Rename] = []
    try:
        for source, target, rename in moves:
            if source.name.casefold() == target.name.casefold():
                # Two steps, so a case-only rename also works on case-insensitive file systems.
                staging = source.with_name(f".{source.name}.check_dataset-tmp")
                source.rename(staging)
                staging.rename(target)
            else:
                source.rename(target)
            undo.append(
                Rename(
                    rename.recording.with_name(rename.new_name),
                    source.name,
                    rename.pairs_with,
                    "undo",
                )
            )
    finally:
        if undo:
            write_plan(undo, undo_path)
    return undo_path


# --- command line ------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Usage:" + __doc__.split("Usage:", 1)[1],
    )
    parser.add_argument("--dataset", required=True, help="Dataset folder name under --corpus-root.")
    parser.add_argument("--corpus-root", type=Path, default=Path("corpus"), help="Default: corpus")
    parser.add_argument(
        "--input-type",
        choices=("midi", "audio"),
        help="Check a MIDI-input or audio-input run. Default: audio when recordings/ has audio.",
    )
    parser.add_argument(
        "--bpm", type=float, default=120.0, help="render_pipeline.bpm of your config (MIDI-input mode). Default: 120"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", type=Path, help="Write the safe renames to this CSV for review.")
    action.add_argument("--apply", type=Path, help="Apply a reviewed plan CSV (writes <plan>.undo.csv).")
    parser.add_argument("-v", "--verbose", action="store_true", help="List every finding, not the first 10.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    dataset_dir = args.corpus_root / args.dataset
    if not dataset_dir.is_dir():
        print(f"No dataset folder at {dataset_dir} (check --dataset and --corpus-root).", file=sys.stderr)
        return 1

    corpus_logger = logging.getLogger("sonitra.corpus")
    previous_level = corpus_logger.level
    corpus_logger.setLevel(logging.ERROR)  # the report lists pairing problems itself
    try:
        if args.apply is not None:
            try:
                count = len(read_plan(args.apply))
                undo = apply_plan(args.apply, dataset_dir / "recordings")
            except (PlanError, OSError) as exc:
                print(f"Plan not applied:\n{exc}", file=sys.stderr)
                return 1
            print(f"Renamed {count} recording{'s' if count != 1 else ''}. To undo, run:")
            print(
                f"  python scripts/check_dataset.py --dataset {args.dataset} "
                f"--corpus-root {args.corpus_root} --apply {undo}"
            )
            print()

        report = check_dataset(dataset_dir, input_type=args.input_type, bpm=args.bpm)
        print(format_report(report, verbose=args.verbose))

        n = len(report.renames)
        if args.plan is not None:
            if not n:
                print(f"No safe renames to write, so {args.plan} was not created.")
            else:
                try:
                    write_plan(report.renames, args.plan)
                except PlanError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                print(f"Wrote {n} safe rename{'s' if n != 1 else ''} to {args.plan}. Review it, then run with --apply {args.plan}.")
        elif n and args.apply is None:
            print(
                f"{n} safe rename{'s' if n != 1 else ''} found. Write them to a plan you can review with "
                "--plan FILE, then apply it with --apply FILE."
            )
        return report.exit_code
    finally:
        corpus_logger.setLevel(previous_level)


if __name__ == "__main__":
    sys.exit(main())
