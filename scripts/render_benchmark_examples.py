#!/usr/bin/env python3
r"""Render one excerpt of one source piece through every benchmark condition.

Writes one audio file per condition, plus the resolved config and a JSON
manifest, per config under ``--output``, so the conditions of a benchmark can be
compared by ear. Nothing is transcribed and nothing is scored. Conditions render
sequentially in one process: DawDreamer and JUCE state cannot be shared across
threads, and each condition already rebuilds the source and effects chain.

Examples:
    # render every config under config/benchmark without asking
    python scripts/render_benchmark_examples.py --output output/renders/examples --yes

    # check the plan for two reverb studies before rendering anything
    python scripts/render_benchmark_examples.py --output output/renders/examples \
        --include "*reverb*" --duration 10 --dry-run

    # choose the configs to render from the discovered tree
    python scripts/render_benchmark_examples.py --output output/renders/examples

    # re-render conditions whose audio file is already there
    python scripts/render_benchmark_examples.py --output output/renders/examples --yes --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions  # noqa: E402
from sonitra.config import (  # noqa: E402
    ConfigError,
    PipelineConfig,
    SelectionSample,
    load_config,
    resolve_corpus_paths,
)
from sonitra.corpus import discover_audio_files, discover_midi_files, pair_audio_to_reference  # noqa: E402
from sonitra.midi_reader import parse_midi  # noqa: E402
from sonitra.midi_writer import write_midi  # noqa: E402
from sonitra.notes import make_note  # noqa: E402
from sonitra.pipeline import run_pipeline  # noqa: E402
from sonitra.selection import (  # noqa: E402
    SelectionError,
    sample_units,
    select_audio,
    select_references,
)
from sonitra.storage import read_audio, write_wav  # noqa: E402

_MIDI = "midi"
_AUDIO = "audio"


class ExcerptError(Exception):
    """An excerpt cannot be cut; the config is skipped instead of aborting the run."""


class EmptyWindow(ExcerptError):
    """The requested excerpt window holds nothing to render."""


class NoRecording(ExcerptError):
    """No recording in the dataset pairs with the picked reference."""


class UsageError(Exception):
    """A command-line value cannot be honoured, so the run stops with exit 2."""


class ToggleError(Exception):
    """A typed line cannot be applied; the picker re-asks instead of guessing."""


# --- data ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """The one source piece every config on a dataset renders from."""

    dataset: str
    reference: Path
    recording: Path | None
    pinned: bool
    seed: int


@dataclass(frozen=True)
class Target:
    """A selected benchmark config: its conditions, or why it cannot run."""

    path: Path
    config: PipelineConfig | None
    conditions: list[Condition]
    dataset_source: str
    skip_reason: str | None
    error: str | None = None

    @property
    def dataset(self) -> str:
        """The dataset to render from; empty when the config could not be loaded."""
        if self.config is None or self.config.io.dataset is None:
            return ""
        return self.config.io.dataset

    @property
    def input_type(self) -> str:
        return _MIDI if self.config is None else self.config.render_pipeline.input_type.value


@dataclass(frozen=True)
class ConditionReport:
    """What one condition did, and where it wrote to."""

    name: str
    slug: str
    overrides: dict[str, Any]
    status: str
    output: str | None
    error: str | None
    render_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "overrides": self.overrides,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "render_seconds": self.render_seconds,
        }


@dataclass
class ConfigReport:
    """The summary row for one config, mirroring ``render_summary.json``."""

    config: Path
    status: str
    reason: str | None
    succeeded: int
    failed: int
    skipped_existing: int
    out_dir: Path | None
    conditions: list[ConditionReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": str(self.config),
            "status": self.status,
            "reason": self.reason,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped_existing": self.skipped_existing,
            "out_dir": None if self.out_dir is None else str(self.out_dir),
        }


@dataclass(frozen=True)
class PlanRow:
    """One config resolved to its piece and source file, or to a skip reason."""

    target: Target
    piece: Piece | None
    source: Path | None
    reason: str | None
    detail: str | None = None


# --- discovery and selection --------------------------------------------------------


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def discover_configs(configs_dir: Path) -> list[Path]:
    """Return every ``*.yaml`` file under *configs_dir*, sorted.

    Paths keep the form *configs_dir* was given in, so a relative
    ``--configs-dir`` yields relative paths.

    Args:
        configs_dir: Folder searched recursively.

    Returns:
        Sorted list of config file paths; ``.yml`` and non-files are ignored.
    """
    return sorted(path for path in configs_dir.rglob("*.yaml") if path.is_file())


def filter_configs(
    paths: Sequence[Path],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    configs_dir: Path,
) -> list[Path]:
    """Match each path against *include* then *exclude* glob patterns.

    Args:
        paths: Config paths as returned by :func:`discover_configs`.
        include: Globs a config must match; empty or ``None`` keeps everything.
        exclude: Globs that drop a config; they win over *include*.
        configs_dir: The folder the globs are relative to.

    Returns:
        The kept paths, in their original order.
    """
    kept: list[Path] = []
    for path in paths:
        relative = _relative(path, configs_dir)
        if include and not any(fnmatch(relative, pattern) for pattern in include):
            continue
        if exclude and any(fnmatch(relative, pattern) for pattern in exclude):
            continue
        kept.append(path)
    return kept


def load_targets(paths: Sequence[Path], default_dataset: str | None) -> list[Target]:
    """Load each config and expand its conditions.

    ``io.dataset`` wins over *default_dataset*, as it does for the CLI. A
    config that cannot be read, loaded, or that names no dataset at all,
    becomes a target carrying its skip reason instead of an exception: one
    broken config must not stop the others.

    Args:
        paths: Config paths to load.
        default_dataset: Dataset for configs that set no ``io.dataset``.

    Returns:
        One :class:`Target` per path, in order.
    """
    targets: list[Target] = []
    for path in paths:
        try:
            config = load_config(path)
            if config.io.dataset is None and default_dataset is not None:
                config.io.dataset = default_dataset
                dataset_source = "--default-dataset"
            else:
                dataset_source = "io.dataset"
            if config.io.dataset is None:
                targets.append(Target(path, config, [], dataset_source, "no_dataset"))
                continue
            conditions = expand_conditions(config.benchmark)
        except (ConfigError, FileNotFoundError, ValueError, OSError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            targets.append(Target(path, None, [], "", "load_error", message))
            continue
        targets.append(Target(path, config, conditions, dataset_source, None))
    return targets


def _pair_recordings(
    config: PipelineConfig,
    recordings_dir: Path,
    references: Sequence[Path],
    group: Sequence[Target],
) -> dict[Path, Path] | None:
    """Map each candidate reference to the recording that pairs with it.

    Only datasets that carry at least one audio-input config need this: a MIDI
    config renders the reference itself. The recording side goes through the
    same corpus pairing the benchmark uses, so an audio config is guaranteed a
    reference for its sampled recording.

    Args:
        config: Config copy whose filters are already cleared.
        recordings_dir: ``<corpus_root>/<dataset>/recordings``.
        references: Every candidate reference MIDI file.
        group: The targets that share the dataset.

    Returns:
        Reference to recording for every paired reference, or ``None`` when the
        dataset needs no recording or has none that pair.
    """
    if not any(target.input_type == _AUDIO for target in group):
        return None
    recordings = discover_audio_files(recordings_dir) if recordings_dir.is_dir() else []
    if not recordings:
        return None
    try:
        selected = select_audio(
            config,
            recordings,
            references,
            unit_kind="recording",
            apply_sample=False,
            always_pair=True,
        )
    except SelectionError:
        return None
    pairing = pair_audio_to_reference(sorted(selected.units), references)
    inverted: dict[Path, Path] = {}
    for recording, reference in sorted(pairing.mapping.items()):
        inverted.setdefault(reference, recording)
    return inverted or None


def pick_pieces(
    targets: Sequence[Target],
    seed: int,
    pins: dict[str, str],
    corpus_root: Path | None = None,
) -> dict[str, Piece]:
    """Choose one reference (and recording) per dataset, shared by its configs.

    Configs on the same dataset must render the same music, otherwise the audio
    files cannot be compared by ear. The pick therefore comes from one loaded
    config with ``io.where`` and ``io.sample`` cleared: one config's filters
    must not narrow what the others may render.

    Args:
        targets: Loaded targets; those with a skip reason are ignored.
        seed: Seed for the sampled pick.
        pins: ``DATASET`` to reference stem, for ``--source``.
        corpus_root: Overrides ``io.corpus_root`` when given.

    Returns:
        Dataset name to :class:`Piece`. A dataset with no reference MIDI is
        absent, which makes its configs skip as ``no_source``; every other
        dataset is unaffected.

    Raises:
        UsageError: If a ``--source`` pin names no file in its dataset.
    """
    by_dataset: dict[str, list[Target]] = {}
    for target in targets:
        if target.config is None or not target.dataset:
            continue
        by_dataset.setdefault(target.dataset, []).append(target)

    pieces: dict[str, Piece] = {}
    for dataset in sorted(by_dataset):
        group = by_dataset[dataset]
        config = group[0].config
        assert config is not None, "a loaded target always carries its config"
        config = config.model_copy(deep=True)
        config.io.where = {}
        config.io.sample = None
        if corpus_root is not None:
            config.io.corpus_root = corpus_root
        paths = resolve_corpus_paths(config)
        references = discover_midi_files(paths.midi) if paths.midi.is_dir() else []
        if not references:
            # No reference MIDI to render from, so this dataset gets no piece
            # and its configs are skipped while the other datasets carry on.
            continue
        candidates = list(select_references(config, references, apply_sample=False).units)

        recordings = _pair_recordings(config, paths.recordings, references, group)
        if recordings is not None:
            paired = [reference for reference in candidates if reference in recordings]
            if paired:
                candidates = paired
            else:
                recordings = None

        pin = pins.get(dataset)
        if pin is not None:
            matches = [reference for reference in candidates if reference.stem == pin]
            if not matches:
                raise UsageError(f"--source {dataset}={pin} matched no file in {paths.midi}")
            reference_path, pinned = matches[0], True
        else:
            (reference_path,) = sample_units(candidates, SelectionSample(n=1, seed=seed))
            pinned = False
        pieces[dataset] = Piece(
            dataset=dataset,
            reference=reference_path,
            recording=None if recordings is None else recordings.get(reference_path),
            pinned=pinned,
            seed=seed,
        )
    return pieces


# --- excerpts -----------------------------------------------------------------------


def _clip_notes(
    notes: Sequence[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    """Keep the notes overlapping ``[start_sec, end_sec)``, clipped and shifted."""
    window: list[dict[str, Any]] = []
    for note in notes:
        note_start = float(note["start_sec"])
        note_end = note_start + float(note["duration_sec"])
        if note_start >= end_sec or note_end <= start_sec:
            continue
        clipped_start = max(note_start, start_sec) - start_sec
        clipped_end = min(note_end, end_sec) - start_sec
        clipped = make_note(
            pitch=note["pitch"],
            velocity=note.get("velocity", 64),
            start_sec=clipped_start,
            duration_sec=clipped_end - clipped_start,
        )
        if clipped is not None:
            window.append(clipped)
    return window


def _cut_midi_excerpt(
    reference: Path,
    start_sec: float,
    duration_sec: float,
    path: Path,
) -> Path:
    """Write the window of *reference* to *path*, dropping the notes outside it."""
    meta = parse_midi(reference, return_meta=True)
    notes = _clip_notes(meta["notes"], start_sec, start_sec + duration_sec)
    if not notes:
        raise EmptyWindow(
            f"no notes in {start_sec:.1f}-{start_sec + duration_sec:.1f}s of {reference}"
        )
    programs = meta["programs"]
    # One program carries over; several would need one channel each, so the
    # excerpt drops them and lets the synth pick its default, as MidiSource does.
    program = programs[0] if len(programs) == 1 else None
    write_midi(notes, path, program=program)
    return path


def _cut_audio_excerpt(
    recording: Path,
    start_sec: float,
    duration_sec: float,
    path: Path,
) -> Path:
    """Write the window of *recording* to *path* at the recording's own rate."""
    audio, sample_rate = read_audio(recording)
    begin = int(start_sec * sample_rate)
    end = int((start_sec + duration_sec) * sample_rate)
    window = audio[..., begin:end]
    if window.shape[-1] == 0:
        raise EmptyWindow(
            f"no audio in {start_sec:.1f}-{start_sec + duration_sec:.1f}s of {recording}"
        )
    # normalize=False: AudioSource.load reads the recording as it is, so rescaling
    # the excerpt here would shift the drive level the effects chain sees.
    write_wav(window, path, sample_rate=sample_rate, normalize=False)
    return path


def cut_excerpt(
    piece: Piece,
    input_type: str,
    start_sec: float,
    duration_sec: float | None,
    tmp_dir: Path,
) -> Path:
    """Return the excerpt of *piece* that every condition of a dataset renders.

    Args:
        piece: The shared piece for the dataset.
        input_type: ``"midi"`` for the reference, ``"audio"`` for its recording.
        start_sec: Where the window starts inside the piece.
        duration_sec: Window length, or ``None`` for the whole piece.
        tmp_dir: Directory the excerpt is cut into.

    Returns:
        The excerpt path, or the original source file for a full-length run.

    Raises:
        NoRecording: An audio run on a piece that pairs with no recording.
        EmptyWindow: The window holds no notes and no samples.
        ValueError: *input_type* is neither ``"midi"`` nor ``"audio"``.
    """
    if input_type == _AUDIO:
        if piece.recording is None:
            raise NoRecording(f"no recording pairs with {piece.reference} on {piece.dataset}")
        original = piece.recording
    elif input_type == _MIDI:
        original = piece.reference
    else:
        raise ValueError(f"input_type must be 'midi' or 'audio', got {input_type!r}")
    if duration_sec is None:
        return original
    stem = f"{piece.reference.stem}_{input_type}"
    if input_type == _AUDIO:
        # Always WAV: the samples were read whatever the recording's extension,
        # and storage.read_audio picks its decoder from that extension.
        return _cut_audio_excerpt(original, start_sec, duration_sec, tmp_dir / f"{stem}.wav")
    return _cut_midi_excerpt(original, start_sec, duration_sec, tmp_dir / f"{stem}.mid")


def _resolve_source(
    target: Target,
    piece: Piece,
    cache: dict[tuple[str, str], Path],
    tmp_dir: Path,
    start_sec: float,
    duration_sec: float | None,
) -> Path:
    """Return the excerpt for *target*, cutting it once per dataset and input type."""
    key = (piece.dataset, target.input_type)
    if key not in cache:
        cache[key] = cut_excerpt(piece, target.input_type, start_sec, duration_sec, tmp_dir)
    return cache[key]


def build_plan(
    targets: Sequence[Target],
    pieces: dict[str, Piece],
    cache: dict[tuple[str, str], Path],
    tmp_dir: Path,
    start_sec: float,
    duration_sec: float | None,
) -> list[PlanRow]:
    """Resolve every target to its piece and source file, or to a skip reason.

    Used by both ``--dry-run`` and the real run so the printed plan is exactly
    what would happen. An unusable source file (unreadable MIDI, undecodable
    recording) skips that one config instead of aborting the run.
    """
    rows: list[PlanRow] = []
    for target in targets:
        if target.skip_reason is not None:
            rows.append(PlanRow(target, None, None, target.skip_reason))
            continue
        piece = pieces.get(target.dataset)
        if piece is None:
            rows.append(PlanRow(target, None, None, "no_source"))
            continue
        try:
            source = _resolve_source(target, piece, cache, tmp_dir, start_sec, duration_sec)
        except NoRecording:
            rows.append(PlanRow(target, piece, None, "no_recording"))
        except EmptyWindow:
            rows.append(PlanRow(target, piece, None, "empty_window"))
        except Exception as exc:  # noqa: BLE001 - a bad corpus file skips one config
            rows.append(
                PlanRow(target, piece, None, "no_source", f"{type(exc).__name__}: {exc}")
            )
        else:
            rows.append(PlanRow(target, piece, source, None))
    return rows


# --- rendering ----------------------------------------------------------------------


def _render_condition(
    target: Target,
    condition: Condition,
    source: Path,
    out_dir: Path,
    overwrite: bool,
) -> ConditionReport:
    """Render one condition, keeping a rejected override from stopping the rest."""
    config = target.config
    assert config is not None, "a rendered target always carries its config"
    output = f"{condition.slug}.{config.io.output_format}"
    try:
        # The naming and manifest switches only label this render; the manifest
        # keeps the condition's own overrides so it still says what it varied.
        overrides = {
            **condition.overrides,
            "io.file_naming": condition.slug,
            "observability.write_manifest": False,
        }
        if overwrite:
            # _render_file skips a file that already exists whenever the config
            # says so, so --overwrite has to override that key as well.
            overrides["render_pipeline.overwrite"] = True
        condition_config = apply_overrides(config, overrides)
        output = f"{condition.slug}.{condition_config.io.output_format}"
        if (out_dir / output).exists() and not overwrite:
            return ConditionReport(
                name=condition.name,
                slug=condition.slug,
                overrides=dict(condition.overrides),
                status="skipped_existing",
                output=output,
                error=None,
                render_seconds=0.0,
            )
        result = run_pipeline([source], out_dir, config=condition_config, corpus_root=None)
        entry = result.log[0] if result.log else {}
        raw_status = str(entry.get("status", "failed"))
        if raw_status == "succeeded":
            status, error = "succeeded", None
        elif raw_status == "skipped":
            status, error = "skipped_existing", None
        else:
            status = "failed"
            # A quality-gate rejection reports its reasons in quality_flags
            # rather than in an error string.
            reported = entry.get("error") or entry.get("quality_flags")
            error = str(reported) if reported else f"render reported status '{raw_status}'"
        return ConditionReport(
            name=condition.name,
            slug=condition.slug,
            overrides=dict(condition.overrides),
            status=status,
            output=output,
            error=error,
            render_seconds=float(entry.get("elapsed_seconds", 0.0)),
        )
    except Exception as exc:  # noqa: BLE001 - one bad condition must not stop the rest
        return ConditionReport(
            name=condition.name,
            slug=condition.slug,
            overrides=dict(condition.overrides),
            status="failed",
            output=output,
            error=f"{type(exc).__name__}: {exc}",
            render_seconds=0.0,
        )


def render_target(
    target: Target,
    source: Path,
    out_dir: Path,
    overwrite: bool,
) -> ConfigReport:
    """Render *source* once per condition of *target* into *out_dir*.

    Args:
        target: The config whose conditions are rendered.
        source: The excerpt every condition starts from.
        out_dir: ``<output>/<config stem>``.
        overwrite: Re-render conditions whose WAV already exists.

    Returns:
        The per-condition reports folded into one summary row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        _render_condition(target, condition, source, out_dir, overwrite)
        for condition in target.conditions
    ]
    counts = {
        status: sum(1 for report in reports if report.status == status)
        for status in ("succeeded", "failed", "skipped_existing")
    }
    return ConfigReport(
        config=target.path,
        status="partial" if counts["failed"] else "rendered",
        reason=None,
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        skipped_existing=counts["skipped_existing"],
        out_dir=out_dir,
        conditions=reports,
    )


def render_config_dir(
    row: PlanRow,
    out_dir: Path,
    overwrite: bool,
    excerpt: dict[str, float | None],
) -> ConfigReport:
    """Render one config and write its config.yaml and manifest.json.

    A write error is reported instead of raised, so the rest of the run
    continues; the config is then ``partial``, because its conditions are on
    disk but the run did not finish cleanly.
    """
    target = row.target
    assert row.source is not None and row.piece is not None
    try:
        report = render_target(target, row.source, out_dir, overwrite)
    except OSError as exc:
        print(f"error: could not render {target.path.stem} into {out_dir}: {exc}", file=sys.stderr)
        return ConfigReport(
            config=target.path,
            status="partial",
            reason=str(exc),
            succeeded=0,
            failed=len(target.conditions),
            skipped_existing=0,
            out_dir=out_dir,
        )
    try:
        write_config_dir(target, row.piece, report, out_dir, excerpt)
    except OSError as exc:
        print(f"error: could not write the manifest for {target.path.stem}: {exc}", file=sys.stderr)
        report.status = "partial"
        report.reason = str(exc)
    return report


def write_config_dir(
    target: Target,
    piece: Piece,
    report: ConfigReport,
    out_dir: Path,
    excerpt: dict[str, float | None],
) -> None:
    """Write the resolved ``config.yaml`` and ``manifest.json`` next to the audio."""
    config = target.config
    assert config is not None, "a rendered target always carries its config"
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    sample = config.io.sample
    manifest = {
        "config": str(target.path),
        "dataset": target.dataset,
        "dataset_source": target.dataset_source,
        "input_type": target.input_type,
        "source": {
            "reference": str(piece.reference),
            "recording": None if piece.recording is None else str(piece.recording),
            "pinned": piece.pinned,
            "seed": piece.seed,
        },
        "filters_ignored": {
            "where": dict(config.io.where),
            "sample": None if sample is None else sample.model_dump(),
        },
        "excerpt": excerpt,
        "conditions": [condition.to_dict() for condition in report.conditions],
    }
    # default=str: a YAML date or Path in an override must not cost the run the
    # manifest after its audio files are already written.
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )


def write_summary(
    path: Path,
    args_payload: dict[str, Any],
    pieces: dict[str, Piece],
    reports: Sequence[ConfigReport],
    rows: Sequence[PlanRow],
) -> None:
    """Write ``render_summary.json`` for the whole run."""
    used = sorted(
        {row.piece.dataset for row in rows if row.reason is None and row.piece is not None}
    )
    summary = {
        "created": datetime.now(timezone.utc).isoformat(),
        "args": args_payload,
        "pieces": {
            dataset: {
                "reference": str(pieces[dataset].reference),
                "recording": None
                if pieces[dataset].recording is None
                else str(pieces[dataset].recording),
                "pinned": pieces[dataset].pinned,
            }
            for dataset in used
        },
        "configs": [report.to_dict() for report in reports],
    }
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


# --- reporting ----------------------------------------------------------------------


def print_config_tree(
    paths: Sequence[Path],
    targets: Sequence[Target],
    configs_dir: Path,
    console: Console,
) -> None:
    """Print every selected config with its condition count, dataset and mode."""
    by_path = {target.path: target for target in targets}
    table = Table(title=f"benchmark configs in {configs_dir}")
    table.add_column("config", overflow="fold")
    table.add_column("conditions", justify="right")
    table.add_column("dataset")
    table.add_column("input")
    table.add_column("note")
    for path in paths:
        target = by_path.get(path)
        if target is None:
            continue
        if target.skip_reason is None:
            note = ""
        elif target.skip_reason == "load_error":
            note = (target.error or "").partition("\n")[0][:120]
        else:
            note = target.skip_reason
        table.add_row(
            _relative(path, configs_dir),
            str(len(target.conditions)),
            target.dataset or "-",
            target.input_type,
            note,
        )
    console.print(table)


def print_plan(
    rows: Sequence[PlanRow],
    configs_dir: Path,
    output: Path,
    console: Console,
    *,
    header: str | None = None,
) -> None:
    """Print what a run would do.

    Args:
        rows: One row per selected config, as :func:`build_plan` resolved it.
        configs_dir: Folder the config labels are relative to.
        output: Folder the run writes into.
        console: Where the plan is printed.
        header: Line above the plan; defaults to the ``--dry-run`` wording, which
            promises that nothing is written.
    """
    if header is None:
        header = f"dry run: {len(rows)} config(s); nothing is written to {output}"
    console.print(header)
    for row in rows:
        label = _relative(row.target.path, configs_dir)
        if row.reason is not None:
            console.print(f"  [yellow]skip[/yellow] {label}: {row.reason}")
            if row.detail:
                console.print(f"    [dim]{row.detail}[/dim]")
            continue
        assert row.piece is not None
        piece = row.piece.reference.name
        if row.piece.recording is not None:
            piece += f" + {row.piece.recording.name}"
        console.print(
            f"  {label}  dataset={row.target.dataset}  piece={piece}  "
            f"{len(row.target.conditions)} condition(s)  -> {output / row.target.path.stem}"
        )


def print_summary_table(reports: Sequence[ConfigReport], console: Console) -> None:
    """Print the final table, one row per config, as in ``render_summary.json``."""
    table = Table(title="render summary")
    table.add_column("config")
    table.add_column("status")
    table.add_column("reason")
    table.add_column("succeeded", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("skipped", justify="right")
    for report in reports:
        table.add_row(
            report.config.stem,
            report.status,
            report.reason or "-",
            str(report.succeeded),
            str(report.failed),
            str(report.skipped_existing),
        )
    console.print(table)


# --- interactive selection ------------------------------------------------------------

_SUBMIT = "submit"
_QUIT = "quit"
_ALL = "all"
_NONE = "none"
_INVERT = "invert"
_ROW = "row"
_RANGE = "range"
_FOLDER = "folder"

_COMMANDS = {"q": _QUIT, "a": _ALL, "n": _NONE, "i": _INVERT}

TOGGLE_HELP = (
    "Toggle: numbers / ranges (3 5-9) · folder name (sweeps) · a=all · n=none · "
    "i=invert · Enter=run · q=quit"
)


@dataclass(frozen=True)
class Token:
    """One word of a picker line: what it asks for, not what it should toggle."""

    kind: str
    first: int = 0
    last: int = 0
    name: str = ""


@dataclass(frozen=True)
class TreeRow:
    """One line of the picker: a folder with its configs, or a single config."""

    number: int
    label: str
    configs: tuple[Path, ...]
    conditions: int
    note: str = ""
    is_folder: bool = False

    @property
    def path(self) -> Path | None:
        """The config of a config row, or ``None`` for a folder row."""
        return None if self.is_folder else self.configs[0]


@dataclass(frozen=True)
class ConfigTree:
    """The selected configs as numbered rows, every folder above its configs."""

    configs_dir: Path
    rows: tuple[TreeRow, ...]

    @property
    def configs(self) -> tuple[Path, ...]:
        """Every config in the tree, in row order."""
        return tuple(row.path for row in self.rows if row.path is not None)

    def selected_conditions(self, selected: frozenset[Path]) -> int:
        """The conditions of *selected*, for the header count."""
        return sum(
            row.conditions for row in self.rows if row.path is not None and row.path in selected
        )

    def row(self, number: int) -> TreeRow:
        """The row numbered *number*.

        Raises:
            ToggleError: The tree has no such row.
        """
        for row in self.rows:
            if row.number == number:
                return row
        raise ToggleError(f"no row {number}; the tree has {len(self.rows)} row(s)")

    def folder(self, name: str) -> TreeRow:
        """The folder row labelled *name*, case- and slash-insensitively.

        Raises:
            ToggleError: No folder in the tree carries that name.
        """
        wanted = name.casefold().rstrip("/")
        for row in self.rows:
            if row.is_folder and row.label.casefold().rstrip("/") == wanted:
                return row
        raise ToggleError(f"no folder named {name!r}; type a row number instead")


def _plural(count: int, word: str) -> str:
    """``count`` and *word*, pluralised unless it is exactly one."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _config_note(target: Target | None) -> str:
    """The note of a config row: an audio config names the dataset it needs."""
    if target is None or target.input_type != _AUDIO:
        return ""
    return f"audio · {target.dataset or '-'}"


def build_tree(paths: Sequence[Path], targets: Sequence[Target], configs_dir: Path) -> ConfigTree:
    """Group the selected configs into folder and config rows numbered from 1.

    A folder row holds every config whose path starts with that folder, so a
    config nested more deeply still appears and is labelled with its file name.
    Folders and configs share one numbering, in the order the paths were given.

    Args:
        paths: The already filtered config paths.
        targets: Loaded targets, for condition counts and datasets.
        configs_dir: Folder the labels are relative to.

    Returns:
        The rows to show and toggle.
    """
    by_path = {target.path: target for target in targets}
    groups: dict[str, list[Path]] = {}
    for path in paths:
        parts = Path(_relative(path, configs_dir)).parts
        groups.setdefault(f"{parts[0]}/" if len(parts) > 1 else "", []).append(path)

    rows: list[TreeRow] = []
    number = 0
    for folder, members in groups.items():
        counts = [len(by_path[path].conditions) if path in by_path else 0 for path in members]
        total = sum(counts)
        if folder:
            number += 1
            note = f"({_plural(len(members), 'config')}, {_plural(total, 'condition')})"
            rows.append(
                TreeRow(
                    number=number,
                    label=folder,
                    configs=tuple(members),
                    conditions=total,
                    note=note,
                    is_folder=True,
                )
            )
        for path, count in zip(members, counts):
            number += 1
            rows.append(
                TreeRow(
                    number=number,
                    label=path.name,
                    configs=(path,),
                    conditions=count,
                    note=_config_note(by_path.get(path)),
                )
            )
    return ConfigTree(configs_dir, tuple(rows))


def _row_number(word: str) -> int | None:
    """The number in *word*, or ``None`` when it is not a plain decimal integer."""
    return int(word) if word.isdecimal() else None


def _parse_word(word: str) -> Token:
    """The single token *word* stands for.

    Raises:
        ToggleError: The word is malformed, so guessing an intent would toggle
            a config the user did not name.
    """
    command = _COMMANDS.get(word.casefold())
    if command is not None:
        return Token(command)
    if "-" in word:
        head, _, tail = word.partition("-")
        first, last = _row_number(head), _row_number(tail)
        if first is None or last is None:
            raise ToggleError(f"{word!r} is not a row range; write it as first-last, e.g. 5-9")
        if first < 1 or last < first:
            raise ToggleError(f"range {word!r} must ascend from row 1, e.g. 5-9")
        return Token(_RANGE, first=first, last=last)
    number = _row_number(word)
    if number is not None:
        if number < 1:
            raise ToggleError(f"row numbers start at 1, got {word!r}")
        return Token(_ROW, first=number)
    if any(char.isdecimal() for char in word):
        raise ToggleError(f"{word!r} is neither a row number nor a range, e.g. 5 or 5-9")
    return Token(_FOLDER, name=word)


def parse_tokens(text: str) -> list[Token]:
    """Read one picker line into tokens, left to right.

    A blank line is a submit. Each word is a command (``q``, ``a``, ``n``, ``i``),
    a row number, an inclusive ``first-last`` range, or a folder name.

    Args:
        text: The line as typed.

    Returns:
        One token per word, or a single submit token for a blank line.

    Raises:
        ToggleError: A word is not one of those.
    """
    words = text.split()
    if not words:
        return [Token(_SUBMIT)]
    return [_parse_word(word) for word in words]


def row_mark(selected: frozenset[Path], configs: Sequence[Path]) -> str:
    """Return ``[x]``, ``[ ]`` or ``[-]`` for *configs* against *selected*."""
    if not configs:
        return "[ ]"
    hits = sum(1 for path in configs if path in selected)
    if hits == len(configs):
        return "[x]"
    return "[ ]" if hits == 0 else "[-]"


def _toggle_row(selected: set[Path], row: TreeRow) -> None:
    """In place: switch every config of *row* the way the whole row is not.

    A config row holds one path, so this is its plain toggle. A folder row moves
    as one, which is what makes a partly selected folder end up fully selected.
    """
    turn_on = not all(path in selected for path in row.configs)
    for path in row.configs:
        if turn_on:
            selected.add(path)
        else:
            selected.discard(path)


def apply_toggle(
    state: frozenset[Path], tokens: Sequence[Token], tree: ConfigTree
) -> frozenset[Path]:
    """Return *state* with every token applied left to right.

    *state* is never touched: the words are applied to a copy and the new
    selection is only returned once every word is understood, so a rejected
    word leaves the caller's selection exactly as it was.

    Args:
        state: The config paths currently selected.
        tokens: As returned by :func:`parse_tokens`.
        tree: The rows the numbers and folder names refer to.

    Returns:
        The new selection.

    Raises:
        ToggleError: A word names no row or folder, or is the submit or quit
            token, which only the prompt loop acts on.
    """
    selected = set(state)
    for token in tokens:
        if token.kind == _ALL:
            selected = set(tree.configs)
        elif token.kind == _NONE:
            selected.clear()
        elif token.kind == _INVERT:
            selected = {path for path in tree.configs if path not in selected}
        elif token.kind == _ROW:
            _toggle_row(selected, tree.row(token.first))
        elif token.kind == _RANGE:
            for number in range(token.first, token.last + 1):
                _toggle_row(selected, tree.row(number))
        elif token.kind == _FOLDER:
            _toggle_row(selected, tree.folder(token.name))
        else:
            raise ToggleError(f"{token.kind} is not a toggle; type rows, ranges or a folder name")
    return frozenset(selected)


# A folder is a group header and a config is a leaf, so the two are styled
# apart rather than left to be read from the trailing slash alone. Bold doubles
# the cue, so the distinction survives a terminal that renders no colour.
_FOLDER_STYLE = "bold cyan"
_META_STYLE = "dim"


def print_tree(tree: ConfigTree, selected: frozenset[Path], console: Console) -> None:
    """Print the picker: the header count, one line per row, the accepted words.

    Folders are drawn in :data:`_FOLDER_STYLE` and their configs in the default
    colour, so a group header is distinguishable from its contents at a glance.
    Every cell is a :class:`~rich.text.Text` rather than a string: a config whose
    name contains brackets would otherwise be read as console markup and vanish.
    """
    console.print(
        f"[bold]Benchmark configs in {tree.configs_dir}[/bold]  "
        f"[dim]selected {len(selected)}/{len(tree.configs)} · "
        f"{tree.selected_conditions(selected)} conditions[/dim]"
    )
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    table.add_column("mark", width=3, no_wrap=True)
    table.add_column("row", justify="right", no_wrap=True)
    table.add_column("config", overflow="fold")
    table.add_column("conditions", justify="right", no_wrap=True)
    table.add_column("note", overflow="fold")
    for row in tree.rows:
        table.add_row(
            Text(row_mark(selected, row.configs)),
            Text(str(row.number), style=_META_STYLE),
            Text(row.label, style=_FOLDER_STYLE if row.is_folder else _META_STYLE),
            Text("" if row.is_folder else str(row.conditions), style=_META_STYLE),
            Text(row.note, style=_META_STYLE),
        )
    console.print(table)
    # soft_wrap: the help stays on one line instead of being folded mid-word
    console.print(f"[dim]{TOGGLE_HELP}[/dim]", soft_wrap=True)


def _read_line(prompt: str) -> str:
    """Read one line from the terminal; the picker's only input point."""
    return input(prompt)


def prompt_tree(tree: ConfigTree, console: Console) -> frozenset[Path] | None:
    """Re-render the picker until the user submits a selection or quits.

    A rejected word prints its reason and asks again, and so does a submit with
    nothing selected: neither ends the run, because both are recoverable.

    Args:
        tree: The rows to show and toggle.
        console: Where the picker is printed.

    Returns:
        The selected config paths, or ``None`` when the user quit or the input
        ended.
    """
    selected = frozenset(tree.configs)
    while True:
        print_tree(tree, selected, console)
        try:
            line = _read_line("> ")
        except (EOFError, KeyboardInterrupt):
            return None
        try:
            tokens = parse_tokens(line)
        except ToggleError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        kinds = {token.kind for token in tokens}
        if _QUIT in kinds:
            return None
        if _SUBMIT in kinds:
            if not selected:
                console.print("[red]nothing selected: type a row number, or q to quit[/red]")
                continue
            return selected
        try:
            selected = apply_toggle(selected, tokens, tree)
        except ToggleError as exc:
            console.print(f"[red]{exc}[/red]")


def _confirm_render(output: Path, conditions: int) -> bool:
    """Ask whether to render *conditions* into *output*; only yes proceeds."""
    try:
        answer = _read_line(f"Render {conditions} conditions into {output}? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().casefold() in {"y", "yes"}


# --- command line -------------------------------------------------------------------


def _duration_value(text: str) -> float | None:
    """Parse ``--duration``: a positive number of seconds, or ``full``."""
    if text.strip().lower() == "full":
        return None
    try:
        seconds = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--duration takes a positive number of seconds or 'full', got {text!r}"
        ) from None
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError(
            f"--duration must be a positive number of seconds or 'full', got {text!r}"
        )
    return seconds


def _start_value(text: str) -> float:
    """Parse ``--start``: zero or a positive number of seconds."""
    try:
        seconds = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--start takes seconds, got {text!r}") from None
    if not math.isfinite(seconds) or seconds < 0.0:
        raise argparse.ArgumentTypeError(
            f"--start must be zero or more seconds, got {text!r}"
        )
    return seconds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Folder for the rendered audio files, manifests and render_summary.json.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("config/benchmark"),
        metavar="DIR",
        help="Folder searched recursively for benchmark configs. Default: config/benchmark",
    )
    parser.add_argument(
        "--include",
        action="extend",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="Keep only configs whose path relative to --configs-dir matches GLOB; repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="extend",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="Drop configs matching GLOB; repeatable, wins over --include.",
    )
    parser.add_argument("--yes", action="store_true", help="Render every matching config without asking.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the discovered configs and their condition counts, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and write nothing.",
    )
    parser.add_argument(
        "--duration",
        type=_duration_value,
        action="extend",
        nargs="+",
        default=[],
        metavar="SECONDS|full",
        help="Exactly one excerpt length in seconds, or 'full' for the whole piece. Default: 30",
    )
    parser.add_argument(
        "--start",
        type=_start_value,
        default=0.0,
        metavar="SECONDS",
        help="Where the excerpt starts in the piece. Default: 0",
    )
    parser.add_argument(
        "--default-dataset",
        metavar="NAME",
        help="Dataset for configs that set no io.dataset.",
    )
    parser.add_argument(
        "--source",
        action="extend",
        nargs="+",
        default=[],
        metavar="DATASET=STEM",
        help="Render this exact piece for a dataset instead of a sampled one; repeatable.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the piece sampling. Default: 0")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render conditions whose audio file is already there.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        metavar="PATH",
        help="Override io.corpus_root for every config.",
    )
    return parser


def _output_conflict(output: Path, configs_dir: Path, roots: Iterable[Path]) -> str | None:
    """Return why *output* may not be used, or ``None`` when it is safe.

    The render writes many files per config, so *output* may neither sit inside
    a folder the run reads nor wrap one: writing to the corpus root would drop
    ``render_summary.json`` beside the inputs and overwrite what is there.
    """
    resolved = output.resolve()
    for root in [configs_dir, *roots]:
        candidate = Path(root).resolve()
        if (
            resolved == candidate
            or candidate in resolved.parents
            or resolved in candidate.parents
        ):
            return f"error: --output {output} and {candidate} are the same folder tree"
    return None


def _corpus_roots(targets: Sequence[Target], corpus_root: Path | None) -> list[Path]:
    """Every corpus folder the run reads from and must not write into."""
    roots: list[Path] = []
    if corpus_root is not None:
        roots.append(corpus_root)
    for target in targets:
        if target.config is None or not target.dataset:
            continue
        paths = resolve_corpus_paths(target.config)
        roots.extend((paths.midi, paths.recordings))
    return roots


def _args_payload(args: argparse.Namespace, duration: float | None) -> dict[str, Any]:
    """Return the CLI options as a JSON-serialisable mapping."""
    return {
        "output": str(args.output),
        "configs_dir": str(args.configs_dir),
        "include": list(args.include),
        "exclude": list(args.exclude),
        "yes": bool(args.yes),
        "list": bool(args.list),
        "dry_run": bool(args.dry_run),
        "duration_sec": duration,
        "start_sec": args.start,
        "default_dataset": args.default_dataset,
        "source": list(args.source),
        "seed": args.seed,
        "overwrite": bool(args.overwrite),
        "corpus_root": None if args.corpus_root is None else str(args.corpus_root),
    }


def _resolve_duration(values: Sequence[float | None]) -> float | None:
    """Return the one ``--duration`` value, or the default when it was not given.

    Raises:
        UsageError: More than one length was asked for, which would make the
            rendered files ambiguous; compare lengths by running again.
    """
    if not values:
        return 30.0
    if len(values) > 1:
        raise UsageError(
            f"--duration takes exactly one value, got {len(values)}; to compare "
            "lengths, run again with another --output"
        )
    return values[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Render the selected benchmark examples. Returns 0, 1 or 2."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    console = Console()

    try:
        duration = _resolve_duration(args.duration)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pins: dict[str, str] = {}
    for item in args.source:
        dataset, separator, stem = item.partition("=")
        if not separator or not dataset or not stem:
            print(f"error: --source takes DATASET=STEM, got {item!r}", file=sys.stderr)
            return 2
        pins[dataset] = stem

    conflict = _output_conflict(args.output, args.configs_dir, [])
    if conflict is not None:
        print(conflict, file=sys.stderr)
        return 2
    if args.output.exists() and not args.output.is_dir():
        print(f"error: --output {args.output} exists and is not a directory", file=sys.stderr)
        return 2

    paths = filter_configs(
        discover_configs(args.configs_dir), args.include, args.exclude, args.configs_dir
    )
    if not paths:
        print(
            f"error: no benchmark config in {args.configs_dir} matches the given "
            "--include/--exclude",
            file=sys.stderr,
        )
        return 2

    interactive = not (args.yes or args.list or args.dry_run)
    if interactive and not sys.stdin.isatty():
        print(
            "error: not a terminal: pass --yes to render every matching config",
            file=sys.stderr,
        )
        return 2

    targets = load_targets(paths, args.default_dataset)
    if args.corpus_root is not None:
        for target in targets:
            if target.config is not None:
                target.config.io.corpus_root = args.corpus_root
    conflict = _output_conflict(
        args.output, args.configs_dir, _corpus_roots(targets, args.corpus_root)
    )
    if conflict is not None:
        print(conflict, file=sys.stderr)
        return 2

    if interactive:
        selection = prompt_tree(build_tree(paths, targets, args.configs_dir), console)
        if selection is None:
            return 0
        paths = [path for path in paths if path in selection]
        targets = [target for target in targets if target.path in selection]

    stems: dict[str, Path] = {}
    for path in paths:
        other = stems.get(path.stem)
        if other is not None:
            print(
                f"error: {path} and {other} share the stem '{path.stem}', "
                "so they would render into the same folder",
                file=sys.stderr,
            )
            return 2
        stems[path.stem] = path

    if args.list:
        try:
            print_config_tree(paths, targets, args.configs_dir, console)
        except OSError as exc:
            print(f"error: could not print the config list: {exc}", file=sys.stderr)
            return 2
        return 0

    try:
        pieces = pick_pieces(targets, args.seed, pins, args.corpus_root)
    except (UsageError, SelectionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    excerpt: dict[str, float | None] = {"start_sec": args.start, "duration_sec": duration}
    reports: list[ConfigReport] = []
    with tempfile.TemporaryDirectory(
        prefix="sonitra-render-examples-", ignore_cleanup_errors=True
    ) as tmp:
        # Excerpts are cut once per dataset and input type: every config sharing
        # a dataset renders the very same music, which is the point of the run.
        excerpts: dict[tuple[str, str], Path] = {}
        rows = build_plan(targets, pieces, excerpts, Path(tmp), args.start, duration)
        if args.dry_run:
            try:
                print_plan(rows, args.configs_dir, args.output, console)
            except OSError as exc:
                print(f"error: could not print the plan: {exc}", file=sys.stderr)
                return 2
            return 0

        if interactive:
            # A different header than --dry-run, because this run does write.
            try:
                print_plan(
                    rows,
                    args.configs_dir,
                    args.output,
                    console,
                    header=f"rendering plan: {len(rows)} config(s) into {args.output}",
                )
            except OSError as exc:
                print(f"error: could not print the plan: {exc}", file=sys.stderr)
                return 2
            conditions = sum(len(row.target.conditions) for row in rows if row.reason is None)
            if not _confirm_render(args.output, conditions):
                try:
                    console.print(f"nothing written to {args.output}")
                except OSError as exc:
                    print(f"error: could not print the notice: {exc}", file=sys.stderr)
                    return 1
                return 0

        try:
            args.output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"error: could not create {args.output}: {exc}", file=sys.stderr)
            return 2
        for row in rows:
            if row.reason is not None:
                reports.append(
                    ConfigReport(
                        config=row.target.path,
                        status="skipped",
                        reason=row.reason,
                        succeeded=0,
                        failed=0,
                        skipped_existing=0,
                        out_dir=None,
                    )
                )
                try:
                    console.print(
                        f"[warning]skipping {row.target.path.stem}: {row.reason}[/warning]"
                    )
                    if row.detail:
                        console.print(f"  [dim]{row.detail}[/dim]")
                except OSError as exc:
                    print(f"error: could not print the skip notice: {exc}", file=sys.stderr)
                    return 1
                continue
            out_dir = args.output / row.target.path.stem
            report = render_config_dir(row, out_dir, args.overwrite, excerpt)
            reports.append(report)
            try:
                console.print(
                    f"{row.target.path.stem}: {report.succeeded}/{len(report.conditions)} rendered"
                )
            except OSError as exc:
                print(f"error: could not print the progress line: {exc}", file=sys.stderr)
                return 1

    summary_path = args.output / "render_summary.json"
    try:
        write_summary(summary_path, _args_payload(args, duration), pieces, reports, rows)
    except OSError as exc:
        print(f"error: could not write {summary_path}: {exc}", file=sys.stderr)
        return 1
    try:
        print_summary_table(reports, console)
    except OSError as exc:
        print(f"error: could not print the summary table: {exc}", file=sys.stderr)
        return 1
    return 1 if any(report.status == "partial" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())