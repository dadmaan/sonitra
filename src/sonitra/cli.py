from __future__ import annotations

import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from sonitra.terminal import (
    FilesPerSecondColumn,
    NullBenchmarkProgress,
    RichBenchmarkProgress,
    configure_framework_logging,
    configure_onednn_opts,
    effective_log_level,
    get_console,
    set_log_level,
    setup_logging,
)

if TYPE_CHECKING:
    # Annotation-only: importing sonitra.selection at module import pulls in
    # the config tree, which the CLI loads lazily per command.
    from sonitra.selection import SelectionResult

logger = logging.getLogger(__name__)

app = typer.Typer(name="sonitra")

_CLI_VERBOSE = False


def _apply_numeric_env(cfg: Any) -> None:
    """Export transcription numeric settings for backend builders/workers.

    Backends read SONITRA_NUMERIC_MODE / SONITRA_GPU_MEMORY_GROWTH at build
    time; benchmark pool workers inherit the env via fork. setdefault keeps
    an explicit user export winning over YAML.
    """
    os.environ.setdefault("SONITRA_NUMERIC_MODE", cfg.transcription.numeric_mode)
    os.environ.setdefault(
        "SONITRA_GPU_MEMORY_GROWTH",
        "1" if cfg.transcription.gpu_memory_growth else "0",
    )


def _progress_enabled(cfg) -> bool:
    console = get_console()
    return cfg.observability.progress and console.is_terminal and not console.quiet


def _progress_columns() -> list[ProgressColumn | str]:
    """Shared column set for per-file progress bars (render/transcribe/evaluate)."""
    return [
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        FilesPerSecondColumn(),
        "ETA:",
        TimeRemainingColumn(),
    ]


def _discover_midi_files(directory: Path) -> list[Path]:
    """Thin re-export of :func:`sonitra.corpus.discover_midi_files`.

    Kept as a module-level name in ``cli.py`` (rather than importing the
    corpus module at every call site) so existing imports of
    ``sonitra.cli._discover_midi_files`` keep working unmodified.
    """
    from sonitra.corpus import discover_midi_files

    return discover_midi_files(directory)


def _apply_subset(
    files: list[Path], limit: int | None, seed: int | None
) -> list[Path]:
    """Backward-compatible alias for :func:`sonitra.selection.sample_units`."""
    from sonitra.config import SelectionSample
    from sonitra.selection import sample_units

    sample = SelectionSample(n=limit, seed=seed or 0) if limit is not None else None
    return sample_units(files, sample)


def _stderr_console() -> Console:
    """Return a stderr-bound console matching the main console's flags."""
    console = get_console()
    return Console(stderr=True, quiet=console.quiet, no_color=console.no_color)


def _fail_selection(exc: Exception) -> None:
    """Print a selection error to stderr and exit with code 1."""
    _stderr_console().print(f"[red]error: {exc}[/red]")
    raise typer.Exit(code=1)


def _load_config_or_exit(config: Path) -> PipelineConfig:
    """Load *config*, or print a one-line error to stderr and exit with code 1."""
    from rich.markup import escape

    from sonitra.config import ConfigError, load_config

    try:
        return load_config(config)
    except FileNotFoundError as exc:
        _stderr_console().print(f"[red]error: {escape(str(exc))}[/red]")
    except ConfigError as exc:
        _stderr_console().print(
            f"[red]error: invalid config {escape(str(config))}:[/red]\n{escape(str(exc))}"
        )
    raise typer.Exit(code=1)


def _print_selection(result: SelectionResult) -> None:
    """Print resolver warnings to stderr and the notice/summary to stdout."""
    console = get_console()
    warnings = _stderr_console()
    if result.notice:
        console.print(f"[dim]{result.notice}[/dim]")
    if result.unmatched:
        names = ", ".join(path.stem for path in result.unmatched[:5])
        warnings.print(
            f"[yellow]warning: {len(result.unmatched)} file(s) have no metadata "
            f"row and were excluded (e.g. {names})[/yellow]"
        )
    if result.unpaired_audio:
        names = ", ".join(path.stem for path in result.unpaired_audio[:5])
        warnings.print(
            f"[yellow]warning: {len(result.unpaired_audio)} audio file(s) did not "
            f"pair to a reference and were excluded (e.g. {names})[/yellow]"
        )
    console.print(result.summary_line())


_MAX_FAILURE_ROWS = 5


def _print_benchmark_failures(console: Console, failed: list[Any]) -> None:
    """Print failed benchmark records grouped by error message, most common first."""
    groups: dict[str, list[Any]] = {}
    for record in failed:
        groups.setdefault(record.error or record.status, []).append(record)
    ranked = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)

    table = Table(title="Benchmark failures", title_style="bold red")
    table.add_column("error", style="red")
    table.add_column("count", justify="right")
    table.add_column("example")
    for error, group in ranked[:_MAX_FAILURE_ROWS]:
        example = group[0]
        table.add_row(
            error,
            str(len(group)),
            f"{example.condition} / {Path(example.midi_path).name}",
        )
    if len(ranked) > _MAX_FAILURE_ROWS:
        table.add_row("", "", f"{len(ranked) - _MAX_FAILURE_ROWS} more distinct errors")
    console.print(table)


def _apply_dataset(cfg: PipelineConfig, dataset: str | None) -> None:
    """Inject *dataset* into *cfg* when provided.

    Args:
        cfg: Loaded pipeline configuration to mutate in place.
        dataset: Dataset name supplied via the CLI ``--dataset`` flag, or
            ``None`` when the flag was not set.

    Raises:
        SelectionError: when the YAML both sets ``io.dataset`` and filters it
            with ``io.where``, and *dataset* names a different one; the
            filter would otherwise be applied to another dataset's metadata.
    """
    if dataset is None:
        return
    if cfg.io.where and cfg.io.dataset is not None and cfg.io.dataset != dataset:
        from sonitra.selection import SelectionError

        raise SelectionError(
            f"this config filters dataset '{cfg.io.dataset}' (io.where); "
            f"--dataset '{dataset}' would apply that filter to another dataset"
        )
    cfg.io.dataset = dataset


def _apply_selection_overrides(
    cfg: PipelineConfig, limit: int | None, seed: int | None
) -> None:
    """Write ``--limit``/``--seed`` into ``cfg.io.sample``.

    Raises:
        SelectionError: when the resulting sample fails validation, e.g.
            ``--limit 0``.
    """
    if limit is None and seed is None:
        return

    from pydantic import ValidationError

    from sonitra.config import SelectionSample
    from sonitra.selection import SelectionError

    existing = cfg.io.sample
    if limit is not None:
        payload = {"n": limit, "seed": seed or 0}
    else:
        if existing is None:
            _stderr_console().print(
                "[yellow]warning: --seed has no effect without --limit or "
                "io.sample[/yellow]"
            )
            return
        payload = {"n": existing.n, "seed": seed}

    try:
        # Validate a rebuilt sample: attribute assignment skips validation, so
        # --limit 0 would reach the resolver unchecked.
        cfg.io.sample = SelectionSample.model_validate(payload)
    except ValidationError as exc:
        raise SelectionError(f"invalid selection from --limit/--seed: {exc}") from exc


@app.command()
def render(
    config: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to pipeline config YAML"
    ),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", "-i", help="Directory of MIDI files to render"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output audio directory"
    ),
    overwrite: bool = typer.Option(
        True, "--overwrite/--no-overwrite", help="Overwrite existing output files"
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers (overrides config)"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Dataset name (overrides io.dataset); reads inputs from and writes "
            "outputs under corpus/{dataset}/"
        ),
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum MIDI files to render (random subset)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for --limit sampling."
    ),
) -> None:
    """Run the MIDI-to-audio rendering pipeline (MIDI mode) or the audio
    read -> effects -> quality-gate pipeline (audio mode, ``pipeline.input_type:
    audio``)."""
    from sonitra.config import InputType, resolve_corpus_paths
    from sonitra.corpus import discover_audio_files
    from sonitra.pipeline import run_pipeline
    from sonitra.selection import SelectionError, select_audio, select_references

    console = get_console()
    cfg = _load_config_or_exit(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
        configure_framework_logging(effective_log_level(cfg))
        configure_onednn_opts()
    try:
        _apply_dataset(cfg, dataset)
    except SelectionError as exc:
        _fail_selection(exc)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)
    audio_mode = cfg.render_pipeline.input_type == InputType.AUDIO

    if corpus is not None:
        actual_corpus = corpus
    elif audio_mode:
        actual_corpus = paths.recordings
    else:
        actual_corpus = paths.midi
    actual_output = output if output is not None else paths.audio

    if audio_mode:
        midi_paths = discover_audio_files(actual_corpus)
        source_label = "audio files"
    else:
        midi_paths = _discover_midi_files(actual_corpus)
        source_label = "MIDI files"
    if not midi_paths:
        console.print(f"[red]No {source_label} found in[/red] [dim]{actual_corpus}[/dim]")
        raise typer.Exit(code=1)

    try:
        _apply_selection_overrides(cfg, limit, seed)
        if audio_mode:
            # Pairing needs the reference list; without a filter there is
            # nothing to pair for, so recordings pass through unchanged.
            references = (
                _discover_midi_files(paths.midi)
                if cfg.io.where
                else []
            )
            selection_result = select_audio(
                cfg, midi_paths, references, unit_kind="recording"
            )
        else:
            selection_result = select_references(cfg, midi_paths)
    except SelectionError as exc:
        _fail_selection(exc)
    _print_selection(selection_result)
    midi_paths = selection_result.units

    progress: Progress | None = None
    task_id: Any = None
    if _progress_enabled(cfg):
        progress = Progress(*_progress_columns(), refresh_per_second=10)
        task_id = progress.add_task("render", total=len(midi_paths))

    def _on_file_done(entry: dict[str, Any]) -> None:
        if progress is not None:
            progress.update(task_id, advance=1)

    try:
        if progress is not None:
            with progress:
                result = run_pipeline(
                    midi_paths,
                    out_dir=actual_output,
                    config=cfg,
                    corpus_root=actual_corpus,
                    on_file_done=_on_file_done,
                )
        else:
            result = run_pipeline(
                midi_paths,
                out_dir=actual_output,
                config=cfg,
                corpus_root=actual_corpus,
            )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted — partial renders kept[/yellow]")
        raise typer.Exit(130)

    msg = (
        f"Done: [green]{result.succeeded} succeeded[/], "
        f"[red]{result.failed} failed[/], "
        f"[yellow]{result.skipped} skipped[/] ({result.elapsed_seconds:.2f}s)"
    )
    if result.failed > 0:
        msg = f"[red]{msg}[/]"
    console.print(msg)
    if result.failed:
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    config: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to pipeline config YAML"
    ),
    audio: Optional[Path] = typer.Option(
        None, "--audio", "-i", help="Directory of audio files to transcribe"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output MIDI directory"
    ),
    transcriber: Optional[str] = typer.Option(
        None, "--transcriber", "-t", help="Only run the transcriber with this name/type"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Dataset name (overrides io.dataset); reads inputs from and writes "
            "outputs under corpus/{dataset}/"
        ),
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum audio files to transcribe (random subset)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for --limit sampling."
    ),
) -> None:
    """Transcribe audio files to MIDI with the configured transcribers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sonitra.config import resolve_corpus_paths
    from sonitra.midi_writer import write_transcription_outputs
    from sonitra.selection import SelectionError, select_audio
    from sonitra.transcribe.protocol import make_transcriber

    console = get_console()
    cfg = _load_config_or_exit(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
        configure_framework_logging(effective_log_level(cfg))
        configure_onednn_opts()
    _apply_numeric_env(cfg)
    try:
        _apply_dataset(cfg, dataset)
    except SelectionError as exc:
        _fail_selection(exc)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    if audio is not None:
        actual_audio = audio
    elif cfg.io.dataset is not None:
        actual_audio = paths.audio
    else:
        console.print(
            "[red]--audio is required when no dataset is set "
            "(--dataset or io.dataset)[/red]"
        )
        raise typer.Exit(code=1)

    if output is not None:
        actual_output = output
    elif cfg.io.dataset is not None:
        actual_output = paths.transcription
    else:
        actual_output = Path("transcriptions")

    transcriber_configs = [t for t in cfg.transcription.transcribers if t.enabled]
    if transcriber is not None:
        transcriber_configs = [
            t for t in transcriber_configs if transcriber in {t.name, t.type}
        ]
    if not transcriber_configs:
        console.print("[red]No matching enabled transcribers in config[/red]")
        raise typer.Exit(code=1)

    audio_paths = sorted(
        path for ext in ("*.wav", "*.flac", "*.mp3") for path in actual_audio.rglob(ext)
    )
    if not audio_paths:
        console.print(f"[red]No audio files found in[/red] [dim]{actual_audio}[/dim]")
        raise typer.Exit(code=1)

    try:
        _apply_selection_overrides(cfg, limit, seed)
        # References are only needed to pair rendered audio back to MIDI when
        # a filter selects a subset; without one, --audio works as before.
        references = (
            _discover_midi_files(paths.midi)
            if cfg.io.where
            else []
        )
        selection_result = select_audio(cfg, audio_paths, references, unit_kind="audio")
    except SelectionError as exc:
        _fail_selection(exc)
    _print_selection(selection_result)
    audio_paths = selection_result.units

    failures = 0
    failure_details: list[tuple[str, str, str]] = []
    n_workers = cfg.transcription.max_workers
    show_progress = _progress_enabled(cfg)

    def _transcribe_one(backend_name: str, backend_transcribe, audio_path: Path) -> tuple[str, str | None]:
        rel = audio_path.relative_to(actual_audio)
        midi_path = actual_output / backend_name / rel.with_suffix(".mid")
        try:
            result = backend_transcribe(audio_path)
            write_transcription_outputs(result, midi_path)
            return f"{backend_name}: {audio_path.name} -> {midi_path}", None
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            return f"{backend_name}: {audio_path.name} FAILED ({exc})", str(exc)

    try:
        for transcriber_cfg in transcriber_configs:
            backend = make_transcriber(transcriber_cfg)
            failed_this = 0
            progress: Progress | None = None
            task_id: Any = None
            if show_progress:
                progress = Progress(*_progress_columns(), refresh_per_second=10)
                task_id = progress.add_task(backend.name, total=len(audio_paths))

            with progress or nullcontext():
                if n_workers > 1:
                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        future_to_path = {
                            executor.submit(_transcribe_one, backend.name, backend.transcribe, ap): ap
                            for ap in audio_paths
                        }
                        for future in as_completed(future_to_path):
                            _, err = future.result()
                            if progress is not None:
                                progress.update(task_id, advance=1)
                            if err is not None:
                                failures += 1
                                failed_this += 1
                                if len(failure_details) < 10:
                                    failure_details.append(
                                        (backend.name, future_to_path[future].name, err)
                                    )
                else:
                    for audio_path in audio_paths:
                        _, err = _transcribe_one(backend.name, backend.transcribe, audio_path)
                        if progress is not None:
                            progress.update(task_id, advance=1)
                        if err is not None:
                            failures += 1
                            failed_this += 1
                            if len(failure_details) < 10:
                                failure_details.append((backend.name, audio_path.name, err))

            console.print(
                f"[cyan]{backend.name}[/]: [green]{len(audio_paths) - failed_this} ok[/], "
                f"[red]{failed_this} failed[/]"
            )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted — partial transcriptions kept[/yellow]")
        raise typer.Exit(130)

    if failures:
        table = Table(title="Transcription failures", title_style="bold red")
        table.add_column("transcriber", style="red")
        table.add_column("file")
        table.add_column("error")
        for transcriber_name, file_name, error in failure_details:
            table.add_row(transcriber_name, file_name, error)
        if failures > len(failure_details):
            table.add_row(
                "...",
                "",
                f"{failures - len(failure_details)} more failures not shown",
            )
        console.print(table)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    reference: Optional[Path] = typer.Option(
        None, "--reference", "-r", help="Directory of reference MIDI files"
    ),
    estimate: Optional[Path] = typer.Option(
        None, "--estimate", "-e", help="Directory of estimated/transcribed MIDI files"
    ),
    config: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to pipeline config YAML"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write per-file results to this JSONL path"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Dataset name (overrides io.dataset); reads inputs from and writes "
            "outputs under corpus/{dataset}/"
        ),
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum reference files to evaluate (random subset)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for --limit sampling."
    ),
) -> None:
    """Score estimated MIDI against reference MIDI, paired by file stem."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    from sonitra.config import resolve_corpus_paths
    from sonitra.evaluation.protocol import evaluate_notes, make_symbolic_metrics
    from sonitra.evaluation.types import notes_from_dicts
    from sonitra.midi_reader import parse_midi
    from sonitra.selection import SelectionError, sample_units, select_references

    console = get_console()
    cfg = _load_config_or_exit(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
        configure_framework_logging(effective_log_level(cfg))
        configure_onednn_opts()
    try:
        _apply_dataset(cfg, dataset)
    except SelectionError as exc:
        _fail_selection(exc)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)
    section = cfg.evaluation

    if cfg.io.dataset is not None:
        midi_dir: Path | None = paths.midi
        # Transcribers may omit the optional `name`; fall back to the backend
        # type exactly as the transcribe command's output dirs are named.
        first_transcriber = "basic_pitch"
        if cfg.transcription.transcribers:
            first_transcriber = (
                cfg.transcription.transcribers[0].name
                or cfg.transcription.transcribers[0].type
            )
        transcription_dir: Path | None = paths.transcription
    else:
        midi_dir = None
        first_transcriber = None
        transcription_dir = None

    metrics = make_symbolic_metrics(section)

    # Resolve reference path.
    actual_reference: Path
    if reference is not None:
        actual_reference = reference
    elif cfg.io.dataset is not None:
        actual_reference = midi_dir  # type: ignore[assignment]
    else:
        console.print(
            "[red]--reference is required when no dataset is set "
            "(--dataset or io.dataset)[/red]"
        )
        raise typer.Exit(code=1)

    # Resolve estimate path.
    actual_estimate: Path
    if estimate is not None:
        actual_estimate = estimate
    elif cfg.io.dataset is not None:
        actual_estimate = transcription_dir / first_transcriber  # type: ignore[operator]
    else:
        console.print(
            "[red]--estimate is required when no dataset is set "
            "(--dataset or io.dataset)[/red]"
        )
        raise typer.Exit(code=1)

    reference_paths = _discover_midi_files(actual_reference)
    if not reference_paths:
        console.print(f"[red]No MIDI files found in[/red] [dim]{actual_reference}[/dim]")
        raise typer.Exit(code=1)

    def _find_estimate(ref_path: Path) -> Path | None:
        rel = ref_path.relative_to(actual_reference)
        return next(
            (
                candidate
                for ext in (".mid", ".midi")
                if (candidate := actual_estimate / rel.with_suffix(ext)).exists()
            ),
            None,
        )

    try:
        _apply_selection_overrides(cfg, limit, seed)
        result = select_references(cfg, reference_paths, apply_sample=False)
        selection_sample = cfg.io.sample
        if selection_sample is not None:
            # Keep the existing ordering: refs with an estimate are filtered
            # before sampling, so every sampled unit is scoreable.
            with_estimates = [
                p for p in result.units if _find_estimate(p) is not None
            ]
            if not with_estimates:
                console.print("[red]No reference files have matching estimates[/red]")
                raise typer.Exit(code=1)
            result = result.with_sampled_units(
                sample_units(with_estimates, selection_sample)
            )
    except SelectionError as exc:
        _fail_selection(exc)
    reference_paths = result.units
    _print_selection(result)

    # Pairs by stem; assumes globally unique filenames across the reference corpus.
    # See .local/notes/TODO.md for the known limitation with nested datasets.
    def _eval_one(ref_path: Path) -> dict | None:
        est_path = _find_estimate(ref_path)
        if est_path is None:
            return None
        rel = ref_path.relative_to(actual_reference)
        try:
            values = evaluate_notes(
                notes_from_dicts(parse_midi(ref_path)),
                notes_from_dicts(parse_midi(est_path)),
                metrics,
            )
        except Exception as exc:  # noqa: BLE001 - evaluate logs and continues
            # Unreadable MIDI (parse_midi raises OSError on a missing MTrk
            # header) or a contract violation must not abort the batch.
            logger.warning("evaluate failed for %s: %s", rel, exc)
            return {"file": str(rel), "error": str(exc)}
        return {"file": str(rel), **values}

    show_progress = console.is_terminal and not console.quiet
    show_progress = show_progress and cfg.observability.progress

    rows: list[dict] = []
    failures: list[dict] = []
    skips = 0
    progress: Progress | None = None
    task_id: Any = None
    if show_progress:
        progress = Progress(*_progress_columns(), refresh_per_second=8)
        task_id = progress.add_task("evaluate", total=len(reference_paths))

    def _consume(result: dict | None) -> None:
        nonlocal skips
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)
        if result is None:
            skips += 1
            if progress is not None and task_id is not None:
                progress.update(task_id, description=f"evaluate - {skips} skipped")
        elif "error" in result:
            failures.append(result)
            if progress is not None and task_id is not None:
                progress.update(task_id, description=f"evaluate - {len(failures)} failed")
        else:
            rows.append(result)

    n_eval_workers = section.max_workers
    try:
        with progress or nullcontext():
            if n_eval_workers > 1:
                with ThreadPoolExecutor(max_workers=n_eval_workers) as executor:
                    for result in executor.map(_eval_one, reference_paths):
                        _consume(result)
            else:
                for ref_path in reference_paths:
                    _consume(_eval_one(ref_path))
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted — partial results kept[/yellow]")
        raise typer.Exit(130)

    # Failure records are written to the JSONL so the run leaves a per-item
    # trace, but they are kept out of `rows` so means stay over scored pairs.
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in [*rows, *failures]:
                handle.write(json.dumps(row) + "\n")

    if failures:
        console.print(f"[yellow]{len(failures)} pair(s) failed to evaluate[/yellow]")
        for row in failures:
            console.print(f"  [dim]{row['file']}: {row['error']}[/dim]")

    if not rows:
        console.print("[red]No reference/estimate pairs evaluated[/red]")
        raise typer.Exit(code=1)

    metric_names = sorted({key for row in rows for key in row if key != "file"})
    console.print(f"Evaluated {len(rows)} pairs (mean over files):")
    for name in metric_names:
        values = [row[name] for row in rows if name in row and not math.isnan(row[name])]
        mean = sum(values) / len(values) if values else float("nan")
        if math.isnan(mean):
            console.print(f"  [cyan]{name}[/]: [dim]NaN[/dim]")
        else:
            console.print(f"  [cyan]{name}[/]: {mean:.4f}")


@app.command()
def benchmark(
    config: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to pipeline config YAML"
    ),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", "-i", help="Directory of reference MIDI files"
    ),
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        "-w",
        help=(
            "Full run directory; overrides benchmark.benchmark_dir and the "
            "default corpus/{dataset}/benchmark/{config stem}"
        ),
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Dataset name (overrides io.dataset); reads inputs from and writes "
            "outputs under corpus/{dataset}/"
        ),
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum MIDI files to benchmark (random subset)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for --limit sampling."
    ),
) -> None:
    """Run the full AMT benchmark: render, transcribe, and evaluate per condition."""
    from sonitra.benchmark.runner import run_benchmark
    from rich.markup import escape

    from sonitra.config import InputType, resolve_benchmark_dir, resolve_corpus_paths
    from sonitra.corpus import discover_audio_files
    from sonitra.selection import SelectionError, select_audio, select_references
    from sonitra.transcribe.base import TranscriptionError

    console = get_console()
    cfg = _load_config_or_exit(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
        configure_framework_logging(effective_log_level(cfg))
        configure_onednn_opts()
    _apply_numeric_env(cfg)
    yaml_dataset = cfg.io.dataset
    try:
        if (
            cfg.benchmark.benchmark_dir is not None
            and dataset is not None
            and dataset != yaml_dataset
            and workdir is None
        ):
            raise SelectionError(
                f"benchmark.benchmark_dir is fixed ({cfg.benchmark.benchmark_dir}) "
                f"but --dataset '{dataset}' differs from io.dataset "
                f"'{yaml_dataset}'; pass --workdir or edit benchmark_dir"
            )
        _apply_dataset(cfg, dataset)
    except SelectionError as exc:
        _fail_selection(exc)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)
    audio_mode = cfg.render_pipeline.input_type == InputType.AUDIO

    # --corpus keeps meaning "reference MIDI dir" in both modes -- it is
    # never used to locate audio-mode recordings, which always come from
    # <corpus_root>/<dataset>/recordings.
    actual_corpus = corpus if corpus is not None else paths.midi
    # corpus_root passed to run_benchmark must be an ancestor of the paths
    # that are actually rendered/evaluated as source_path -- the recordings
    # (paths.recordings) in audio mode, not the reference-MIDI dir (which
    # --corpus/actual_corpus denotes); otherwise Path.relative_to() in
    # _resolve_output_path/_evaluate_one raises ValueError.
    render_corpus_root = paths.recordings if audio_mode else actual_corpus
    actual_workdir = (
        workdir if workdir is not None else resolve_benchmark_dir(cfg, config.stem)
    )

    midi_paths = _discover_midi_files(actual_corpus)
    if not midi_paths:
        console.print(f"[red]No MIDI files found in[/red] [dim]{actual_corpus}[/dim]")
        raise typer.Exit(code=1)

    audio_paths: list[Path] | None = None
    try:
        _apply_selection_overrides(cfg, limit, seed)
        if audio_mode:
            # --limit/--seed applies to the recordings (the per-cell unit)
            # *after* pairing, so unpaired recordings never consume the
            # sample budget; midi_paths is re-derived from the sampled pairs
            # and never sampled independently, keeping the two lists
            # mutually consistent.
            recordings = discover_audio_files(paths.recordings)
            if not recordings:
                console.print(
                    f"[red]No audio files found in[/red] [dim]{paths.recordings}[/dim]"
                )
                raise typer.Exit(code=1)
            selection_result = select_audio(
                cfg,
                recordings,
                midi_paths,
                unit_kind="recording",
                always_pair=True,
            )
            audio_paths = selection_result.units
            midi_paths = selection_result.references
        else:
            selection_result = select_references(cfg, midi_paths)
            midi_paths = selection_result.units
    except SelectionError as exc:
        _fail_selection(exc)
    _print_selection(selection_result)
    # Relative to the directory the units were discovered from, so the hash
    # stays host-independent and honours --corpus in MIDI mode.
    selection_provenance = selection_result.provenance(
        unit_root=paths.recordings if audio_mode else actual_corpus
    )

    show_progress = _progress_enabled(cfg)
    devices = {
        t.name or t.type: t.device
        for t in cfg.transcription.transcribers
        if t.enabled and hasattr(t, "device")
    }
    prog = (
        RichBenchmarkProgress(
            get_console(),
            n_workers=cfg.benchmark.max_workers,
            devices=devices,
        )
        if show_progress
        else NullBenchmarkProgress()
    )
    cm = prog if show_progress else nullcontext(prog)
    try:
        with cm:
            result = run_benchmark(
                midi_paths,
                actual_workdir,
                cfg,
                corpus_root=render_corpus_root,
                audio_paths=audio_paths,
                progress=prog,
                selection=selection_provenance,
            )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted — partial results kept in manifests[/yellow]")
        raise typer.Exit(130)
    except TranscriptionError as exc:
        # Raised by the device preflight before any render (missing optional
        # backend or absent accelerator). Per-file failures never reach here:
        # the run records them and continues.
        _stderr_console().print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    succeeded = sum(1 for record in result.records if record.status == "succeeded")
    total = len(result.records)
    if succeeded < total:
        console.print(
            f"Benchmark finished: [red]{succeeded}/{total} evaluations succeeded[/] "
            f"({result.elapsed_seconds:.1f}s)"
        )
    else:
        console.print(
            f"Benchmark finished: [green]{succeeded}[/]/{total} evaluations succeeded "
            f"({result.elapsed_seconds:.1f}s)"
        )
    console.print(f"Results: [dim]{result.results_path}[/dim]")
    console.print(f"Summary: [dim]{result.summary_path}[/dim]")

    failed_records = [record for record in result.records if record.status != "succeeded"]
    if failed_records:
        _print_benchmark_failures(console, failed_records)

    if result.summary:
        table = Table(title="Benchmark summary")
        table.add_column("condition")
        table.add_column("transcriber")
        table.add_column("files", justify="right")
        table.add_column("ok", justify="right")
        table.add_column("failed", justify="right")
        metric_keys = sorted(
            {
                key
                for row in result.summary
                for key in row
                if key not in {"condition", "transcriber", "n_files", "n_succeeded"}
            }
        )
        f1_keys = [key for key in metric_keys if key == "f1" or key.endswith(".f1")]
        for name in f1_keys:
            table.add_column(name, justify="right")
        for row in result.summary:
            n_files = int(row.get("n_files", 0))
            n_ok = int(row.get("n_succeeded", 0))
            n_failed = n_files - n_ok
            cells = [
                str(row.get("condition", "")),
                str(row.get("transcriber", "")),
                str(n_files),
                f"[green]{n_ok}[/]",
                f"[red]{n_failed}[/]" if n_failed else "0",
            ]
            for name in f1_keys:
                value = row.get(name, float("nan"))
                if isinstance(value, (int, float)) and math.isnan(value):
                    cells.append("[dim]NaN[/dim]")
                elif isinstance(value, (int, float)):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            table.add_row(*cells)
        console.print(table)

    timing_conditions = (
        result.timing.get("conditions") if result.timing is not None else None
    )
    if timing_conditions:
        timing_table = Table(title="Benchmark timing (seconds)")
        timing_table.add_column("condition")
        timing_table.add_column("wall (sec)", justify="right")
        timing_table.add_column("render (sec)", justify="right")
        # separate (sec) only when at least one condition has a real value
        # (all-NaN when separation is disabled).
        has_separate = any(
            isinstance(entry.get("separate_seconds"), (int, float))
            and not math.isnan(entry["separate_seconds"])
            for entry in timing_conditions
        )
        if has_separate:
            timing_table.add_column("separate (sec)", justify="right")
        timing_table.add_column("transcribe (sec)", justify="right")
        timing_table.add_column("evaluate (sec)", justify="right")
        timing_keys = (
            "wall_seconds",
            "render_seconds",
            "separate_seconds",
            "transcribe_seconds",
            "evaluate_seconds",
        )
        for entry in timing_conditions:
            cells = [str(entry.get("condition", ""))]
            for key in timing_keys:
                if key == "separate_seconds" and not has_separate:
                    continue
                value = entry.get(key, float("nan"))
                if isinstance(value, (int, float)) and math.isnan(value):
                    cells.append("[dim]NaN[/dim]")
                elif isinstance(value, (int, float)):
                    cells.append(f"{value:.1f}")
                else:
                    cells.append(str(value))
            timing_table.add_row(*cells)
        console.print(timing_table)

        # Per-transcriber breakdown (Option A): transcribe/evaluate are
        # per-cell measurements, so they can be attributed per backend from
        # the `per_transcriber` roll-up already stored in summary.json.
        # wall/render/separate stay on the condition table above: wall is a
        # condition stopwatch and render/separate are shared per file.
        by_transcriber_rows: list[tuple[str, str, object, object, object]] = []
        for entry in timing_conditions:
            condition_name = str(entry.get("condition", ""))
            for pt in entry.get("per_transcriber") or []:
                by_transcriber_rows.append(
                    (
                        condition_name,
                        str(pt.get("transcriber", "")),
                        pt.get("transcribe_seconds", float("nan")),
                        pt.get("evaluate_seconds", float("nan")),
                        pt.get("n_succeeded", 0),
                    )
                )
        if by_transcriber_rows:
            bt_table = Table(title="Benchmark timing by transcriber (seconds)")
            bt_table.add_column("condition")
            bt_table.add_column("transcriber")
            bt_table.add_column("transcribe (sec)", justify="right")
            bt_table.add_column("evaluate (sec)", justify="right")
            bt_table.add_column("ok", justify="right")
            for cond, name, t_sec, e_sec, n_ok in by_transcriber_rows:
                bt_cells = [cond, name]
                for value in (t_sec, e_sec):
                    if isinstance(value, (int, float)) and math.isnan(value):
                        bt_cells.append("[dim]NaN[/dim]")
                    elif isinstance(value, (int, float)):
                        bt_cells.append(f"{value:.1f}")
                    else:
                        bt_cells.append(str(value))
                bt_cells.append(
                    str(int(n_ok)) if isinstance(n_ok, (int, float)) else str(n_ok)
                )
                bt_table.add_row(*bt_cells)
            console.print(bt_table)
            console.print(
                "[dim]per-transcriber values sum succeeded runs only.[/dim]"
            )

    if result.degradation:
        deg_table = Table(title="Benchmark degradation (delta vs baseline)")
        deg_table.add_column("condition")
        deg_table.add_column("transcriber")
        all_delta_keys = sorted(
            {
                key
                for row in result.degradation
                for key in row
                if key not in {"condition", "transcriber"}
            }
        )
        delta_keys = [key for key in all_delta_keys if key.endswith(".f1")]
        if not delta_keys:
            delta_keys = all_delta_keys
        for key in delta_keys:
            deg_table.add_column(key, justify="right")
        for row in result.degradation:
            cells = [str(row.get("condition", "")), str(row.get("transcriber", ""))]
            for key in delta_keys:
                value = row.get(key, float("nan"))
                if isinstance(value, (int, float)) and math.isnan(value):
                    cells.append("[dim]NaN[/dim]")
                elif isinstance(value, (int, float)):
                    cells.append(
                        f"[yellow]{value:.4f}[/]" if value < 0 else f"{value:.4f}"
                    )
                else:
                    cells.append(str(value))
            deg_table.add_row(*cells)
        console.print(deg_table)

    if succeeded < total:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port"),
    reload: bool = typer.Option(
        False, "--reload", help="Auto-reload on file changes"
    ),
) -> None:
    """Start the FastAPI management server."""
    import uvicorn
    from sonitra.api.app import create_app

    console = get_console()
    console.print(
        f"[bold cyan]sonitra[/] API server on [dim]http://{host}:{port}[/dim]"
    )
    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def init(
    path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Output path for default config"
    ),
) -> None:
    """Write a starter config.yaml to the given path."""
    from sonitra.config import (
        DawDreamerSection,
        EffectsChain,
        FluidSynthSection,
        IOSection,
        NormalisationSection,
        ObservabilitySection,
        PipelineConfig,
        PipelineSection,
        QualityGatesSection,
        SynthBackend,
        TranscriptionSection,
    )
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    cfg = PipelineConfig(
        render_pipeline=PipelineSection(
            synth_backend=SynthBackend.DAWDREAMER_FAUST,
            effects_chain=EffectsChain.NONE,
            bpm=120,
            sample_rate=44100,
            bit_depth=24,
            channels=2,
            duration_padding_sec=2.0,
            overwrite=False,
            resume=False,
            max_workers=1,
            log_level="INFO",
        ),
        fluidsynth=FluidSynthSection(soundfont_path=None),
        io=IOSection(
            corpus_root="corpus",
            output_format="wav",
            mp3_bitrate_kbps=192,
            file_naming="{stem}",
        ),
        dawdreamer=DawDreamerSection(),
        normalisation=NormalisationSection(
            enabled=True,
            mode="peak",
            target_db=-1.0,
            pre_effects=False,
        ),
        quality_gates=QualityGatesSection(
            silence_threshold_rms=0.001,
            min_duration_sec=0.1,
            max_duration_deviation_sec=1.0,
            clip_threshold=1.0,
        ),
        observability=ObservabilitySection(
            write_manifest=True,
            manifest_path="renders.jsonl",
            write_failed_list=True,
            emit_sse_events=False,
            progress=True,
            log_level=None,
        ),
        transcription=TranscriptionSection(
            transcribers=[
                BasicPitchTranscriberConfig(
                    enabled=True,
                    name="basic_pitch",
                    onset_threshold=0.5,
                    frame_threshold=0.3,
                )
            ]
        ),
    )
    cfg.save(path)
    console = get_console()
    console.print(f"Starter config written to [dim]{path}[/dim]")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit", is_eager=True
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all console output (manifests/results files are still written)",
    ),
) -> None:
    from importlib.metadata import version as pkg_version

    if version:
        console = get_console()
        console.print(f"sonitra v{pkg_version('sonitra')}")
        raise typer.Exit()

    global _CLI_VERBOSE
    _CLI_VERBOSE = verbose
    console = get_console(quiet=quiet)
    setup_logging("DEBUG" if verbose else "INFO", console=console)

    # TF C++ logs follow the log level (respects user overrides). Set before
    # any TensorFlow import; also inherited by benchmark pool workers.
    configure_framework_logging("DEBUG" if verbose else "INFO")
    configure_onednn_opts()


if __name__ == "__main__":
    app()
