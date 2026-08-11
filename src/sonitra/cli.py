from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import typer
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
    effective_log_level,
    get_console,
    set_log_level,
    setup_logging,
)

app = typer.Typer(name="sonitra")

_MIDI_SUFFIXES: frozenset[str] = frozenset({".mid", ".midi"})

_CLI_VERBOSE = False


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
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _MIDI_SUFFIXES
    )


def _apply_subset(
    files: list[Path], limit: int | None, seed: int | None
) -> list[Path]:
    if limit is None or limit >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, limit))


def _apply_dataset(cfg: PipelineConfig, dataset: str | None, config: Path) -> None:
    """Inject *dataset* into *cfg* when provided.

    Args:
        cfg: Loaded pipeline configuration to mutate in place.
        dataset: Dataset name supplied via the CLI ``--dataset`` flag, or
            ``None`` when the flag was not set.
        config: Path to the YAML config file (kept for symmetry; unused here
            but documents the call-site contract).
    """
    if dataset is not None:
        cfg.io.dataset = dataset


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
            "Dataset name; scopes corpus paths to corpus/midi/{dataset}/ and "
            "outputs to corpus/{subdir}/{dataset}/{config}/"
        ),
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum MIDI files to render (random subset)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for --limit sampling."
    ),
) -> None:
    """Run the MIDI-to-audio rendering pipeline."""
    from sonitra.config import load_config, resolve_corpus_paths
    from sonitra.pipeline import run_pipeline

    console = get_console()
    cfg = load_config(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
    _apply_dataset(cfg, dataset, config)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    actual_corpus = corpus if corpus is not None else paths.midi
    actual_output = output if output is not None else paths.audio

    midi_paths = _discover_midi_files(actual_corpus)
    if not midi_paths:
        console.print(f"[red]No MIDI files found in[/red] [dim]{actual_corpus}[/dim]")
        raise typer.Exit(code=1)
    midi_paths = _apply_subset(midi_paths, limit, seed)

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
            "Dataset name; scopes corpus paths to corpus/midi/{dataset}/ and "
            "outputs to corpus/{subdir}/{dataset}/{config}/"
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

    from sonitra.config import load_config, resolve_corpus_paths
    from sonitra.midi_writer import write_transcription_outputs
    from sonitra.transcribe.protocol import make_transcriber

    console = get_console()
    cfg = load_config(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
    _apply_dataset(cfg, dataset, config)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    if audio is not None:
        actual_audio = audio
    elif dataset is not None:
        actual_audio = paths.audio
    else:
        console.print("[red]--audio or --dataset must be provided[/red]")
        raise typer.Exit(code=1)

    if output is not None:
        actual_output = output
    elif dataset is not None:
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
    audio_paths = _apply_subset(audio_paths, limit, seed)

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
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Pipeline config YAML (for metric settings)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write per-file results to this JSONL path"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Dataset name; scopes corpus paths to corpus/midi/{dataset}/ and "
            "outputs to corpus/{subdir}/{dataset}/{config}/"
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

    from sonitra.config import EvaluationSection, load_config, resolve_corpus_paths
    from sonitra.evaluation.protocol import evaluate_notes, make_symbolic_metrics
    from sonitra.evaluation.types import notes_from_dicts
    from sonitra.midi_reader import parse_midi

    console = get_console()

    eval_cfg = None
    # Derive metric settings and dataset-scoped paths in one config load.
    if dataset is not None and config is not None:
        full_cfg = load_config(config)
        if not _CLI_VERBOSE:
            set_log_level(effective_log_level(full_cfg))
        full_cfg.io.dataset = dataset
        section = full_cfg.evaluation
        eval_paths = resolve_corpus_paths(full_cfg, config_name=config.stem)
        midi_dir: Path | None = eval_paths.midi
        first_transcriber = (
            full_cfg.transcription.transcribers[0].name
            if full_cfg.transcription.transcribers
            else "basic_pitch"
        )
        transcription_dir: Path | None = eval_paths.transcription
        config_stem: str | None = config.stem
        eval_cfg = full_cfg
    elif dataset is not None:
        # No config provided; use defaults for metric settings and path bases.
        section = EvaluationSection()
        midi_dir = Path("corpus") / dataset / "midi"
        first_transcriber = "basic_pitch"
        transcription_dir = Path("corpus") / dataset / "transcription"
        config_stem = None
    else:
        if config is not None:
            loaded_cfg = load_config(config)
            if not _CLI_VERBOSE:
                set_log_level(effective_log_level(loaded_cfg))
            section = loaded_cfg.evaluation
            eval_cfg = loaded_cfg
        else:
            section = EvaluationSection()
        midi_dir = None
        first_transcriber = None
        transcription_dir = None
        config_stem = None

    metrics = make_symbolic_metrics(section)

    # Resolve reference path.
    actual_reference: Path
    if reference is not None:
        actual_reference = reference
    elif dataset is not None:
        actual_reference = midi_dir  # type: ignore[assignment]
    else:
        console.print("[red]--reference is required when --dataset is not set[/red]")
        raise typer.Exit(code=1)

    # Resolve estimate path.
    actual_estimate: Path
    if estimate is not None:
        actual_estimate = estimate
    elif dataset is not None:
        actual_estimate = transcription_dir / first_transcriber  # type: ignore[operator]
    else:
        console.print("[red]--estimate is required when --dataset is not set[/red]")
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

    if limit is not None:
        reference_paths = [p for p in reference_paths if _find_estimate(p) is not None]
        if not reference_paths:
            console.print("[red]No reference files have matching estimates[/red]")
            raise typer.Exit(code=1)
    reference_paths = _apply_subset(reference_paths, limit, seed)

    # Pairs by stem; assumes globally unique filenames across the reference corpus.
    # See .local/notes/TODO.md for the known limitation with nested datasets.
    def _eval_one(ref_path: Path) -> dict | None:
        est_path = _find_estimate(ref_path)
        if est_path is None:
            return None
        rel = ref_path.relative_to(actual_reference)
        values = evaluate_notes(
            notes_from_dicts(parse_midi(ref_path)),
            notes_from_dicts(parse_midi(est_path)),
            metrics,
        )
        return {"file": str(rel), **values}

    show_progress = console.is_terminal and not console.quiet
    if eval_cfg is not None:
        show_progress = show_progress and eval_cfg.observability.progress

    rows: list[dict] = []
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

    if not rows:
        console.print("[red]No reference/estimate pairs evaluated[/red]")
        raise typer.Exit(code=1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

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
        None, "--workdir", "-w", help="Working directory for audio/transcriptions/results"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Dataset name; scopes corpus paths to corpus/midi/{dataset}/ and "
            "outputs to corpus/{subdir}/{dataset}/{config}/"
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
    from sonitra.config import load_config, resolve_corpus_paths

    console = get_console()
    cfg = load_config(config)
    if not _CLI_VERBOSE:
        set_log_level(effective_log_level(cfg))
    _apply_dataset(cfg, dataset, config)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    actual_corpus = corpus if corpus is not None else paths.midi
    if workdir is not None:
        actual_workdir = workdir
    elif dataset is not None:
        actual_workdir = Path(cfg.io.corpus_root) / dataset / "benchmark" / config.stem
    else:
        actual_workdir = Path("benchmark") / config.stem

    midi_paths = _discover_midi_files(actual_corpus)
    if not midi_paths:
        console.print(f"[red]No MIDI files found in[/red] [dim]{actual_corpus}[/dim]")
        raise typer.Exit(code=1)
    midi_paths = _apply_subset(midi_paths, limit, seed)

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
                corpus_root=actual_corpus,
                progress=prog,
            )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted — partial results kept in manifests[/yellow]")
        raise typer.Exit(130)

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
        pipeline=PipelineSection(
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

    # Suppress TF/absl C++ logs (idempotent, respects user overrides). Set
    # before any TensorFlow import; also inherited by benchmark pool workers.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


if __name__ == "__main__":
    app()
