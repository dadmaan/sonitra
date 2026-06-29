from __future__ import annotations

from pathlib import Path
from typing import Optional

import random
import typer

app = typer.Typer(name="sonitra")

_MIDI_SUFFIXES: frozenset[str] = frozenset({".mid", ".midi"})


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

    cfg = load_config(config)
    _apply_dataset(cfg, dataset, config)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    actual_corpus = corpus if corpus is not None else paths.midi
    actual_output = output if output is not None else paths.audio

    midi_paths = _discover_midi_files(actual_corpus)
    if not midi_paths:
        typer.echo(f"No MIDI files found in {actual_corpus}")
        raise typer.Exit(code=1)
    midi_paths = _apply_subset(midi_paths, limit, seed)

    result = run_pipeline(midi_paths, out_dir=actual_output, config=cfg, corpus_root=actual_corpus)
    typer.echo(
        f"Done: {result.succeeded} succeeded, {result.failed} failed, "
        f"{result.skipped} skipped ({result.elapsed_seconds:.2f}s)"
    )
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
) -> None:
    """Transcribe audio files to MIDI with the configured transcribers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sonitra.config import load_config, resolve_corpus_paths
    from sonitra.midi_writer import write_midi
    from sonitra.transcribe.protocol import make_transcriber

    cfg = load_config(config)
    _apply_dataset(cfg, dataset, config)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    if audio is not None:
        actual_audio = audio
    elif dataset is not None:
        actual_audio = paths.audio
    else:
        typer.echo("--audio or --dataset must be provided")
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
        typer.echo("No matching enabled transcribers in config")
        raise typer.Exit(code=1)

    audio_paths = sorted(
        path for ext in ("*.wav", "*.flac", "*.mp3") for path in actual_audio.rglob(ext)
    )
    if not audio_paths:
        typer.echo(f"No audio files found in {actual_audio}")
        raise typer.Exit(code=1)

    failures = 0
    n_workers = cfg.transcription.max_workers

    def _transcribe_one(backend_name: str, backend_transcribe, audio_path: Path) -> tuple[str, str | None]:
        rel = audio_path.relative_to(actual_audio)
        midi_path = actual_output / backend_name / rel.with_suffix(".mid")
        try:
            result = backend_transcribe(audio_path)
            write_midi(result.notes, midi_path)
            return f"{backend_name}: {audio_path.name} -> {midi_path}", None
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            return f"{backend_name}: {audio_path.name} FAILED ({exc})", str(exc)

    for transcriber_cfg in transcriber_configs:
        backend = make_transcriber(transcriber_cfg)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_path = {
                    executor.submit(_transcribe_one, backend.name, backend.transcribe, ap): ap
                    for ap in audio_paths
                }
                for future in as_completed(future_to_path):
                    msg, err = future.result()
                    typer.echo(msg)
                    if err is not None:
                        failures += 1
        else:
            for audio_path in audio_paths:
                msg, err = _transcribe_one(backend.name, backend.transcribe, audio_path)
                typer.echo(msg)
                if err is not None:
                    failures += 1
    if failures:
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
) -> None:
    """Score estimated MIDI against reference MIDI, paired by file stem."""
    import json
    import math
    from concurrent.futures import ThreadPoolExecutor

    from sonitra.config import EvaluationSection, load_config, resolve_corpus_paths
    from sonitra.evaluation.protocol import evaluate_notes, make_symbolic_metrics
    from sonitra.evaluation.types import notes_from_dicts
    from sonitra.midi_reader import parse_midi

    # Derive metric settings and dataset-scoped paths in one config load.
    if dataset is not None and config is not None:
        full_cfg = load_config(config)
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
    elif dataset is not None:
        # No config provided; use defaults for metric settings and path bases.
        section = EvaluationSection()
        midi_dir = Path("corpus") / dataset / "midi"
        first_transcriber = "basic_pitch"
        transcription_dir = Path("corpus") / dataset / "transcription"
        config_stem = None
    else:
        section = load_config(config).evaluation if config else EvaluationSection()
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
        typer.echo("--reference is required when --dataset is not set")
        raise typer.Exit(code=1)

    # Resolve estimate path.
    actual_estimate: Path
    if estimate is not None:
        actual_estimate = estimate
    elif dataset is not None:
        actual_estimate = transcription_dir / first_transcriber  # type: ignore[operator]
    else:
        typer.echo("--estimate is required when --dataset is not set")
        raise typer.Exit(code=1)

    reference_paths = _discover_midi_files(actual_reference)
    if not reference_paths:
        typer.echo(f"No MIDI files found in {actual_reference}")
        raise typer.Exit(code=1)

    # Pairs by stem; assumes globally unique filenames across the reference corpus.
    # See .local/notes/TODO.md for the known limitation with nested datasets.
    def _eval_one(ref_path: Path) -> dict | None:
        rel = ref_path.relative_to(actual_reference)
        est_path = next(
            (
                candidate
                for ext in (".mid", ".midi")
                if (candidate := actual_estimate / rel.with_suffix(ext)).exists()
            ),
            None,
        )
        if est_path is None:
            typer.echo(f"{rel}: no estimate found, skipping")
            return None
        values = evaluate_notes(
            notes_from_dicts(parse_midi(ref_path)),
            notes_from_dicts(parse_midi(est_path)),
            metrics,
        )
        return {"file": str(rel), **values}

    n_eval_workers = section.max_workers
    if n_eval_workers > 1:
        with ThreadPoolExecutor(max_workers=n_eval_workers) as executor:
            rows = [r for r in executor.map(_eval_one, reference_paths) if r is not None]
    else:
        rows = [r for ref_path in reference_paths if (r := _eval_one(ref_path)) is not None]

    if not rows:
        typer.echo("No reference/estimate pairs evaluated")
        raise typer.Exit(code=1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    metric_names = sorted({key for row in rows for key in row if key != "file"})
    typer.echo(f"Evaluated {len(rows)} pairs (mean over files):")
    for name in metric_names:
        values = [row[name] for row in rows if name in row and not math.isnan(row[name])]
        mean = sum(values) / len(values) if values else float("nan")
        typer.echo(f"  {name}: {mean:.4f}")


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
) -> None:
    """Run the full AMT benchmark: render, transcribe, and evaluate per condition."""
    from sonitra.benchmark.runner import run_benchmark
    from sonitra.config import load_config, resolve_corpus_paths

    cfg = load_config(config)
    _apply_dataset(cfg, dataset, config)
    paths = resolve_corpus_paths(cfg, config_name=config.stem)

    actual_corpus = corpus if corpus is not None else paths.midi
    if workdir is not None:
        actual_workdir = workdir
    elif dataset is not None:
        actual_workdir = Path(cfg.io.corpus_root) / dataset / "benchmark"
    else:
        actual_workdir = Path("benchmark")

    midi_paths = _discover_midi_files(actual_corpus)
    if not midi_paths:
        typer.echo(f"No MIDI files found in {actual_corpus}")
        raise typer.Exit(code=1)

    result = run_benchmark(midi_paths, actual_workdir, cfg, corpus_root=actual_corpus)
    succeeded = sum(1 for record in result.records if record.status == "succeeded")
    typer.echo(
        f"Benchmark finished: {succeeded}/{len(result.records)} evaluations succeeded "
        f"({result.elapsed_seconds:.1f}s)"
    )
    typer.echo(f"Results: {result.results_path}")
    typer.echo(f"Summary: {result.summary_path}")
    if succeeded < len(result.records):
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
    typer.echo(f"Starter config written to {path}")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit", is_eager=True
    ),
) -> None:
    from importlib.metadata import version as pkg_version

    if version:
        typer.echo(f"sonitra v{pkg_version('sonitra')}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
