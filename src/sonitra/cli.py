from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="sonitra")


@app.command()
def render(
    config: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to pipeline config YAML"
    ),
    corpus: Path = typer.Option(
        ..., "--corpus", "-i", help="Directory of MIDI files to render"
    ),
    output: Path = typer.Option(
        "output", "--output", "-o", help="Output audio directory"
    ),
    overwrite: bool = typer.Option(
        True, "--overwrite/--no-overwrite", help="Overwrite existing output files"
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers (overrides config)"
    ),
) -> None:
    """Run the MIDI-to-audio rendering pipeline."""
    from sonitra.config import load_config
    from sonitra.pipeline import run_pipeline

    cfg = load_config(config)

    midi_paths = sorted(corpus.glob("*.mid"))
    if not midi_paths:
        typer.echo(f"No .mid files found in {corpus}")
        raise typer.Exit(code=1)

    result = run_pipeline(midi_paths, out_dir=output, config=cfg)
    typer.echo(
        f"Done: {result.succeeded} succeeded, {result.failed} failed, "
        f"{result.skipped} skipped ({result.elapsed_seconds:.2f}s)"
    )
    if result.failed:
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
    """Write a default config.yaml to the given path."""
    from sonitra.config import PipelineConfig, PipelineSection, RenderingMode
    from sonitra.config import IOSection, ObservabilitySection

    cfg = PipelineConfig(
        pipeline=PipelineSection(
            rendering_mode=RenderingMode.PEDALBOARD_ONLY,
            sample_rate=44100,
            bit_depth=24,
            channels=2,
            duration_padding_sec=2.0,
            overwrite=False,
            resume=False,
            max_workers=4,
            log_level="INFO",
        ),
        io=IOSection(
            midi_dir="corpus/midi",
            output_dir="corpus/audio",
            output_format="wav",
            mp3_bitrate_kbps=320,
            file_naming="{stem}",
        ),
        observability=ObservabilitySection(
            write_manifest=True,
            manifest_path="renders.jsonl",
            write_failed_list=True,
            emit_sse_events=False,
        ),
    )
    cfg.save(path)
    typer.echo(f"Default config written to {path}")


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
