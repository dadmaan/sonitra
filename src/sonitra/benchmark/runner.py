from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions
from sonitra.benchmark.results import BenchmarkRecord, ResultsWriter, degradation, summarise
from sonitra.config import PipelineConfig
from sonitra.evaluation.protocol import (
    AudioMetric,
    evaluate_notes,
    make_audio_metrics,
    make_symbolic_metrics,
)
from sonitra.evaluation.types import NoteEvent, notes_from_dicts
from sonitra.midi_reader import parse_midi
from sonitra.midi_writer import write_midi
from sonitra.pipeline import run_pipeline
from sonitra.separation.protocol import make_separator
from sonitra.storage import read_audio
from sonitra.synth.protocol import make_synth
from sonitra.transcribe.protocol import TranscriberProtocol, make_transcriber

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    records: list[BenchmarkRecord]
    summary: list[dict[str, Any]]
    degradation: list[dict[str, Any]]
    results_path: Path
    summary_path: Path
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary,
            "degradation": self.degradation,
            "results_path": str(self.results_path),
            "summary_path": str(self.summary_path),
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_benchmark(
    midi_paths: Iterable[Path | str],
    work_dir: Path | str,
    config: PipelineConfig,
    corpus_root: Path | None = None,
) -> BenchmarkResult:
    """Run the full benchmark: render -> (separate) -> transcribe -> evaluate.

    For every condition (baseline, explicit conditions, sweep values) the
    corpus is rendered with the overridden config, each enabled transcriber is
    run on the audio, and the transcription is scored against the source MIDI
    with the configured metric suite. Per-file records go to a JSONL file; the
    aggregate summary and the degradation-vs-baseline table go to summary.json.
    """
    start = time.perf_counter()
    midi_paths = [Path(path) for path in midi_paths]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    transcriber_configs = [t for t in config.transcription.transcribers if t.enabled]
    if not transcriber_configs:
        raise ValueError("No enabled transcribers configured under 'transcription.transcribers'")
    transcribers = [make_transcriber(cfg) for cfg in transcriber_configs]

    conditions = expand_conditions(config.benchmark)
    symbolic_metrics = make_symbolic_metrics(config.evaluation)
    audio_metrics = make_audio_metrics(config.evaluation)

    writer = ResultsWriter(work_dir / config.benchmark.results_path)
    references: dict[Path, list[NoteEvent] | None] = {}
    for path in midi_paths:
        try:
            references[path] = notes_from_dicts(parse_midi(path))
        except Exception as exc:  # noqa: BLE001 - recorded as a per-file failure
            logger.warning("Cannot parse reference MIDI %s: %s", path, exc)
            references[path] = None

    records: list[BenchmarkRecord] = []
    for condition in conditions:
        logger.info("Benchmark condition '%s' (%d overrides)", condition.name, len(condition.overrides))
        condition_config = apply_overrides(config, condition.overrides)
        records.extend(
            _run_condition(
                condition,
                condition_config,
                midi_paths,
                references,
                transcribers,
                symbolic_metrics,
                audio_metrics,
                work_dir,
                writer,
                corpus_root=corpus_root,
            )
        )

    summary = summarise(records)
    degradation_rows = degradation(summary, baseline=config.benchmark.baseline_name)
    summary_path = work_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "degradation": degradation_rows}, indent=2)
    )

    return BenchmarkResult(
        records=records,
        summary=summary,
        degradation=degradation_rows,
        results_path=writer.path,
        summary_path=summary_path,
        elapsed_seconds=time.perf_counter() - start,
    )


def _run_condition(
    condition: Condition,
    condition_config: PipelineConfig,
    midi_paths: Sequence[Path],
    references: dict[Path, list[NoteEvent] | None],
    transcribers: Sequence[TranscriberProtocol],
    symbolic_metrics: Sequence,
    audio_metrics: Sequence[AudioMetric],
    work_dir: Path,
    writer: ResultsWriter,
    corpus_root: Path | None = None,
) -> list[BenchmarkRecord]:
    audio_dir = work_dir / "audio" / condition.slug
    render_result = run_pipeline(midi_paths, audio_dir, config=condition_config, corpus_root=corpus_root)
    render_log = {entry["midi"]: entry for entry in render_result.log}
    separator = (
        make_separator(condition_config.separation)
        if condition_config.separation.enabled
        else None
    )

    records: list[BenchmarkRecord] = []
    for midi_path in midi_paths:
        entry = render_log.get(str(midi_path), {})
        audio_path = Path(entry["output"]) if "output" in entry else None
        if (
            audio_path is None
            or entry.get("status") == "failed"
            or not audio_path.exists()
            or references[midi_path] is None
        ):
            reason = entry.get("error", "render failed or produced no output")
            for transcriber in transcribers:
                record = BenchmarkRecord(
                    condition=condition.name,
                    transcriber=transcriber.name,
                    midi_path=str(midi_path),
                    audio_path=str(audio_path) if audio_path else "",
                    status="render_failed",
                    overrides=condition.overrides,
                    error=str(reason),
                )
                writer.write(record)
                records.append(record)
            continue

        transcribe_input = audio_path
        if separator is not None:
            stems = separator.separate(
                audio_path, work_dir / "stems" / condition.slug
            )
            stem_name = condition_config.separation.stem
            selected = stems.get(stem_name) if stem_name else next(iter(stems.values()), None)
            if selected is None:
                raise ValueError(
                    f"Separator produced no stem named '{stem_name}'; available: {sorted(stems)}"
                )
            transcribe_input = selected

        for transcriber in transcribers:
            records.append(
                _evaluate_one(
                    condition,
                    condition_config,
                    midi_path,
                    audio_path,
                    transcribe_input,
                    references[midi_path],
                    transcriber,
                    symbolic_metrics,
                    audio_metrics,
                    work_dir,
                    writer,
                    corpus_root=corpus_root,
                )
            )
    return records


def _evaluate_one(
    condition: Condition,
    condition_config: PipelineConfig,
    midi_path: Path,
    audio_path: Path,
    transcribe_input: Path,
    reference: list[NoteEvent],
    transcriber: TranscriberProtocol,
    symbolic_metrics: Sequence,
    audio_metrics: Sequence[AudioMetric],
    work_dir: Path,
    writer: ResultsWriter,
    corpus_root: Path | None = None,
) -> BenchmarkRecord:
    try:
        result = transcriber.transcribe(transcribe_input)
        estimate = notes_from_dicts(result.notes)
        rel = midi_path.relative_to(corpus_root) if corpus_root is not None else Path(midi_path.stem)
        transcription_path = (
            work_dir
            / "transcriptions"
            / condition.slug
            / transcriber.name
            / rel.with_suffix(".mid")
        )
        write_midi(result.notes, transcription_path)

        metrics = evaluate_notes(reference, estimate, symbolic_metrics)
        if audio_metrics:
            metrics.update(
                _audio_metric_values(
                    audio_path, result.notes, condition_config, audio_metrics
                )
            )
        record = BenchmarkRecord(
            condition=condition.name,
            transcriber=transcriber.name,
            midi_path=str(midi_path),
            audio_path=str(audio_path),
            status="succeeded",
            metrics=metrics,
            overrides=condition.overrides,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark logs and continues
        logger.exception("Transcription failed: %s on %s", transcriber.name, audio_path)
        record = BenchmarkRecord(
            condition=condition.name,
            transcriber=transcriber.name,
            midi_path=str(midi_path),
            audio_path=str(audio_path),
            status="failed",
            overrides=condition.overrides,
            error=str(exc),
        )
    writer.write(record)
    return record


def _audio_metric_values(
    audio_path: Path,
    estimate_notes: list[dict],
    condition_config: PipelineConfig,
    audio_metrics: Sequence[AudioMetric],
) -> dict[str, float]:
    """Compare rendered audio against a re-synthesis of the transcription.

    Following Bradshaw et al. (2024): the transcription is rendered back to
    audio with the same synthesiser configuration, and audio metrics (DTW)
    score the divergence from the audio the transcriber actually heard.
    """
    reference_audio, sample_rate = read_audio(audio_path)
    synth = make_synth(condition_config)
    duration = max(
        (float(n["start_sec"]) + float(n["duration_sec"]) for n in estimate_notes),
        default=0.0,
    ) + condition_config.pipeline.duration_padding_sec
    estimate_audio = np.asarray(synth.render(estimate_notes, duration_sec=duration))

    values: dict[str, float] = {}
    for metric in audio_metrics:
        for key, value in metric.compute(reference_audio, estimate_audio, sample_rate).items():
            values[f"{metric.name}.{key}"] = value
    return values
