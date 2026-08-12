from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import numpy as np

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions
from sonitra.benchmark.results import (
    BenchmarkRecord,
    ResultsWriter,
    compute_fingerprint,
    degradation,
    load_records,
    order_by_condition,
    summarise,
)
from sonitra.config import PipelineConfig
from sonitra.evaluation.protocol import (
    AudioMetric,
    evaluate_notes,
    make_audio_metrics,
    make_symbolic_metrics,
)
from sonitra.evaluation.types import NoteEvent, notes_from_dicts
from sonitra.midi_reader import parse_midi
from sonitra.midi_writer import write_transcription_outputs
from sonitra.pipeline import run_pipeline
from sonitra.separation.protocol import make_separator
from sonitra.storage import read_audio
from sonitra.synth.protocol import make_synth
from sonitra.transcribe.protocol import TranscriberProtocol, make_transcriber

if TYPE_CHECKING:
    from sonitra.terminal import BenchmarkProgress

logger = logging.getLogger(__name__)


# Shared queue through which worker subprocesses stream per-record events to
# the parent. Set by the ProcessPoolExecutor initializer; None in the parent
# (and in serial mode, where no subprocesses exist).
_WORKER_EVENTS: multiprocessing.Queue | None = None


def _drain_events(event_queue: multiprocessing.Queue, progress: BenchmarkProgress) -> None:
    """Daemon thread body: forward worker events from the queue to the display."""
    while True:
        event = event_queue.get()
        if event is None:
            return
        progress.on_worker_event(event)


def _worker_event_init(event_queue: multiprocessing.Queue | None, log_dir: Path) -> None:
    """ProcessPoolExecutor worker initializer: contain output, neutralise logging.

    Redirects worker fd 1/2 to a per-worker log file under *log_dir* so neither
    TF C++ stderr output nor Python-level print/logging can corrupt the shared
    terminal (the rich Live display), while still surviving for debugging.
    """
    global _WORKER_EVENTS
    _WORKER_EVENTS = event_queue
    # Suppress TF/absl C++ logs (belt-and-braces; env is inherited from parent too)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    # Tee fd 1 and fd 2 to a per-worker log file (survives for debugging)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker-{os.getpid()}.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 2)
    os.dup2(fd, 1)
    # Neutralize Python-level output so rich/FileProxy/print never touch the display
    sys.stdout = sys.__stdout__   # wrapper over fd 1 -> now the log file
    sys.stderr = sys.__stderr__   # wrapper over fd 2 -> now the log file
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(logging.NullHandler())


@contextmanager
def _contain_serial_output(log_path: Path) -> Iterator[None]:
    """Redirect stdout/stderr to *log_path* for the duration of the block.

    Serial mode (``benchmark.max_workers == 1``) runs in the same process as
    an active Rich Live display. Third-party backends that print directly
    (e.g. basic-pitch's bare ``print()`` in ``predict()``) write straight to
    the real terminal and corrupt the display's cursor tracking. Pool workers
    avoid this via fd-level redirection in a separate process (see
    ``_worker_event_init``); that trick isn't available here since redirecting
    fd 1/2 in-process would also blind the display's own output. Instead this
    only reassigns the ``sys.stdout``/``sys.stderr`` *names* that ``print()``
    looks up — the display's console keeps writing to the real terminal
    because :func:`sonitra.terminal.get_console` pins its file to the ``sys.
    stdout`` object present at console-construction time, before any
    redirection happens.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        yield


def _emit_worker_event(progress: BenchmarkProgress | None, event: WorkerEvent) -> None:
    """Route a worker event to the shared queue (parallel) or the display directly.

    In parallel mode ``_WORKER_EVENTS`` is set by the worker initializer, so
    events stream to the parent's drainer thread. In serial mode there are no
    subprocesses and ``_WORKER_EVENTS`` is None, so events go straight to the
    display.
    """
    if _WORKER_EVENTS is not None:
        _WORKER_EVENTS.put(event)
    elif progress is not None:
        progress.on_worker_event(event)


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
    *,
    progress: BenchmarkProgress | None = None,
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
    transcriber_names = [t.name for t in transcribers]

    conditions = expand_conditions(config.benchmark)
    condition_order = [condition.name for condition in conditions]
    symbolic_metrics = make_symbolic_metrics(config.evaluation)
    audio_metrics = make_audio_metrics(config.evaluation)
    n_workers = config.benchmark.max_workers

    results_file = work_dir / config.benchmark.results_path
    fingerprint_file = results_file.with_name(results_file.name + ".fingerprint")
    current_fingerprint = compute_fingerprint(config)

    records: list[BenchmarkRecord] = []
    if config.benchmark.resume and results_file.exists():
        if fingerprint_file.exists():
            stored_fingerprint = fingerprint_file.read_text().strip()
            if stored_fingerprint != current_fingerprint:
                raise ValueError(
                    "Cannot resume: the config no longer matches the one that produced "
                    f"'{results_file}' (fingerprint mismatch). Either revert the config "
                    "change or start a new work_dir."
                )
        else:
            logger.warning(
                "Resuming from '%s' with no stored fingerprint to verify against; "
                "proceeding without a config-consistency check.",
                results_file,
            )
        records = load_records(results_file)
    elif results_file.exists():
        logger.info(
            "benchmark.resume is false; discarding existing results at '%s'", results_file
        )
        results_file.unlink()
        fingerprint_file.unlink(missing_ok=True)

    fingerprint_file.write_text(current_fingerprint)

    completed_by_condition: dict[str, set[tuple[str, str]]] = {}
    for record in records:
        if record.status in {"succeeded", "failed", "render_failed"}:
            completed_by_condition.setdefault(record.condition, set()).add(
                (record.midi_path, record.transcriber)
            )

    expected_pairs = {(str(mp), name) for mp in midi_paths for name in transcriber_names}
    pending_conditions = []
    for condition in conditions:
        done = completed_by_condition.get(condition.name, set())
        if expected_pairs and done >= expected_pairs:
            logger.info("Skipping already-completed condition '%s' (resume)", condition.name)
            continue
        pending_conditions.append(condition)
    conditions = pending_conditions

    writer = ResultsWriter(results_file)
    references: dict[Path, list[NoteEvent] | None] = {}
    for path in midi_paths:
        try:
            references[path] = notes_from_dicts(parse_midi(path))
        except Exception as exc:  # noqa: BLE001 - recorded as a per-file failure
            logger.warning("Cannot parse reference MIDI %s: %s", path, exc)
            references[path] = None

    if n_workers > 1:
        event_queue: multiprocessing.Queue | None = None
        drainer: Thread | None = None
        if progress is not None:
            event_queue = multiprocessing.Queue()
            drainer = Thread(
                target=_drain_events, args=(event_queue, progress), daemon=True
            )
            drainer.start()
        log_dir = work_dir / "logs"
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_event_init,
            initargs=(event_queue, log_dir),
        ) as executor:
            futures: dict[Future, Condition] = {}
            for condition in conditions:
                if progress is not None:
                    progress.on_condition_started(
                        condition.name,
                        condition.overrides,
                        len(midi_paths),
                        transcriber_names,
                    )
                futures[
                    executor.submit(
                        _condition_worker,
                        condition,
                        apply_overrides(config, condition.overrides),
                        midi_paths,
                        references,
                        transcriber_configs,
                        work_dir,
                        corpus_root,
                        completed_by_condition.get(condition.name, set()),
                    )
                ] = condition
            for future in as_completed(futures):
                condition = futures[future]
                condition_records = future.result()
                for record in condition_records:
                    writer.write(record)
                    records.append(record)
                if progress is not None:
                    progress.on_condition_done(condition.name)
        if event_queue is not None and drainer is not None:
            event_queue.put(None)  # sentinel: drainer thread may exit
            drainer.join()
    else:
        for condition in conditions:
            logger.info("Benchmark condition '%s' (%d overrides)", condition.name, len(condition.overrides))
            if progress is not None:
                progress.on_condition_started(
                    condition.name,
                    condition.overrides,
                    len(midi_paths),
                    transcriber_names,
                )
            condition_config = apply_overrides(config, condition.overrides)
            output_guard = (
                _contain_serial_output(work_dir / "logs" / "serial.log")
                if progress is not None
                else nullcontext()
            )
            with output_guard:
                condition_records = _run_condition(
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
                    completed=completed_by_condition.get(condition.name, set()),
                    progress=progress,
                )
            records.extend(condition_records)
            if progress is not None:
                progress.on_condition_done(condition.name)

    summary = order_by_condition(summarise(records), condition_order)
    degradation_rows = order_by_condition(
        degradation(summary, baseline=config.benchmark.baseline_name), condition_order
    )
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


def _condition_worker(
    condition: Condition,
    condition_config: PipelineConfig,
    midi_paths: list[Path],
    references: dict[Path, list[NoteEvent] | None],
    transcriber_cfgs: list,
    work_dir: Path,
    corpus_root: Path | None,
    completed: set[tuple[str, str]] = frozenset(),
) -> list[BenchmarkRecord]:
    """Run one benchmark condition in a subprocess (for ProcessPoolExecutor).

    Recreates transcribers and metrics from configs so no non-picklable state
    crosses the process boundary.  Returns records without writing to disk so
    the parent process can stream-write them to the shared ResultsWriter.
    """
    transcribers = [make_transcriber(cfg) for cfg in transcriber_cfgs]
    symbolic_metrics = make_symbolic_metrics(condition_config.evaluation)
    audio_metrics = make_audio_metrics(condition_config.evaluation)
    return _run_condition(
        condition,
        condition_config,
        midi_paths,
        references,
        transcribers,
        symbolic_metrics,
        audio_metrics,
        work_dir,
        writer=None,
        corpus_root=corpus_root,
        completed=completed,
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
    writer: ResultsWriter | None = None,
    corpus_root: Path | None = None,
    completed: set[tuple[str, str]] = frozenset(),
    progress: BenchmarkProgress | None = None,
) -> list[BenchmarkRecord]:
    # Imported lazily: WorkerEvent is added to sonitra.benchmark.results by the
    # display lane; this keeps the module importable regardless of edit order.
    from sonitra.benchmark.results import WorkerEvent

    audio_dir = work_dir / "audio" / condition.slug
    _emit_worker_event(
        progress,
        WorkerEvent(
            worker_id=os.getpid(),
            condition=condition.name,
            transcriber="",
            midi_path="",
            status="stage",
            stage="render",
            ok=True,
        ),
    )
    render_result = run_pipeline(midi_paths, audio_dir, config=condition_config, corpus_root=corpus_root)
    render_log = {entry["midi"]: entry for entry in render_result.log}
    separator = (
        make_separator(condition_config.separation)
        if condition_config.separation.enabled
        else None
    )

    records: list[BenchmarkRecord] = []
    for midi_path in midi_paths:
        pending_transcribers = [
            t for t in transcribers if (str(midi_path), t.name) not in completed
        ]
        if not pending_transcribers:
            continue

        entry = render_log.get(str(midi_path), {})
        audio_path = Path(entry["output"]) if "output" in entry else None
        if (
            audio_path is None
            or entry.get("status") == "failed"
            or not audio_path.exists()
            or references[midi_path] is None
        ):
            reason = entry.get("error", "render failed or produced no output")
            for transcriber in pending_transcribers:
                record = BenchmarkRecord(
                    condition=condition.name,
                    transcriber=transcriber.name,
                    midi_path=str(midi_path),
                    audio_path=str(audio_path) if audio_path else "",
                    status="render_failed",
                    overrides=condition.overrides,
                    error=str(reason),
                )
                if writer is not None:
                    writer.write(record)
                records.append(record)
                _emit_worker_event(
                    progress,
                    WorkerEvent(
                        worker_id=os.getpid(),
                        condition=record.condition,
                        transcriber=record.transcriber,
                        midi_path=record.midi_path,
                        status="done",
                        ok=False,
                    ),
                )
            continue

        transcribe_input = audio_path
        if separator is not None:
            _emit_worker_event(
                progress,
                WorkerEvent(
                    worker_id=os.getpid(),
                    condition=condition.name,
                    transcriber="",
                    midi_path=str(midi_path),
                    status="stage",
                    stage="separate",
                    ok=True,
                ),
            )
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

        for transcriber in pending_transcribers:
            _emit_worker_event(
                progress,
                WorkerEvent(
                    worker_id=os.getpid(),
                    condition=condition.name,
                    transcriber=transcriber.name,
                    midi_path=str(midi_path),
                    status="start",
                    stage="transcribe",
                    ok=True,
                ),
            )
            record = _evaluate_one(
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
            records.append(record)
            _emit_worker_event(
                progress,
                WorkerEvent(
                    worker_id=os.getpid(),
                    condition=record.condition,
                    transcriber=record.transcriber,
                    midi_path=record.midi_path,
                    status="done",
                    ok=record.status == "succeeded",
                ),
            )
    _cleanup_condition_audio(condition_config, work_dir, condition, separator is not None)
    return records


def _cleanup_condition_audio(
    condition_config: PipelineConfig,
    work_dir: Path,
    condition: Condition,
    had_separator: bool,
) -> None:
    """Remove a condition's rendered audio (and derived stems) once it's done.

    Only the audio/stems that fed this condition's transcription+evaluation
    pass are ever read again, and only during that pass (see _run_condition
    and _audio_metric_values) - safe to delete once it returns.
    """
    if condition_config.benchmark.save_audio:
        return
    shutil.rmtree(work_dir / "audio" / condition.slug, ignore_errors=True)
    if had_separator:
        shutil.rmtree(work_dir / "stems" / condition.slug, ignore_errors=True)


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
    writer: ResultsWriter | None = None,
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
        write_transcription_outputs(result, transcription_path)

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
    if writer is not None:
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
