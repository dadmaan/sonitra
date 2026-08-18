from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from sonitra.config import PipelineConfig


@dataclass
class BenchmarkRecord:
    """One transcription evaluation: a (condition, transcriber, file) cell.

    Also records per-cell wall-clock timing (render/separate/transcribe/
    evaluate) in seconds; timing fields default to NaN for backward
    compatibility with pre-upgrade JSONL.
    """

    condition: str
    transcriber: str
    midi_path: str
    audio_path: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    source_path: str | None = None
    """The recording that produced this record, in audio mode. ``midi_path``
    always names the reference MIDI (evaluation key) in both modes; in MIDI
    mode ``source_path`` stays ``None`` (``source_path == midi_path`` would
    be redundant there). Optional/defaulted so ``load_records``'s
    ``BenchmarkRecord(**json.loads(line))`` still loads pre-upgrade JSONL."""
    render_seconds: float = float("nan")
    """Per-file render wall-clock (same value on every transcriber row of the file)."""
    separate_seconds: float = float("nan")
    """Per-file separation wall-clock; NaN when separation is disabled."""
    transcribe_seconds: float = float("nan")
    """Per-cell transcription wall-clock."""
    evaluate_seconds: float = float("nan")
    """Per-cell evaluation wall-clock."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerEvent:
    """One fine-grained benchmark progress event streamed from a worker.

    Lifecycle per condition: ``stage(render)`` -> ``[stage(separate)]`` ->
    ``start(transcribe)`` -> ``done``. ``status == "stage"`` marks a
    non-cell-boundary transition (render fires once per condition, separate
    fires once per file); ``status == "start"`` marks a worker beginning one
    (file, transcriber) cell; ``status == "done"`` marks a record produced for
    that cell, with ``ok`` reporting whether the evaluation succeeded.
    """

    worker_id: int  # os.getpid() of the worker process (or parent pid in serial mode)
    condition: str
    transcriber: str
    midi_path: str
    status: str  # "stage" (mid-lifecycle marker) | "start" | "done"
    ok: bool  # meaningful only when status == "done"
    stage: str = ""  # "render" | "separate" | "transcribe" | ""


class ResultsWriter:
    """Appends benchmark records to a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: BenchmarkRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")


_FINGERPRINT_EXCLUDE = {
    ("benchmark", "resume"),
    ("benchmark", "max_workers"),
    ("benchmark", "save_audio"),
    ("render_pipeline", "max_workers"),
    ("transcription", "max_workers"),
    ("evaluation", "max_workers"),
}


def compute_fingerprint(config: PipelineConfig) -> str:
    """Hash of the config, excluding fields that don't affect result semantics.

    Used by benchmark resume to detect a config that changed between runs
    into the same work_dir, which would otherwise silently mix results
    computed under two different meanings of "condition"/"record".
    """
    data = config.model_dump(mode="json")
    for section, key in _FINGERPRINT_EXCLUDE:
        data.get(section, {}).pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def load_records(path: Path | str) -> list[BenchmarkRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(BenchmarkRecord(**json.loads(line)))
    return records


def summarise(records: Iterable[BenchmarkRecord]) -> list[dict[str, Any]]:
    """Aggregate metric means per (condition, transcriber), NaN-aware.

    Summary rows are METRICS-ONLY: condition, transcriber, n_files,
    n_succeeded, and the metric means. Per-stage timing totals live
    exclusively in the ``"timing"`` block produced by :func:`timing_block`
    (aggregated there with different semantics: condition-level totals over
    ALL cells, deduped render/separate per file) -- keeping them out of
    summary rows avoids conflating process time with evaluation time.
    """
    groups: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        groups.setdefault((record.condition, record.transcriber), []).append(record)

    summary = []
    for (condition, transcriber), members in groups.items():
        succeeded = [record for record in members if record.status == "succeeded"]
        row: dict[str, Any] = {
            "condition": condition,
            "transcriber": transcriber,
            "n_files": len(members),
            "n_succeeded": len(succeeded),
        }
        metric_names = sorted({name for record in succeeded for name in record.metrics})
        for name in metric_names:
            values = [
                record.metrics[name]
                for record in succeeded
                if name in record.metrics and not math.isnan(record.metrics[name])
            ]
            row[name] = sum(values) / len(values) if values else float("nan")
        summary.append(row)
    return summary


def order_by_condition(
    rows: Sequence[dict[str, Any]], condition_order: Sequence[str]
) -> list[dict[str, Any]]:
    """Sort summary/degradation rows by declared condition order, then transcriber.

    Benchmark conditions may complete in a nondeterministic order (parallel
    mode dispatches conditions across a process pool and gathers results via
    ``as_completed``), which otherwise leaks into the row order of the
    aggregate tables. This restores the order conditions were declared in the
    config so repeated runs of the same config produce identically-ordered
    output. Conditions not present in *condition_order* sort after all known
    ones.
    """
    index = {name: i for i, name in enumerate(condition_order)}
    return sorted(
        rows,
        key=lambda row: (index.get(row["condition"], len(index)), row["transcriber"]),
    )


def degradation(
    summary: Sequence[dict[str, Any]], *, baseline: str = "baseline"
) -> list[dict[str, Any]]:
    """Per-condition metric deltas relative to the baseline condition.

    This is the degradation table of Edwards et al. (2024): how much each
    perturbation condition moves each metric for each transcriber.
    """
    baselines = {
        row["transcriber"]: row for row in summary if row["condition"] == baseline
    }
    rows = []
    for row in summary:
        if row["condition"] == baseline:
            continue
        reference = baselines.get(row["transcriber"])
        if reference is None:
            continue
        delta_row: dict[str, Any] = {
            "condition": row["condition"],
            "transcriber": row["transcriber"],
        }
        for key, value in row.items():
            if key in {"condition", "transcriber", "n_files", "n_succeeded"}:
                continue
            base_value = reference.get(key)
            if (
                isinstance(value, (int, float))
                and isinstance(base_value, (int, float))
                and not math.isnan(value)
                and not math.isnan(base_value)
            ):
                delta_row[f"delta_{key}"] = value - base_value
        rows.append(delta_row)
    return rows


def _nanaware_sum(values: Iterable[float]) -> float:
    """NaN-aware sum of *values*; NaN when no finite value is present."""
    total = 0.0
    seen = 0
    for value in values:
        if not math.isnan(value):
            total += value
            seen += 1
    return total if seen else float("nan")


def timing_block(
    records: Iterable[BenchmarkRecord],
    *,
    overall_seconds: float,
    condition_order: Sequence[str],
    host: dict[str, Any],
    condition_wall_seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the per-condition timing block for a benchmark summary.

    Conditions are derived from *records* (grouped by ``condition``) and
    ordered by *condition_order* using the same index rule as
    ``order_by_condition``: listed conditions appear in declared order,
    unlisted ones sort after them. Within a condition, ``per_transcriber``
    entries are ordered by transcriber name.

    Timing semantics:

    - ``render_seconds``/``separate_seconds`` are recorded once per
      (condition, file) but duplicated across transcriber rows (the runner
      keys cells by ``source_path or midi_path``), so the condition-level
      total dedupes records by file key and sums the first value seen per
      file, avoiding transcriber double-counting.
    - ``transcribe_seconds``/``evaluate_seconds`` at condition level sum
      over ALL records of the condition (failed cells still consumed time).
    - ``per_transcriber`` totals sum only over records whose ``status`` is
      ``"succeeded"``.
    - All sums are NaN-aware (NaN values skipped; a group with no finite
      values yields NaN). ``wall_seconds`` comes from
      *condition_wall_seconds*, NaN when a condition is missing from it.
    - Empty *records* produce ``"conditions": []``.
    """
    condition_wall_seconds = condition_wall_seconds or {}

    by_condition: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)

    index = {name: i for i, name in enumerate(condition_order)}
    condition_names = sorted(
        by_condition, key=lambda name: index.get(name, len(index))
    )

    condition_rows: list[dict[str, Any]] = []
    for condition in condition_names:
        members = by_condition[condition]

        # render/separate are recorded once per (condition, file) but repeated
        # on every transcriber row: dedupe by file key, take the first value.
        per_file: dict[str, list[BenchmarkRecord]] = {}
        for record in members:
            per_file.setdefault(record.source_path or record.midi_path, []).append(record)
        render_seconds = _nanaware_sum(
            file_records[0].render_seconds for file_records in per_file.values()
        )
        separate_seconds = _nanaware_sum(
            file_records[0].separate_seconds for file_records in per_file.values()
        )
        transcribe_seconds = _nanaware_sum(
            record.transcribe_seconds for record in members
        )
        evaluate_seconds = _nanaware_sum(record.evaluate_seconds for record in members)

        per_transcriber: list[dict[str, Any]] = []
        for name in sorted({record.transcriber for record in members}):
            succeeded = [
                record
                for record in members
                if record.status == "succeeded" and record.transcriber == name
            ]
            per_transcriber.append(
                {
                    "transcriber": name,
                    "transcribe_seconds": _nanaware_sum(
                        record.transcribe_seconds for record in succeeded
                    ),
                    "evaluate_seconds": _nanaware_sum(
                        record.evaluate_seconds for record in succeeded
                    ),
                    "n_succeeded": len(succeeded),
                }
            )

        condition_rows.append(
            {
                "condition": condition,
                "wall_seconds": condition_wall_seconds.get(condition, float("nan")),
                "render_seconds": render_seconds,
                "separate_seconds": separate_seconds,
                "transcribe_seconds": transcribe_seconds,
                "evaluate_seconds": evaluate_seconds,
                "per_transcriber": per_transcriber,
            }
        )

    return {
        "overall_seconds": overall_seconds,
        "host": host,
        "conditions": condition_rows,
    }
