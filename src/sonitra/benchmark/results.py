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
    """One transcription evaluation: a (condition, transcriber, file) cell."""

    condition: str
    transcriber: str
    midi_path: str
    audio_path: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    ("pipeline", "max_workers"),
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
    """Aggregate metric means per (condition, transcriber), NaN-aware."""
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
