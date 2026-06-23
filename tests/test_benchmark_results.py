from __future__ import annotations

import math
from pathlib import Path

import pytest

from sonitra.benchmark.results import (
    BenchmarkRecord,
    ResultsWriter,
    degradation,
    load_records,
    summarise,
)


def _record(condition: str, transcriber: str, f1: float, status: str = "succeeded") -> BenchmarkRecord:
    return BenchmarkRecord(
        condition=condition,
        transcriber=transcriber,
        midi_path="a.mid",
        audio_path="a.wav",
        status=status,
        metrics={"note.onset_f1": f1} if status == "succeeded" else {},
    )


def test_writer_round_trip(tmp_path: Path) -> None:
    writer = ResultsWriter(tmp_path / "results.jsonl")
    writer.write(_record("baseline", "bp", 0.9))
    writer.write(_record("noise", "bp", 0.7))
    records = load_records(writer.path)
    assert len(records) == 2
    assert records[0].metrics["note.onset_f1"] == 0.9


def test_summarise_means_per_group() -> None:
    records = [
        _record("baseline", "bp", 0.8),
        _record("baseline", "bp", 1.0),
        _record("baseline", "mt3", 0.6),
    ]
    summary = summarise(records)
    by_key = {(row["condition"], row["transcriber"]): row for row in summary}
    assert by_key[("baseline", "bp")]["note.onset_f1"] == pytest.approx(0.9)
    assert by_key[("baseline", "bp")]["n_files"] == 2
    assert by_key[("baseline", "mt3")]["note.onset_f1"] == pytest.approx(0.6)


def test_summarise_skips_nan_and_failures() -> None:
    records = [
        _record("baseline", "bp", 0.8),
        _record("baseline", "bp", float("nan")),
        _record("baseline", "bp", 0.0, status="failed"),
    ]
    summary = summarise(records)
    row = summary[0]
    assert row["note.onset_f1"] == pytest.approx(0.8)
    assert row["n_files"] == 3
    assert row["n_succeeded"] == 2


def test_summarise_all_nan_metric() -> None:
    records = [_record("baseline", "bp", float("nan"))]
    assert math.isnan(summarise(records)[0]["note.onset_f1"])


def test_degradation_vs_baseline() -> None:
    summary = summarise(
        [
            _record("baseline", "bp", 0.9),
            _record("reverb=0.5", "bp", 0.6),
            _record("reverb=0.5", "mt3", 0.5),
        ]
    )
    rows = degradation(summary, baseline="baseline")
    deltas = {(row["condition"], row["transcriber"]): row for row in rows}
    assert deltas[("reverb=0.5", "bp")]["delta_note.onset_f1"] == pytest.approx(-0.3)
    # mt3 has no baseline row -> excluded
    assert ("reverb=0.5", "mt3") not in deltas
