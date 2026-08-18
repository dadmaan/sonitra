from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sonitra.benchmark.results import (
    BenchmarkRecord,
    ResultsWriter,
    WorkerEvent,
    degradation,
    load_records,
    order_by_condition,
    summarise,
    timing_block,
)


def _record(
    condition: str,
    transcriber: str,
    f1: float,
    status: str = "succeeded",
    *,
    midi_path: str = "a.mid",
    render_seconds: float = float("nan"),
    separate_seconds: float = float("nan"),
    transcribe_seconds: float = float("nan"),
    evaluate_seconds: float = float("nan"),
) -> BenchmarkRecord:
    return BenchmarkRecord(
        condition=condition,
        transcriber=transcriber,
        midi_path=midi_path,
        audio_path="a.wav",
        status=status,
        metrics={"note.onset_f1": f1} if status == "succeeded" else {},
        render_seconds=render_seconds,
        separate_seconds=separate_seconds,
        transcribe_seconds=transcribe_seconds,
        evaluate_seconds=evaluate_seconds,
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


def test_order_by_condition_restores_declared_order() -> None:
    # Simulates parallel-mode completion order scrambling the declared order.
    summary = summarise(
        [
            _record("wet_level=0.6", "bp", 0.5),
            _record("baseline", "bp", 0.9),
            _record("no_reverb", "bp", 0.8),
        ]
    )
    ordered = order_by_condition(
        summary, ["baseline", "no_reverb", "wet_level=0.6"]
    )
    assert [row["condition"] for row in ordered] == [
        "baseline",
        "no_reverb",
        "wet_level=0.6",
    ]


def test_order_by_condition_breaks_ties_by_transcriber() -> None:
    summary = summarise(
        [
            _record("baseline", "mt3", 0.5),
            _record("baseline", "bp", 0.9),
        ]
    )
    ordered = order_by_condition(summary, ["baseline"])
    assert [row["transcriber"] for row in ordered] == ["bp", "mt3"]


def test_worker_event_stage_field_defaults_to_empty_string() -> None:
    """``stage`` is display-only and optional: every existing construction
    site (and any not yet updated to pass it) must keep working."""
    event = WorkerEvent(
        worker_id=123,
        condition="baseline",
        transcriber="oracle",
        midi_path="a.mid",
        status="done",
        ok=True,
    )
    assert event.stage == ""


def test_order_by_condition_unknown_condition_sorts_last() -> None:
    summary = summarise(
        [
            _record("mystery", "bp", 0.5),
            _record("baseline", "bp", 0.9),
        ]
    )
    ordered = order_by_condition(summary, ["baseline"])
    assert [row["condition"] for row in ordered] == ["baseline", "mystery"]


def test_summarise_is_metrics_only_no_timing_keys() -> None:
    """Summary rows are METRICS-ONLY: the per-stage timing totals that used to
    live in summary rows are produced by ``timing_block`` instead."""
    records = [
        _record("baseline", "bp", 0.8, render_seconds=1.5, separate_seconds=2.0, transcribe_seconds=3.0, evaluate_seconds=0.5),
        _record("baseline", "bp", 1.0, render_seconds=4.5, separate_seconds=float("nan"), transcribe_seconds=7.0, evaluate_seconds=1.5),
        _record("baseline", "bp", 0.0, status="failed", render_seconds=10.0),
    ]
    row = summarise(records)[0]
    assert set(row) == {
        "condition", "transcriber", "n_files", "n_succeeded", "note.onset_f1"
    }
    assert row["n_files"] == 3
    assert row["n_succeeded"] == 2
    assert row["note.onset_f1"] == pytest.approx(0.9)

    # The same records' timing data is produced by timing_block: per-condition
    # transcribe/evaluate sums over ALL records (failed cells still consumed
    # time), render/separate deduped per file (all records share midi_path
    # "a.mid", so the first row's values win).
    block = timing_block(
        records, overall_seconds=99.0, condition_order=["baseline"], host={}
    )
    condition = block["conditions"][0]
    assert condition["render_seconds"] == pytest.approx(1.5)
    assert condition["separate_seconds"] == pytest.approx(2.0)  # NaN skipped
    assert condition["transcribe_seconds"] == pytest.approx(10.0)
    assert condition["evaluate_seconds"] == pytest.approx(2.0)


def test_summarise_never_emits_timing_keys() -> None:
    """Even records carrying timing values never leak timing keys into summary
    rows -- with or without NaN defaults."""
    records = [
        _record("baseline", "bp", 0.8, render_seconds=1.0),
        _record("baseline", "bp", 1.0),
    ]
    row = summarise(records)[0]
    for key in (
        "render_seconds",
        "separate_seconds",
        "transcribe_seconds",
        "evaluate_seconds",
    ):
        assert key not in row
    assert set(row) == {
        "condition", "transcriber", "n_files", "n_succeeded", "note.onset_f1"
    }


def test_degradation_skips_timing_columns() -> None:
    summary = summarise(
        [
            _record("baseline", "bp", 0.9, render_seconds=1.0, transcribe_seconds=2.0),
            _record("reverb=0.5", "bp", 0.6, render_seconds=5.0, transcribe_seconds=9.0),
        ]
    )
    rows = degradation(summary, baseline="baseline")
    assert len(rows) == 1
    assert "delta_render_seconds" not in rows[0]
    assert "delta_transcribe_seconds" not in rows[0]
    assert rows[0]["delta_note.onset_f1"] == pytest.approx(-0.3)


def test_timing_block_overall_and_condition_order() -> None:
    records = [
        _record("wet_level=0.6", "bp", 0.5, render_seconds=2.0),
        _record("baseline", "bp", 0.9, render_seconds=1.0),
        _record("no_reverb", "bp", 0.8, render_seconds=3.0),
    ]
    block = timing_block(
        records,
        overall_seconds=12.5,
        condition_order=["baseline", "no_reverb", "wet_level=0.6"],
        host={"cpu_count": 8},
        condition_wall_seconds={"baseline": 4.0, "no_reverb": 5.0},
    )
    assert block["overall_seconds"] == pytest.approx(12.5)
    assert block["host"] == {"cpu_count": 8}
    assert [c["condition"] for c in block["conditions"]] == [
        "baseline",
        "no_reverb",
        "wet_level=0.6",
    ]
    assert block["conditions"][0]["wall_seconds"] == pytest.approx(4.0)
    assert block["conditions"][1]["wall_seconds"] == pytest.approx(5.0)
    assert math.isnan(block["conditions"][2]["wall_seconds"])


def test_timing_block_unlisted_condition_sorts_last() -> None:
    block = timing_block(
        [
            _record("mystery", "bp", 0.5),
            _record("baseline", "bp", 0.9),
        ],
        overall_seconds=1.0,
        condition_order=["baseline"],
        host={},
    )
    assert [c["condition"] for c in block["conditions"]] == ["baseline", "mystery"]


def test_timing_block_render_dedupes_by_file() -> None:
    # render/separate are recorded once per (condition, file) but duplicated
    # across transcriber rows; the condition total must not double-count.
    block = timing_block(
        [
            _record("baseline", "bp", 0.9, midi_path="a.mid", render_seconds=5.0, separate_seconds=2.0),
            _record("baseline", "mt3", 0.8, midi_path="a.mid", render_seconds=5.0, separate_seconds=2.0),
        ],
        overall_seconds=10.0,
        condition_order=["baseline"],
        host={},
    )
    condition = block["conditions"][0]
    assert condition["render_seconds"] == pytest.approx(5.0)
    assert condition["separate_seconds"] == pytest.approx(2.0)


def test_timing_block_condition_totals_include_failed_cells() -> None:
    block = timing_block(
        [
            _record("baseline", "bp", 0.9, transcribe_seconds=1.0, evaluate_seconds=0.5),
            _record("baseline", "mt3", 0.8, transcribe_seconds=2.0, evaluate_seconds=1.0),
            _record("baseline", "bp", 0.0, status="failed", transcribe_seconds=9.0, evaluate_seconds=4.0),
        ],
        overall_seconds=30.0,
        condition_order=["baseline"],
        host={},
    )
    condition = block["conditions"][0]
    assert condition["transcribe_seconds"] == pytest.approx(12.0)
    assert condition["evaluate_seconds"] == pytest.approx(5.5)


def test_timing_block_per_transcriber_over_succeeded() -> None:
    block = timing_block(
        [
            _record("baseline", "mt3", 0.8, transcribe_seconds=2.0, evaluate_seconds=1.0),
            _record("baseline", "bp", 0.9, transcribe_seconds=1.0, evaluate_seconds=0.5),
            _record("baseline", "bp", 0.0, status="failed", transcribe_seconds=9.0, evaluate_seconds=4.0),
        ],
        overall_seconds=30.0,
        condition_order=["baseline"],
        host={},
    )
    condition = block["conditions"][0]
    assert [entry["transcriber"] for entry in condition["per_transcriber"]] == ["bp", "mt3"]
    by_name = {entry["transcriber"]: entry for entry in condition["per_transcriber"]}
    assert by_name["bp"]["transcribe_seconds"] == pytest.approx(1.0)
    assert by_name["bp"]["evaluate_seconds"] == pytest.approx(0.5)
    assert by_name["bp"]["n_succeeded"] == 1
    assert by_name["mt3"]["transcribe_seconds"] == pytest.approx(2.0)
    assert by_name["mt3"]["evaluate_seconds"] == pytest.approx(1.0)
    assert by_name["mt3"]["n_succeeded"] == 1


def test_timing_block_per_transcriber_nan_when_no_timing() -> None:
    block = timing_block(
        [_record("baseline", "bp", 0.9)],
        overall_seconds=1.0,
        condition_order=["baseline"],
        host={},
    )
    entry = block["conditions"][0]["per_transcriber"][0]
    assert math.isnan(entry["transcribe_seconds"])
    assert math.isnan(entry["evaluate_seconds"])
    assert entry["n_succeeded"] == 1


def test_timing_block_empty_records() -> None:
    block = timing_block(
        [], overall_seconds=0.0, condition_order=[], host={}
    )
    assert block["conditions"] == []
    assert block["overall_seconds"] == pytest.approx(0.0)


def test_load_records_defaults_timing_fields(tmp_path: Path) -> None:
    # A pre-upgrade JSONL line without the timing fields must load with NaN defaults.
    old_line = json.dumps(
        {
            "condition": "baseline",
            "transcriber": "bp",
            "midi_path": "a.mid",
            "audio_path": "a.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.9},
            "overrides": {},
            "error": None,
            "source_path": None,
        }
    )
    path = tmp_path / "old.jsonl"
    path.write_text(old_line + "\n")
    records = load_records(path)
    assert len(records) == 1
    assert math.isnan(records[0].render_seconds)
    assert math.isnan(records[0].separate_seconds)
    assert math.isnan(records[0].transcribe_seconds)
    assert math.isnan(records[0].evaluate_seconds)
