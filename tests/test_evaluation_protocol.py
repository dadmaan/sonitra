from __future__ import annotations

from sonitra.config import EvaluationSection
from sonitra.evaluation.protocol import (
    evaluate_notes,
    make_audio_metrics,
    make_symbolic_metrics,
)
from sonitra.evaluation.types import NoteEvent


def test_default_section_builds_symbolic_metrics() -> None:
    metrics = make_symbolic_metrics(EvaluationSection())
    assert {metric.name for metric in metrics} == {"note", "frame", "expressive"}


def test_disabled_metrics_are_skipped() -> None:
    section = EvaluationSection.model_validate(
        {
            "note_metrics": {"enabled": False},
            "frame_metrics": {"enabled": False},
            "expressive_metrics": {"enabled": True},
        }
    )
    metrics = make_symbolic_metrics(section)
    assert {metric.name for metric in metrics} == {"expressive"}


def test_dtw_disabled_by_default_and_toggleable() -> None:
    assert make_audio_metrics(EvaluationSection()) == []
    section = EvaluationSection.model_validate({"dtw": {"enabled": True}})
    assert [metric.name for metric in make_audio_metrics(section)] == ["dtw"]


def test_evaluate_notes_prefixes_keys() -> None:
    notes = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(64, 1.0, 2.0, 90)]
    metrics = make_symbolic_metrics(EvaluationSection())
    results = evaluate_notes(notes, notes, metrics)
    assert results["note.onset_f1"] == 1.0
    assert results["frame.f1"] == 1.0
    assert "expressive.onset_mae" in results


def test_metric_parameters_flow_from_config() -> None:
    section = EvaluationSection.model_validate(
        {"note_metrics": {"onset_tolerance_sec": 0.2}}
    )
    metric = next(m for m in make_symbolic_metrics(section) if m.name == "note")
    assert metric.onset_tolerance_sec == 0.2
