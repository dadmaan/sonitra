from __future__ import annotations

import pytest

from sonitra.evaluation.frame_metrics import FrameMetrics, rasterise
from sonitra.evaluation.types import NoteEvent


def test_perfect_match() -> None:
    notes = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(64, 0.5, 1.5, 80)]
    results = FrameMetrics().compute(notes, notes)
    assert results == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_disjoint_pitches_score_zero() -> None:
    ref = [NoteEvent(60, 0.0, 1.0, 80)]
    est = [NoteEvent(72, 0.0, 1.0, 80)]
    results = FrameMetrics().compute(ref, est)
    assert results["f1"] == 0.0


def test_half_overlap() -> None:
    ref = [NoteEvent(60, 0.0, 1.0, 80)]
    est = [NoteEvent(60, 0.0, 0.5, 80)]
    results = FrameMetrics(hop_sec=0.01).compute(ref, est)
    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(0.5)


def test_rasterise_minimum_one_frame() -> None:
    cells = rasterise([NoteEvent(60, 0.0, 0.0001, 80)], hop_sec=0.01)
    assert cells == {(60, 0)}


def test_empty_inputs() -> None:
    assert FrameMetrics().compute([], [])["f1"] == 0.0
