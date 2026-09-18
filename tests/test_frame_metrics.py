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


def test_rasterise_zero_duration_exactly_one_frame() -> None:
    """Part C Stage 15: clamp `last <= first` pins zero-duration to one cell.

    The existing test uses 0.0001 s (positive duration). Only the 0.0 case
    proves the `<=` rather than `<` branch in frame_metrics.py:18-19.
    Flipping `<=` to `<` would produce 0 cells and fail this test.
    """
    cells = rasterise([NoteEvent(60, 0.0, 0.0, 80)], hop_sec=0.01)
    assert cells == {(60, 0)}
    # Also pin degenerate at a non-zero onset to ensure frame math, not
    # just origin special-casing.
    cells_mid = rasterise([NoteEvent(60, 0.5, 0.5, 80)], hop_sec=0.01)
    assert cells_mid == {(60, 50)}


def test_rasterise_short_note_shorter_than_hop_one_frame() -> None:
    """Part C Stage 15: notes shorter than one hop still occupy one cell.

    The clamp is only reached at zero/negative duration (any positive
    duration already yields ceil > floor), but the broader deliberate
    behaviour is that a sub-hop note rasterises to ≥1 cell rather than
    disappearing. Pin it explicitly.
    """
    # 2 ms note with 10 ms hop — shorter than one hop, not degenerate.
    cells = rasterise([NoteEvent(60, 0.0, 0.002, 80)], hop_sec=0.01)
    assert cells == {(60, 0)}
    # Non-zero onset, still sub-hop.
    cells_offset = rasterise([NoteEvent(60, 0.005, 0.007, 80)], hop_sec=0.01)
    assert len(cells_offset) >= 1


def test_frame_metrics_degenerate_note_deflates_precision() -> None:
    """Part C Stage 15: degenerate estimate adds a cell → precision < 1, recall == 1.

    rasterise is tested above in isolation; this pins the consequence at the
    FrameMetrics level — a zero-duration phantom inflates len(est_cells) and
    lowers frame.precision while recall stays perfect when the estimate is
    otherwise a superset of the reference.
    """
    hop = 0.01
    ref = [NoteEvent(60, 0.0, 1.0, 80)]  # 100 frames: 0..99
    # Degenerate phantom on a different pitch adds exactly one extra cell
    # (61, 0) that is not in ref. Without the clamp it would add 0 cells
    # and precision would stay 1.0.
    est = ref + [NoteEvent(61, 0.0, 0.0, 80)]
    results = FrameMetrics(hop_sec=hop).compute(ref, est)
    assert results["recall"] == pytest.approx(1.0)
    assert results["precision"] == pytest.approx(100 / 101)
    assert results["precision"] < 1.0
    assert results["f1"] < 1.0


def test_frame_metrics_degenerate_note_different_frame_deflates_precision() -> None:
    """Part C Stage 15: degenerate at a new frame also inflates est_cells.

    Complements the different-pitch variant — a degenerate note at the frame
    just past the reference (pitch 60, frame 100) is also exactly one new
    cell via the clamp, so precision is deflated even on the same pitch.
    """
    hop = 0.01
    ref = [NoteEvent(60, 0.0, 1.0, 80)]
    est = ref + [NoteEvent(60, 1.0, 1.0, 80)]  # degenerate at 1.0 -> frame 100
    ref_cells = rasterise(ref, hop_sec=hop)
    est_cells = rasterise(est, hop_sec=hop)
    assert len(est_cells) == len(ref_cells) + 1
    assert (60, 100) in est_cells
    results = FrameMetrics(hop_sec=hop).compute(ref, est)
    assert results["recall"] == pytest.approx(1.0)
    assert results["precision"] < 1.0


def test_empty_inputs() -> None:
    assert FrameMetrics().compute([], [])["f1"] == 0.0
