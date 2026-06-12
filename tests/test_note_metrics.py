from __future__ import annotations

import pytest

from sonitra.evaluation.note_metrics import (
    NoteMetrics,
    match_notes,
    match_notes_with_velocity,
    precision_recall_f1,
)
from sonitra.evaluation.types import NoteEvent, notes_from_dicts


REFERENCE = [
    NoteEvent(60, 0.0, 0.5, 80),
    NoteEvent(64, 0.5, 1.0, 90),
    NoteEvent(67, 1.0, 1.6, 70),
    NoteEvent(72, 1.8, 2.5, 100),
]


def test_perfect_match_scores_one() -> None:
    results = NoteMetrics().compute(REFERENCE, REFERENCE)
    for key, value in results.items():
        assert value == pytest.approx(1.0), key


def test_empty_estimate_scores_zero() -> None:
    results = NoteMetrics().compute(REFERENCE, [])
    assert results["onset_f1"] == 0.0
    assert results["onset_offset_f1"] == 0.0


def test_onset_within_tolerance_matches() -> None:
    shifted = [NoteEvent(n.pitch, n.onset_sec + 0.03, n.offset_sec + 0.03, n.velocity) for n in REFERENCE]
    pairs = match_notes(REFERENCE, shifted, onset_tolerance_sec=0.05)
    assert len(pairs) == len(REFERENCE)


def test_onset_beyond_tolerance_does_not_match() -> None:
    shifted = [NoteEvent(n.pitch, n.onset_sec + 0.08, n.offset_sec + 0.08, n.velocity) for n in REFERENCE]
    pairs = match_notes(REFERENCE, shifted, onset_tolerance_sec=0.05)
    assert pairs == []


def test_wrong_pitch_does_not_match() -> None:
    transposed = [NoteEvent(n.pitch + 1, n.onset_sec, n.offset_sec, n.velocity) for n in REFERENCE]
    assert match_notes(REFERENCE, transposed) == []


def test_offset_condition_is_stricter() -> None:
    # offsets pulled way in: onset matching passes, offset matching fails
    truncated = [NoteEvent(n.pitch, n.onset_sec, n.onset_sec + 0.05, n.velocity) for n in REFERENCE]
    onset_pairs = match_notes(REFERENCE, truncated)
    offset_pairs = match_notes(REFERENCE, truncated, with_offset=True)
    assert len(onset_pairs) == len(REFERENCE)
    assert offset_pairs == []


def test_offset_tolerance_uses_duration_ratio() -> None:
    # 20% of a 1s note = 200ms tolerance, so a 150ms offset error still matches
    ref = [NoteEvent(60, 0.0, 1.0, 80)]
    est = [NoteEvent(60, 0.0, 1.15, 80)]
    assert len(match_notes(ref, est, with_offset=True, offset_ratio=0.2)) == 1
    assert match_notes(ref, est, with_offset=True, offset_ratio=0.1) == []


def test_bipartite_matching_is_one_to_one() -> None:
    # two estimates compete for one reference note
    ref = [NoteEvent(60, 0.0, 1.0, 80)]
    est = [NoteEvent(60, 0.01, 1.0, 80), NoteEvent(60, 0.02, 1.0, 80)]
    pairs = match_notes(ref, est)
    assert len(pairs) == 1


def test_velocity_matching_drops_outliers() -> None:
    ref = [NoteEvent(60, 0.0, 1.0, 100), NoteEvent(64, 1.0, 2.0, 50), NoteEvent(67, 2.0, 3.0, 75)]
    # velocities linearly related -> all kept after rescaling
    est_linear = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(64, 1.0, 2.0, 40), NoteEvent(67, 2.0, 3.0, 60)]
    assert len(match_notes_with_velocity(ref, est_linear)) == 3
    # one velocity wildly off the linear relationship -> dropped
    est_outlier = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(64, 1.0, 2.0, 127), NoteEvent(67, 2.0, 3.0, 60)]
    assert len(match_notes_with_velocity(ref, est_outlier)) < 3


def test_precision_recall_f1_edge_cases() -> None:
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    precision, recall, f1 = precision_recall_f1(1, 2, 4)
    assert precision == 0.5
    assert recall == 0.25
    assert f1 == pytest.approx(2 * 0.5 * 0.25 / 0.75)


def test_notes_from_dicts_sorting_and_offsets() -> None:
    events = notes_from_dicts(
        [
            {"pitch": 64, "velocity": 90, "start_sec": 1.0, "duration_sec": 0.5},
            {"pitch": 60, "velocity": 80, "start_sec": 0.0, "duration_sec": 1.0},
        ]
    )
    assert [event.pitch for event in events] == [60, 64]
    assert events[0].offset_sec == pytest.approx(1.0)
    assert events[1].duration_sec == pytest.approx(0.5)
