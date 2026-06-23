from __future__ import annotations

import math

import pytest

from sonitra.evaluation.expressive_metrics import ExpressiveMetrics
from sonitra.evaluation.types import NoteEvent


REFERENCE = [
    NoteEvent(60, 0.0, 0.5, 80),
    NoteEvent(64, 0.5, 1.0, 90),
    NoteEvent(67, 1.2, 1.6, 70),
    NoteEvent(72, 1.8, 2.5, 100),
]


def test_identical_sequences() -> None:
    results = ExpressiveMetrics().compute(REFERENCE, REFERENCE)
    assert results["onset_mae"] == pytest.approx(0.0)
    assert results["onset_bias"] == pytest.approx(0.0)
    assert results["ioi_corr"] == pytest.approx(1.0)
    assert results["kor_corr"] == pytest.approx(1.0)
    assert results["velocity_corr"] == pytest.approx(1.0)
    assert results["harmony_sim"] == pytest.approx(1.0)


def test_constant_onset_shift_gives_bias() -> None:
    shifted = [NoteEvent(n.pitch, n.onset_sec + 0.02, n.offset_sec + 0.02, n.velocity) for n in REFERENCE]
    results = ExpressiveMetrics().compute(REFERENCE, shifted)
    assert results["onset_mae"] == pytest.approx(0.02)
    assert results["onset_bias"] == pytest.approx(0.02)
    # constant shift preserves IOIs exactly
    assert results["ioi_corr"] == pytest.approx(1.0)


def test_no_matches_gives_nan() -> None:
    transposed = [NoteEvent(n.pitch + 1, n.onset_sec, n.offset_sec, n.velocity) for n in REFERENCE]
    results = ExpressiveMetrics().compute(REFERENCE, transposed)
    assert math.isnan(results["onset_mae"])
    assert math.isnan(results["ioi_corr"])
    # both sequences still have tonal content in the same windows, just shifted
    assert not math.isnan(results["harmony_sim"])


def test_empty_inputs_give_nan() -> None:
    results = ExpressiveMetrics().compute([], [])
    assert all(math.isnan(value) for value in results.values())


def test_harmony_sim_detects_pitch_class_divergence() -> None:
    same = ExpressiveMetrics().compute(REFERENCE, REFERENCE)["harmony_sim"]
    tritones = [NoteEvent(n.pitch + 6, n.onset_sec, n.offset_sec, n.velocity) for n in REFERENCE]
    diverged = ExpressiveMetrics().compute(REFERENCE, tritones)["harmony_sim"]
    assert diverged < same


def test_zero_variance_velocity_gives_nan() -> None:
    flat = [NoteEvent(n.pitch, n.onset_sec, n.offset_sec, 64) for n in REFERENCE]
    results = ExpressiveMetrics().compute(flat, flat)
    assert math.isnan(results["velocity_corr"])
