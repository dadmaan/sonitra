"""Tests for the canonical note contract (`sonitra.notes`)."""

from __future__ import annotations

import math

import pytest

from sonitra.notes import make_note, normalise_notes
from tests.helpers import assert_notes_satisfy_contract


# ── baseline contract (pitch / velocity / start / duration) ───────────────


def test_make_note_returns_normalised_dict() -> None:
    note = make_note(pitch=60, velocity=100, start_sec=1.5, duration_sec=0.5)
    assert note == {
        "pitch": 60,
        "velocity": 100,
        "start_sec": 1.5,
        "duration_sec": 0.5,
    }


def test_make_note_clamps_velocity_into_1_127() -> None:
    assert make_note(pitch=60, velocity=0, start_sec=0.0, duration_sec=1.0)["velocity"] == 1
    assert make_note(pitch=60, velocity=200, start_sec=0.0, duration_sec=1.0)["velocity"] == 127


def test_make_note_clamps_start_to_zero() -> None:
    note = make_note(pitch=60, velocity=64, start_sec=-1.0, duration_sec=1.0)
    assert note["start_sec"] == 0.0


def test_make_note_drops_non_positive_duration() -> None:
    assert make_note(pitch=60, velocity=64, start_sec=0.0, duration_sec=0.0) is None
    assert make_note(pitch=60, velocity=64, start_sec=0.0, duration_sec=-1.0) is None


@pytest.mark.parametrize("pitch", [-1, 128])
def test_make_note_rejects_out_of_range_pitch(pitch: int) -> None:
    with pytest.raises(ValueError, match="pitch"):
        make_note(pitch=pitch, velocity=64, start_sec=0.0, duration_sec=1.0)


# ── non-finite (NaN / +-Inf) rejection ─────────────────────────────────────
#
# NaN and Inf defeat make_note's own comparisons (`nan <= 0`, `nan < 0`,
# `inf < 0` are all False), so a bare float() cast lets them slip through
# either as a kept note with a poisoned field, or as a silently "clamped"
# 0.0. Both are producer bugs, never a legitimately empty or negative
# note, so they must raise -- exactly like an out-of-range pitch already
# does.


@pytest.mark.parametrize("bad_duration", [math.nan, math.inf, -math.inf])
def test_make_note_rejects_non_finite_duration(bad_duration: float) -> None:
    with pytest.raises(ValueError, match="duration_sec"):
        make_note(pitch=60, velocity=64, start_sec=0.0, duration_sec=bad_duration)


@pytest.mark.parametrize("bad_start", [math.nan, math.inf, -math.inf])
def test_make_note_rejects_non_finite_start(bad_start: float) -> None:
    with pytest.raises(ValueError, match="start_sec"):
        make_note(pitch=60, velocity=64, start_sec=bad_start, duration_sec=1.0)


@pytest.mark.parametrize("bad_velocity", [math.nan, math.inf, -math.inf])
def test_make_note_rejects_non_finite_velocity(bad_velocity: float) -> None:
    with pytest.raises(ValueError, match="velocity"):
        make_note(pitch=60, velocity=bad_velocity, start_sec=0.0, duration_sec=1.0)


@pytest.mark.parametrize("bad_pitch", [math.inf, -math.inf])
def test_make_note_rejects_non_finite_pitch(bad_pitch: float) -> None:
    with pytest.raises(ValueError, match="pitch"):
        make_note(pitch=bad_pitch, velocity=64, start_sec=0.0, duration_sec=1.0)  # type: ignore[arg-type]


def test_make_note_rejects_nan_pitch() -> None:
    # Already covered by int()'s own ValueError before this fix; pinned here
    # so the whole non-finite surface is documented in one file.
    with pytest.raises(ValueError, match="pitch"):
        make_note(pitch=math.nan, velocity=64, start_sec=0.0, duration_sec=1.0)  # type: ignore[arg-type]


# ── normalise_notes: batch propagation ─────────────────────────────────────


def test_normalise_notes_sorts_and_satisfies_contract() -> None:
    notes = normalise_notes(
        [
            {"pitch": 64, "velocity": 90, "start_sec": 1.0, "duration_sec": 0.5},
            {"pitch": 60, "velocity": 80, "start_sec": 0.0, "duration_sec": 1.0},
            {"pitch": 62, "velocity": 80, "start_sec": 0.0, "duration_sec": 0.0},  # dropped
        ]
    )
    assert_notes_satisfy_contract(notes)
    assert [note["pitch"] for note in notes] == [60, 64]


def test_normalise_notes_propagates_valueerror_for_non_finite_note() -> None:
    # One corrupt note aborts the whole batch -- the existing, deliberate
    # behaviour for out-of-range pitch, now also true for non-finite
    # duration/start/velocity/pitch. Callers must catch this at the
    # per-file boundary (see cli.py / benchmark/runner.py), not here.
    with pytest.raises(ValueError, match="duration_sec"):
        normalise_notes(
            [
                {"pitch": 60, "velocity": 80, "start_sec": 0.0, "duration_sec": 1.0},
                {"pitch": 64, "velocity": 90, "start_sec": 1.0, "duration_sec": math.nan},
            ]
        )
