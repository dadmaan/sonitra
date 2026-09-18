"""Canonical note helpers for producers and consumers.

Validates and normalises note dictionaries. Drops zero or negative
durations, clamps velocity to 1..127 and start to >=0, and rejects
out-of-range pitches. Non-finite (NaN/+-Inf) pitch, velocity, start_sec
or duration_sec always raise: a non-finite value is always a producer
bug, never a legitimately empty or clamped note.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def _finite_float(value: Any, field: str) -> float:
    """Coerce *value* to a finite float, raising ValueError otherwise.

    Covers both non-numeric input and NaN/+-Inf, since a bare ``float()``
    cast happily accepts either and non-finite values then defeat every
    downstream ``<``/``<=`` comparison (``nan <= 0`` and ``inf < 0`` are
    both False).
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(as_float):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    return as_float


def make_note(
    pitch: int,
    velocity: int | float,
    start_sec: float,
    duration_sec: float,
) -> dict[str, Any] | None:
    """Create a normalised note dict or drop it.

    Returns None when duration is (finite and) not positive. Clamps
    velocity and start time. Raises ValueError for pitches outside
    0..127, and for any non-finite (NaN/+-Inf) pitch, velocity,
    start_sec or duration_sec.
    """
    try:
        pitch_int = int(pitch)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"pitch must be in 0..127, got {pitch!r}") from exc
    if not 0 <= pitch_int <= 127:
        raise ValueError(f"pitch must be in 0..127, got {pitch_int}")

    duration = _finite_float(duration_sec, "duration_sec")
    if duration <= 0.0:
        return None

    velocity_float = _finite_float(velocity, "velocity")
    velocity_clamped = max(1, min(127, int(velocity_float)))

    start = _finite_float(start_sec, "start_sec")
    if start < 0.0:
        start = 0.0

    return {
        "pitch": pitch_int,
        "velocity": velocity_clamped,
        "start_sec": start,
        "duration_sec": duration,
    }


def normalise_notes(
    notes: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise an iterable of note dicts.

    Filters notes with non-positive duration via make_note and returns
    the remainder sorted by start time then pitch. Propagates ValueError
    for out-of-range pitch or any non-finite pitch/velocity/start_sec/
    duration_sec -- one bad note aborts the whole batch rather than being
    silently dropped or corrupted; callers own retrying at the per-file
    granularity.
    """
    out: list[dict[str, Any]] = []
    for note in notes:
        normalised = make_note(
            pitch=note["pitch"],
            velocity=note.get("velocity", 64),
            start_sec=note["start_sec"],
            duration_sec=note["duration_sec"],
        )
        if normalised is not None:
            out.append(normalised)
    out.sort(key=lambda n: (n["start_sec"], n["pitch"]))
    return out
