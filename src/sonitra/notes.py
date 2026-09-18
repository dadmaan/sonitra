"""Canonical note helpers for producers and consumers.

Validates and normalises note dictionaries. Drops zero or negative
durations, clamps velocity to 1..127 and start to >=0, and rejects
out-of-range pitches.
"""

from __future__ import annotations

from typing import Any, Iterable


def make_note(
    pitch: int,
    velocity: int | float,
    start_sec: float,
    duration_sec: float,
) -> dict[str, Any] | None:
    """Create a normalised note dict or drop it.

    Returns None when duration is not positive. Clamps velocity and
    start time. Raises ValueError for pitches outside 0..127.
    """
    try:
        pitch_int = int(pitch)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pitch must be in 0..127, got {pitch!r}") from exc
    if not 0 <= pitch_int <= 127:
        raise ValueError(f"pitch must be in 0..127, got {pitch_int}")

    duration = float(duration_sec)
    if duration <= 0.0:
        return None

    try:
        velocity_int = int(velocity)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # allow float-like strings
        try:
            velocity_int = int(float(velocity))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"velocity must be int-coercible, got {velocity!r}") from exc
    velocity_clamped = max(1, min(127, velocity_int))

    start = float(start_sec)
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
    for out-of-range pitches.
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
