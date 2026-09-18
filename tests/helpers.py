"""Shared test helper for note-dict contract."""

from __future__ import annotations

from typing import Any, Iterable


def assert_notes_satisfy_contract(notes: Iterable[dict[str, Any]]) -> None:
    """Assert notes satisfy the canonical contract.

    Checks that each note has required keys, that numeric fields are
    within bounds, durations are positive, and the list is sorted by
    (start_sec, pitch). Raises AssertionError with a clear message on
    failure; never ValueError.
    """
    notes_list = list(notes)
    # sortedness check needs list; also validates iteration
    for idx, note in enumerate(notes_list):
        assert isinstance(note, dict), f"note {idx} is not a dict: {type(note).__name__}"
        for key in ("pitch", "velocity", "start_sec", "duration_sec"):
            assert key in note, f"note {idx} missing key {key!r}: {note!r}"

        pitch = note["pitch"]
        # pitch must be comparable and in 0..127
        try:
            pitch_ok = 0 <= pitch <= 127  # type: ignore[operator]
        except Exception as exc:
            raise AssertionError(f"note {idx}: pitch {pitch!r} not comparable: {exc}") from exc
        assert pitch_ok, f"note {idx}: pitch {pitch!r} out of range 0..127"

        velocity = note["velocity"]
        try:
            vel_ok = 1 <= velocity <= 127  # type: ignore[operator]
        except Exception as exc:
            raise AssertionError(f"note {idx}: velocity {velocity!r} not comparable: {exc}") from exc
        assert vel_ok, f"note {idx}: velocity {velocity!r} out of range 1..127"

        start = note["start_sec"]
        try:
            start_ok = start >= 0  # type: ignore[operator]
        except Exception as exc:
            raise AssertionError(f"note {idx}: start_sec {start!r} not comparable: {exc}") from exc
        assert start_ok, f"note {idx}: start_sec {start!r} must be >= 0"

        duration = note["duration_sec"]
        try:
            dur_ok = duration > 0  # type: ignore[operator]
        except Exception as exc:
            raise AssertionError(f"note {idx}: duration_sec {duration!r} not comparable: {exc}") from exc
        assert dur_ok, f"note {idx}: duration_sec {duration!r} must be > 0"

    # sorted by (start_sec, pitch); allow equal start with pitch ordering
    for i in range(1, len(notes_list)):
        prev = notes_list[i - 1]
        cur = notes_list[i]
        prev_key = (prev["start_sec"], prev["pitch"])
        cur_key = (cur["start_sec"], cur["pitch"])
        try:
            is_sorted = prev_key <= cur_key  # type: ignore[operator]
        except Exception as exc:
            raise AssertionError(
                f"notes not comparable for sortedness at index {i}: {prev_key!r} vs {cur_key!r}: {exc}"
            ) from exc
        assert is_sorted, (
            f"notes not sorted by (start_sec, pitch) at index {i}: "
            f"prev {prev_key!r} > cur {cur_key!r}"
        )
