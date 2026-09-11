"""Detect and remove unison notes.

A unison is a note with the same pitch as, and an onset within 50 ms of, the
first note of its pitch group, regardless of instrument.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Onset tolerance for unison detection (two strings, same pitch). Matches the
#: onset tolerance used by the note metrics, so a counted unison is exactly a
#: pair the evaluator cannot match 1-to-1.
_UNISON_ONSET_TOLERANCE_SEC = 0.05


def count_unisons(
    notes: List[Dict[str, Any]],
    tolerance: float = _UNISON_ONSET_TOLERANCE_SEC,
) -> int:
    """Count notes sharing a pitch within *tolerance* of a group onset.

    Notes are grouped per pitch by ascending onset; the first note opens a
    group and every later note within *tolerance* of the group onset is one
    detected unison. A note further than *tolerance* away opens a new group.
    """
    by_pitch: Dict[int, List[float]] = {}
    for note in notes:
        by_pitch.setdefault(note["pitch"], []).append(note["onset"])
    detected = 0
    for onsets in by_pitch.values():
        onsets.sort()
        group_start: Optional[float] = None
        for onset in onsets:
            if group_start is None or onset - group_start > tolerance:
                group_start = onset
            else:
                detected += 1
    return detected


def dedupe_unisons(
    notes: List[Dict[str, Any]],
    tolerance: float = _UNISON_ONSET_TOLERANCE_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop unison duplicates, keeping the earliest onset per pitch group.

    Grouping is identical to :func:`count_unisons`; the return value is
    ``(kept_notes, n_removed)`` with kept notes re-sorted by onset.
    """
    by_pitch: Dict[int, List[Dict[str, Any]]] = {}
    for note in notes:
        by_pitch.setdefault(note["pitch"], []).append(note)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for group in by_pitch.values():
        group.sort(key=lambda note: note["onset"])
        group_start: Optional[float] = None
        for note in group:
            if group_start is None or note["onset"] - group_start > tolerance:
                group_start = note["onset"]
                kept.append(note)
            else:
                removed += 1
    kept.sort(key=lambda note: (note["onset"], note["pitch"]))
    return (kept, removed)
