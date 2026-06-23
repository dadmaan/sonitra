from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class NoteEvent:
    """A single note event in absolute time, the unit of all symbolic metrics."""

    pitch: int
    onset_sec: float
    offset_sec: float
    velocity: int = 0

    @property
    def duration_sec(self) -> float:
        return self.offset_sec - self.onset_sec


def notes_from_dicts(notes: Iterable[dict[str, Any]]) -> list[NoteEvent]:
    """Convert midi_reader-style note dicts into sorted NoteEvents."""
    events: list[NoteEvent] = []
    for note in notes:
        onset = float(note["start_sec"])
        events.append(
            NoteEvent(
                pitch=int(note["pitch"]),
                onset_sec=onset,
                offset_sec=onset + float(note["duration_sec"]),
                velocity=int(note.get("velocity", 0)),
            )
        )
    return sorted(events, key=lambda e: (e.onset_sec, e.pitch))


def notes_to_dicts(events: Sequence[NoteEvent]) -> list[dict[str, Any]]:
    return [
        {
            "pitch": event.pitch,
            "velocity": event.velocity,
            "start_sec": event.onset_sec,
            "duration_sec": event.duration_sec,
        }
        for event in events
    ]
