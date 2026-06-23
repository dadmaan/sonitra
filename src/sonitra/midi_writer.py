from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import mido

DEFAULT_TICKS_PER_BEAT = 480


def write_midi(
    notes: Iterable[dict[str, Any]],
    path: Path | str,
    *,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    tempo_bpm: float = 120.0,
) -> Path:
    """Write midi_reader-style note dicts to a single-track MIDI file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tempo = mido.bpm2tempo(tempo_bpm)
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[float, int, mido.Message]] = []
    for note in notes:
        pitch = int(note["pitch"])
        velocity = max(1, min(127, int(note.get("velocity", 64))))
        start = max(0.0, float(note["start_sec"]))
        duration = float(note["duration_sec"])
        if duration <= 0.0:
            continue
        # note_off sorts before note_on at the same instant so retriggers of
        # the same pitch survive the round trip through midi_reader
        events.append((start, 1, mido.Message("note_on", note=pitch, velocity=velocity, time=0)))
        events.append((start + duration, 0, mido.Message("note_off", note=pitch, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for time_sec, _, message in events:
        tick = int(round(mido.second2tick(time_sec, ticks_per_beat, tempo)))
        track.append(message.copy(time=max(0, tick - previous_tick)))
        previous_tick = tick

    midi.save(output_path)
    return output_path
