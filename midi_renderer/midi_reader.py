from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import mido

DEFAULT_TEMPO = 500000


def parse_midi(path: Path | str, return_meta: bool = False) -> List[Dict[str, Any]] | Dict[str, Any]:
    midi_path = Path(path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    midi = mido.MidiFile(midi_path)
    notes: List[Dict[str, Any]] = []
    active_notes: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    time_sec = 0.0
    bpm = mido.tempo2bpm(DEFAULT_TEMPO)
    time_signature = (4, 4)

    for message in midi:
        time_sec += float(message.time)
        if message.type == "set_tempo":
            bpm = float(mido.tempo2bpm(message.tempo))
        elif message.type == "time_signature":
            time_signature = (message.numerator, message.denominator)
        elif message.type == "note_on" and message.velocity > 0:
            active_notes.setdefault((message.channel, message.note), []).append((time_sec, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            key = (message.channel, message.note)
            if key not in active_notes or not active_notes[key]:
                continue
            start_sec, velocity = active_notes[key].pop(0)
            duration_sec = max(0.0, time_sec - start_sec)
            if duration_sec <= 0.0:
                continue
            notes.append(
                {
                    "pitch": int(message.note),
                    "velocity": int(velocity),
                    "start_sec": float(start_sec),
                    "duration_sec": float(duration_sec),
                }
            )

    if return_meta:
        return {"notes": notes, "bpm": bpm, "time_signature": time_signature}

    return notes