from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import mido
import pytest

from sonitra.midi_reader import parse_midi


def _write_midi_with_programs(
    path: Path, programs: List[Tuple[int, int]], with_note: bool = True
) -> Path:
    """Build a minimal type-1 MIDI with the given program changes.

    Args:
        path: Destination ``.mid`` file path.
        programs: Sequence of ``(channel, program)`` pairs, each emitted
            as a ``program_change`` message with delta time 0.
        with_note: Whether to append a single middle-C note so the file
            also carries note content.

    Returns:
        The ``path`` passed in, for chaining.
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for channel, program in programs:
        track.append(
            mido.Message(
                "program_change", channel=channel, program=program, time=0
            )
        )
    if with_note:
        track.append(
            mido.Message("note_on", channel=0, note=60, velocity=64, time=0)
        )
        track.append(
            mido.Message("note_off", channel=0, note=60, velocity=64, time=480)
        )
    mid.save(path)
    return path


def test_parse_returns_note_list(midi_fixture: Any) -> None:
    notes = parse_midi(midi_fixture("test_c4.mid"))
    assert isinstance(notes, list)
    assert len(notes) == 1


def test_note_fields_present(midi_fixture: Any) -> None:
    note = parse_midi(midi_fixture("test_c4.mid"))[0]
    assert {"pitch", "velocity", "start_sec", "duration_sec"}.issubset(note.keys())


def test_note_values_in_range(midi_fixture: Any) -> None:
    note = parse_midi(midi_fixture("test_c4.mid"))[0]
    assert 0 <= note["pitch"] <= 127
    assert 0 < note["velocity"] <= 127
    assert note["start_sec"] >= 0.0
    assert note["duration_sec"] > 0.0


def test_empty_midi_returns_empty_list(midi_fixture: Any) -> None:
    notes = parse_midi(midi_fixture("test_empty.mid"))
    assert notes == []


def test_polyphonic_midi_multiple_notes(midi_fixture: Any) -> None:
    notes = parse_midi(midi_fixture("test_polyphonic.mid"))
    assert len(notes) > 1


def test_bpm_extracted(midi_fixture: Any) -> None:
    result = parse_midi(midi_fixture("test_c4.mid"), return_meta=True)
    assert result["bpm"] > 0


def test_invalid_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_midi(tmp_path / "ghost.mid")


def test_programs_absent_returns_empty(tmp_path: Path) -> None:
    path = _write_midi_with_programs(tmp_path / "no_program.mid", [])
    result = parse_midi(path, return_meta=True)
    assert isinstance(result, dict)
    assert result["programs"] == []


def test_programs_single(tmp_path: Path) -> None:
    path = _write_midi_with_programs(tmp_path / "single.mid", [(0, 24)])
    result = parse_midi(path, return_meta=True)
    assert result["programs"] == [24]


def test_programs_repeated_same_deduped(tmp_path: Path) -> None:
    path = _write_midi_with_programs(
        tmp_path / "repeated.mid", [(0, 24), (0, 24)]
    )
    result = parse_midi(path, return_meta=True)
    assert result["programs"] == [24]


def test_programs_multiple_sorted_distinct(tmp_path: Path) -> None:
    path = _write_midi_with_programs(
        tmp_path / "multi.mid", [(0, 73), (1, 43), (2, 71), (3, 70), (0, 70)]
    )
    result = parse_midi(path, return_meta=True)
    assert result["programs"] == [43, 70, 71, 73]


def test_programs_nonzero_channel_collected(tmp_path: Path) -> None:
    path = _write_midi_with_programs(tmp_path / "ch9.mid", [(5, 10)])
    result = parse_midi(path, return_meta=True)
    assert result["programs"] == [10]


def test_return_meta_false_unchanged_with_program_change(tmp_path: Path) -> None:
    path = _write_midi_with_programs(tmp_path / "notes.mid", [(0, 24)])
    notes = parse_midi(path, return_meta=False)
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert set(notes[0].keys()) == {
        "pitch",
        "velocity",
        "start_sec",
        "duration_sec",
    }
