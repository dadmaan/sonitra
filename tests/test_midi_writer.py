from __future__ import annotations

from pathlib import Path

import pytest

from sonitra.midi_reader import parse_midi
from sonitra.midi_writer import write_midi


def test_round_trip_preserves_notes(midi_fixture, tmp_path: Path) -> None:
    original = parse_midi(midi_fixture("test_polyphonic.mid"))
    assert original

    output = write_midi(original, tmp_path / "out.mid")
    rebuilt = parse_midi(output)

    assert len(rebuilt) == len(original)
    original_sorted = sorted(original, key=lambda n: (n["start_sec"], n["pitch"]))
    rebuilt_sorted = sorted(rebuilt, key=lambda n: (n["start_sec"], n["pitch"]))
    for ref, out in zip(original_sorted, rebuilt_sorted):
        assert out["pitch"] == ref["pitch"]
        assert out["velocity"] == ref["velocity"]
        assert out["start_sec"] == pytest.approx(ref["start_sec"], abs=0.005)
        assert out["duration_sec"] == pytest.approx(ref["duration_sec"], abs=0.01)


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    notes = [{"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0}]
    output = write_midi(notes, tmp_path / "nested" / "dir" / "out.mid")
    assert output.exists()


def test_zero_duration_notes_are_dropped(tmp_path: Path) -> None:
    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 0.0},
        {"pitch": 64, "velocity": 100, "start_sec": 0.5, "duration_sec": 0.5},
    ]
    output = write_midi(notes, tmp_path / "out.mid")
    rebuilt = parse_midi(output)
    assert [note["pitch"] for note in rebuilt] == [64]


def test_same_pitch_retrigger_survives_round_trip(tmp_path: Path) -> None:
    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 0.5},
        {"pitch": 60, "velocity": 80, "start_sec": 0.5, "duration_sec": 0.5},
    ]
    output = write_midi(notes, tmp_path / "out.mid")
    rebuilt = parse_midi(output)
    assert len(rebuilt) == 2
