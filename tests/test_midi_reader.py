import pytest

from sonitra.midi_reader import parse_midi


def test_parse_returns_note_list(midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    assert isinstance(notes, list)
    assert len(notes) == 1


def test_note_fields_present(midi_fixture):
    note = parse_midi(midi_fixture("test_c4.mid"))[0]
    assert {"pitch", "velocity", "start_sec", "duration_sec"}.issubset(note.keys())


def test_note_values_in_range(midi_fixture):
    note = parse_midi(midi_fixture("test_c4.mid"))[0]
    assert 0 <= note["pitch"] <= 127
    assert 0 < note["velocity"] <= 127
    assert note["start_sec"] >= 0.0
    assert note["duration_sec"] > 0.0


def test_empty_midi_returns_empty_list(midi_fixture):
    notes = parse_midi(midi_fixture("test_empty.mid"))
    assert notes == []


def test_polyphonic_midi_multiple_notes(midi_fixture):
    notes = parse_midi(midi_fixture("test_polyphonic.mid"))
    assert len(notes) > 1


def test_bpm_extracted(midi_fixture):
    result = parse_midi(midi_fixture("test_c4.mid"), return_meta=True)
    assert result["bpm"] > 0


def test_invalid_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_midi(tmp_path / "ghost.mid")
