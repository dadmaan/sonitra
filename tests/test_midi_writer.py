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


# ── Raw model-output CSV sidecars ─────────────────────────────────────

def test_write_raw_outputs_writes_wide_piano_roll_csv(tmp_path: Path) -> None:
    import csv

    import numpy as np
    from basic_pitch.note_creation import model_frames_to_time

    from sonitra.midi_writer import write_raw_outputs

    raw = {
        "onset": np.zeros((5, 88)),
        "contour": np.zeros((5, 264)),
        "note": np.zeros((5, 88)),
    }
    write_raw_outputs(raw, tmp_path / "stem.mid")

    sidecar = tmp_path / "stem.model_outputs.csv"
    assert sidecar.exists()

    lines = sidecar.read_text().splitlines()
    header = lines[0]
    assert header.startswith("# time_sec,onset_21,")
    assert "contour_bin_000" in header
    assert "contour_bin_263" in header
    assert header.endswith(",note_108")
    assert len(lines) == 6  # header + 5 data rows

    with sidecar.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 6
    assert len(rows[0]) == 441  # 1 time + 88 onset + 264 contour + 88 note

    data_rows = rows[1:]
    for row in data_rows:
        assert len(row) == 441
        # all probability cells are zero, hence exactly "%.6f"-formatted
        for cell in row[1:]:
            assert cell == "0.000000"

    times = np.array([float(row[0]) for row in data_rows])
    np.testing.assert_allclose(
        times, model_frames_to_time(5), rtol=0.0, atol=1e-6
    )


def test_write_raw_outputs_rejects_mismatched_frames(tmp_path: Path) -> None:
    import numpy as np

    from sonitra.midi_writer import write_raw_outputs

    mismatched = {
        "onset": np.zeros((5, 88)),
        "contour": np.zeros((4, 264)),
        "note": np.zeros((5, 88)),
    }
    with pytest.raises(ValueError):
        write_raw_outputs(mismatched, tmp_path / "stem.mid")

    # Companion: differing axis-1 widths (88 / 264 / 88) with a shared frame
    # count is the by-design case and must NOT raise.
    well_formed = {
        "onset": np.zeros((5, 88)),
        "contour": np.zeros((5, 264)),
        "note": np.zeros((5, 88)),
    }
    write_raw_outputs(well_formed, tmp_path / "stem_ok.mid")
    assert (tmp_path / "stem_ok.model_outputs.csv").exists()


def test_write_transcription_outputs_writes_midi_and_sidecar(tmp_path: Path) -> None:
    import numpy as np

    from sonitra.midi_writer import write_transcription_outputs
    from sonitra.transcribe.base import TranscriptionResult

    notes = [{"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0}]
    raw = {
        "onset": np.zeros((2, 88)),
        "contour": np.zeros((2, 264)),
        "note": np.zeros((2, 88)),
    }
    result = TranscriptionResult(notes=notes, transcriber="t", raw_outputs=raw)

    write_transcription_outputs(result, tmp_path / "out.mid")

    assert (tmp_path / "out.mid").exists()
    assert (tmp_path / "out.model_outputs.csv").exists()


def test_write_transcription_outputs_without_raw_outputs(tmp_path: Path) -> None:
    from sonitra.midi_writer import write_transcription_outputs
    from sonitra.transcribe.base import TranscriptionResult

    notes = [{"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0}]
    result = TranscriptionResult(notes=notes, transcriber="t", raw_outputs=None)

    write_transcription_outputs(result, tmp_path / "out.mid")

    assert (tmp_path / "out.mid").exists()
    assert not (tmp_path / "out.model_outputs.csv").exists()


def test_write_transcription_outputs_sidecar_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    import sonitra.midi_writer
    from sonitra.midi_writer import write_transcription_outputs
    from sonitra.transcribe.base import TranscriptionResult

    def _boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        sonitra.midi_writer, "write_raw_outputs", _boom, raising=False
    )

    notes = [{"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0}]
    raw = {
        "onset": np.zeros((2, 88)),
        "contour": np.zeros((2, 264)),
        "note": np.zeros((2, 88)),
    }
    result = TranscriptionResult(notes=notes, transcriber="t", raw_outputs=raw)

    # The sidecar CSV write is isolated; a failure must not propagate.
    write_transcription_outputs(result, tmp_path / "out.mid")

    assert (tmp_path / "out.mid").exists()


# ── GM program selection ──────────────────────────────────────────────

def test_write_midi_golden_sha256(tmp_path: Path) -> None:
    import hashlib

    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0},
        {"pitch": 64, "velocity": 90, "start_sec": 0.5, "duration_sec": 0.5},
        {"pitch": 67, "velocity": 80, "start_sec": 1.0, "duration_sec": 0.25},
    ]
    output = write_midi(notes, tmp_path / "golden.mid", ticks_per_beat=480, tempo_bpm=120.0, program=24)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert digest == "efd17410272557684f3af5416b0fe1715818020fe29d1a5dc95601a0cd11d5a8"


def test_write_multi_program_midi_allocates_channels_ascending_skipping_drum(tmp_path: Path) -> None:
    import mido

    from sonitra.midi_writer import write_multi_program_midi

    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 0.5, "program": 41},
        {"pitch": 62, "velocity": 100, "start_sec": 0.5, "duration_sec": 0.5, "program": 1},
        {"pitch": 64, "velocity": 100, "start_sec": 1.0, "duration_sec": 0.5, "program": 71},
    ]
    out = write_multi_program_midi(notes, tmp_path / "multi.mid", ticks_per_beat=480, tempo_bpm=120.0)
    midi = mido.MidiFile(out)
    # Programs sorted: 1,41,71 -> channels 0,1,2
    prog_changes = [m for m in midi.tracks[0] if m.type == "program_change"]
    assert [(m.program, m.channel) for m in prog_changes] == [(1, 0), (41, 1), (71, 2)]
    # Notes channels follow same mapping
    note_ons = [m for m in midi.tracks[0] if m.type == "note_on"]
    # order by time: 60@1, 62@0, 64@2? Actually times 0.0 prog41 ch1, 0.5 prog1 ch0, 1.0 prog71 ch2
    # Check channels correspond to program mapping
    # We'll map pitch to expected channel: 60->1, 62->0, 64->2
    pitch_to_channel = {60: 1, 62: 0, 64: 2}
    for msg in note_ons:
        assert msg.channel == pitch_to_channel[msg.note]


def test_write_multi_program_midi_never_uses_channel_9(tmp_path: Path) -> None:
    import mido

    from sonitra.midi_writer import write_multi_program_midi

    # 10 distinct programs -> channels 0-8,10 (skip 9)
    notes = [
        {"pitch": 60 + i, "velocity": 100, "start_sec": float(i), "duration_sec": 0.5, "program": i}
        for i in range(10)
    ]
    out = write_multi_program_midi(notes, tmp_path / "multi10.mid")
    midi = mido.MidiFile(out)
    channels_used = {m.channel for m in midi.tracks[0] if m.type in ("note_on", "note_off", "program_change")}
    assert 9 not in channels_used
    # Should use 0-8 and 10
    assert channels_used == {0, 1, 2, 3, 4, 5, 6, 7, 8, 10}


def test_write_multi_program_midi_rejects_more_than_15_programs(tmp_path: Path) -> None:
    from sonitra.midi_writer import write_multi_program_midi

    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 0.5, "program": i}
        for i in range(16)
    ]
    with pytest.raises(ValueError, match="too many programs"):
        write_multi_program_midi(notes, tmp_path / "too_many.mid")


def test_write_multi_program_midi_program_changes_at_time_zero(tmp_path: Path) -> None:
    import mido

    from sonitra.midi_writer import write_multi_program_midi

    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.5, "duration_sec": 0.5, "program": 5},
        {"pitch": 62, "velocity": 100, "start_sec": 1.0, "duration_sec": 0.5, "program": 10},
    ]
    out = write_multi_program_midi(notes, tmp_path / "prog.mid", write_programs=True)
    midi = mido.MidiFile(out)
    prog_msgs = [m for m in midi.tracks[0] if m.type == "program_change"]
    assert len(prog_msgs) == 2
    for msg in prog_msgs:
        assert msg.time == 0
    # when write_programs=False, no program_change
    out2 = write_multi_program_midi(notes, tmp_path / "noprogram.mid", write_programs=False)
    midi2 = mido.MidiFile(out2)
    assert not [m for m in midi2.tracks[0] if m.type == "program_change"]


def test_write_multi_program_midi_round_trip(tmp_path: Path) -> None:
    from sonitra.midi_writer import write_multi_program_midi

    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0, "program": 0},
        {"pitch": 64, "velocity": 90, "start_sec": 0.5, "duration_sec": 0.5, "program": 40},
        {"pitch": 67, "velocity": 80, "start_sec": 0.75, "duration_sec": 0.25, "program": 0},
    ]
    out = write_multi_program_midi(notes, tmp_path / "round.mid", ticks_per_beat=480, tempo_bpm=120.0)
    rebuilt = parse_midi(out)
    # parse_midi merges channels, should recover all notes
    assert len(rebuilt) == len(notes)
    rebuilt_sorted = sorted(rebuilt, key=lambda n: (n["start_sec"], n["pitch"]))
    notes_sorted = sorted(notes, key=lambda n: (n["start_sec"], n["pitch"]))
    for ref, out_note in zip(notes_sorted, rebuilt_sorted):
        assert out_note["pitch"] == ref["pitch"]
        assert out_note["start_sec"] == pytest.approx(ref["start_sec"], abs=0.01)
        assert out_note["duration_sec"] == pytest.approx(ref["duration_sec"], abs=0.02)


def test_no_program_change_by_default(tmp_path: Path) -> None:
    import mido

    notes = [{"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0}]
    output = write_midi(notes, tmp_path / "out.mid")

    midi = mido.MidiFile(output)
    assert not [m for m in midi.tracks[0] if m.type == "program_change"]


def test_program_change_precedes_first_note(tmp_path: Path) -> None:
    import mido

    notes = [
        {"pitch": 60, "velocity": 100, "start_sec": 0.5, "duration_sec": 1.0},
        {"pitch": 64, "velocity": 90, "start_sec": 0.0, "duration_sec": 1.0},
    ]
    output = write_midi(notes, tmp_path / "out.mid", program=24)

    midi = mido.MidiFile(output)
    messages = list(midi.tracks[0])
    program_index = next(
        i for i, m in enumerate(messages) if m.type == "program_change"
    )
    first_note_index = next(i for i, m in enumerate(messages) if m.type == "note_on")
    assert program_index < first_note_index
    program = messages[program_index]
    assert program.program == 24
    assert program.channel == 0
    assert program.time == 0
    # The note stream is unchanged by the added program change.
    rebuilt = parse_midi(output)
    assert sorted(n["pitch"] for n in rebuilt) == [60, 64]
    assert min(n["start_sec"] for n in rebuilt) == pytest.approx(0.0, abs=0.005)


@pytest.mark.parametrize("program", [-1, 128])
def test_out_of_range_program_rejected(tmp_path: Path, program: int) -> None:
    notes = [{"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0}]
    with pytest.raises(ValueError, match="program"):
        write_midi(notes, tmp_path / "out.mid", program=program)
