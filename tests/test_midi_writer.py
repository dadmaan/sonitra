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
