from __future__ import annotations

import mido
import pytest
from pathlib import Path

from sonitra.synth.fluid_synth import _write_notes_to_midi


def _parse_onset_sec(midi_path: Path) -> float:
    """Decode MIDI file and return the onset time of the first note_on in seconds."""
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat
    tempo_us = 500_000
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break
    ticks_per_sec = ticks_per_beat * 1_000_000 / tempo_us
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            return abs_tick / ticks_per_sec
    raise AssertionError("No note_on found in MIDI file")


# ── Timing invariant: note onset is always at the correct second ─────────────

def test_bpm_120_note_at_1s_timing_preserved(tmp_path: Path) -> None:
    path = tmp_path / "out.mid"
    _write_notes_to_midi(path,
        [{"pitch": 60, "velocity": 64, "start_sec": 1.0, "duration_sec": 0.5}],
        bpm=120)
    assert abs(_parse_onset_sec(path) - 1.0) < 1e-3


def test_bpm_90_note_at_1s_timing_preserved(tmp_path: Path) -> None:
    path = tmp_path / "out.mid"
    _write_notes_to_midi(path,
        [{"pitch": 60, "velocity": 64, "start_sec": 1.0, "duration_sec": 0.5}],
        bpm=90)
    assert abs(_parse_onset_sec(path) - 1.0) < 1e-3


def test_bpm_200_note_at_2_5s_timing_preserved(tmp_path: Path) -> None:
    path = tmp_path / "out.mid"
    _write_notes_to_midi(path,
        [{"pitch": 60, "velocity": 64, "start_sec": 2.5, "duration_sec": 0.5}],
        bpm=200)
    assert abs(_parse_onset_sec(path) - 2.5) < 1e-3


def test_bpm_60_note_at_0s_timing_preserved(tmp_path: Path) -> None:
    path = tmp_path / "out.mid"
    _write_notes_to_midi(path,
        [{"pitch": 60, "velocity": 64, "start_sec": 0.0, "duration_sec": 0.5}],
        bpm=60)
    assert abs(_parse_onset_sec(path) - 0.0) < 1e-3


# ── Tempo meta message reflects actual BPM ───────────────────────────────────

def test_tempo_meta_matches_bpm_90(tmp_path: Path) -> None:
    path = tmp_path / "out.mid"
    _write_notes_to_midi(path,
        [{"pitch": 60, "velocity": 64, "start_sec": 0.0, "duration_sec": 0.5}],
        bpm=90)
    mid = mido.MidiFile(str(path))
    tempos = [msg.tempo for msg in mid.tracks[0] if msg.type == "set_tempo"]
    assert len(tempos) == 1
    assert abs(tempos[0] - round(60_000_000 / 90)) <= 1


def test_tempo_meta_matches_bpm_120(tmp_path: Path) -> None:
    path = tmp_path / "out.mid"
    _write_notes_to_midi(path,
        [{"pitch": 60, "velocity": 64, "start_sec": 0.0, "duration_sec": 0.5}],
        bpm=120)
    mid = mido.MidiFile(str(path))
    tempos = [msg.tempo for msg in mid.tracks[0] if msg.type == "set_tempo"]
    assert len(tempos) == 1
    assert abs(tempos[0] - 500_000) <= 1   # 60_000_000 / 120 = 500_000
