from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import mido
import numpy as np
import pedalboard
import pytest

from sonitra.config import load_config
from sonitra.effects.chain_builder import compute_chain_hash
from sonitra.midi_reader import parse_midi
from sonitra.pipeline import (
    _compute_duration,
    _init_thread_source_chain,
    _render_file,
)


def test_parse_midi_returns_initial_bpm(midi_fixture: Any) -> None:
    result = parse_midi(midi_fixture("test_c4.mid"), return_meta=True)
    assert isinstance(result["bpm"], float)
    assert result["bpm"] > 0


def test_render_file_bpm_scaling_integration(
    midi_fixture: Any, config_fixture: Any, tmp_path: Any
) -> None:
    """bpm=60 plus a 120 BPM file must send unscaled notes to synth.render.

    After the tempo fix render_pipeline.bpm is host tempo only; notes are
    never moved. This is the exact invariant benchmark scores against.
    """
    cfg = load_config(config_fixture("config_no_effects.yaml"))
    cfg.render_pipeline.bpm = 60
    cfg.render_pipeline.overwrite = True

    _meta = parse_midi(midi_fixture("test_c4.mid"), return_meta=True)
    raw_notes: List[Dict[str, Any]] = _meta["notes"]

    mock_synth = MagicMock()
    n_samples = int(cfg.render_pipeline.sample_rate * 1.0)
    mock_synth.render.return_value = np.ones((2, n_samples), dtype=np.float32) * 0.1

    with patch("sonitra.source.make_synth", return_value=mock_synth), patch(
        "sonitra.pipeline.build_effects_chain_from_config",
        return_value=pedalboard.Pedalboard([]),
    ):
        _init_thread_source_chain(cfg)
        chain_hash = compute_chain_hash(cfg.pedalboard.effects)
        _render_file(midi_fixture("test_c4.mid"), tmp_path, cfg, chain_hash, None, None)

    captured_notes: List[Dict[str, Any]] = mock_synth.render.call_args.args[0]
    assert len(captured_notes) == len(raw_notes)
    for raw, captured in zip(raw_notes, captured_notes):
        np.testing.assert_allclose(captured["start_sec"], raw["start_sec"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(captured["duration_sec"], raw["duration_sec"], rtol=1e-9, atol=1e-9)


def test_midi_source_load_is_unscaled_for_multi_tempo(tmp_path: Any, config_fixture: Any) -> None:
    """Multi-tempo MIDI: MidiSource.load passes parse_midi notes unscaled.

    Built with mido: 90 BPM then 150 BPM mid-file. The invariant is
    MidiSource.load notes == parse_midi(path) exactly, regardless of bpm.
    Also checks duration == last_note_end + padding.
    """
    from sonitra.source import MidiSource

    # Build a two-tempo MIDI file in tmp_path
    ticks_per_beat = 480
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    # Initial tempo 90 BPM
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90), time=0))
    # Note 0: starts at tick 0, duration 480 ticks (1 beat at 90 BPM = 0.666s)
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    # Tempo change to 150 BPM at tick 960 (2 beats in)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(150), time=480))
    # Note 1: starts at tick 960, duration 480 ticks (1 beat at 150 BPM = 0.4s)
    track.append(mido.Message("note_on", note=64, velocity=90, time=0))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    mid_path = tmp_path / "multi_tempo.mid"
    midi.save(mid_path)

    cfg = load_config(config_fixture("config_no_effects.yaml"))
    cfg.render_pipeline.bpm = 60  # deliberately off-tempo; must not affect notes
    cfg.render_pipeline.duration_padding_sec = 1.5

    expected_notes = parse_midi(mid_path)
    expected_meta = parse_midi(mid_path, return_meta=True)
    # sanity: file is indeed multi-tempo via parse_midi (follows full tempo map)
    assert len(expected_notes) == 2
    assert expected_meta["bpm"] == pytest.approx(90.0)

    mock_synth = MagicMock()
    # Return dummy audio of correct duration
    expected_duration = _compute_duration(expected_notes, cfg.render_pipeline.duration_padding_sec)
    n_samples = int(cfg.render_pipeline.sample_rate * expected_duration)
    mock_synth.render.return_value = np.zeros((2, n_samples), dtype=np.float32)

    with patch("sonitra.source.make_synth", return_value=mock_synth):
        source = MidiSource(cfg)
        audio, sr = source.load(mid_path)

    assert sr == cfg.render_pipeline.sample_rate
    captured_notes: List[Dict[str, Any]] = mock_synth.render.call_args.args[0]
    captured_kwargs = mock_synth.render.call_args.kwargs
    # Notes must equal parse_midi exactly
    assert len(captured_notes) == len(expected_notes)
    for exp, cap in zip(expected_notes, captured_notes):
        assert exp["pitch"] == cap["pitch"]
        np.testing.assert_allclose(cap["start_sec"], exp["start_sec"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(cap["duration_sec"], exp["duration_sec"], rtol=1e-9, atol=1e-9)
    # Duration must be last note end + padding
    last_end = max(n["start_sec"] + n["duration_sec"] for n in expected_notes)
    assert captured_kwargs["duration_sec"] == pytest.approx(last_end + cfg.render_pipeline.duration_padding_sec)
    # Also direct _compute_duration matches
    assert expected_duration == pytest.approx(last_end + cfg.render_pipeline.duration_padding_sec)
