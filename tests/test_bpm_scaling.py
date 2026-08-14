from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pedalboard
import pytest

import sonitra.pipeline as pipeline_module
from sonitra.config import load_config
from sonitra.effects.chain_builder import compute_chain_hash
from sonitra.midi_reader import parse_midi
from sonitra.pipeline import (
    _compute_duration,
    _init_thread_source_chain,
    _render_file,
    _scale_note_timings,
)


def _make_notes() -> List[Dict[str, Any]]:
    return [
        {"pitch": 60, "velocity": 80, "start_sec": 1.0, "duration_sec": 0.5},
        {"pitch": 64, "velocity": 90, "start_sec": 2.0, "duration_sec": 1.0},
    ]


def test_scale_note_timings_doubles_at_half_speed() -> None:
    notes = _make_notes()
    result = _scale_note_timings(notes, 2.0)
    np.testing.assert_allclose(result[0]["start_sec"], 2.0)
    np.testing.assert_allclose(result[0]["duration_sec"], 1.0)
    np.testing.assert_allclose(result[1]["start_sec"], 4.0)
    np.testing.assert_allclose(result[1]["duration_sec"], 2.0)


def test_scale_note_timings_halves_at_double_speed() -> None:
    notes = _make_notes()
    result = _scale_note_timings(notes, 0.5)
    np.testing.assert_allclose(result[0]["start_sec"], 0.5)
    np.testing.assert_allclose(result[0]["duration_sec"], 0.25)
    np.testing.assert_allclose(result[1]["start_sec"], 1.0)
    np.testing.assert_allclose(result[1]["duration_sec"], 0.5)


def test_scale_note_timings_noop_when_equal() -> None:
    notes = _make_notes()
    result = _scale_note_timings(notes, 1.0)
    assert result is notes


def test_scale_note_timings_zero_native_bpm_skips_scaling() -> None:
    notes = _make_notes()
    native_bpm: float = 0.0
    target_bpm: float = 120.0
    if native_bpm > 0:
        notes = _scale_note_timings(notes, native_bpm / target_bpm)
    assert notes[0]["start_sec"] == 1.0
    assert notes[0]["duration_sec"] == 0.5
    assert notes[1]["start_sec"] == 2.0
    assert notes[1]["duration_sec"] == 1.0


def test_parse_midi_returns_initial_bpm(midi_fixture: Any) -> None:
    result = parse_midi(midi_fixture("test_c4.mid"), return_meta=True)
    assert isinstance(result["bpm"], float)
    assert result["bpm"] > 0


def test_render_file_bpm_scaling_integration(
    midi_fixture: Any, config_fixture: Any, tmp_path: Any
) -> None:
    cfg = load_config(config_fixture("config_no_effects.yaml"))
    cfg.render_pipeline.bpm = 60
    cfg.render_pipeline.overwrite = True

    _meta = parse_midi(midi_fixture("test_c4.mid"), return_meta=True)
    raw_notes: List[Dict[str, Any]] = _meta["notes"]
    native_bpm: float = _meta["bpm"]

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
    expected_scale = native_bpm / 60.0
    for raw, captured in zip(raw_notes, captured_notes):
        np.testing.assert_allclose(
            captured["start_sec"], raw["start_sec"] * expected_scale, rtol=1e-5
        )
        np.testing.assert_allclose(
            captured["duration_sec"], raw["duration_sec"] * expected_scale, rtol=1e-5
        )
