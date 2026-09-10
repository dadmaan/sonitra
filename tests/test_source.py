"""MidiSource program-inheritance tests.

Covers the ``source.py`` slice of the fluidsynth program-support plan:
``MidiSource.load`` derives a per-file program from the ``programs`` key of
the ``parse_midi(..., return_meta=True)`` meta dict and passes it as a
keyword-only ``program=`` argument to ``synth.render``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

from sonitra.source import MidiSource


def _make_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.render_pipeline.bpm = 120.0
    cfg.render_pipeline.duration_padding_sec = 2.0
    cfg.render_pipeline.sample_rate = 44100
    return cfg


def _make_notes() -> List[Dict[str, Any]]:
    return [
        {"pitch": 60, "velocity": 80, "start_sec": 1.0, "duration_sec": 0.5},
    ]


def _make_source(mock_synth: MagicMock) -> MidiSource:
    with patch("sonitra.source.make_synth", return_value=mock_synth):
        return MidiSource(_make_cfg())


def _mock_audio(mock_synth: MagicMock) -> None:
    mock_synth.render.return_value = np.zeros((2, 44100), dtype=np.float32)


def test_single_program_passed_to_render(tmp_path: Path) -> None:
    mock_synth = MagicMock()
    _mock_audio(mock_synth)
    source = _make_source(mock_synth)
    meta = {"notes": _make_notes(), "bpm": 120.0, "programs": [24]}
    path = tmp_path / "single.mid"
    path.touch()
    with patch("sonitra.source.parse_midi", return_value=meta):
        source.load(path)
    assert mock_synth.render.call_args.kwargs["program"] == 24


def test_no_program_passes_none(tmp_path: Path) -> None:
    mock_synth = MagicMock()
    _mock_audio(mock_synth)
    source = _make_source(mock_synth)
    meta = {"notes": _make_notes(), "bpm": 120.0, "programs": []}
    path = tmp_path / "none.mid"
    path.touch()
    with patch("sonitra.source.parse_midi", return_value=meta):
        source.load(path)
    assert mock_synth.render.call_args.kwargs["program"] is None


def test_missing_programs_key_passes_none(tmp_path: Path) -> None:
    """Backward compat: meta dicts without ``programs`` (old midi_reader)."""
    mock_synth = MagicMock()
    _mock_audio(mock_synth)
    source = _make_source(mock_synth)
    meta = {"notes": _make_notes(), "bpm": 120.0}
    path = tmp_path / "legacy.mid"
    path.touch()
    with patch("sonitra.source.parse_midi", return_value=meta):
        source.load(path)
    assert mock_synth.render.call_args.kwargs["program"] is None


def test_multi_program_warns_once(tmp_path: Path, caplog: Any) -> None:
    mock_synth = MagicMock()
    _mock_audio(mock_synth)
    source = _make_source(mock_synth)
    first = tmp_path / "orch_a.mid"
    second = tmp_path / "orch_b.mid"
    first.touch()
    second.touch()
    with caplog.at_level(logging.WARNING):
        with patch(
            "sonitra.source.parse_midi",
            return_value={"notes": _make_notes(), "bpm": 120.0, "programs": [43, 70, 71, 73]},
        ):
            source.load(first)
            source.load(second)
    # Both renders fall back to the SoundFont default.
    assert mock_synth.render.call_args_list[0].kwargs["program"] is None
    assert mock_synth.render.call_args_list[1].kwargs["program"] is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "orch_a.mid" in warnings[0].getMessage()
    assert "fluidsynth.program" in warnings[0].getMessage()


def test_render_call_uses_keyword_only_program(tmp_path: Path) -> None:
    """Positional ``args[0]`` stays the note list (cf. test_bpm_scaling:94)."""
    mock_synth = MagicMock()
    _mock_audio(mock_synth)
    source = _make_source(mock_synth)
    notes = _make_notes()
    meta = {"notes": notes, "bpm": 120.0, "programs": [24]}
    path = tmp_path / "kw.mid"
    path.touch()
    with patch("sonitra.source.parse_midi", return_value=meta):
        source.load(path)
    assert mock_synth.render.call_args.args[0] == notes
    assert len(mock_synth.render.call_args.args) == 1
    assert mock_synth.render.call_args.kwargs["program"] == 24
