from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from sonitra.benchmark import runner as runner_module


class _FakeSynth:
    """Captures notes and duration passed to render; returns silent audio."""

    def __init__(self) -> None:
        self.notes: list[dict] | None = None
        self.duration_sec: float | None = None

    def render(self, notes, duration_sec: float, **kwargs):
        self.notes = list(notes)
        self.duration_sec = float(duration_sec)
        # Return silent audio matching the requested duration at 22050 Hz, mono->stereo shape (channels, samples)
        # The actual sample_rate of reference_audio is independent; we just need a valid array.
        sr = 22050
        samples = int(duration_sec * sr)
        return np.zeros((1, max(0, samples)), dtype=np.float32)


class _FakeDTW:
    name = "dtw"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def compute(self, reference_audio, estimate_audio, sample_rate):
        self.calls.append((reference_audio, estimate_audio, sample_rate))
        return {"distance": 0.0}


def _fake_config(duration_padding_sec: float = 0.5):
    return types.SimpleNamespace(
        render_pipeline=types.SimpleNamespace(duration_padding_sec=duration_padding_sec)
    )


def _patch_audio(monkeypatch: pytest.MonkeyPatch, synth: _FakeSynth) -> None:
    monkeypatch.setattr(runner_module, "make_synth", lambda cfg: synth)
    monkeypatch.setattr(
        runner_module,
        "read_audio",
        lambda path: (np.zeros((1, 22050), dtype=np.float32), 22050),
    )


def test_audio_metric_values_drops_zero_duration_phantom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resynthesis path is pinned through the canonical contract.

    _audio_metric_values is the one metric input that never passed through
    notes_from_dicts. It receives make_note output and is clean,
    but nothing said so. This test pins that a degenerate note is dropped before
    synthesis, not leaked to the synth.

    Without the normalise_notes filter the phantom ``{duration_sec: 0.0}``
    would be forwarded to ``synth.render`` (it would still be silent-dropped
    inside FluidSynth/Pedalboard, but the runner path itself would be
    dishonest: duration would be inflated and the second path would diverge
    from notes_from_dicts). With the filter the phantom never reaches the
    synth and never contributes to the duration calculation.

    Would fail (phantom leaks) if the ``normalise_notes`` call were removed
    from runner.py.
    """
    synth = _FakeSynth()
    _patch_audio(monkeypatch, synth)
    metric = _FakeDTW()
    config = _fake_config(duration_padding_sec=0.5)

    estimate_notes = [
        {"pitch": 60, "velocity": 80, "start_sec": 0.0, "duration_sec": 0.5},
        {"pitch": 60, "velocity": 80, "start_sec": 0.5, "duration_sec": 0.0},  # degenerate phantom
    ]

    runner_module._audio_metric_values(Path("/tmp/fake.wav"), estimate_notes, config, [metric])

    assert synth.notes is not None
    assert len(synth.notes) == 1, "degenerate note should be dropped before synth.render"
    assert synth.notes[0]["pitch"] == 60
    assert synth.notes[0]["duration_sec"] == pytest.approx(0.5)
    # Without filtering the phantom would still be present (len==2). That is the red condition.


def test_audio_metric_values_duration_uses_filtered_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duration for resynthesis is computed from filtered notes only.

    A degenerate note far in the future would inflate the max(start+duration)
    if duration were computed before filtering. Pin that it does not.

    Would fail if duration were computed from raw estimate_notes: the
    degenerate at start 100 s would inflate duration to ~100.5 instead of 1.0.
    """
    synth = _FakeSynth()
    _patch_audio(monkeypatch, synth)
    metric = _FakeDTW()
    config = _fake_config(duration_padding_sec=0.5)

    estimate_notes = [
        {"pitch": 60, "velocity": 80, "start_sec": 0.0, "duration_sec": 0.5},
        {"pitch": 61, "velocity": 80, "start_sec": 100.0, "duration_sec": 0.0},  # degenerate far future
    ]

    runner_module._audio_metric_values(Path("/tmp/fake.wav"), estimate_notes, config, [metric])

    assert synth.notes is not None
    assert len(synth.notes) == 1
    # filtered max is 0.0+0.5 = 0.5; plus padding 0.5 = 1.0
    assert synth.duration_sec == pytest.approx(1.0)
    # unfiltered max would be 100.0 + 0.0 = 100.0 + 0.5 = 100.5
    assert synth.duration_sec != pytest.approx(100.5)


def test_audio_metric_values_clamps_velocity_and_start_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    """normalise_notes clamps velocity/start and sorts, even on the resynthesis path.

    Mirrors the contract in notes.py: velocity 999->127, start -1->0, and the
    output is sorted by (start_sec, pitch) before reaching the synth.
    """
    synth = _FakeSynth()
    _patch_audio(monkeypatch, synth)
    metric = _FakeDTW()
    config = _fake_config(duration_padding_sec=0.0)

    estimate_notes = [
        {"pitch": 64, "velocity": 80, "start_sec": 1.0, "duration_sec": 0.5},
        {"pitch": 60, "velocity": 999, "start_sec": -1.0, "duration_sec": 0.5},  # clamped
        {"pitch": 62, "velocity": 80, "start_sec": 0.5, "duration_sec": 0.0},  # dropped
    ]

    runner_module._audio_metric_values(Path("/tmp/fake.wav"), estimate_notes, config, [metric])

    assert synth.notes is not None
    assert len(synth.notes) == 2
    # Sorted by (start_sec, pitch); the clamped note (start 0.0, pitch 60) comes first
    assert synth.notes[0]["pitch"] == 60
    assert synth.notes[0]["velocity"] == 127
    assert synth.notes[0]["start_sec"] == pytest.approx(0.0)
    assert synth.notes[1]["pitch"] == 64


def test_audio_metric_values_raises_on_bad_pitch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad pitch (0..127 violation) raises ValueError via normalise_notes.

    The resynthesis path must not silently coerce an out-of-range pitch that
    notes_from_dicts would reject.
    """
    synth = _FakeSynth()
    _patch_audio(monkeypatch, synth)
    metric = _FakeDTW()
    config = _fake_config()

    estimate_notes = [
        {"pitch": 200, "velocity": 80, "start_sec": 0.0, "duration_sec": 0.5},
    ]

    with pytest.raises(ValueError, match="pitch"):
        runner_module._audio_metric_values(Path("/tmp/fake.wav"), estimate_notes, config, [metric])


def test_audio_metric_values_all_degenerate_yields_padding_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-degenerate input renders silence of length padding_sec and no phantom notes."""
    synth = _FakeSynth()
    _patch_audio(monkeypatch, synth)
    metric = _FakeDTW()
    config = _fake_config(duration_padding_sec=1.0)

    estimate_notes = [
        {"pitch": 60, "velocity": 80, "start_sec": 0.0, "duration_sec": 0.0},
        {"pitch": 61, "velocity": 80, "start_sec": 1.0, "duration_sec": -0.5},
    ]

    result = runner_module._audio_metric_values(Path("/tmp/fake.wav"), estimate_notes, config, [metric])
    assert synth.notes == []
    assert synth.duration_sec == pytest.approx(1.0)
    assert result == {"dtw.distance": pytest.approx(0.0)}
