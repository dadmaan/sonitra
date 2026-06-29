from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unittest.mock import patch

from sonitra.midi_reader import parse_midi
from sonitra.synth.fluid_synth import FluidSynth
from sonitra.synth.protocol import SynthesiserProtocol

_DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/default-GM.sf2")
_SOUNDFONT_AVAILABLE = _DEFAULT_SOUNDFONT.exists() and _DEFAULT_SOUNDFONT.is_file()


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    """Return the strongest frequency component in the first channel."""
    signal = audio[0] if audio.ndim > 1 else audio
    n = len(signal)
    if n == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return float(freqs[np.argmax(spectrum)])


@pytest.mark.skipif(
    not _SOUNDFONT_AVAILABLE,
    reason="Default system SoundFont is not installed",
)
def test_fluid_synth_implements_protocol() -> None:
    synth = FluidSynth(
        sample_rate=44100,
        channels=2,
        soundfont_path=_DEFAULT_SOUNDFONT,
    )
    assert isinstance(synth, SynthesiserProtocol)


def test_fluid_synth_missing_soundfont_raises() -> None:
    with pytest.raises(FileNotFoundError):
        FluidSynth(
            sample_rate=44100,
            channels=2,
            soundfont_path="/nonexistent/soundfont.sf2",
        )


@pytest.mark.skipif(
    not _SOUNDFONT_AVAILABLE,
    reason="Default system SoundFont is not installed",
)
def test_fluid_synth_renders_c4_with_soundfont(midi_fixture) -> None:
    synth = FluidSynth(
        sample_rate=44100,
        channels=2,
        soundfont_path=_DEFAULT_SOUNDFONT,
    )
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=2.0)

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 2
    assert audio.shape[0] == 2
    assert audio.shape[1] == int(2.0 * 44100)
    assert audio.max() > 0.0

    dominant = _dominant_frequency(audio, sample_rate=44100)
    # SoundFont instruments can have a stronger first harmonic than fundamental,
    # so accept either C4 (~262 Hz) or its octave (~523 Hz).
    assert (
        np.allclose(dominant, 262.0, rtol=0.05)
        or np.allclose(dominant, 523.0, rtol=0.05)
    )


def test_fluid_synth_accepts_bpm_in_constructor(tmp_path: Path) -> None:
    dummy_sf2 = tmp_path / "dummy.sf2"
    dummy_sf2.touch()
    fs = FluidSynth(sample_rate=44100, soundfont_path=dummy_sf2, bpm=90)
    assert fs.bpm == 90


def test_fluid_synth_bpm_defaults_to_120(tmp_path: Path) -> None:
    dummy_sf2 = tmp_path / "dummy.sf2"
    dummy_sf2.touch()
    fs = FluidSynth(sample_rate=44100, soundfont_path=dummy_sf2)
    assert fs.bpm == 120


def test_fluid_synth_render_passes_bpm_to_write_notes(tmp_path: Path) -> None:
    dummy_sf2 = tmp_path / "dummy.sf2"
    dummy_sf2.touch()
    fs = FluidSynth(sample_rate=44100, soundfont_path=dummy_sf2, bpm=100)
    with patch("sonitra.synth.fluid_synth._write_notes_to_midi") as mock_write:
        with patch("sonitra.synth.fluid_synth._run_fluidsynth"):
            with patch("sonitra.synth.fluid_synth.wavfile.read") as mock_read:
                mock_read.return_value = (44100, __import__("numpy").zeros((132300, 2), dtype=np.int16))
                mock_write.return_value = None
                fs.render([], duration_sec=1.0)
    _, call_kwargs = mock_write.call_args
    assert call_kwargs.get("bpm") == 100
