import numpy as np
import pytest

from sonitra.midi_reader import parse_midi
from sonitra.synth.pedalboard_synth import PedalboardSynth
from sonitra.synth.protocol import SynthesiserProtocol


# ── Protocol conformance ─────────────────────────────────────────────

def test_pedalboard_synth_implements_synthesiser_protocol():
    synth = PedalboardSynth(sample_rate=44100, channels=2)
    assert isinstance(synth, SynthesiserProtocol)


# ── Faust fallback (no VST needed) ───────────────────────────────────

def test_synth_with_no_plugin_uses_silence_fallback(midi_fixture):
    synth = PedalboardSynth(sample_rate=44100, channels=2, plugin_path=None)
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=2.0)
    assert audio.shape == (2, 44100 * 2)
    assert np.allclose(audio, 0.0)


# ── Instrument rendering (requires VST) ──────────────────────────────

@pytest.mark.skip_if_no_vst

def test_synth_with_vst_produces_audio(vst_path, midi_fixture):
    synth = PedalboardSynth(
        sample_rate=44100,
        channels=2,
        plugin_path=str(vst_path),
        reload_plugin_per_file=False,
    )
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=3.0)
    assert audio.max() > 0.001


@pytest.mark.skip_if_no_vst

def test_midi_messages_converted_correctly(vst_path, midi_fixture):
    synth = PedalboardSynth(sample_rate=44100, channels=2, plugin_path=str(vst_path))
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=3.0)
    rms = np.sqrt(np.mean(audio**2))
    assert rms > 0.001


@pytest.mark.skip_if_no_vst

def test_silence_flush_appended(vst_path, midi_fixture):
    synth = PedalboardSynth(
        sample_rate=44100,
        channels=2,
        plugin_path=str(vst_path),
        silence_flush_sec=1.0,
    )
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=2.0)
    assert audio.shape[1] >= 44100 * 2


@pytest.mark.skip_if_no_vst

def test_reload_plugin_per_file_does_not_bleed_state(vst_path, midi_fixture):
    synth = PedalboardSynth(
        sample_rate=44100,
        channels=2,
        plugin_path=str(vst_path),
        reload_plugin_per_file=True,
    )
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio1 = synth.render(notes, duration_sec=2.0)
    audio2 = synth.render(notes, duration_sec=2.0)
    assert np.allclose(audio1, audio2, atol=1e-4)


# ── Output shape ─────────────────────────────────────────────────────

@pytest.mark.skip_if_no_vst

def test_render_output_is_stereo(vst_path, midi_fixture):
    synth = PedalboardSynth(sample_rate=44100, channels=2, plugin_path=str(vst_path))
    audio = synth.render(parse_midi(midi_fixture("test_c4.mid")), duration_sec=2.0)
    assert audio.ndim == 2
    assert audio.shape[0] == 2


@pytest.mark.skip_if_no_vst

def test_render_duration_close_to_requested(vst_path, midi_fixture):
    synth = PedalboardSynth(sample_rate=44100, channels=2, plugin_path=str(vst_path))
    audio = synth.render(parse_midi(midi_fixture("test_c4.mid")), duration_sec=3.0)
    expected = 44100 * 3
    assert abs(audio.shape[1] - expected) < 1024


def test_pedalboard_synth_accepts_bpm_in_constructor() -> None:
    ps = PedalboardSynth(sample_rate=44100, plugin_path=None, bpm=130)
    assert ps.bpm == 130


def test_pedalboard_synth_bpm_defaults_to_120() -> None:
    ps = PedalboardSynth(sample_rate=44100, plugin_path=None)
    assert ps.bpm == 120
