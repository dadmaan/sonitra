import threading

import numpy as np
import pytest

from sonitra.midi_reader import parse_midi
from sonitra.renderer import render_notes_faust, render_notes_vst


def test_notes_fed_to_faust_processor_produce_audio(session_engine, midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = render_notes_faust(notes, engine=session_engine, duration_sec=3.0)
    assert audio.ndim == 2
    assert audio.shape[1] > 0


def test_empty_notes_produces_silence(session_engine):
    audio = render_notes_faust([], engine=session_engine, duration_sec=2.0)
    assert np.allclose(audio, 0.0, atol=1e-6)


def test_polyphonic_render_no_crash(session_engine, midi_fixture):
    notes = parse_midi(midi_fixture("test_polyphonic.mid"))
    audio = render_notes_faust(notes, engine=session_engine, duration_sec=5.0)
    assert audio.max() > 0.0


def test_audio_normalised_within_clip_range(session_engine, midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = render_notes_faust(notes, engine=session_engine, duration_sec=2.0)
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_clear_midi_between_renders(session_engine, midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    render_notes_faust(notes, engine=session_engine, duration_sec=2.0)
    audio = render_notes_faust([], engine=session_engine, duration_sec=2.0)
    assert np.allclose(audio, 0.0, atol=1e-6)


def test_render_requires_engine_thread(session_engine, midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    errors = []

    def run():
        try:
            render_notes_faust(notes, engine=session_engine, duration_sec=1.0)
        except Exception as exc:  # noqa: BLE001 - test wants any exception
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()

    assert errors
    assert isinstance(errors[0], RuntimeError)


@pytest.mark.skip_if_no_vst

def test_vst_render_produces_nonzero_audio(session_engine, vst_path, midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = render_notes_vst(notes, engine=session_engine, plugin_path=vst_path, duration_sec=3.0)
    assert audio.max() > 0.0


def test_vst_render_follows_midi_pitch(session_engine, vital_vst_path, midi_fixture):
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = render_notes_vst(notes, engine=session_engine, plugin_path=vital_vst_path, duration_sec=2.0)
    assert audio.ndim == 2
    assert audio.shape[1] > 0
    assert audio.max() > 0.0
    dominant = _dominant_frequency(audio, sample_rate=44100)
    np.testing.assert_allclose(dominant, 261.6, rtol=0.05)


def test_vst_render_with_preset_loads_vital_state(session_engine, vital_vst_path, midi_fixture):
    preset = Path("/workspace/preset/vital/Keys - Goodies.vital")
    notes = parse_midi(midi_fixture("test_c4.mid"))

    # Render with the preset and verify we get non-silent audio. The preset
    # timbre may shift the dominant frequency, so we only check amplitude.
    audio = render_notes_vst(
        notes,
        engine=session_engine,
        plugin_path=vital_vst_path,
        preset_path=preset,
        duration_sec=2.0,
    )
    assert audio.ndim == 2
    assert audio.shape[1] > 0
    assert audio.max() > 0.0


def test_vital_preset_changes_rendered_audio(vital_vst_path, midi_fixture):
    # Vital's output is stochastic, so compare long-term amplitude envelopes
    # rather than sample-level correlation. A successfully loaded preset must
    # produce a clearly different RMS than the default init patch.
    preset = Path("/workspace/preset/vital/Keys - Goodies.vital")
    notes = parse_midi(midi_fixture("test_c4.mid"))

    engine = RendererEngine(sample_rate=44100, block_size=512)
    default_audio = render_notes_vst(
        notes, engine=engine, plugin_path=vital_vst_path, duration_sec=2.0
    )
    preset_audio = render_notes_vst(
        notes, engine=engine, plugin_path=vital_vst_path, preset_path=preset, duration_sec=2.0
    )

    default_rms = float(np.sqrt(np.mean(default_audio**2)))
    preset_rms = float(np.sqrt(np.mean(preset_audio**2)))

    assert default_rms > 0.05
    assert preset_rms > 0.01
    assert preset_rms < default_rms * 0.7
