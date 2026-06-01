import pytest

from midi_renderer.engine import RendererEngine


def test_engine_initialises():
    eng = RendererEngine(sample_rate=44100, block_size=512)
    assert eng.sample_rate == 44100


def test_engine_rejects_invalid_sample_rate():
    with pytest.raises(ValueError):
        RendererEngine(sample_rate=0, block_size=512)


def test_faust_sine_produces_nonzero_audio(session_engine):
    audio = session_engine.render_faust_sine(freq=440, duration_sec=1.0)
    assert audio.shape[0] == 2
    assert audio.shape[1] > 0
    assert audio.max() > 0.0


def test_render_produces_correct_length(session_engine):
    audio = session_engine.render_faust_sine(freq=440, duration_sec=2.0)
    expected_samples = 44100 * 2
    assert abs(audio.shape[1] - expected_samples) < 512


def test_set_bpm_changes_beat_length(session_engine):
    audio_120 = session_engine.render_faust_sine(freq=440, duration_sec=None, beats=4, bpm=120)
    audio_60 = session_engine.render_faust_sine(freq=440, duration_sec=None, beats=4, bpm=60)
    assert audio_60.shape[1] > audio_120.shape[1]


@pytest.mark.skip_if_no_vst

def test_vst_loads_without_crash(vst_path, session_engine):
    proc = session_engine.load_plugin(vst_path)
    assert proc is not None
