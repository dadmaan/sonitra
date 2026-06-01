import numpy as np
import pytest
from scipy.io import wavfile

from midi_renderer.engine import RendererEngine
from midi_renderer.midi_reader import parse_midi
from midi_renderer.pipeline import run_pipeline
from midi_renderer.renderer import render_notes_vst


@pytest.mark.integration
@pytest.mark.skip_if_no_vst

def test_full_pipeline_with_real_vst(vst_path, corpus_dir, tmp_path):
    midis = list(corpus_dir.glob("*.mid"))[:10]
    result = run_pipeline(midis, tmp_path, engine=RendererEngine(44100, 512), plugin_path=vst_path)
    assert result.failed == 0
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == len(midis)


@pytest.mark.integration
@pytest.mark.skip_if_no_vst

def test_rendered_audio_not_silent(vst_path, midi_fixture, tmp_path):
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, RendererEngine(44100, 512), plugin_path=vst_path)
    sr, data = wavfile.read(list(tmp_path.glob("*.wav"))[0])
    rms = np.sqrt(np.mean(data.astype(float) ** 2))
    assert rms > 0.001


@pytest.mark.integration
@pytest.mark.skip_if_no_vst
@pytest.mark.timeout(30)

def test_engine_does_not_hang_on_reuse(vst_path, session_engine, midi_fixture):
    for _ in range(5):
        notes = parse_midi(midi_fixture("test_c4.mid"))
        render_notes_vst(notes, engine=session_engine, plugin_path=vst_path, duration_sec=2.0)
