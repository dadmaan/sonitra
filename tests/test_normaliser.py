import numpy as np
import pytest

from sonitra.config import load_config
from sonitra.normaliser import normalise, normalise_from_config


def test_peak_normalise_scales_to_target():
    audio = np.array([[0.5, -0.5, 0.25], [0.3, -0.3, 0.1]], dtype=np.float32)
    out = normalise(audio, mode="peak", target_db=-1.0)
    peak = np.max(np.abs(out))
    expected_linear = 10 ** (-1.0 / 20)
    assert abs(peak - expected_linear) < 1e-5


def test_peak_normalise_silent_audio_unchanged():
    audio = np.zeros((2, 1000), dtype=np.float32)
    out = normalise(audio, mode="peak", target_db=-1.0)
    assert np.allclose(out, 0.0)


def test_rms_normalise_adjusts_energy():
    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.1, 0.1, (2, 44100)).astype(np.float32)
    out = normalise(audio, mode="rms", target_db=-18.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(out**2)) + 1e-9)
    assert abs(rms_db - (-18.0)) < 0.5


def test_normalise_does_not_clip():
    audio = np.array([[2.0, -2.0]], dtype=np.float32)
    out = normalise(audio, mode="peak", target_db=-0.1)
    assert out.max() <= 1.0


def test_normalise_preserves_shape():
    audio = np.random.randn(2, 88200).astype(np.float32)
    out = normalise(audio, mode="peak", target_db=-3.0)
    assert out.shape == audio.shape


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        normalise(np.zeros((2, 100)), mode="lufs", target_db=-14.0)


def test_normalise_respects_pre_effects_flag_ordering(config_fixture, dummy_audio):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.normalisation.pre_effects = True
    order = []
    normalised_audio = normalise_from_config(dummy_audio, cfg, stage="pre", call_log=order)
    assert order == ["normalise"]
    assert normalised_audio.shape == dummy_audio.shape
