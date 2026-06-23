from __future__ import annotations

import math

import numpy as np
import pytest

from sonitra.evaluation.dtw_metric import DTWAudioMetric, chroma_features, dtw_distance


SAMPLE_RATE = 22050


def _tone(freq: float, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def test_identical_audio_distance_near_zero() -> None:
    audio = _tone(440.0)
    result = DTWAudioMetric().compute(audio, audio, SAMPLE_RATE)
    assert result["distance"] == pytest.approx(0.0, abs=1e-9)


def test_different_pitch_increases_distance() -> None:
    same = DTWAudioMetric().compute(_tone(440.0), _tone(440.0), SAMPLE_RATE)["distance"]
    different = DTWAudioMetric().compute(_tone(440.0), _tone(660.0), SAMPLE_RATE)["distance"]
    assert different > same + 0.5


def test_time_warping_is_forgiven() -> None:
    # the same pitch played longer should align much better than a different pitch
    stretched = DTWAudioMetric().compute(_tone(440.0, 1.0), _tone(440.0, 1.5), SAMPLE_RATE)["distance"]
    different = DTWAudioMetric().compute(_tone(440.0, 1.0), _tone(660.0, 1.0), SAMPLE_RATE)["distance"]
    assert stretched < different


def test_chroma_shape_and_normalisation() -> None:
    features = chroma_features(_tone(440.0), SAMPLE_RATE, frame_size=2048, hop_size=1024)
    assert features.shape[1] == 12
    norms = np.linalg.norm(features, axis=1)
    assert np.all(norms <= 1.0 + 1e-9)
    # A440 energy lands on pitch class 9 (A)
    assert int(np.argmax(features.sum(axis=0))) == 9


def test_chroma_accepts_multichannel() -> None:
    mono = _tone(440.0)
    stereo = np.stack([mono, mono])
    assert chroma_features(stereo, SAMPLE_RATE).shape == chroma_features(mono, SAMPLE_RATE).shape


def test_dtw_distance_empty_features_nan() -> None:
    assert math.isnan(dtw_distance(np.zeros((0, 12)), np.zeros((4, 12))))


def test_max_frames_decimation() -> None:
    metric = DTWAudioMetric(frame_size=1024, hop_size=256, max_frames=10)
    audio = _tone(440.0, duration=2.0)
    assert metric._features(audio, SAMPLE_RATE).shape[0] <= 10
