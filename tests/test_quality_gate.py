import json

import numpy as np

from sonitra.config import QualityGatesSection
from sonitra.quality_gate import check_quality


def test_silent_audio_flagged(dummy_silent_audio):
    result = check_quality(dummy_silent_audio, sample_rate=44100, cfg=QualityGatesSection(silence_threshold_rms=0.001))
    assert result.is_silent is True
    assert result.passed is False


def test_normal_audio_not_flagged_as_silent(dummy_audio):
    result = check_quality(dummy_audio, sample_rate=44100, cfg=QualityGatesSection(silence_threshold_rms=0.001))
    assert result.is_silent is False


def test_clipping_detected():
    clipped = [[1.0] * 1000, [1.0] * 1000]
    result = check_quality(clipped, sample_rate=44100, cfg=QualityGatesSection(clip_threshold=0.999))
    assert result.is_clipped is True
    assert result.passed is False


def test_normal_audio_not_clipped(dummy_audio):
    result = check_quality(dummy_audio, sample_rate=44100, cfg=QualityGatesSection(clip_threshold=0.999))
    assert result.is_clipped is False


def test_too_short_audio_flagged():
    short = [[0.0] * 100, [0.0] * 100]
    result = check_quality(short, sample_rate=44100, cfg=QualityGatesSection(min_duration_sec=0.1))
    assert result.too_short is True
    assert result.passed is False


def test_quality_result_contains_rms_and_peak(dummy_audio):
    result = check_quality(dummy_audio, sample_rate=44100, cfg=QualityGatesSection())
    assert result.rms is not None
    assert result.peak is not None
    assert result.duration_sec > 0.0


def test_all_checks_pass_for_good_audio(dummy_audio):
    result = check_quality(
        dummy_audio,
        sample_rate=44100,
        cfg=QualityGatesSection(
            silence_threshold_rms=0.001,
            clip_threshold=0.999,
            min_duration_sec=0.1,
        ),
    )
    assert result.passed is True


def test_quality_result_serialises_to_dict(dummy_audio):
    result = check_quality(dummy_audio, 44100, QualityGatesSection())
    data = result.to_dict()
    assert data["duration_expected_sec"] is None
    assert data["duration_mismatch"] is False
    json.dumps(data)


# ── Planned-duration gate ─────────────────────────────────────────────


def _audio_of_duration(duration_sec: float, sample_rate: int = 100) -> np.ndarray:
    return np.ones((1, int(duration_sec * sample_rate)), dtype=np.float32)


def _duration_cfg(*, tolerance: float | None = 1.0) -> QualityGatesSection:
    return QualityGatesSection(max_duration_deviation_sec=tolerance)


def test_duration_within_tolerance_passes():
    result = check_quality(
        _audio_of_duration(10.9), 100, _duration_cfg(), expected_duration_sec=10.0
    )
    assert result.duration_expected_sec == 10.0
    assert result.duration_mismatch is False
    assert result.passed is True


def test_duration_too_long_fails():
    result = check_quality(
        _audio_of_duration(11.1), 100, _duration_cfg(), expected_duration_sec=10.0
    )
    assert result.duration_mismatch is True
    assert result.passed is False


def test_duration_too_short_fails():
    result = check_quality(
        _audio_of_duration(8.9), 100, _duration_cfg(), expected_duration_sec=10.0
    )
    assert result.duration_mismatch is True
    assert result.passed is False


def test_missing_expected_duration_is_never_a_mismatch():
    result = check_quality(
        _audio_of_duration(100.0), 100, _duration_cfg(), expected_duration_sec=None
    )
    assert result.duration_expected_sec is None
    assert result.duration_mismatch is False
    assert result.passed is True


def test_missing_tolerance_disables_the_duration_gate():
    result = check_quality(
        _audio_of_duration(100.0),
        100,
        _duration_cfg(tolerance=None),
        expected_duration_sec=10.0,
    )
    assert result.duration_mismatch is False
    assert result.passed is True
