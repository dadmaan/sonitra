import json

import numpy as np
import pytest

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


# ── Clip gate and the output guard ────────────────────────────────────


def test_pre_guard_peak_drives_the_clip_gate():
    written = np.full((1, 44100), 0.9, dtype=np.float32)
    result = check_quality(
        written, 44100, QualityGatesSection(clip_threshold=1.0), peak_before_guard=1.5
    )
    assert result.is_clipped is True
    assert result.passed is False
    # ``peak`` keeps describing the written signal, not the gate's input.
    assert result.peak == pytest.approx(0.9)


def test_engaged_guard_suppresses_the_clip_flag():
    written = np.full((1, 44100), 0.89, dtype=np.float32)
    result = check_quality(
        written,
        44100,
        QualityGatesSection(clip_threshold=1.0),
        peak_before_guard=1.5,
        guard_engaged=True,
    )
    assert result.is_clipped is False
    assert result.passed is True
    assert result.peak == pytest.approx(0.89)


# ── Silence judged on the input ───────────────────────────────────────


def test_input_below_threshold_is_silent_despite_output_level():
    # A post stage lifted this to the configured target, but the source
    # itself was inaudible, so the render may not be scored.
    audio = np.full((1, 44100), 10 ** (-33 / 20), dtype=np.float32)
    result = check_quality(
        audio,
        44100,
        QualityGatesSection(silence_threshold_rms=0.001),
        input_rms=3.0e-5,
        output_level_set=True,
    )
    assert result.is_silent is True
    assert result.passed is False


def test_input_above_threshold_with_low_post_target_is_not_silent():
    # The final level is the quiet configured target, not a dead signal.
    audio = np.full((1, 44100), 10 ** (-63 / 20), dtype=np.float32)
    result = check_quality(
        audio,
        44100,
        QualityGatesSection(silence_threshold_rms=0.001),
        input_rms=0.05,
        output_level_set=True,
    )
    assert result.is_silent is False
    assert result.passed is True


def test_input_above_threshold_with_killed_output_is_silent(dummy_silent_audio):
    # No post stage set the level, so a stage that zeroed the signal is caught.
    result = check_quality(
        dummy_silent_audio,
        44100,
        QualityGatesSection(silence_threshold_rms=0.001),
        input_rms=0.05,
        output_level_set=False,
    )
    assert result.is_silent is True
    assert result.passed is False


def test_input_absent_falls_back_to_the_output_rms(dummy_silent_audio, dummy_audio):
    cfg = QualityGatesSection(silence_threshold_rms=0.001)
    silent = check_quality(dummy_silent_audio, 44100, cfg, input_rms=None)
    assert silent.is_silent is True
    loud = check_quality(dummy_audio, 44100, cfg, input_rms=None)
    assert loud.is_silent is False


def test_input_absent_checks_the_output_even_when_level_set(
    dummy_silent_audio, dummy_audio
):
    # No input level means there is nothing else to go on, so the final RMS
    # decides even when a post stage set the level.
    cfg = QualityGatesSection(silence_threshold_rms=0.001)
    silent = check_quality(
        dummy_silent_audio, 44100, cfg, input_rms=None, output_level_set=True
    )
    assert silent.is_silent is True
    assert silent.passed is False
    loud = check_quality(
        dummy_audio, 44100, cfg, input_rms=None, output_level_set=True
    )
    assert loud.is_silent is False


def test_input_levels_in_to_dict_default_to_none(dummy_audio):
    data = check_quality(dummy_audio, 44100, QualityGatesSection()).to_dict()
    assert data["input_rms"] is None
    assert data["input_peak"] is None


def test_input_levels_in_to_dict_are_recorded(dummy_audio):
    data = check_quality(
        dummy_audio,
        44100,
        QualityGatesSection(),
        input_rms=0.05,
        input_peak=0.2,
    ).to_dict()
    assert data["input_rms"] == pytest.approx(0.05)
    assert data["input_peak"] == pytest.approx(0.2)
    json.dumps(data)
