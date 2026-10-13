from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonitra.config import QualityGatesSection


@dataclass
class QualityResult:
    is_silent: bool
    is_clipped: bool
    too_short: bool
    rms: float
    peak: float
    duration_sec: float
    passed: bool
    duration_expected_sec: float | None = None
    duration_mismatch: bool = False
    input_rms: float | None = None
    input_peak: float | None = None

    def to_dict(self) -> dict:
        return {
            "is_silent": self.is_silent,
            "is_clipped": self.is_clipped,
            "too_short": self.too_short,
            "rms": self.rms,
            "peak": self.peak,
            "duration_sec": self.duration_sec,
            "duration_expected_sec": self.duration_expected_sec,
            "duration_mismatch": self.duration_mismatch,
            "passed": self.passed,
            "input_rms": self.input_rms,
            "input_peak": self.input_peak,
        }


def check_quality(
    audio: np.ndarray,
    sample_rate: int,
    cfg: QualityGatesSection,
    expected_duration_sec: float | None = None,
    *,
    peak_before_guard: float | None = None,
    guard_engaged: bool = False,
    input_rms: float | None = None,
    input_peak: float | None = None,
    output_level_set: bool = False,
) -> QualityResult:
    """Judge the final *audio* against the quality gates.

    *input_rms* is the RMS of the audio as loaded, before any stage ran;
    when it is below ``silence_threshold_rms`` the render is silent even if
    a post stage lifted it to the configured target. The final RMS is only
    used for the silence gate when no post stage set the output level
    (*output_level_set* false): a stage that killed the signal is caught,
    a deliberately quiet target is not. Callers that pass no input level
    keep the output-only check, whatever *output_level_set* says.
    """
    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.ndim == 1:
        audio_arr = np.expand_dims(audio_arr, axis=0)
    samples = audio_arr.shape[1] if audio_arr.ndim == 2 else 0
    duration_sec = float(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio_arr**2))) if audio_arr.size else 0.0
    peak = float(np.max(np.abs(audio_arr))) if audio_arr.size else 0.0

    input_silent = input_rms is not None and input_rms < cfg.silence_threshold_rms
    output_silent = (input_rms is None or not output_level_set) and rms < cfg.silence_threshold_rms
    is_silent = input_silent or output_silent
    for_clip = peak_before_guard if peak_before_guard is not None else peak
    is_clipped = for_clip > cfg.clip_threshold and not guard_engaged
    too_short = duration_sec < cfg.min_duration_sec
    duration_mismatch = (
        expected_duration_sec is not None
        and cfg.max_duration_deviation_sec is not None
        and abs(duration_sec - expected_duration_sec) > cfg.max_duration_deviation_sec
    )
    passed = not (is_silent or is_clipped or too_short or duration_mismatch)

    return QualityResult(
        is_silent=is_silent,
        is_clipped=is_clipped,
        too_short=too_short,
        rms=rms,
        peak=peak,
        duration_sec=duration_sec,
        passed=passed,
        duration_expected_sec=expected_duration_sec,
        duration_mismatch=duration_mismatch,
        input_rms=input_rms,
        input_peak=input_peak,
    )
