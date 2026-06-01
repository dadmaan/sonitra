from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from midi_renderer.config import QualityGatesSection


@dataclass
class QualityResult:
    is_silent: bool
    is_clipped: bool
    too_short: bool
    rms: float
    peak: float
    duration_sec: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "is_silent": self.is_silent,
            "is_clipped": self.is_clipped,
            "too_short": self.too_short,
            "rms": self.rms,
            "peak": self.peak,
            "duration_sec": self.duration_sec,
            "passed": self.passed,
        }


def check_quality(audio: np.ndarray, sample_rate: int, cfg: QualityGatesSection) -> QualityResult:
    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.ndim == 1:
        audio_arr = np.expand_dims(audio_arr, axis=0)
    samples = audio_arr.shape[1] if audio_arr.ndim == 2 else 0
    duration_sec = float(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio_arr**2))) if audio_arr.size else 0.0
    peak = float(np.max(np.abs(audio_arr))) if audio_arr.size else 0.0

    is_silent = rms < cfg.silence_threshold_rms
    is_clipped = peak > cfg.clip_threshold
    too_short = duration_sec < cfg.min_duration_sec
    passed = not (is_silent or is_clipped or too_short)

    return QualityResult(
        is_silent=is_silent,
        is_clipped=is_clipped,
        too_short=too_short,
        rms=rms,
        peak=peak,
        duration_sec=duration_sec,
        passed=passed,
    )
