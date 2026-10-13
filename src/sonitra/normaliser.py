from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonitra.config import PipelineConfig


def _level(audio: np.ndarray, mode: str) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if mode == "peak":
        return float(np.max(np.abs(arr)))
    if mode == "rms":
        return float(np.sqrt(np.mean(arr**2)))
    raise ValueError(f"Unsupported normalisation mode: {mode}")


def apply_stage(
    audio: np.ndarray, *, mode: str, target_db: float
) -> tuple[np.ndarray, float | None]:
    """Apply one levelling gain to *audio* and return ``(audio, gain_db)``.

    The gain is one scalar for the whole file and the stage never clips:
    a peak-crunching RMS target may push samples above full scale, which the
    effects chain and the output guard handle downstream. Silent input leaves
    the stage a no-op with no gain.
    """
    audio_arr = np.array(audio, copy=True)
    if audio_arr.size == 0:
        return audio_arr, None
    level = _level(audio_arr, mode)
    if level <= 0.0:
        return audio_arr, None
    gain_db = float(target_db) - float(20.0 * np.log10(level))
    scale = 10.0 ** (gain_db / 20.0)
    return audio_arr * scale, gain_db


def apply_guard(
    audio: np.ndarray, *, ceiling_db: float | None
) -> tuple[np.ndarray, float | None]:
    """Scale the whole file down to *ceiling_db* when its peak exceeds it.

    Returns ``(audio, gain_db)``: ``None`` when the ceiling is disabled,
    ``0.0`` when the guard did not engage, and the (negative) applied gain
    when it did. The comparison carries a small tolerance so a float32 peak
    normalised exactly to the ceiling does not engage it.
    """
    audio_arr = np.array(audio, copy=True)
    if ceiling_db is None:
        return audio_arr, None
    if audio_arr.size == 0:
        return audio_arr, 0.0
    peak = float(np.max(np.abs(np.asarray(audio_arr, dtype=np.float64))))
    ceiling_linear = 10.0 ** (float(ceiling_db) / 20.0)
    if peak <= ceiling_linear * (1.0 + 1e-6):
        return audio_arr, 0.0
    gain_db = float(ceiling_db) - float(20.0 * np.log10(peak))
    scale = 10.0 ** (gain_db / 20.0)
    return audio_arr * scale, gain_db


@dataclass
class NormalisationReport:
    pre_gain_db: float | None = None
    post_gain_db: float | None = None
    peak_before_guard: float | None = None
    guard_gain_db: float | None = None
    guard_engaged: bool = False

    def to_dict(self) -> dict:
        return {
            "pre_gain_db": self.pre_gain_db,
            "post_gain_db": self.post_gain_db,
            "peak_before_guard": self.peak_before_guard,
            "guard_gain_db": self.guard_gain_db,
            "guard_engaged": self.guard_engaged,
        }


def normalise_from_config(
    audio: np.ndarray,
    cfg: PipelineConfig,
    *,
    stage: str,
    call_log: list[str] | None = None,
) -> tuple[np.ndarray, NormalisationReport]:
    """Run the *stage* levelling pass selected by the config's placement.

    Returns ``(audio, report)`` where ``audio`` is unchanged when the stage
    does not run and the report carries the stage's gain, ``None`` when the
    stage was skipped.
    """
    report = NormalisationReport()
    section = cfg.normalisation
    if not section.enabled:
        return audio, report
    stage = stage.lower()
    if stage == "pre":
        applies = section.placement in ("pre", "both")
        stage_cfg = section.pre
    elif stage == "post":
        applies = section.placement in ("post", "both")
        stage_cfg = section.post
    else:
        raise ValueError(f"Unsupported normalisation stage: {stage}")
    if not applies:
        return audio, report
    if call_log is not None:
        call_log.append(stage)
    out, gain_db = apply_stage(
        audio, mode=stage_cfg.mode, target_db=stage_cfg.target_db
    )
    if stage == "pre":
        report.pre_gain_db = gain_db
    else:
        report.post_gain_db = gain_db
    return out, report
