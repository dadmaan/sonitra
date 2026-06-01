from __future__ import annotations

from typing import Iterable

import numpy as np

from sonitra.config import PipelineConfig


def normalise(audio: np.ndarray, *, mode: str, target_db: float) -> np.ndarray:
    audio_arr = np.array(audio, copy=True)
    if audio_arr.size == 0:
        return audio_arr
    mode = mode.lower()
    target_linear = 10 ** (float(target_db) / 20.0)
    if mode == "peak":
        peak = float(np.max(np.abs(audio_arr)))
        if peak <= 0.0:
            return audio_arr
        scale = target_linear / peak
        audio_arr = audio_arr * scale
    elif mode == "rms":
        rms = float(np.sqrt(np.mean(audio_arr**2)))
        if rms <= 0.0:
            return audio_arr
        scale = target_linear / rms
        audio_arr = audio_arr * scale
    else:
        raise ValueError(f"Unsupported normalisation mode: {mode}")
    return np.clip(audio_arr, -1.0, 1.0)


def normalise_from_config(
    audio: np.ndarray,
    cfg: PipelineConfig,
    *,
    stage: str,
    call_log: list[str] | None = None,
) -> np.ndarray:
    if not cfg.normalisation.enabled:
        return audio
    stage = stage.lower()
    should_apply = cfg.normalisation.pre_effects if stage == "pre" else not cfg.normalisation.pre_effects
    if not should_apply:
        return audio
    if call_log is not None:
        call_log.append("normalise")
    return normalise(audio, mode=cfg.normalisation.mode, target_db=cfg.normalisation.target_db)
