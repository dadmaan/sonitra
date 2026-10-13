from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sonitra.config import PipelineConfig
from sonitra.normaliser import (
    NormalisationReport,
    apply_guard,
    apply_stage,
    normalise_from_config,
)


def _base_dict(normalisation: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "render_pipeline": {
            "synth_backend": "dawdreamer_faust",
            "effects_chain": "none",
            "bpm": 120,
            "sample_rate": 44100,
            "bit_depth": 24,
            "channels": 2,
            "duration_padding_sec": 2.0,
            "overwrite": False,
            "resume": True,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": {
            "corpus_root": ".",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
    }
    if normalisation is not None:
        data["normalisation"] = normalisation
    return data


def _config(normalisation: dict[str, Any] | None = None) -> PipelineConfig:
    return PipelineConfig.model_validate(_base_dict(normalisation))


def _crest_signal(crest_db: float = 25.0) -> np.ndarray:
    sample_rate = 44100
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    sine = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    audio = np.stack([sine, sine]).astype(np.float64)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = rms * 10.0 ** (crest_db / 20.0)
    audio[0, 0] = peak
    audio[1, 0] = -peak
    return audio.astype(np.float32)


def _level_db(audio: np.ndarray, mode: str) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if mode == "rms":
        level = float(np.sqrt(np.mean(arr**2)))
    else:
        level = float(np.max(np.abs(arr)))
    return 20.0 * np.log10(level)


# ── apply_stage ───────────────────────────────────────────────────────


def test_apply_stage_rms_reports_gain_and_never_clips() -> None:
    audio = _crest_signal(crest_db=25.0)

    out, gain_db = apply_stage(audio, mode="rms", target_db=-18.0)

    assert gain_db == pytest.approx(-18.0 - _level_db(audio, "rms"), abs=1e-6)
    assert float(np.max(np.abs(out))) > 1.0
    assert _level_db(out, "rms") == pytest.approx(-18.0, abs=1e-5)


def test_apply_stage_peak_reports_gain() -> None:
    audio = _crest_signal(crest_db=25.0)

    out, gain_db = apply_stage(audio, mode="peak", target_db=-1.0)

    assert gain_db == pytest.approx(-1.0 - _level_db(audio, "peak"), abs=1e-6)
    assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-1.0 / 20.0), abs=1e-5)


def test_apply_stage_silent_input_is_unchanged_without_gain() -> None:
    audio = np.zeros((2, 1000), dtype=np.float32)

    out, gain_db = apply_stage(audio, mode="rms", target_db=-18.0)

    assert gain_db is None
    assert np.array_equal(out, audio)
    assert out is not audio


def test_apply_stage_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        apply_stage(np.zeros((1, 10), dtype=np.float32), mode="lufs", target_db=-14.0)


# ── apply_guard ───────────────────────────────────────────────────────


def test_apply_guard_scales_peak_to_ceiling() -> None:
    audio = np.array([[2.0, -1.5, 0.25]], dtype=np.float32)

    out, gain_db = apply_guard(audio, ceiling_db=-1.0)

    assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-1.0 / 20.0), abs=1e-6)
    assert gain_db == pytest.approx(-(20.0 * np.log10(2.0) + 1.0), abs=1e-9)


def test_apply_guard_peak_below_ceiling_is_noop() -> None:
    audio = np.array([[0.5, -0.25, 0.0]], dtype=np.float32)

    out, gain_db = apply_guard(audio, ceiling_db=-1.0)

    assert gain_db == 0.0
    assert np.array_equal(out, audio)


def test_apply_guard_float32_peak_at_ceiling_does_not_engage() -> None:
    audio = np.array([[10 ** (-1.0 / 20.0), -0.5]], dtype=np.float32)

    _out, gain_db = apply_guard(audio, ceiling_db=-1.0)

    assert gain_db == 0.0


def test_apply_guard_none_ceiling_disables_guard() -> None:
    audio = np.array([[2.0, -2.0]], dtype=np.float32)

    out, gain_db = apply_guard(audio, ceiling_db=None)

    assert gain_db is None
    assert np.array_equal(out, audio)


def test_apply_guard_is_a_single_scalar() -> None:
    audio = _crest_signal(crest_db=25.0)

    out, gain_db = apply_guard(audio, ceiling_db=-6.0)

    assert gain_db is not None
    assert gain_db < 0.0
    nonzero = audio != 0
    ratios = out[nonzero] / audio[nonzero]
    assert np.allclose(ratios, ratios[0], rtol=0.0, atol=1e-6)


# ── NormalisationReport ───────────────────────────────────────────────


def test_normalisation_report_to_dict_carries_every_diagnostic_key() -> None:
    report = NormalisationReport()

    data = report.to_dict()

    assert set(data) == {
        "pre_gain_db",
        "post_gain_db",
        "peak_before_guard",
        "guard_gain_db",
        "guard_engaged",
    }
    assert data["pre_gain_db"] is None
    assert data["post_gain_db"] is None
    assert data["peak_before_guard"] is None
    assert data["guard_gain_db"] is None
    assert data["guard_engaged"] is False


# ── normalise_from_config placement dispatch ──────────────────────────


def _stage_call(
    audio: np.ndarray, cfg: PipelineConfig, stage: str
) -> tuple[np.ndarray, NormalisationReport, list[str]]:
    call_log: list[str] = []
    out, report = normalise_from_config(audio, cfg, stage=stage, call_log=call_log)
    return out, report, call_log


def test_normalise_from_config_disabled_skips_both_stages() -> None:
    audio = _crest_signal()
    cfg = _config()

    for stage in ("pre", "post"):
        out, report, call_log = _stage_call(audio, cfg, stage)
        assert call_log == []
        assert np.array_equal(out, audio)
        assert report.pre_gain_db is None
        assert report.post_gain_db is None


def test_normalise_from_config_pre_placement_applies_only_pre() -> None:
    audio = _crest_signal()
    cfg = _config(
        {
            "enabled": True,
            "placement": "pre",
            "pre": {"mode": "rms", "target_db": -18.0},
        }
    )

    pre_out, pre_report, pre_log = _stage_call(audio, cfg, "pre")
    assert pre_log == ["pre"]
    assert pre_report.pre_gain_db == pytest.approx(
        -18.0 - _level_db(audio, "rms"), abs=1e-6
    )
    assert pre_report.post_gain_db is None

    post_out, post_report, post_log = _stage_call(audio, cfg, "post")
    assert post_log == []
    assert np.array_equal(post_out, audio)
    assert post_report.pre_gain_db is None
    assert post_report.post_gain_db is None


def test_normalise_from_config_post_placement_applies_only_post() -> None:
    audio = _crest_signal()
    cfg = _config(
        {
            "enabled": True,
            "placement": "post",
            "post": {"mode": "rms", "target_db": -18.0},
        }
    )

    pre_out, pre_report, pre_log = _stage_call(audio, cfg, "pre")
    assert pre_log == []
    assert np.array_equal(pre_out, audio)
    assert pre_report.pre_gain_db is None

    post_out, post_report, post_log = _stage_call(audio, cfg, "post")
    assert post_log == ["post"]
    assert post_report.post_gain_db == pytest.approx(
        -18.0 - _level_db(audio, "rms"), abs=1e-6
    )
    assert post_report.pre_gain_db is None


def test_normalise_from_config_both_placement_applies_both_stages() -> None:
    audio = _crest_signal()
    cfg = _config(
        {
            "enabled": True,
            "placement": "both",
            "pre": {"mode": "peak", "target_db": -1.0},
            "post": {"mode": "rms", "target_db": -18.0},
        }
    )

    after_pre, pre_report, pre_log = _stage_call(audio, cfg, "pre")
    assert pre_log == ["pre"]
    assert pre_report.pre_gain_db == pytest.approx(
        -1.0 - _level_db(audio, "peak"), abs=1e-6
    )

    after_post, post_report, post_log = _stage_call(after_pre, cfg, "post")
    assert post_log == ["post"]
    assert post_report.post_gain_db == pytest.approx(
        -18.0 - _level_db(after_pre, "rms"), abs=1e-6
    )
    assert post_report.pre_gain_db is None
