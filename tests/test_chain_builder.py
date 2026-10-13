import numpy as np
import pytest
import pedalboard
from pydantic import ValidationError

from sonitra.benchmark.conditions import apply_overrides
from sonitra.config import PedalboardSection, load_config
from sonitra.effects import EffectsChain
from sonitra.effects.builtin_effects import (
    ChorusConfig,
    ClippingConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    GainConfig,
    HighpassFilterConfig,
    HighShelfFilterConfig,
    LimiterConfig,
    LowpassFilterConfig,
    LowShelfFilterConfig,
    PeakFilterConfig,
    ReverbConfig,
    TuningOffsetConfig,
    VST3PluginConfig,
)
from sonitra.effects.chain_builder import (
    TuningOffsetStage,
    build_effects_chain,
    build_effects_chain_from_config,
)
from tests.test_pipeline_audio import _audio_config


# ── Basic construction ───────────────────────────────────────────────

def test_build_empty_chain_returns_empty_effects_chain():
    board = build_effects_chain([])
    assert isinstance(board, EffectsChain)
    assert len(board) == 0


def test_build_compressor_from_config():
    cfg = [CompressorConfig(threshold_db=-18, ratio=4.0, attack_ms=5.0, release_ms=100.0, enabled=True)]
    board = build_effects_chain(cfg)
    assert len(board) == 1
    assert isinstance(board[0], pedalboard.Compressor)


def test_compressor_parameters_applied():
    cfg = [CompressorConfig(threshold_db=-12.0, ratio=8.0, attack_ms=10.0, release_ms=200.0, enabled=True)]
    board = build_effects_chain(cfg)
    comp = board[0]
    assert comp.threshold_db == pytest.approx(-12.0)
    assert comp.ratio == pytest.approx(8.0)


def test_build_reverb_from_config():
    cfg = [
        ReverbConfig(
            room_size=0.6,
            damping=0.3,
            wet_level=0.2,
            dry_level=0.8,
            width=1.0,
            freeze_mode=False,
            enabled=True,
        )
    ]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.Reverb)
    assert board[0].room_size == pytest.approx(0.6)


def test_build_limiter_from_config():
    cfg = [LimiterConfig(threshold_db=-1.0, release_ms=100.0, enabled=True)]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.Limiter)


def test_build_highpass_filter_from_config():
    cfg = [HighpassFilterConfig(cutoff_frequency_hz=80.0, enabled=True)]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.HighpassFilter)
    assert board[0].cutoff_frequency_hz == pytest.approx(80.0)


def test_build_lowpass_filter_from_config():
    cfg = [LowpassFilterConfig(cutoff_frequency_hz=8000.0, enabled=True)]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.LowpassFilter)
    assert board[0].cutoff_frequency_hz == pytest.approx(8000.0)


def test_build_high_shelf_filter_from_config():
    cfg = [
        HighShelfFilterConfig(
            cutoff_frequency_hz=5000.0, gain_db=-6.0, q=0.7, enabled=True
        )
    ]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.HighShelfFilter)
    assert board[0].cutoff_frequency_hz == pytest.approx(5000.0)
    assert board[0].gain_db == pytest.approx(-6.0)
    assert board[0].q == pytest.approx(0.7)


def test_build_low_shelf_filter_from_config():
    cfg = [
        LowShelfFilterConfig(
            cutoff_frequency_hz=100.0, gain_db=3.0, q=0.7, enabled=True
        )
    ]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.LowShelfFilter)
    assert board[0].cutoff_frequency_hz == pytest.approx(100.0)
    assert board[0].gain_db == pytest.approx(3.0)
    assert board[0].q == pytest.approx(0.7)


def test_build_peak_filter_from_config():
    cfg = [
        PeakFilterConfig(cutoff_frequency_hz=2500.0, gain_db=3.0, q=1.0, enabled=True)
    ]
    board = build_effects_chain(cfg)
    assert isinstance(board[0], pedalboard.PeakFilter)
    assert board[0].cutoff_frequency_hz == pytest.approx(2500.0)
    assert board[0].gain_db == pytest.approx(3.0)
    assert board[0].q == pytest.approx(1.0)


def test_cascaded_lowpass_filters_preserve_order_and_identity():
    cfg = [
        LowpassFilterConfig(cutoff_frequency_hz=8000.0, enabled=True),
        LowpassFilterConfig(cutoff_frequency_hz=8000.0, enabled=True),
    ]
    board = build_effects_chain(cfg)
    assert len(board) == 2
    assert isinstance(board[0], pedalboard.LowpassFilter)
    assert isinstance(board[1], pedalboard.LowpassFilter)
    assert board[0].cutoff_frequency_hz == pytest.approx(8000.0)
    assert board[1].cutoff_frequency_hz == pytest.approx(8000.0)
    assert board[0] is not board[1]


def test_build_full_chain_correct_order():
    cfg = [
        CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=100, enabled=True),
        ReverbConfig(
            room_size=0.4,
            damping=0.5,
            wet_level=0.15,
            dry_level=0.85,
            width=1.0,
            freeze_mode=False,
            enabled=True,
        ),
        LimiterConfig(threshold_db=-1.0, release_ms=100.0, enabled=True),
    ]
    board = build_effects_chain(cfg)
    assert len(board) == 3
    assert isinstance(board[0], pedalboard.Compressor)
    assert isinstance(board[1], pedalboard.Reverb)
    assert isinstance(board[2], pedalboard.Limiter)


# ── enabled flag ─────────────────────────────────────────────────────

def test_disabled_effect_excluded_from_chain():
    cfg = [
        CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=100, enabled=False),
        LimiterConfig(threshold_db=-1.0, release_ms=100.0, enabled=True),
    ]
    board = build_effects_chain(cfg)
    assert len(board) == 1
    assert isinstance(board[0], pedalboard.Limiter)


def test_all_effects_disabled_returns_empty_chain():
    cfg = [
        CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=100, enabled=False),
    ]
    board = build_effects_chain(cfg)
    assert len(board) == 0


# ── All built-in types ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "effect_cfg,expected_cls",
    [
        (
            ChorusConfig(
                rate_hz=1.0,
                depth=0.25,
                centre_delay_ms=7.0,
                feedback=0.0,
                mix=0.5,
                enabled=True,
            ),
            pedalboard.Chorus,
        ),
        (DelayConfig(delay_seconds=0.5, feedback=0.0, mix=0.5, enabled=True), pedalboard.Delay),
        (DistortionConfig(drive_db=25.0, enabled=True), pedalboard.Distortion),
        (ClippingConfig(threshold_db=-6.0, enabled=True), pedalboard.Clipping),
        (GainConfig(gain_db=0.0, enabled=True), pedalboard.Gain),
        (
            HighpassFilterConfig(cutoff_frequency_hz=80.0, enabled=True),
            pedalboard.HighpassFilter,
        ),
        (
            LowpassFilterConfig(cutoff_frequency_hz=8000.0, enabled=True),
            pedalboard.LowpassFilter,
        ),
        (
            HighShelfFilterConfig(
                cutoff_frequency_hz=5000.0, gain_db=-6.0, q=0.7, enabled=True
            ),
            pedalboard.HighShelfFilter,
        ),
        (
            LowShelfFilterConfig(
                cutoff_frequency_hz=100.0, gain_db=3.0, q=0.7, enabled=True
            ),
            pedalboard.LowShelfFilter,
        ),
        (
            PeakFilterConfig(
                cutoff_frequency_hz=2500.0, gain_db=3.0, q=1.0, enabled=True
            ),
            pedalboard.PeakFilter,
        ),
        (TuningOffsetConfig(cents=10.0, enabled=True), TuningOffsetStage),
    ],
)
def test_all_builtin_effect_types_instantiate(effect_cfg, expected_cls):
    board = build_effects_chain([effect_cfg])
    assert isinstance(board[0], expected_cls)


# ── VST3 plugin effect ────────────────────────────────────────────────

@pytest.mark.skip_if_no_vst

def test_vst3_effect_loads_into_chain(vst_path):
    cfg = [VST3PluginConfig(plugin_path=str(vst_path), enabled=True)]
    board = build_effects_chain(cfg)
    assert len(board) == 1


def test_vst3_effect_guard_rejects_instrument(vital_vst_path):
    """Loading an instrument VST3 as VST3PluginConfig must raise ValueError."""
    cfg = [VST3PluginConfig(plugin_path=str(vital_vst_path), enabled=True)]
    with pytest.raises(ValueError, match="not an effect"):
        build_effects_chain(cfg)


# ── Factory from full config ─────────────────────────────────────────

def test_build_from_pipeline_config(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    board = build_effects_chain_from_config(cfg)
    assert isinstance(board, EffectsChain)
    assert len(board) == 3


# ── Clipping ─────────────────────────────────────────────────────────

def test_clipping_config_accepts_a_ceiling_at_or_below_full_scale():
    """A ceiling above full scale can never engage, so it must not validate."""
    section = PedalboardSection.model_validate(
        {"effects": [{"type": "Clipping", "threshold_db": -6.0}]}
    )
    assert isinstance(section.effects[0], ClippingConfig)
    assert section.effects[0].threshold_db == pytest.approx(-6.0)

    at_full_scale = PedalboardSection.model_validate(
        {"effects": [{"type": "Clipping", "threshold_db": 0.0}]}
    )
    assert at_full_scale.effects[0].threshold_db == pytest.approx(0.0)

    for above_full_scale in (0.5, 6.0):
        with pytest.raises(ValidationError):
            PedalboardSection.model_validate(
                {"effects": [{"type": "Clipping", "threshold_db": above_full_scale}]}
            )


def test_clipping_config_rejects_unknown_keys():
    """A misspelled knob would otherwise be dropped silently."""
    with pytest.raises(ValidationError):
        PedalboardSection.model_validate(
            {"effects": [{"type": "Clipping", "threshold_db": -6.0, "drive_db": 10.0}]}
        )


def test_build_clipping_from_config():
    board = build_effects_chain([ClippingConfig(threshold_db=-6.0, enabled=True)])
    assert len(board) == 1
    assert isinstance(board[0], pedalboard.Clipping)
    assert board[0].threshold_db == pytest.approx(-6.0)


def test_clipping_clamps_at_the_threshold_and_leaves_quieter_samples_alone():
    sample_rate = 44100
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = np.sin(2.0 * np.pi * 440.0 * samples).astype(np.float32)[None, :]
    ceiling = 10.0 ** (-6.0 / 20.0)

    board = build_effects_chain([ClippingConfig(threshold_db=-6.0)])
    clipped = board(sine, sample_rate)

    assert float(np.max(np.abs(clipped))) == pytest.approx(ceiling, abs=1e-6)
    quiet = np.abs(sine) < ceiling
    np.testing.assert_allclose(clipped[quiet], sine[quiet], atol=1e-7)


def test_disabled_clipping_is_excluded_from_the_chain():
    board = build_effects_chain([ClippingConfig(threshold_db=-6.0, enabled=False)])
    assert len(board) == 0


def test_apply_overrides_sets_a_clipping_threshold(tmp_path):
    """The dotted-path benchmark override path resolves for a Clipping slot."""
    cfg = _audio_config(
        tmp_path,
        effects_chain="pedalboard",
        pedalboard_effects=[{"type": "Clipping", "threshold_db": -6.0}],
    )
    updated = apply_overrides(cfg, {"pedalboard.effects.0.threshold_db": -12.0})
    assert updated.pedalboard.effects[0].threshold_db == pytest.approx(-12.0)
