import pytest
import pedalboard

from sonitra.config import load_config
from sonitra.effects.builtin_effects import (
    ChorusConfig,
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
    VST3PluginConfig,
)
from sonitra.effects.chain_builder import (
    build_effects_chain,
    build_effects_chain_from_config,
)


# ── Basic construction ───────────────────────────────────────────────

def test_build_empty_chain_returns_pedalboard():
    board = build_effects_chain([])
    assert isinstance(board, pedalboard.Pedalboard)
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
    assert isinstance(board, pedalboard.Pedalboard)
    assert len(board) == 3
