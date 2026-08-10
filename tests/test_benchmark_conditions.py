from __future__ import annotations

from pathlib import Path

import pytest

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions
from sonitra.config import BenchmarkSection, PipelineConfig, SynthBackend, load_config
from sonitra.effects.builtin_effects import (
    CompressorConfig,
    DistortionConfig,
    HighpassFilterConfig,
    LowpassFilterConfig,
    PeakFilterConfig,
)

_CONFIGS = Path(__file__).parent.parent / "config"


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig.model_validate({
        "pipeline": {
            "synth_backend": "pedalboard_instrument",
            "effects_chain": "pedalboard",
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 2,
            "duration_padding_sec": 1.0,
            "overwrite": True,
            "resume": False,
            "max_workers": 1,
            "log_level": "INFO",
            "bpm": 120,
            **overrides.pop("pipeline", {}),
        },
        "io": {
            "corpus_root": ".",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
        "dawdreamer": {
            "block_size": 512,
            "plugin_path": None,
            "preset_path": None,
            "faust_code": None,
            "clear_midi_between_renders": True,
        },
        "pedalboard": {
            "instrument": {"plugin_path": None, "preset_path": None,
                           "reload_plugin_per_file": False, "silence_flush_sec": 0.5},
            "effects": [
                {"type": "Gain", "gain_db": 0.0},
                {
                    "type": "Reverb",
                    "room_size": 0.4,
                    "damping": 0.5,
                    "wet_level": 0.1,
                    "dry_level": 0.9,
                    "width": 1.0,
                    "freeze_mode": False,
                },
            ],
        },
        **overrides,
    })


def test_expand_includes_baseline_conditions_and_sweeps() -> None:
    section = BenchmarkSection.model_validate(
        {
            "conditions": [{"name": "loud", "overrides": {"pedalboard.effects.0.gain_db": 6.0}}],
            "sweeps": [
                {"parameter": "pedalboard.effects.1.wet_level", "values": [0.2, 0.5], "name": "reverb"}
            ],
        }
    )
    conditions = expand_conditions(section)
    assert [condition.name for condition in conditions] == [
        "baseline",
        "loud",
        "reverb=0.2",
        "reverb=0.5",
    ]
    assert conditions[2].overrides == {"pedalboard.effects.1.wet_level": 0.2}


def test_expand_without_baseline() -> None:
    section = BenchmarkSection.model_validate(
        {"include_baseline": False, "sweeps": [{"parameter": "pipeline.sample_rate", "values": [22050]}]}
    )
    conditions = expand_conditions(section)
    assert [condition.name for condition in conditions] == ["sample_rate=22050"]


def test_expand_rejects_duplicate_names() -> None:
    section = BenchmarkSection.model_validate(
        {"conditions": [{"name": "baseline", "overrides": {}}]}
    )
    with pytest.raises(ValueError, match="Duplicate condition names"):
        expand_conditions(section)


def test_apply_overrides_scalar() -> None:
    updated = apply_overrides(_config(), {"pipeline.sample_rate": 22050})
    assert updated.pipeline.sample_rate == 22050
    # original untouched
    assert _config().pipeline.sample_rate == 44100


def test_apply_overrides_list_index() -> None:
    updated = apply_overrides(_config(), {"pedalboard.effects.1.wet_level": 0.7})
    assert updated.pedalboard.effects[1].wet_level == 0.7
    assert updated.pedalboard.effects[1].dry_level == 0.9


def test_apply_overrides_enum_field() -> None:
    updated = apply_overrides(_config(), {"pipeline.synth_backend": "dawdreamer_faust"})
    assert updated.pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST


def test_apply_overrides_unknown_key_raises() -> None:
    with pytest.raises(KeyError, match="unknown key"):
        apply_overrides(_config(), {"pipeline.nonexistent": 1})


def test_apply_overrides_bad_index_raises() -> None:
    with pytest.raises(KeyError, match="out of range"):
        apply_overrides(_config(), {"pedalboard.effects.9.gain_db": 1.0})


def test_apply_overrides_invalid_value_fails_validation() -> None:
    from sonitra.config import ConfigError

    with pytest.raises(ConfigError):
        apply_overrides(_config(), {"io.output_format": "ogg"})


def test_condition_slug_is_filesystem_safe() -> None:
    condition = Condition(name="reverb wet/dry=0.5")
    assert "/" not in condition.slug
    assert " " not in condition.slug


def test_apply_overrides_triggers_cross_field_validator_on_vst_without_plugin() -> None:
    from sonitra.config import ConfigError
    with pytest.raises(ConfigError, match="plugin_path"):
        apply_overrides(_config(), {"pipeline.synth_backend": "dawdreamer_vst"})


def test_vintage_scenarios_config_expands_and_resolves_all_overrides() -> None:
    cfg = load_config(_CONFIGS / "benchmark" / "old_recording" / "vintage_scenarios.yaml")

    conditions = expand_conditions(cfg.benchmark)
    assert [condition.name for condition in conditions] == [
        "baseline",
        "shellac_bandlimit_mild",
        "shellac_bandlimit_pronounced",
        "tape_bandlimit_mild",
        "tape_bandlimit_pronounced",
        "am_bandlimit_mild",
        "am_bandlimit_pronounced",
    ]

    for condition in conditions:
        apply_overrides(cfg, condition.overrides)

    effects = cfg.pedalboard.effects
    assert isinstance(effects[0], HighpassFilterConfig)
    assert isinstance(effects[2], LowpassFilterConfig)
    assert isinstance(effects[4], PeakFilterConfig)
    assert isinstance(effects[5], PeakFilterConfig)
    assert isinstance(effects[6], DistortionConfig)
    assert isinstance(effects[7], CompressorConfig)


def test_telephone_scenarios_config_expands_and_resolves_all_overrides() -> None:
    cfg = load_config(_CONFIGS / "benchmark" / "telephone_channel" / "telephone_scenarios.yaml")

    conditions = expand_conditions(cfg.benchmark)
    assert [condition.name for condition in conditions] == [
        "baseline",
        "voip_wideband",
        "pstn_narrowband",
        "intercom_lofi",
    ]

    for condition in conditions:
        apply_overrides(cfg, condition.overrides)


def test_venue_scenarios_config_expands_and_resolves_all_overrides() -> None:
    cfg = load_config(_CONFIGS / "benchmark" / "venue_acoustics" / "venue_scenarios.yaml")

    conditions = expand_conditions(cfg.benchmark)
    assert [condition.name for condition in conditions] == [
        "baseline",
        "studio_dry",
        "recital_hall",
        "symphony_hall",
        "cathedral",
    ]

    for condition in conditions:
        apply_overrides(cfg, condition.overrides)


def test_rotary_scenarios_config_expands_and_resolves_all_overrides() -> None:
    cfg = load_config(_CONFIGS / "benchmark" / "rotary_speaker" / "rotary_scenarios.yaml")

    conditions = expand_conditions(cfg.benchmark)
    assert [condition.name for condition in conditions] == [
        "baseline",
        "leslie_chorale",
        "leslie_tremolo",
    ]

    for condition in conditions:
        apply_overrides(cfg, condition.overrides)
