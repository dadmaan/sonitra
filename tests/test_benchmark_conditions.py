from __future__ import annotations

import pytest

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions
from sonitra.config import BenchmarkSection, PipelineConfig, SynthBackend
from sonitra.effects.builtin_effects import (
    CompressorConfig,
    DistortionConfig,
    HighpassFilterConfig,
    LowpassFilterConfig,
    PeakFilterConfig,
)


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig.model_validate({
        "render_pipeline": {
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
            **overrides.pop("render_pipeline", {}),
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
        {"include_baseline": False, "sweeps": [{"parameter": "render_pipeline.sample_rate", "values": [22050]}]}
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
    updated = apply_overrides(_config(), {"render_pipeline.sample_rate": 22050})
    assert updated.render_pipeline.sample_rate == 22050
    # original untouched
    assert _config().render_pipeline.sample_rate == 44100


def test_apply_overrides_list_index() -> None:
    updated = apply_overrides(_config(), {"pedalboard.effects.1.wet_level": 0.7})
    assert updated.pedalboard.effects[1].wet_level == 0.7
    assert updated.pedalboard.effects[1].dry_level == 0.9


def test_apply_overrides_enum_field() -> None:
    updated = apply_overrides(_config(), {"render_pipeline.synth_backend": "dawdreamer_faust"})
    assert updated.render_pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST


def test_apply_overrides_unknown_key_raises() -> None:
    with pytest.raises(KeyError, match="unknown key"):
        apply_overrides(_config(), {"render_pipeline.nonexistent": 1})


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
        apply_overrides(_config(), {"render_pipeline.synth_backend": "dawdreamer_vst"})


def test_condition_overrides_resolve_dotted_paths_to_their_effect_type() -> None:
    effects = [
        {"type": "HighpassFilter", "cutoff_frequency_hz": 80.0},
        {"type": "HighpassFilter", "cutoff_frequency_hz": 120.0},
        {"type": "LowpassFilter", "cutoff_frequency_hz": 9000.0},
        {"type": "LowpassFilter", "cutoff_frequency_hz": 12000.0},
        {"type": "PeakFilter", "cutoff_frequency_hz": 2500.0, "gain_db": 3.0, "q": 1.0},
        {"type": "PeakFilter", "cutoff_frequency_hz": 4000.0, "gain_db": -2.0, "q": 0.7},
        {"type": "Distortion", "drive_db": 6.0},
        {
            "type": "Compressor",
            "threshold_db": -12.0,
            "ratio": 2.0,
            "attack_ms": 5.0,
            "release_ms": 50.0,
        },
    ]
    cfg = _config(pedalboard={"effects": effects})

    section = BenchmarkSection.model_validate(
        {
            "conditions": [
                {
                    "name": "mixed",
                    "overrides": {
                        "pedalboard.effects.0.enabled": False,
                        "pedalboard.effects.2.cutoff_frequency_hz": 6500.0,
                        "pedalboard.effects.4.gain_db": 8.0,
                        "pedalboard.effects.5.q": 2.5,
                        "pedalboard.effects.6.drive_db": 18.0,
                        "pedalboard.effects.7.threshold_db": -24.0,
                    },
                }
            ],
        }
    )
    conditions = expand_conditions(section)
    assert conditions[0].name == cfg.benchmark.baseline_name

    updated = apply_overrides(cfg, conditions[1].overrides)
    resolved = updated.pedalboard.effects

    assert isinstance(resolved[0], HighpassFilterConfig)
    assert isinstance(resolved[1], HighpassFilterConfig)
    assert isinstance(resolved[2], LowpassFilterConfig)
    assert isinstance(resolved[3], LowpassFilterConfig)
    assert isinstance(resolved[4], PeakFilterConfig)
    assert isinstance(resolved[5], PeakFilterConfig)
    assert isinstance(resolved[6], DistortionConfig)
    assert isinstance(resolved[7], CompressorConfig)

    assert resolved[0].enabled is False
    assert resolved[2].cutoff_frequency_hz == 6500.0
    assert resolved[4].gain_db == 8.0
    assert resolved[5].q == 2.5
    assert resolved[6].drive_db == 18.0
    assert resolved[7].threshold_db == -24.0

    # Siblings the condition did not name keep their baseline values.
    assert resolved[0].cutoff_frequency_hz == 80.0
    assert resolved[1].cutoff_frequency_hz == 120.0
    assert resolved[2].enabled is True
    assert resolved[4].cutoff_frequency_hz == 2500.0
    assert resolved[4].q == 1.0
    assert resolved[5].gain_db == -2.0
    assert resolved[6].enabled is True
    assert resolved[7].ratio == 2.0
