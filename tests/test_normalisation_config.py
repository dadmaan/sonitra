from __future__ import annotations

from typing import Any

import pytest
import yaml

from sonitra.config import ConfigError, PipelineConfig, load_config


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


# ── Defaults ──────────────────────────────────────────────────────────


def test_omitted_section_uses_defaults() -> None:
    cfg = PipelineConfig.model_validate(_base_dict())

    assert cfg.normalisation.enabled is False
    assert cfg.normalisation.placement == "post"
    assert cfg.normalisation.post.mode == "peak"
    assert cfg.normalisation.post.target_db == pytest.approx(-1.0)
    assert cfg.normalisation.peak_ceiling_db == pytest.approx(-1.0)


def test_empty_section_uses_defaults() -> None:
    cfg = PipelineConfig.model_validate(_base_dict(normalisation={}))

    assert cfg.normalisation.enabled is False
    assert cfg.normalisation.placement == "post"
    assert cfg.normalisation.post.mode == "peak"
    assert cfg.normalisation.post.target_db == pytest.approx(-1.0)
    assert cfg.normalisation.peak_ceiling_db == pytest.approx(-1.0)


# ── Stage target defaults by mode ─────────────────────────────────────


def test_rms_stage_target_defaults_to_rms_value() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict(normalisation={"placement": "pre", "pre": {"mode": "rms"}})
    )

    assert cfg.normalisation.pre.target_db == pytest.approx(-33.0)


def test_peak_stage_target_defaults_to_peak_value() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict(normalisation={"placement": "pre", "pre": {"mode": "peak"}})
    )

    assert cfg.normalisation.pre.target_db == pytest.approx(-1.0)


def test_explicit_target_wins_over_mode_default() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict(
            normalisation={
                "placement": "pre",
                "pre": {"mode": "rms", "target_db": -18.0},
            }
        )
    )

    assert cfg.normalisation.pre.target_db == pytest.approx(-18.0)


def test_post_stage_target_defaults_by_mode() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict(normalisation={"post": {"mode": "rms"}})
    )

    assert cfg.normalisation.post.target_db == pytest.approx(-33.0)


# ── Mode validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["RMS ", "loudness"])
def test_unknown_stage_mode_rejected(mode: str) -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(
            _base_dict(normalisation={"placement": "pre", "pre": {"mode": mode}})
        )


# ── Legacy section-level keys ─────────────────────────────────────────

_LEGACY_CASES = [
    (
        {"pre_effects": False},
        (
            "normalisation.pre_effects was replaced by normalisation.placement",
            "false → post, true → pre",
        ),
    ),
    ({"mode": "peak"}, ("normalisation.mode moved under normalisation.<stage>",)),
    (
        {"target_db": -1.0},
        ("normalisation.target_db moved under normalisation.<stage>",),
    ),
]


@pytest.mark.parametrize(("legacy", "fragments"), _LEGACY_CASES)
def test_legacy_section_keys_rejected_by_model_validate(
    legacy: dict[str, Any], fragments: tuple[str, ...]
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        PipelineConfig.model_validate(_base_dict(normalisation=dict(legacy)))

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, message


@pytest.mark.parametrize(("legacy", "fragments"), _LEGACY_CASES)
def test_legacy_section_keys_rejected_by_load_config(
    tmp_path, legacy: dict[str, Any], fragments: tuple[str, ...]
) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(_base_dict(normalisation=dict(legacy))))

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, message


def test_all_legacy_keys_together_get_one_migration_message() -> None:
    legacy = {"pre_effects": True, "mode": "rms", "target_db": -18.0}
    with pytest.raises(ConfigError) as excinfo:
        PipelineConfig.model_validate(_base_dict(normalisation=legacy))

    message = str(excinfo.value)
    assert "normalisation.pre_effects" in message
    assert "normalisation.mode" in message
    assert "normalisation.target_db" in message


# ── Placement / stage consistency ─────────────────────────────────────


def test_explicit_pre_stage_with_post_placement_rejected() -> None:
    with pytest.raises(ConfigError, match="placement"):
        PipelineConfig.model_validate(
            _base_dict(
                normalisation={"placement": "post", "pre": {"mode": "rms"}}
            )
        )


def test_explicit_post_stage_with_pre_placement_rejected() -> None:
    with pytest.raises(ConfigError, match="placement"):
        PipelineConfig.model_validate(
            _base_dict(
                normalisation={"placement": "pre", "post": {"mode": "rms"}}
            )
        )


def test_both_placement_without_stages_is_valid_with_defaults() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict(normalisation={"placement": "both", "enabled": True})
    )

    assert cfg.normalisation.placement == "both"
    assert cfg.normalisation.pre.mode == "peak"
    assert cfg.normalisation.pre.target_db == pytest.approx(-1.0)
    assert cfg.normalisation.post.mode == "peak"
    assert cfg.normalisation.post.target_db == pytest.approx(-1.0)


def test_matching_explicit_stage_and_placement_is_valid() -> None:
    pre = PipelineConfig.model_validate(
        _base_dict(normalisation={"placement": "pre", "pre": {"mode": "rms"}})
    )
    post = PipelineConfig.model_validate(
        _base_dict(normalisation={"placement": "post", "post": {"mode": "rms"}})
    )

    assert pre.normalisation.pre.target_db == pytest.approx(-33.0)
    assert post.normalisation.post.target_db == pytest.approx(-33.0)


# ── Peak ceiling ──────────────────────────────────────────────────────


def test_null_peak_ceiling_is_valid() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict(normalisation={"peak_ceiling_db": None})
    )

    assert cfg.normalisation.peak_ceiling_db is None


def test_positive_peak_ceiling_rejected() -> None:
    with pytest.raises(ConfigError, match="peak_ceiling_db"):
        PipelineConfig.model_validate(
            _base_dict(normalisation={"peak_ceiling_db": 0.5})
        )
