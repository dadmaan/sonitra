from __future__ import annotations

from pathlib import Path

import pytest

from sonitra.config import (
    ConfigError,
    PipelineConfig,
    RenderingMode,
    default_config_path,
    load_config,
)
from sonitra.pipeline import run_pipeline


def _minimal_config_dict() -> dict:
    return {
        "pipeline": {
            "rendering_mode": "dawdreamer_only",
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
            "midi_dir": ".",
            "output_dir": ".",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
    }


# ── Loading ─────────────────────────────────────────────────────────-

def test_load_valid_yaml_returns_config(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    assert cfg is not None


def test_config_type_is_pipelineconfig(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    assert isinstance(cfg, PipelineConfig)


def test_load_nonexistent_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "ghost.yaml")


def test_load_malformed_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("pipeline: [invalid: yaml: {")
    with pytest.raises(ConfigError):
        load_config(bad)


# ── Rendering mode ───────────────────────────────────────────────────

def test_rendering_mode_parsed_as_enum(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    assert cfg.pipeline.rendering_mode == RenderingMode.DAWDREAMER_SYNTH_PEDALBOARD_FX


def test_invalid_rendering_mode_raises(config_fixture):
    with pytest.raises(ConfigError):
        load_config(config_fixture("config_invalid_mode.yaml"))


def test_all_rendering_modes_valid():
    for mode in RenderingMode:
        cfg = PipelineConfig.model_validate(
            {
                **_minimal_config_dict(),
                "pipeline": {
                    **_minimal_config_dict()["pipeline"],
                    "rendering_mode": mode.value,
                },
            }
        )
        assert cfg.pipeline.rendering_mode == mode


# ── max_workers clamp ────────────────────────────────────────────────

def test_dawdreamer_mode_forces_max_workers_1(config_fixture, caplog):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pipeline.rendering_mode = RenderingMode.DAWDREAMER_ONLY
    cfg.pipeline.max_workers = 8
    validated = cfg.validate_worker_constraint()
    assert validated.pipeline.max_workers == 1
    assert "max_workers forced to 1" in caplog.text


def test_pedalboard_mode_allows_multiple_workers(config_fixture):
    cfg = load_config(config_fixture("config_pedalboard_only.yaml"))
    cfg.pipeline.max_workers = 4
    validated = cfg.validate_worker_constraint()
    assert validated.pipeline.max_workers == 4


# ── Effects list ─────────────────────────────────────────────────────

def test_effects_list_parsed_correctly(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    effects = cfg.pedalboard.effects
    assert len(effects) == 3
    assert effects[0].type == "Compressor"
    assert effects[1].type == "Reverb"
    assert effects[2].type == "Limiter"


def test_unknown_effect_type_raises(tmp_path):
    bad_cfg = tmp_path / "bad_effect.yaml"
    bad_cfg.write_text("pedalboard:\n  effects:\n    - type: DrumMachine\n      enabled: true\n")
    with pytest.raises(ConfigError):
        load_config(bad_cfg)


def test_disabled_effect_preserved_in_config(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pedalboard.effects[0].enabled = False
    assert cfg.pedalboard.effects[0].enabled is False


# ── Output format ────────────────────────────────────────────────────

def test_output_format_wav_valid(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    assert cfg.io.output_format == "wav"


def test_invalid_output_format_raises():
    bad = _minimal_config_dict()
    bad["io"] = {**bad["io"], "output_format": "aiff"}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


# ── Serialisation round-trip ─────────────────────────────────────────

def test_config_serialises_to_dict(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    d = cfg.model_dump()
    assert "pipeline" in d
    assert "pedalboard" in d


def test_config_round_trip_yaml(config_fixture, tmp_path):
    cfg = load_config(config_fixture("config_valid.yaml"))
    out = tmp_path / "round_trip.yaml"
    cfg.save(out)
    cfg2 = load_config(out)
    assert cfg == cfg2


# ── Default config regression ─────────────────────────────────────────

def test_default_config_renders_fixtures(corpus_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(default_config_path())
    result = run_pipeline(sorted(corpus_dir.glob("*.mid")), tmp_path, config=cfg)
    assert result.succeeded >= 2
