from __future__ import annotations

from pathlib import Path

import pytest

from sonitra.config import (
    ConfigError,
    EffectsChain,
    PipelineConfig,
    SynthBackend,
    default_config_path,
    load_config,
)
from sonitra.pipeline import run_pipeline


def _minimal_config_dict() -> dict:
    return {
        "pipeline": {
            "synth_backend": "dawdreamer_faust",
            "effects_chain": "none",
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


# ── Synth backend ───────────────────────────────────────────────────

def test_synth_backend_parsed_as_enum(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    assert cfg.pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST


def test_invalid_synth_backend_raises(config_fixture):
    with pytest.raises(ConfigError):
        load_config(config_fixture("config_invalid_mode.yaml"))


def test_all_synth_backends_valid():
    for backend in SynthBackend:
        extra: dict = {}
        if backend == SynthBackend.FLUIDSYNTH:
            extra = {"fluidsynth": {"soundfont_path": "/tmp/dummy.sf2"}}
        elif backend == SynthBackend.DAWDREAMER_VST:
            extra = {"dawdreamer": {"plugin_path": "/tmp/dummy.vst3"}}
        cfg = PipelineConfig.model_validate(
            {
                **_minimal_config_dict(),
                "pipeline": {
                    **_minimal_config_dict()["pipeline"],
                    "synth_backend": backend.value,
                },
                **extra,
            }
        )
        assert cfg.pipeline.synth_backend == backend


# ── max_workers clamp ────────────────────────────────────────────────

def test_dawdreamer_mode_forces_max_workers_1(config_fixture, caplog):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pipeline.synth_backend = SynthBackend.DAWDREAMER_FAUST
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


def test_unknown_key_in_filter_effect_raises(tmp_path):
    bad_cfg = tmp_path / "bad_filter_effect.yaml"
    bad_cfg.write_text(
        "pedalboard:\n"
        "  effects:\n"
        "    - type: LowpassFilter\n"
        "      cutoff_frequency_hz: 8000.0\n"
        "      resonance: 0.7\n"
        "      enabled: true\n"
    )
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


def test_filter_effects_round_trip_yaml(tmp_path):
    cfg = PipelineConfig.model_validate(
        {
            **_minimal_config_dict(),
            "pedalboard": {
                "effects": [
                    {"type": "HighpassFilter", "cutoff_frequency_hz": 80.0},
                    {"type": "LowpassFilter", "cutoff_frequency_hz": 8000.0},
                    {
                        "type": "HighShelfFilter",
                        "cutoff_frequency_hz": 5000.0,
                        "gain_db": -6.0,
                        "q": 0.7,
                    },
                    {
                        "type": "LowShelfFilter",
                        "cutoff_frequency_hz": 100.0,
                        "gain_db": 3.0,
                        "q": 0.7,
                    },
                    {
                        "type": "PeakFilter",
                        "cutoff_frequency_hz": 2500.0,
                        "gain_db": 3.0,
                        "q": 1.0,
                    },
                ]
            },
        }
    )
    out = tmp_path / "filter_round_trip.yaml"
    cfg.save(out)
    cfg2 = load_config(out)
    assert cfg == cfg2


# ── Default config regression ─────────────────────────────────────────

def test_default_config_renders_fixtures(corpus_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(default_config_path())
    result = run_pipeline(sorted(corpus_dir.glob("*.mid")), tmp_path, config=cfg)
    assert result.succeeded >= 2
