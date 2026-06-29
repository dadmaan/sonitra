from __future__ import annotations

import pytest

from sonitra.config import (
    ConfigError,
    EffectsChain,
    FluidSynthSection,
    PipelineConfig,
    SynthBackend,
    load_config,
)


def _base_dict(synth_backend: str, effects_chain: str, **overrides) -> dict:
    base = {
        "pipeline": {
            "synth_backend": synth_backend,
            "effects_chain": effects_chain,
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
    base.update(overrides)
    return base


def test_synth_backend_enum_values() -> None:
    assert SynthBackend.FLUIDSYNTH.value == "fluidsynth"
    assert SynthBackend.DAWDREAMER_FAUST.value == "dawdreamer_faust"
    assert SynthBackend.DAWDREAMER_VST.value == "dawdreamer_vst"
    assert SynthBackend.PEDALBOARD_INSTRUMENT.value == "pedalboard_instrument"

def test_effects_chain_enum_values() -> None:
    assert EffectsChain.NONE.value == "none"
    assert EffectsChain.PEDALBOARD.value == "pedalboard"

def test_all_synth_backends_parse() -> None:
    for backend in SynthBackend:
        extra: dict = {}
        if backend == SynthBackend.FLUIDSYNTH:
            extra = {"fluidsynth": {"soundfont_path": "/tmp/dummy.sf2"}}
        elif backend == SynthBackend.DAWDREAMER_VST:
            extra = {"dawdreamer": {"plugin_path": "/tmp/dummy.vst3"}}
        cfg = PipelineConfig.model_validate(
            _base_dict(backend.value, "none", **extra)
        )
        assert cfg.pipeline.synth_backend == backend

def test_all_effects_chains_parse() -> None:
    for chain in EffectsChain:
        cfg = PipelineConfig.model_validate(
            _base_dict("dawdreamer_faust", chain.value)
        )
        assert cfg.pipeline.effects_chain == chain

def test_invalid_synth_backend_raises() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(_base_dict("invalid_mode", "none"))

def test_fluidsynth_without_soundfont_raises() -> None:
    data = _base_dict("fluidsynth", "none", fluidsynth={"soundfont_path": None})
    with pytest.raises(ConfigError, match="soundfont_path"):
        PipelineConfig.model_validate(data)

def test_fluidsynth_with_soundfont_passes() -> None:
    data = _base_dict("fluidsynth", "none",
                      fluidsynth={"soundfont_path": "/tmp/dummy.sf2"})
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.synth_backend == SynthBackend.FLUIDSYNTH

def test_fluidsynth_without_soundfont_section_raises() -> None:
    data = _base_dict("fluidsynth", "none")
    with pytest.raises(ConfigError, match="soundfont_path"):
        PipelineConfig.model_validate(data)

def test_dawdreamer_vst_without_plugin_path_raises() -> None:
    data = _base_dict("dawdreamer_vst", "none")
    with pytest.raises(ConfigError, match="plugin_path"):
        PipelineConfig.model_validate(data)

def test_dawdreamer_vst_with_plugin_path_passes() -> None:
    data = _base_dict("dawdreamer_vst", "none",
                      dawdreamer={"plugin_path": "/tmp/dummy.vst3"})
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.synth_backend == SynthBackend.DAWDREAMER_VST

def test_dawdreamer_faust_with_plugin_path_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none",
                      dawdreamer={"plugin_path": "/tmp/dummy.vst3"})
    with pytest.raises(ConfigError, match="plugin_path"):
        PipelineConfig.model_validate(data)

def test_dawdreamer_faust_without_plugin_path_passes() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("dawdreamer_faust", "none"))
    assert cfg.pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST

def test_pedalboard_instrument_without_plugin_path_passes() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("pedalboard_instrument", "none"))
    assert cfg.pipeline.synth_backend == SynthBackend.PEDALBOARD_INSTRUMENT

def test_rendering_mode_in_pipeline_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["rendering_mode"] = "dawdreamer_only"
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_dawdreamer_enabled_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["dawdreamer"] = {"enabled": True, "block_size": 512}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_pedalboard_enabled_raises() -> None:
    data = _base_dict("pedalboard_instrument", "pedalboard")
    data["pedalboard"] = {"enabled": True}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_dawdreamer_soundfont_path_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["dawdreamer"] = {"soundfont_path": "/tmp/a.sf2"}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_fluidsynthsection_extra_key_raises() -> None:
    data = _base_dict("fluidsynth", "none",
                      fluidsynth={"soundfont_path": "/tmp/dummy.sf2", "bogus": True})
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_validate_worker_constraint_forces_1_for_dawdreamer_faust() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("dawdreamer_faust", "none"))
    cfg.pipeline.max_workers = 8
    result = cfg.validate_worker_constraint()
    assert result.pipeline.max_workers == 1

def test_validate_worker_constraint_forces_1_for_dawdreamer_vst() -> None:
    data = _base_dict("dawdreamer_vst", "none",
                      dawdreamer={"plugin_path": "/tmp/x.vst3"})
    cfg = PipelineConfig.model_validate(data)
    cfg.pipeline.max_workers = 4
    assert cfg.validate_worker_constraint().pipeline.max_workers == 1

def test_validate_worker_constraint_allows_multiple_for_pedalboard_instrument() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("pedalboard_instrument", "pedalboard"))
    cfg.pipeline.max_workers = 4
    assert cfg.validate_worker_constraint().pipeline.max_workers == 4

def test_validate_worker_constraint_allows_multiple_for_fluidsynth() -> None:
    data = _base_dict("fluidsynth", "none",
                      fluidsynth={"soundfont_path": "/tmp/x.sf2"})
    cfg = PipelineConfig.model_validate(data)
    cfg.pipeline.max_workers = 4
    assert cfg.validate_worker_constraint().pipeline.max_workers == 4

def test_bpm_defaults_to_120() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"].pop("bpm", None)
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.bpm == 120

def test_bpm_zero_rejected() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["bpm"] = 0
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_bpm_negative_rejected() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["bpm"] = -10
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_bpm_1_is_valid() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["bpm"] = 1
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.bpm == 1
