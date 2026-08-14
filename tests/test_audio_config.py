from __future__ import annotations

import pytest

from sonitra.config import (
    ConfigError,
    InputType,
    PipelineConfig,
    load_config,
)


def _base_dict(synth_backend: str, effects_chain: str, **overrides) -> dict:
    base = {
        "render_pipeline": {
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


def test_input_type_enum_values() -> None:
    assert InputType.MIDI.value == "midi"
    assert InputType.AUDIO.value == "audio"


def test_input_type_defaults_to_midi() -> None:
    data = _base_dict(
        "fluidsynth", "none", fluidsynth={"soundfont_path": "/tmp/dummy.sf2"}
    )
    cfg = PipelineConfig.model_validate(data)
    assert cfg.render_pipeline.input_type == InputType.MIDI


def test_input_type_audio_accepted() -> None:
    data = _base_dict("fluidsynth", "none")
    data["render_pipeline"]["input_type"] = "audio"
    cfg = PipelineConfig.model_validate(data)
    assert cfg.render_pipeline.input_type == InputType.AUDIO


def test_input_type_audio_no_dawdreamer_plugin_required() -> None:
    data = _base_dict("dawdreamer_vst", "none")
    data["render_pipeline"]["input_type"] = "audio"
    cfg = PipelineConfig.model_validate(data)
    assert cfg.render_pipeline.input_type == InputType.AUDIO
    assert cfg.render_pipeline.synth_backend == "dawdreamer_vst"


def test_input_type_midi_still_requires_soundfont() -> None:
    data = _base_dict("fluidsynth", "none")
    data["render_pipeline"]["input_type"] = "midi"
    with pytest.raises(ConfigError, match="soundfont_path"):
        PipelineConfig.model_validate(data)


def test_input_type_midi_still_requires_plugin() -> None:
    data = _base_dict("dawdreamer_vst", "none")
    data["render_pipeline"]["input_type"] = "midi"
    with pytest.raises(ConfigError, match="plugin_path"):
        PipelineConfig.model_validate(data)


def test_input_type_invalid_raises() -> None:
    data = _base_dict("fluidsynth", "none", fluidsynth={"soundfont_path": "/tmp/dummy.sf2"})
    data["render_pipeline"]["input_type"] = "tape"
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)


def test_input_type_unknown_pipeline_key_raises() -> None:
    data = _base_dict("fluidsynth", "none", fluidsynth={"soundfont_path": "/tmp/dummy.sf2"})
    data["render_pipeline"]["input_type_extra"] = "x"
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)


def test_worker_constraint_unaffected_by_input_type() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["render_pipeline"]["input_type"] = "audio"
    cfg = PipelineConfig.model_validate(data)
    cfg.render_pipeline.max_workers = 8
    result = cfg.validate_worker_constraint()
    assert result.render_pipeline.max_workers == 1


def test_input_type_round_trips_through_save_load(tmp_path) -> None:
    data = _base_dict("fluidsynth", "none")
    data["render_pipeline"]["input_type"] = "audio"
    cfg = PipelineConfig.model_validate(data)
    out_path = tmp_path / "roundtrip.yaml"
    cfg.save(out_path)
    reloaded = load_config(out_path)
    assert reloaded.render_pipeline.input_type == InputType.AUDIO


def test_benchmark_sweep_on_input_type_rejected(tmp_path) -> None:
    """A condition/sweep override touching pipeline.input_type must be
    rejected at run_benchmark setup (or apply_overrides), not surfaced as a
    mid-loop ConfigError.
    """
    from sonitra.benchmark.runner import run_benchmark

    data = _base_dict("fluidsynth", "none", fluidsynth={"soundfont_path": "/tmp/dummy.sf2"})
    data["transcription"] = {
        "transcribers": [{"type": "precomputed", "midi_dir": str(tmp_path), "name": "oracle"}]
    }
    data["benchmark"] = {
        "sweeps": [{"parameter": "render_pipeline.input_type", "values": ["audio"]}]
    }
    cfg = PipelineConfig.model_validate(data)

    with pytest.raises(ValueError, match="input_type"):
        run_benchmark([tmp_path / "missing.mid"], tmp_path / "work", cfg)


def test_benchmark_condition_on_input_type_rejected(tmp_path) -> None:
    """Same guard, exercised via an explicit condition override rather than
    a sweep."""
    from sonitra.benchmark.runner import run_benchmark

    data = _base_dict("fluidsynth", "none", fluidsynth={"soundfont_path": "/tmp/dummy.sf2"})
    data["transcription"] = {
        "transcribers": [{"type": "precomputed", "midi_dir": str(tmp_path), "name": "oracle"}]
    }
    data["benchmark"] = {
        "conditions": [
            {"name": "flip", "overrides": {"render_pipeline.input_type": "audio"}}
        ]
    }
    cfg = PipelineConfig.model_validate(data)

    with pytest.raises(ValueError, match="input_type"):
        run_benchmark([tmp_path / "missing.mid"], tmp_path / "work", cfg)
