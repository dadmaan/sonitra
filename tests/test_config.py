from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from sonitra.config import (
    ConfigError,
    EffectsChain,
    InputType,
    PipelineConfig,
    SynthBackend,
    default_config_path,
    load_config,
)
from sonitra.pipeline import run_pipeline


def _minimal_config_dict() -> dict:
    return {
        "render_pipeline": {
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
    assert cfg.render_pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST


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
                "render_pipeline": {
                    **_minimal_config_dict()["render_pipeline"],
                    "synth_backend": backend.value,
                },
                **extra,
            }
        )
        assert cfg.render_pipeline.synth_backend == backend


# ── max_workers clamp ────────────────────────────────────────────────

def test_dawdreamer_mode_forces_max_workers_1(config_fixture, caplog):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.render_pipeline.synth_backend = SynthBackend.DAWDREAMER_FAUST
    cfg.render_pipeline.max_workers = 8
    validated = cfg.validate_worker_constraint()
    assert validated.render_pipeline.max_workers == 1
    assert "max_workers forced to 1" in caplog.text


def test_pedalboard_mode_allows_multiple_workers(config_fixture):
    cfg = load_config(config_fixture("config_pedalboard_only.yaml"))
    cfg.render_pipeline.max_workers = 4
    validated = cfg.validate_worker_constraint()
    assert validated.render_pipeline.max_workers == 4


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
    assert "render_pipeline" in d
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


# ── Basic Pitch config knobs ─────────────────────────────────────────

def test_basic_pitch_config_new_knob_defaults() -> None:
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    cfg = BasicPitchTranscriberConfig()
    assert cfg.melodia_trick is True
    assert cfg.multiple_pitch_bends is False
    assert cfg.save_raw_outputs is False


def test_basic_pitch_config_accepts_new_knobs() -> None:
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    cfg = BasicPitchTranscriberConfig(
        melodia_trick=False,
        multiple_pitch_bends=True,
        save_raw_outputs=True,
    )
    assert cfg.melodia_trick is False
    assert cfg.multiple_pitch_bends is True
    assert cfg.save_raw_outputs is True


def test_basic_pitch_config_still_forbids_unknown_keys() -> None:
    # ConfigError comes from PipelineConfig.model_validate re-raising pydantic's
    # ValidationError when the basic_pitch transcriber block carries an unknown key.
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    bad = {
        **_minimal_config_dict(),
        "transcription": {
            "transcribers": [
                {"type": "basic_pitch", "bogus_key": 1},
            ]
        },
    }
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


def test_all_runnable_configs_carry_new_basic_pitch_keys() -> None:
    config_dir = Path(__file__).parent.parent / "config"
    runnable = sorted(p for p in config_dir.rglob("*.yaml") if p.name != "source.yaml")
    # Guard that the glob still finds both config trees rather than pinning an
    # exact count, which goes stale every time a preset or study is added.
    found = {p.relative_to(config_dir).parts[0] for p in runnable}
    assert found == {"benchmark", "examples"}, found

    checked = 0
    for path in runnable:
        cfg = load_config(path)
        # TransKun-only presets (tk_*) have no basic_pitch block to check.
        for basic_pitch in (
            t for t in cfg.transcription.transcribers if t.type == "basic_pitch"
        ):
            checked += 1
            assert basic_pitch.melodia_trick is True, path
            assert basic_pitch.multiple_pitch_bends is False, path
            relative = path.relative_to(config_dir)
            if relative.parts[0] == "examples":
                assert basic_pitch.save_raw_outputs is True, path
            else:
                assert basic_pitch.save_raw_outputs is False, path
    # Most presets run basic_pitch; guard against the filter matching nothing.
    assert checked >= len(runnable) // 2, (checked, len(runnable))

    # The annotated reference documents all three knobs at the text level.
    source_text = (config_dir / "source.yaml").read_text()
    for key in ("melodia_trick", "multiple_pitch_bends", "save_raw_outputs"):
        assert key in source_text


def test_transkun_baseline_presets_pin_cpu_and_cuda() -> None:
    config_dir = Path(__file__).parent.parent / "config" / "benchmark"
    cpu_cfg = load_config(config_dir / "transkun_baseline.yaml")
    gpu_cfg = load_config(config_dir / "transkun_baseline_gpu.yaml")

    def _devices(cfg: PipelineConfig) -> dict[str, str]:
        return {t.type: t.device for t in cfg.transcription.transcribers}

    assert _devices(cpu_cfg) == {"basic_pitch": "cpu", "transkun": "cpu"}
    assert _devices(gpu_cfg) == {"basic_pitch": "cuda", "transkun": "cuda"}


def test_transcription_numeric_defaults() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict())
    assert cfg.transcription.numeric_mode == "off"
    assert cfg.transcription.gpu_memory_growth is False


def test_transcription_rejects_unknown_key() -> None:
    bad = {**_minimal_config_dict(), "transcription": {"bogus_key": 1}}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


def test_source_yaml_documents_io_filter_keys_and_no_selection_block() -> None:
    text = default_config_path().read_text()
    cfg = load_config(default_config_path())
    assert cfg.io.metadata_csv is None
    assert cfg.io.where == {}
    assert cfg.io.sample is None
    assert not re.search(r"^selection:", text, re.M)
    for key in ("metadata_csv", "join_column", "where", "sample", "n", "seed"):
        assert re.search(rf"^[#\s]+{key}:", text, re.M), key
    for key in ("metadata_csv", "join_column", "where", "sample"):
        assert re.search(rf"^\s+#?\s*{key}:", text, re.M), key


_PAPER_EXPERIMENTS = (
    Path(__file__).resolve().parent.parent / "config" / "benchmark" / "paper_experiments"
)
# bp_* run basic_pitch, as piano_only.yaml does; tk_* are the TransKun twins.
_MAESTRO_TEST_AUDIO = _PAPER_EXPERIMENTS / "bp_piano_only_maestro_test_audio.yaml"
_MAESTRO_TEST_MIDI = _PAPER_EXPERIMENTS / "bp_piano_only_maestro_test_midi.yaml"
_TK_MAESTRO_TEST_AUDIO = _PAPER_EXPERIMENTS / "tk_piano_only_maestro_test_audio.yaml"
_TK_MAESTRO_TEST_MIDI = _PAPER_EXPERIMENTS / "tk_piano_only_maestro_test_midi.yaml"


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _differing_keys(base: dict[str, Any], other: dict[str, Any]) -> set[str]:
    return {key for key in set(base) | set(other) if base.get(key) != other.get(key)}


def test_maestro_test_presets_select_test_split() -> None:
    cases = (
        (_MAESTRO_TEST_AUDIO, InputType.AUDIO),
        (_MAESTRO_TEST_MIDI, InputType.MIDI),
    )
    for path, expected_input in cases:
        cfg = load_config(path)
        assert cfg.io.dataset == "maestro-v3", path
        assert cfg.io.metadata_csv == "maestro-v3.0.0.csv", path
        assert cfg.io.join_column == "midi_filename", path
        assert cfg.io.where == {"split": ["test"]}, path
        assert cfg.io.sample is None, path
        assert cfg.benchmark.benchmark_dir is None, path
        assert cfg.render_pipeline.input_type == expected_input, path


def test_maestro_test_presets_differ_from_piano_only_only_by_io_filter() -> None:
    base = _flatten(
        load_config(_PAPER_EXPERIMENTS / "piano_only.yaml").model_dump(mode="json")
    )

    audio = _flatten(load_config(_MAESTRO_TEST_AUDIO).model_dump(mode="json"))
    audio_diff = _differing_keys(base, audio)
    assert {
        key for key in audio_diff if not key.startswith("io.where.")
    } == {"io.dataset", "io.metadata_csv"}
    assert "io.where.split" in audio_diff

    midi = _flatten(load_config(_MAESTRO_TEST_MIDI).model_dump(mode="json"))
    midi_diff = _differing_keys(base, midi)
    assert {
        key for key in midi_diff if not key.startswith("io.where.")
    } == {"io.dataset", "io.metadata_csv", "render_pipeline.input_type"}
    assert "io.where.split" in midi_diff


def test_transkun_maestro_test_presets_differ_from_basic_pitch_twin_only_by_transcriber() -> None:
    pairs = (
        (_TK_MAESTRO_TEST_AUDIO, _MAESTRO_TEST_AUDIO),
        (_TK_MAESTRO_TEST_MIDI, _MAESTRO_TEST_MIDI),
    )
    for tk_path, bp_path in pairs:
        tk_cfg = load_config(tk_path)
        assert [t.type for t in tk_cfg.transcription.transcribers] == ["transkun"], tk_path
        assert tk_cfg.io.where == {"split": ["test"]}, tk_path
        assert tk_cfg.io.sample is None, tk_path
        assert tk_cfg.benchmark.benchmark_dir is None, tk_path
        diff = _differing_keys(
            _flatten(load_config(bp_path).model_dump(mode="json")),
            _flatten(tk_cfg.model_dump(mode="json")),
        )
        # Worker count may differ: TransKun and basic_pitch load different
        # models, so GPU memory per worker differs.
        assert diff - {"benchmark.max_workers"} == {"transcription.transcribers"}, tk_path


_TK_MAESTRO_TRAIN_PROBE_AUDIO = (
    _PAPER_EXPERIMENTS / "tk_piano_only_maestro_train_probe_audio.yaml"
)
_TK_MAESTRO_TRAIN_PROBE_MIDI = (
    _PAPER_EXPERIMENTS / "tk_piano_only_maestro_train_probe_midi.yaml"
)


def test_transkun_train_probes_sample_train_split() -> None:
    cases = (
        (_TK_MAESTRO_TRAIN_PROBE_AUDIO, InputType.AUDIO),
        (_TK_MAESTRO_TRAIN_PROBE_MIDI, InputType.MIDI),
    )
    for path, expected_input in cases:
        cfg = load_config(path)
        assert cfg.io.dataset == "maestro-v3", path
        assert cfg.io.metadata_csv == "maestro-v3.0.0.csv", path
        assert cfg.io.join_column == "midi_filename", path
        assert cfg.io.where == {"split": ["train"]}, path
        assert cfg.io.sample is not None, path
        assert cfg.io.sample.n == 177, path
        assert cfg.io.sample.seed == 0, path
        assert cfg.render_pipeline.input_type == expected_input, path
        assert [t.type for t in cfg.transcription.transcribers] == ["transkun"], path


def test_transkun_train_probes_differ_from_test_twin_only_by_io_filter() -> None:
    pairs = (
        (_TK_MAESTRO_TRAIN_PROBE_AUDIO, _TK_MAESTRO_TEST_AUDIO),
        (_TK_MAESTRO_TRAIN_PROBE_MIDI, _TK_MAESTRO_TEST_MIDI),
    )
    for probe_path, twin_path in pairs:
        probe = _flatten(load_config(probe_path).model_dump(mode="json"))
        twin = _flatten(load_config(twin_path).model_dump(mode="json"))
        # Flattened, the sample becomes io.sample.n / io.sample.seed.
        assert _differing_keys(twin, probe) == {
            "io.where.split",
            "io.sample.n",
            "io.sample.seed",
        }, probe_path
