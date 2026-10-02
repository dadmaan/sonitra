from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from sonitra.benchmark.conditions import Condition, expand_conditions
from sonitra.config import (
    ConfigError,
    EffectsChain,
    InputType,
    PipelineConfig,
    SynthBackend,
    default_config_path,
    load_config,
)
from sonitra.effects import (
    ChorusConfig,
    DelayConfig,
    DistortionConfig,
    ReverbConfig,
    TuningOffsetConfig,
)
from sonitra.effects.builtin_effects import HighpassFilterConfig, LowpassFilterConfig
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
        # Presets without a basic_pitch transcriber have nothing to check here.
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


def test_benchmark_configs_live_in_subfolders() -> None:
    # Benchmark studies are grouped by kind; only the README sits at the top level.
    config_dir = Path(__file__).parent.parent / "config" / "benchmark"
    assert sorted(p.name for p in config_dir.glob("*.yaml")) == []
    for group in ("smoke", "sweeps", "transkun", "methods"):
        assert sorted(config_dir.glob(f"{group}/*.yaml")), group


def test_transkun_baseline_presets_pin_cpu_and_cuda() -> None:
    config_dir = Path(__file__).parent.parent / "config" / "benchmark"
    cpu_cfg = load_config(config_dir / "transkun" / "transkun_baseline.yaml")
    gpu_cfg = load_config(config_dir / "transkun" / "transkun_baseline_gpu.yaml")

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
# The MAESTRO split presets run basic_pitch (CPU) and TransKun (CUDA) together.
_MAESTRO_TEST_AUDIO = _PAPER_EXPERIMENTS / "piano_only_maestro_test_audio.yaml"
_MAESTRO_TEST_MIDI = _PAPER_EXPERIMENTS / "piano_only_maestro_test_midi.yaml"
_MAESTRO_TRAIN_PROBE_AUDIO = _PAPER_EXPERIMENTS / "piano_only_maestro_train_probe_audio.yaml"
_MAESTRO_TRAIN_PROBE_MIDI = _PAPER_EXPERIMENTS / "piano_only_maestro_train_probe_midi.yaml"
_MAESTRO_SPLIT_PRESETS = (
    _MAESTRO_TEST_AUDIO,
    _MAESTRO_TEST_MIDI,
    _MAESTRO_TRAIN_PROBE_AUDIO,
    _MAESTRO_TRAIN_PROBE_MIDI,
)


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


def test_maestro_split_presets_run_basic_pitch_on_cpu_and_transkun_on_cuda() -> None:
    for path in _MAESTRO_SPLIT_PRESETS:
        cfg = load_config(path)
        transcribers = cfg.transcription.transcribers
        assert [t.type for t in transcribers] == ["basic_pitch", "transkun"], path
        basic_pitch, transkun = transcribers
        assert basic_pitch.device == "cpu", path
        assert basic_pitch.batch_size == 16, path
        assert transkun.device == "cuda", path
        assert cfg.transcription.numeric_mode == "strict", path
        # TensorFlow reserves most GPU memory unless growth is on, which would
        # starve TransKun in the same process even with basic_pitch on CPU.
        assert cfg.transcription.gpu_memory_growth is True, path
        assert cfg.benchmark.benchmark_dir is None, path


def test_maestro_test_presets_differ_from_piano_only_only_by_filter_and_transcription() -> None:
    base_cfg = load_config(_PAPER_EXPERIMENTS / "piano_only.yaml")
    base = _flatten(base_cfg.model_dump(mode="json"))
    shared = {
        "io.dataset",
        "io.metadata_csv",
        "transcription.numeric_mode",
        "transcription.gpu_memory_growth",
        "transcription.transcribers",
    }
    cases = (
        # The audio preset also keeps already-written render files on rerun.
        (_MAESTRO_TEST_AUDIO, {"render_pipeline.resume"}),
        (_MAESTRO_TEST_MIDI, {"render_pipeline.input_type", "benchmark.max_workers"}),
    )
    for path, extra in cases:
        cfg = load_config(path)
        diff = _differing_keys(base, _flatten(cfg.model_dump(mode="json")))
        assert {key for key in diff if not key.startswith("io.where.")} == shared | extra, path
        assert "io.where.split" in diff, path
        # basic_pitch keeps piano_only's settings apart from the device.
        base_bp = base_cfg.transcription.transcribers[0].model_dump(mode="json")
        bp = cfg.transcription.transcribers[0].model_dump(mode="json")
        assert {k for k in base_bp if base_bp[k] != bp[k]} == {"device"}, path


def test_maestro_train_probes_sample_train_split() -> None:
    cases = (
        (_MAESTRO_TRAIN_PROBE_AUDIO, InputType.AUDIO),
        (_MAESTRO_TRAIN_PROBE_MIDI, InputType.MIDI),
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


def test_maestro_train_probes_differ_from_test_twin_only_by_io_filter() -> None:
    pairs = (
        (_MAESTRO_TRAIN_PROBE_AUDIO, _MAESTRO_TEST_AUDIO),
        (_MAESTRO_TRAIN_PROBE_MIDI, _MAESTRO_TEST_MIDI),
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


# ── Global tuning offset ─────────────────────────────────────────────
# The offset leads the chain in two adjacent slots: the instrument's own
# offset, then the return leg the round-trip control alone switches on. Both
# ship disabled so a preset that declares neither stays unprocessed.

_TUNING_KNOB_DEFAULTS = {
    "high_quality": True,
    "transient_mode": "crisp",
    "transient_detector": "compound",
    "retain_phase_continuity": True,
    "use_long_fft_window": None,
    "use_time_domain_smoothing": False,
    "preserve_formants": True,
}

# (name suffix, instrument-slot cents, return-slot cents or None when off)
_TUNING_LEVELS = (
    ("-32c", -32.0, None),
    ("-20c", -20.0, None),
    ("-8c", -8.0, None),
    ("8c", 8.0, None),
    ("12c", 12.0, None),
    ("20c", 20.0, None),
    ("-40c", -40.0, None),
    ("40c", 40.0, None),
    ("rt40c", 40.0, -40.0),
)

_PIANO_CONDITION_PREFIX = "inst=piano_"
_GUITAR_CONDITION_PREFIX = ""

_PAPER_PIANO_CONFIGS = (
    "piano_only.yaml",
    "piano_only_maestro_test_audio.yaml",
    "piano_only_maestro_test_midi.yaml",
    "piano_only_maestro_train_probe_audio.yaml",
    "piano_only_maestro_train_probe_midi.yaml",
)
_PAPER_GUITAR_CONFIGS = ("guitar_only.yaml",)
_PAPER_CONFIGS = _PAPER_PIANO_CONFIGS + _PAPER_GUITAR_CONFIGS

_PAPER_LAYOUT = {
    **{
        name: (
            TuningOffsetConfig,
            TuningOffsetConfig,
            ReverbConfig,
            ChorusConfig,
            DistortionConfig,
        )
        for name in _PAPER_PIANO_CONFIGS
    },
    "guitar_only.yaml": (
        TuningOffsetConfig,
        TuningOffsetConfig,
        DistortionConfig,
        HighpassFilterConfig,
        LowpassFilterConfig,
        LowpassFilterConfig,
        DelayConfig,
        ReverbConfig,
    ),
}

_SMOKE_DIR = Path(__file__).resolve().parent.parent / "config" / "benchmark" / "smoke"
_TUNING_SMOKE_CONFIGS = (
    (_SMOKE_DIR / "tuning_piano_test.yaml", "maestro-v3", _PIANO_CONDITION_PREFIX, "piano_only.yaml"),
    (_SMOKE_DIR / "tuning_guitar_test.yaml", "guitarset", _GUITAR_CONDITION_PREFIX, "guitar_only.yaml"),
)


def _tuning_names(prefix: str, levels: tuple[tuple[str, float, float | None], ...]) -> list[str]:
    return [f"{prefix}tune={suffix}" for suffix, _, _ in levels]


def test_paper_configs_lead_with_two_disabled_tuning_slots() -> None:
    for name in _PAPER_CONFIGS:
        effects = load_config(_PAPER_EXPERIMENTS / name).pedalboard.effects
        for index in (0, 1):
            slot = effects[index]
            assert isinstance(slot, TuningOffsetConfig), (name, index)
            assert slot.enabled is False, (name, index)
            assert slot.cents == 0.0, (name, index)
            knobs = slot.model_dump(mode="json")
            del knobs["type"], knobs["enabled"], knobs["cents"]
            assert knobs == _TUNING_KNOB_DEFAULTS, (name, index, knobs)


def test_paper_configs_slot_layout() -> None:
    for name, expected in _PAPER_LAYOUT.items():
        effects = load_config(_PAPER_EXPERIMENTS / name).pedalboard.effects
        assert tuple(type(effect) for effect in effects) == expected, name


def test_paper_configs_tuning_conditions() -> None:
    prefixes = {name: _PIANO_CONDITION_PREFIX for name in _PAPER_PIANO_CONFIGS}
    prefixes.update({name: _GUITAR_CONDITION_PREFIX for name in _PAPER_GUITAR_CONFIGS})
    for name, prefix in prefixes.items():
        cfg = load_config(_PAPER_EXPERIMENTS / name)
        declared = [condition.name for condition in cfg.benchmark.conditions]
        assert declared[-9:] == _tuning_names(prefix, _TUNING_LEVELS), name
        assert not [n for n in declared[:-9] if "tune=" in n], name

        slot_count = len(cfg.pedalboard.effects)
        enabled_paths = {f"pedalboard.effects.{i}.enabled" for i in range(slot_count)}
        overrides = {c.name: c.overrides for c in cfg.benchmark.conditions}
        for suffix, instrument_cents, return_cents in _TUNING_LEVELS:
            tuning = overrides[f"{prefix}tune={suffix}"]
            assert tuning["pedalboard.effects.0.enabled"] is True, (name, suffix)
            assert tuning["pedalboard.effects.0.cents"] == instrument_cents, (name, suffix)
            if return_cents is None:
                assert tuning["pedalboard.effects.1.enabled"] is False, (name, suffix)
            else:
                assert tuning["pedalboard.effects.1.enabled"] is True, (name, suffix)
                assert tuning["pedalboard.effects.1.cents"] == return_cents, (name, suffix)
            # A clean signal: every slot the offset does not use is switched off.
            switches = {p: v for p, v in tuning.items() if p.endswith(".enabled")}
            assert set(switches) == enabled_paths, (name, suffix)
            assert all(switches[f"pedalboard.effects.{i}.enabled"] is False for i in range(2, slot_count)), (
                name,
                suffix,
            )


def test_paper_configs_non_tuning_conditions_leave_tuning_off() -> None:
    for name in _PAPER_CONFIGS:
        cfg = load_config(_PAPER_EXPERIMENTS / name)
        for condition in cfg.benchmark.conditions:
            if "tune=" in condition.name:
                continue
            for path in condition.overrides:
                assert not path.startswith(("pedalboard.effects.0.", "pedalboard.effects.1.")), (
                    name,
                    condition.name,
                    path,
                )


def test_paper_configs_condition_names_unique_and_slug_stable() -> None:
    for name in _PAPER_CONFIGS:
        cfg = load_config(_PAPER_EXPERIMENTS / name)
        expanded = [condition.name for condition in expand_conditions(cfg.benchmark)]
        assert len(expanded) == len(set(expanded)), name
        for condition in cfg.benchmark.conditions:
            if "tune=" not in condition.name:
                continue
            # A "+" would not survive Condition.slug, so offsets stay signed.
            assert Condition(condition.name).slug == condition.name, (name, condition.name)


def test_tuning_smoke_configs() -> None:
    for path, dataset, prefix, twin_name in _TUNING_SMOKE_CONFIGS:
        cfg = load_config(path)
        twin = load_config(_PAPER_EXPERIMENTS / twin_name)
        assert cfg.io.dataset == dataset, path
        assert cfg.render_pipeline.input_type == InputType.AUDIO, path
        assert [type(e) for e in cfg.pedalboard.effects] == [type(e) for e in twin.pedalboard.effects], path
        assert [c.name for c in cfg.benchmark.conditions] == [
            f"{prefix}tune={suffix}" for suffix in ("-32c", "40c", "rt40c")
        ], path
        assert cfg.benchmark.include_baseline is True, path
        assert [t.type for t in cfg.transcription.transcribers] == ["basic_pitch"], path
        assert cfg.transcription.transcribers[0].device == "cpu", path
        if dataset == "maestro-v3":
            # The test split is capped by an explicit sample, so the smoke run
            # cannot draw one of MAESTRO's tens-of-minutes recordings.
            assert cfg.io.where == {"split": ["test"]}, path
            assert cfg.io.sample is not None, path
            assert (cfg.io.sample.n, cfg.io.sample.seed) == (2, 0), path
        assert expand_conditions(cfg.benchmark)[0].name == cfg.benchmark.baseline_name, path


def test_source_yaml_documents_tuning_offset() -> None:
    text = default_config_path().read_text()
    entry = re.search(
        r"^[ \t]*# - type: TuningOffset$(?P<body>(?:\n[ \t]*#.*)*)", text, re.M
    )
    assert entry, "config/source.yaml has no commented TuningOffset entry"
    body = entry.group("body")
    for field in ("cents", *_TUNING_KNOB_DEFAULTS):
        assert re.search(rf"^[ \t]*#[ \t]+{field}:", body, re.M), field
    # The engine choice is the reason the offset is not a native plugin.
    assert "time_stretch" in body
