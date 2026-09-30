from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sonitra.config import (
    ConfigError,
    CorpusPaths,
    IOSection,
    PipelineConfig,
    load_config,
    resolve_benchmark_dir,
    resolve_corpus_paths,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _minimal_config_dict(*, dataset: str | None = None) -> dict:
    """Return the smallest valid PipelineConfig payload."""
    io: dict = {
        "corpus_root": "corpus",
        "output_format": "wav",
        "mp3_bitrate_kbps": 192,
        "file_naming": "{stem}",
    }
    if dataset is not None:
        io["dataset"] = dataset
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
        "io": io,
    }


# ── IOSection field acceptance ────────────────────────────────────────────────


def test_io_section_accepts_corpus_root_and_dataset() -> None:
    section = IOSection(
        corpus_root="corpus",
        output_format="wav",
        mp3_bitrate_kbps=192,
        file_naming="{stem}",
        dataset="maestro",
    )
    assert section.corpus_root == "corpus"
    assert section.dataset == "maestro"


def test_io_section_corpus_root_defaults_to_corpus() -> None:
    section = IOSection(
        output_format="wav",
        mp3_bitrate_kbps=192,
        file_naming="{stem}",
    )
    assert section.corpus_root == "corpus"


def test_io_section_dataset_defaults_to_none() -> None:
    section = IOSection(
        corpus_root="corpus",
        output_format="wav",
        mp3_bitrate_kbps=192,
        file_naming="{stem}",
    )
    assert section.dataset is None


def test_io_section_rejects_midi_dir() -> None:
    """midi_dir is a removed field; extra="forbid" must reject it."""
    with pytest.raises(Exception):
        IOSection(
            corpus_root="corpus",
            midi_dir="corpus/midi",
            output_format="wav",
            mp3_bitrate_kbps=192,
            file_naming="{stem}",
        )


def test_io_section_rejects_output_dir() -> None:
    """output_dir is a removed field; extra="forbid" must reject it."""
    with pytest.raises(Exception):
        IOSection(
            corpus_root="corpus",
            output_dir="corpus/audio",
            output_format="wav",
            mp3_bitrate_kbps=192,
            file_naming="{stem}",
        )


def test_io_section_extra_field_still_forbidden() -> None:
    """extra="forbid" must still reject truly unknown keys."""
    with pytest.raises(Exception):
        IOSection(
            corpus_root="corpus",
            output_format="wav",
            mp3_bitrate_kbps=192,
            file_naming="{stem}",
            totally_unknown_key="oops",
        )


# ── resolve_corpus_paths — return type ───────────────────────────────────────


def test_resolve_corpus_paths_returns_corpus_paths_instance() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict())
    result = resolve_corpus_paths(cfg)
    assert isinstance(result, CorpusPaths)


# ── resolve_corpus_paths — no dataset ────────────────────────────────────────


def test_resolve_corpus_paths_no_dataset_no_config_name() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict())
    paths = resolve_corpus_paths(cfg)
    assert paths.midi == Path("corpus") / "midi"
    assert paths.audio == Path("corpus") / "audio"
    assert paths.transcription == Path("corpus") / "transcription"
    assert paths.eval_results == Path("corpus") / "eval_results"


def test_resolve_corpus_paths_no_dataset_with_config_name() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict())
    paths = resolve_corpus_paths(cfg, config_name="pedalboard_baseline")
    assert paths.midi == Path("corpus") / "midi"
    assert paths.audio == Path("corpus") / "audio" / "pedalboard_baseline"
    assert paths.transcription == Path("corpus") / "transcription" / "pedalboard_baseline"
    assert paths.eval_results == Path("corpus") / "eval_results"


# ── resolve_corpus_paths — with dataset ──────────────────────────────────────


def test_resolve_corpus_paths_with_dataset_and_config_name() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict(dataset="maestro"))
    paths = resolve_corpus_paths(cfg, config_name="pedalboard_baseline")
    assert paths.midi == Path("corpus") / "maestro" / "midi"
    assert paths.audio == Path("corpus") / "maestro" / "audio" / "pedalboard_baseline"
    assert paths.transcription == Path("corpus") / "maestro" / "transcription" / "pedalboard_baseline"
    assert paths.eval_results == Path("corpus") / "maestro" / "eval_results"


def test_resolve_corpus_paths_with_dataset_no_config_name() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict(dataset="maestro"))
    paths = resolve_corpus_paths(cfg)
    assert paths.midi == Path("corpus") / "maestro" / "midi"
    assert paths.audio == Path("corpus") / "maestro" / "audio"
    assert paths.transcription == Path("corpus") / "maestro" / "transcription"
    assert paths.eval_results == Path("corpus") / "maestro" / "eval_results"


def test_resolve_corpus_paths_dataset_scopes_midi_dir_consistently() -> None:
    """midi path is scoped by dataset regardless of whether config_name is supplied."""
    cfg = PipelineConfig.model_validate(_minimal_config_dict(dataset="jsb"))
    with_name = resolve_corpus_paths(cfg, config_name="some_config")
    without_name = resolve_corpus_paths(cfg)
    assert with_name.midi == Path("corpus") / "jsb" / "midi"
    assert without_name.midi == Path("corpus") / "jsb" / "midi"


# ── YAML round-trip backward compatibility ────────────────────────────────────


def test_existing_config_without_dataset_loads_cleanly(config_fixture) -> None:
    """Existing YAML files that omit 'dataset' must still load without errors."""
    cfg = load_config(config_fixture("config_valid.yaml"))
    assert cfg.io.dataset is None


def test_config_round_trip_preserves_none_dataset(tmp_path: Path, config_fixture) -> None:
    """Save and reload a config that has no dataset; field must survive as None."""
    cfg = load_config(config_fixture("config_valid.yaml"))
    out = tmp_path / "round_trip.yaml"
    cfg.save(out)
    cfg2 = load_config(out)
    assert cfg2.io.dataset is None


def test_config_round_trip_preserves_dataset_value(tmp_path: Path) -> None:
    """Save and reload a config with dataset='maestro'; value must survive."""
    cfg = PipelineConfig.model_validate(_minimal_config_dict(dataset="maestro"))
    out = tmp_path / "with_dataset.yaml"
    cfg.save(out)
    cfg2 = load_config(out)
    assert cfg2.io.dataset == "maestro"


# ── Old keys now raise ConfigError ───────────────────────────────────────────


def test_midi_dir_yaml_key_raises_config_error(tmp_path: Path) -> None:
    """midi_dir in YAML is now an unknown key and must raise ConfigError."""
    bad_yaml = tmp_path / "old_keys.yaml"
    bad_yaml.write_text(
        "pipeline:\n"
        "  synth_backend: dawdreamer_faust\n"
        "  effects_chain: none\n"
        "  sample_rate: 44100\n"
        "  bit_depth: 24\n"
        "  channels: 2\n"
        "  duration_padding_sec: 2.0\n"
        "  overwrite: false\n"
        "  resume: true\n"
        "  max_workers: 1\n"
        "  log_level: INFO\n"
        "io:\n"
        "  midi_dir: corpus/midi\n"
        "  output_dir: corpus/audio\n"
        "  output_format: wav\n"
        "  mp3_bitrate_kbps: 192\n"
        "  file_naming: '{stem}'\n"
    )
    with pytest.raises(ConfigError):
        load_config(bad_yaml)


def test_output_dir_yaml_key_raises_config_error() -> None:
    """output_dir in the io section is now an unknown key and must raise ConfigError."""
    data = {
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
            "corpus_root": "corpus",
            "output_dir": "corpus/audio",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
    }
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)


def test_pipeline_config_unknown_io_field_raises() -> None:
    """extra="forbid" on PipelineConfig/IOSection must reject unknown keys."""
    data = _minimal_config_dict()
    data["io"]["definitely_not_a_real_field"] = "oops"
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)


# ── resolve_benchmark_dir ────────────────────────────────────────────────────


def _cfg_for_run_dir(
    *,
    dataset: str | None = None,
    corpus_root: str = "corpus",
    benchmark_dir: str | None = None,
) -> PipelineConfig:
    payload = _minimal_config_dict(dataset=dataset)
    payload["io"]["corpus_root"] = corpus_root
    if benchmark_dir is not None:
        payload["benchmark"] = {"benchmark_dir": benchmark_dir}
    return PipelineConfig.model_validate(payload)


def test_resolve_benchmark_dir_key_is_used_as_given_and_wins() -> None:
    cfg = _cfg_for_run_dir(dataset="maestro", benchmark_dir="runs/fixed")
    assert resolve_benchmark_dir(cfg, "study") == Path("runs/fixed")


def test_resolve_benchmark_dir_key_absolute_stays_absolute(tmp_path: Path) -> None:
    cfg = _cfg_for_run_dir(benchmark_dir=str(tmp_path / "abs"))
    assert resolve_benchmark_dir(cfg, "study") == tmp_path / "abs"


def test_resolve_benchmark_dir_uses_dataset_under_corpus_root() -> None:
    cfg = _cfg_for_run_dir(dataset="maestro")
    assert resolve_benchmark_dir(cfg, "study") == (
        Path("corpus") / "maestro" / "benchmark" / "study"
    )


def test_resolve_benchmark_dir_honours_custom_corpus_root() -> None:
    cfg = _cfg_for_run_dir(dataset="maestro", corpus_root="/data/root")
    assert resolve_benchmark_dir(cfg, "study") == (
        Path("/data/root") / "maestro" / "benchmark" / "study"
    )


def test_resolve_benchmark_dir_without_dataset_is_cwd_relative() -> None:
    cfg = _cfg_for_run_dir(corpus_root="/data/root")
    resolved = resolve_benchmark_dir(cfg, "study")
    assert resolved == Path("benchmark") / "study"
    assert not resolved.is_absolute()


def test_benchmark_dir_defaults_to_none() -> None:
    assert _cfg_for_run_dir().benchmark.benchmark_dir is None
