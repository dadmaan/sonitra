from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sonitra.benchmark.results import compute_fingerprint
from sonitra.config import ConfigError, PipelineConfig, load_config


def _base_payload() -> dict[str, Any]:
    return {
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


def _payload_with_io(**keys: Any) -> dict[str, Any]:
    payload = _base_payload()
    payload["io"].update(keys)
    return payload


def _filter_keys(where: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "maestro-v3",
        "metadata_csv": "maestro-v3.0.0.csv",
        "where": where,
    }


def test_io_defaults() -> None:
    cfg = PipelineConfig.model_validate(_base_payload())
    assert cfg.io.where == {}
    assert cfg.io.metadata_csv is None
    assert cfg.io.join_column == "midi_filename"
    assert cfg.io.sample is None


def test_io_filter_keys_load() -> None:
    cfg = PipelineConfig.model_validate(
        _payload_with_io(
            dataset="maestro-v3",
            metadata_csv="maestro-v3.0.0.csv",
            join_column="midi_path",
            where={"split": ["validation", "test", "test"], "year": [2018]},
            sample={"n": 2, "seed": 3},
        )
    )
    assert cfg.io.dataset == "maestro-v3"
    assert cfg.io.metadata_csv == "maestro-v3.0.0.csv"
    assert cfg.io.join_column == "midi_path"
    assert cfg.io.where == {"split": ["test", "validation"], "year": ["2018"]}
    assert cfg.io.sample is not None
    assert (cfg.io.sample.n, cfg.io.sample.seed) == (2, 3)


def test_where_lists_sorted_and_deduped_hash_identically() -> None:
    ascending = PipelineConfig.model_validate(
        _payload_with_io(**_filter_keys({"split": ["test", "validation"]}))
    )
    descending = PipelineConfig.model_validate(
        _payload_with_io(**_filter_keys({"split": ["validation", "test"]}))
    )
    assert ascending.io.where == descending.io.where == {"split": ["test", "validation"]}
    assert compute_fingerprint(ascending) == compute_fingerprint(descending)


def test_where_bool_rejected() -> None:
    payload = _payload_with_io(**_filter_keys({"flag": [True]}))
    with pytest.raises(ConfigError, match=r"io\.where.*bool"):
        PipelineConfig.model_validate(payload)


def test_where_empty_list_rejected() -> None:
    payload = _payload_with_io(**_filter_keys({"split": []}))
    with pytest.raises(ConfigError, match=r"io\.where.*empty"):
        PipelineConfig.model_validate(payload)


def test_where_scalar_value_rejected() -> None:
    payload = _payload_with_io(**_filter_keys({"split": "test"}))
    with pytest.raises(ConfigError, match=r"io\.where values must be lists"):
        PipelineConfig.model_validate(payload)


def test_where_non_mapping_rejected() -> None:
    payload = _payload_with_io(
        dataset="maestro-v3", metadata_csv="m.csv", where=["split"]
    )
    with pytest.raises(ConfigError, match=r"io\.where must be a mapping"):
        PipelineConfig.model_validate(payload)


@pytest.mark.parametrize(
    "keys",
    [
        {"dataset": "maestro-v3", "where": {"split": ["test"]}},
        {"metadata_csv": "maestro-v3.0.0.csv", "where": {"split": ["test"]}},
    ],
    ids=["missing-metadata-csv", "missing-dataset"],
)
def test_where_requires_metadata_csv_and_dataset(keys: dict[str, Any]) -> None:
    with pytest.raises(
        ConfigError,
        match=r"io\.where requires io\.metadata_csv and io\.dataset to be set",
    ):
        PipelineConfig.model_validate(_payload_with_io(**keys))


def test_metadata_csv_without_where_rejected() -> None:
    payload = _payload_with_io(dataset="maestro-v3", metadata_csv="maestro-v3.0.0.csv")
    with pytest.raises(
        ConfigError, match=r"io\.metadata_csv has no effect without io\.where"
    ):
        PipelineConfig.model_validate(payload)


def test_join_column_without_where_accepted() -> None:
    cfg = PipelineConfig.model_validate(_payload_with_io(join_column="midi_path"))
    assert cfg.io.join_column == "midi_path"
    assert cfg.io.where == {}


def test_sample_alone_loads() -> None:
    cfg = PipelineConfig.model_validate(_payload_with_io(sample={"n": 2}))
    assert cfg.io.sample is not None
    assert cfg.io.sample.n == 2
    assert cfg.io.sample.seed == 0
    assert cfg.io.dataset is None


def test_sample_n_must_be_positive() -> None:
    payload = _payload_with_io(sample={"n": 0})
    with pytest.raises(ConfigError, match=r"io\.sample\.n"):
        PipelineConfig.model_validate(payload)


def test_top_level_selection_rejected() -> None:
    payload = _base_payload()
    payload["selection"] = {"sample": {"n": 2}}
    with pytest.raises(ConfigError, match=r"selection"):
        PipelineConfig.model_validate(payload)


def test_io_extra_key_forbidden() -> None:
    with pytest.raises(ConfigError, match=r"io\.foo"):
        PipelineConfig.model_validate(_payload_with_io(foo=1))


def test_config_save_roundtrip_with_io_filter(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(
        _payload_with_io(
            **_filter_keys({"split": ["test"]}),
            join_column="midi_filename",
            sample={"n": 2, "seed": 3},
        )
    )
    path = tmp_path / "selection.yaml"
    cfg.save(path)
    reloaded = load_config(path)
    assert reloaded.io == cfg.io
    assert reloaded == cfg


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        (2, "io.sample must be a mapping like {n: 2, seed: 0}, or null (got 2)"),
        ("5", "io.sample must be a mapping like {n: 5, seed: 0}, or null (got '5')"),
        ([2], "io.sample must be a mapping like {n: 2, seed: 0}, or null (got [2])"),
        (True, "io.sample must be a mapping like {n: 2, seed: 0}, or null (got True)"),
    ],
)
def test_sample_scalar_rejected_with_hint(value: Any, hint: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        PipelineConfig.model_validate(_payload_with_io(sample=value))
    assert hint in str(excinfo.value)
