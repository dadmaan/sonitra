from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from sonitra.benchmark.results import compute_fingerprint
from sonitra.config import PipelineConfig, load_config

GOLDEN_VALID_CONFIG = "114b31a404838203a35a46cb8dbc25548b5462b5930a0ac72667c7eb0a403721"
GOLDEN_PIANO_ONLY_PRESET = "3840432d6e2f670ab9372c89d570359d9b2c9401eec5454442ac8f19f082f069"


def test_fingerprint_golden_minimal_config() -> None:
    # A failure here changes every benchmark fingerprint, so every existing work
    # dir stops resuming; re-capture only as a deliberate, CHANGELOG'd decision.
    path = Path(__file__).resolve().parent / "fixtures" / "config_valid.yaml"
    assert compute_fingerprint(load_config(path)) == GOLDEN_VALID_CONFIG


def test_fingerprint_golden_piano_only_preset() -> None:
    # A failure here changes every benchmark fingerprint, so every existing work
    # dir stops resuming; re-capture only as a deliberate, CHANGELOG'd decision.
    path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "benchmark"
        / "paper_experiments"
        / "piano_only.yaml"
    )
    assert compute_fingerprint(load_config(path)) == GOLDEN_PIANO_ONLY_PRESET


def _valid_payload() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "fixtures" / "config_valid.yaml"
    return yaml.safe_load(path.read_text())


def _fingerprint_with_io(**keys: Any) -> str:
    payload = _valid_payload()
    payload["io"].update(keys)
    return compute_fingerprint(PipelineConfig.model_validate(payload))


def test_fingerprint_ignores_inactive_filter_keys() -> None:
    assert _fingerprint_with_io(join_column="midi_path") == GOLDEN_VALID_CONFIG


def test_fingerprint_changes_with_where() -> None:
    keys = {"dataset": "maestro-v3", "metadata_csv": "maestro-v3.0.0.csv"}
    test_split = _fingerprint_with_io(**keys, where={"split": ["test"]})
    validation = _fingerprint_with_io(**keys, where={"split": ["validation"]})
    other_column = _fingerprint_with_io(
        **keys, where={"split": ["test"]}, join_column="midi_path"
    )
    assert len({GOLDEN_VALID_CONFIG, test_split, validation, other_column}) == 4


def test_fingerprint_changes_with_sample() -> None:
    sampled = _fingerprint_with_io(sample={"n": 2, "seed": 0})
    other_seed = _fingerprint_with_io(sample={"n": 2, "seed": 1})
    assert len({GOLDEN_VALID_CONFIG, sampled, other_seed}) == 3


def test_fingerprint_default_io_saved_and_reloaded(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "config_valid.yaml"
    config = load_config(path)
    saved = tmp_path / "saved.yaml"
    config.save(saved)
    text = saved.read_text()
    assert not re.search(r"^selection:", text, re.M)
    assert re.search(r"^  where: \{\}$", text, re.M)
    assert re.search(r"^  sample: null$", text, re.M)
    reloaded = load_config(saved)
    assert compute_fingerprint(reloaded) == GOLDEN_VALID_CONFIG


def test_fingerprint_ignores_benchmark_dir() -> None:
    payload = _valid_payload()
    payload.setdefault("benchmark", {})["benchmark_dir"] = "runs/anywhere"
    config = PipelineConfig.model_validate(payload)
    assert config.benchmark.benchmark_dir == "runs/anywhere"
    assert compute_fingerprint(config) == GOLDEN_VALID_CONFIG
