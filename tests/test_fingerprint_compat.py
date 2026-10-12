from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from sonitra.benchmark.results import compute_fingerprint
from sonitra.config import PipelineConfig, load_config

GOLDEN_VALID_CONFIG = "990237c4c9cffd9f3e9094940a740aa38afd5155ce8a7cf5ad70b8db65e2ae7f"
# Re-captured 2026-10-12 following the normalisation schema migration.
# The preset goldens are captured against frozen snapshots under
# tests/fixtures/fingerprint/, byte copies from the branch cut. Those fixtures are
# never updated to follow the presets, so the goldens detect changes to the
# fingerprint hashing and the schema dump, not preset edits. A golden change
# stops every existing benchmark work dir from resuming; re-capture only as a
# deliberate, CHANGELOG'd decision.
GOLDEN_PIANO_ONLY_PRESET = "8062a87533123f7e55e1656dda922b147cc3e3b7cd1183433fe4b154d7b361fd"
# basic_pitch with an explicit batch_size on CUDA
GOLDEN_BATCH8_PRESET = "a90c497e740e9a9733b58bc0e98748e990defa03ce5b5f45af7e51abdc157da7"
# basic_pitch on CPU with batch_size left unset in the YAML (16 was the schema default)
GOLDEN_CPU_UNSET_BASIC_PITCH = "09cccdd392380e6aa11031cf9b12ac2fd7b94ba1c5e9b6b13ccf9890437ebde9"


def test_fingerprint_golden_minimal_config() -> None:
    # A failure here changes every benchmark fingerprint, so every existing work
    # dir stops resuming; re-capture only as a deliberate, CHANGELOG'd decision.
    path = Path(__file__).resolve().parent / "fixtures" / "config_valid.yaml"
    assert compute_fingerprint(load_config(path)) == GOLDEN_VALID_CONFIG


def test_fingerprint_golden_piano_only_preset() -> None:
    # The fixture is a frozen snapshot of the preset as of the branch cut and must
    # never be updated to follow the preset: the golden detects changes to the
    # fingerprint hashing and the schema dump. Such a change stops every existing
    # benchmark work dir from resuming; re-capture only as a deliberate,
    # CHANGELOG'd decision.
    path = Path(__file__).resolve().parent / "fixtures" / "fingerprint" / "piano_only.yaml"
    assert compute_fingerprint(load_config(path)) == GOLDEN_PIANO_ONLY_PRESET


def test_fingerprint_golden_batch8_preset() -> None:
    # Protects an explicit batch_size on CUDA. The fixture is a frozen snapshot of
    # the preset as of the branch cut and must never be updated to follow the
    # preset: the golden detects changes to the fingerprint hashing and the schema
    # dump. Such a change stops every existing benchmark work dir from resuming;
    # re-capture only as a deliberate, CHANGELOG'd decision.
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "fingerprint"
        / "transkun_baseline_gpu_batch8_strict.yaml"
    )
    assert compute_fingerprint(load_config(path)) == GOLDEN_BATCH8_PRESET


def test_fingerprint_golden_cpu_unset_basic_pitch() -> None:
    # Guards the unset-value normalisation: this preset leaves batch_size unset,
    # so it may only keep hashing as the old schema default while the unset value
    # normalises to it. The fixture is a frozen snapshot of the preset as of the
    # branch cut and must never be updated to follow the preset: the golden
    # detects changes to the fingerprint hashing and the schema dump. Such a
    # change stops every existing benchmark work dir from resuming; re-capture
    # only as a deliberate, CHANGELOG'd decision.
    path = Path(__file__).resolve().parent / "fixtures" / "fingerprint" / "transkun_baseline.yaml"
    assert compute_fingerprint(load_config(path)) == GOLDEN_CPU_UNSET_BASIC_PITCH


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


def _fingerprint_with_basic_pitch(**transcriber_keys: Any) -> str:
    payload = _valid_payload()
    payload["transcription"] = {"transcribers": [{"type": "basic_pitch", **transcriber_keys}]}
    return compute_fingerprint(PipelineConfig.model_validate(payload))


def test_fingerprint_unset_batch_size_hashes_as_16() -> None:
    # An unset value normalises to the old schema default, so a work dir written
    # before the default became "unset" still resumes.
    assert _fingerprint_with_basic_pitch() == _fingerprint_with_basic_pitch(batch_size=16)


def test_fingerprint_explicit_batch_still_counts() -> None:
    assert _fingerprint_with_basic_pitch() != _fingerprint_with_basic_pitch(batch_size=8)
