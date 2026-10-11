"""Checks that config/ preset discovery never goes silently empty."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.presets._presets import (
    CONFIG_DIR,
    PAPER_DIR,
    SMOKE_DIR,
    PaperFamily,
    paper_family,
    paper_presets,
    presets,
    tuning_smoke_pairs,
)

EXPECTED_PAPER_FAMILIES: tuple[PaperFamily, ...] = ("piano", "guitar", "sweep")


def test_config_dir_is_the_repository_config_directory() -> None:
    assert CONFIG_DIR.is_dir(), f"{CONFIG_DIR} is not a directory"
    assert (CONFIG_DIR / "source.yaml").is_file()


def test_presets_are_discovered() -> None:
    # A count is deliberately not pinned: adding or removing a preset should not
    # need a test edit. What matters is that the walk is not silently empty.
    found = presets()
    assert found, "no presets discovered under config/"


def test_presets_exclude_source_and_all_exist() -> None:
    found = presets()
    assert "source.yaml" not in {path.name for path in found}
    for path in found:
        assert path.is_file(), f"{path} does not exist"
        assert path.is_relative_to(CONFIG_DIR), f"{path} is not under {CONFIG_DIR}"


def test_paper_presets_are_discovered() -> None:
    found = paper_presets()
    assert found, f"no presets discovered in {PAPER_DIR}"


@pytest.mark.parametrize("family", EXPECTED_PAPER_FAMILIES)
def test_paper_family_has_presets(family: PaperFamily) -> None:
    matching = [path for path in paper_presets() if paper_family(path) == family]
    assert matching, f"no paper experiment preset classified as {family!r}"


def test_every_paper_preset_classifies() -> None:
    classified = [paper_family(path) for path in paper_presets()]
    assert classified, f"no presets discovered in {PAPER_DIR}"


def test_paper_families_cover_every_paper_preset() -> None:
    assert set(paper_family(path) for path in paper_presets()) == set(
        EXPECTED_PAPER_FAMILIES
    )


def test_tuning_smoke_pairs_are_discovered() -> None:
    pairs = tuning_smoke_pairs()
    assert pairs, f"no tuning smoke presets discovered in {SMOKE_DIR}"
    for smoke, paper in pairs:
        assert smoke.is_file(), f"{smoke} does not exist"
        assert paper.is_file(), f"{paper} does not exist"
        assert smoke.is_relative_to(SMOKE_DIR)
        assert paper.is_relative_to(PAPER_DIR)


def test_paper_family_rejects_unknown_preset() -> None:
    unknown = Path("config/benchmark/paper_experiments/not_a_paper_file.yaml")
    with pytest.raises(ValueError):
        paper_family(unknown)


def test_paper_family_accepts_known_presets() -> None:
    assert paper_family(PAPER_DIR / "piano_only.yaml") == "piano"
    assert paper_family(PAPER_DIR / "guitar_only.yaml") == "guitar"
    assert paper_family(
        PAPER_DIR / "transcriber_params_basic_pitch_maestro_test_audio.yaml"
    ) == "sweep"