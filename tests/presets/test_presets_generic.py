"""Invariants every shipped preset must satisfy, derived from the preset files themselves."""

from __future__ import annotations

import yaml

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions
from sonitra.config import load_config
from tests.presets._presets import CONFIG_DIR, presets

_EXAMPLES_DIR_NAME = "examples"
_TRANSCUN_DIR = CONFIG_DIR / "benchmark" / "transkun"
_SWEEP_STUDY_PREFIX = "transcriber_params_"
_GPU_MARKER = "_gpu"


def test_basic_pitch_entries_keep_the_published_knobs() -> None:
    found = presets()
    assert found, "the preset walk under config/ found nothing"

    offenders: list[str] = []
    with_basic_pitch = 0
    for path in found:
        cfg = load_config(path)
        relative = path.relative_to(CONFIG_DIR)
        sweep_study = path.stem.startswith(_SWEEP_STUDY_PREFIX)
        entries = [
            (index, entry)
            for index, entry in enumerate(cfg.transcription.transcribers)
            if entry.type == "basic_pitch"
        ]
        if entries:
            with_basic_pitch += 1
        for index, entry in entries:
            offender = f"{relative} ({entry.name})"
            if entry.multiple_pitch_bends is not False:
                offenders.append(
                    f"{offender}: multiple_pitch_bends="
                    f"{entry.multiple_pitch_bends}, expected False"
                )
            expected_raw_outputs = relative.parts[0] == _EXAMPLES_DIR_NAME
            if entry.save_raw_outputs is not expected_raw_outputs:
                offenders.append(
                    f"{offender}: save_raw_outputs={entry.save_raw_outputs}, "
                    f"expected {expected_raw_outputs} for a {relative.parts[0]}/ preset"
                )
            # A transcriber-parameter sweep study measures the melodia trick by
            # switching it off on a variant entry, so only that entry may
            # disagree with the published default.
            is_variant_entry = sweep_study and index > 0
            if not is_variant_entry and entry.melodia_trick is not True:
                offenders.append(
                    f"{offender}: melodia_trick={entry.melodia_trick}, expected True"
                )

    assert not offenders, "basic_pitch knobs diverged from the published set:\n" + "\n".join(
        offenders
    )
    # Most presets run basic_pitch, so a filter that silently matches nothing
    # would let the walk above pass without checking a single entry.
    assert with_basic_pitch >= len(found) // 2, (
        f"only {with_basic_pitch} of {len(found)} presets declare a basic_pitch "
        "transcriber"
    )


def test_benchmark_presets_live_in_subfolders() -> None:
    # Benchmark studies are grouped by kind; only the README sits at the top level.
    assert sorted((CONFIG_DIR / "benchmark").glob("*.yaml")) == []


def test_transkun_preset_name_matches_its_devices() -> None:
    found = sorted(_TRANSCUN_DIR.glob("*.yaml"))
    assert found, f"the walk under {_TRANSCUN_DIR} found nothing"

    offenders: list[str] = []
    gpu_files = 0
    cpu_files = 0
    for path in found:
        expected = "cuda" if _GPU_MARKER in path.name else "cpu"
        if expected == "cuda":
            gpu_files += 1
        else:
            cpu_files += 1
        for transcriber in load_config(path).transcription.transcribers:
            if transcriber.device != expected:
                offenders.append(
                    f"{path.relative_to(CONFIG_DIR)} ({transcriber.name}): "
                    f"device={transcriber.device}, expected {expected} because "
                    f"the file name says {_GPU_MARKER!r}"
                )

    assert not offenders, "transkun preset names disagree with their devices:\n" + "\n".join(
        offenders
    )
    # Both naming branches have to exist, otherwise the rule only ever checks
    # one device and the other goes unverified.
    assert gpu_files, f"no {_GPU_MARKER} preset under {_TRANSCUN_DIR}"
    assert cpu_files, f"no non-{_GPU_MARKER} preset under {_TRANSCUN_DIR}"


def test_condition_names_are_unique_and_slug_stable() -> None:
    found = presets()
    assert found, "the preset walk under config/ found nothing"

    offenders: list[str] = []
    slug_checked = 0
    for path in found:
        relative = path.relative_to(CONFIG_DIR)
        cfg = load_config(path)
        # A duplicate name is a hard error at expansion time, so it is reported
        # here as an offender instead of aborting the whole walk.
        try:
            expand_conditions(cfg.benchmark)
        except ValueError as exc:
            offenders.append(f"{relative}: {exc}")
        for condition in cfg.benchmark.conditions:
            if "tune=" not in condition.name:
                continue
            slug_checked += 1
            # A "+" would not survive Condition.slug, which is why the offsets
            # in a tuning condition name stay signed.
            slug = Condition(condition.name).slug
            if slug != condition.name:
                offenders.append(
                    f"{relative}: condition name {condition.name!r} slugs to {slug!r}"
                )

    assert not offenders, "condition names are not unique or not slug-stable:\n" + "\n".join(
        offenders
    )
    assert slug_checked, "no preset declared a tune= condition to check the slug of"


def test_preset_survives_run_level_validation() -> None:
    # This is the run-level guard the CLI applies; it lives in the runner module
    # and is deliberately not exported.
    from sonitra.benchmark import runner as runner_module

    found = presets()
    assert found, "the preset walk under config/ found nothing"

    offenders: list[str] = []
    for path in found:
        try:
            runner_module._validate_no_run_level_override(load_config(path))
        except ValueError as exc:
            offenders.append(f"{path.relative_to(CONFIG_DIR)}: {exc}")

    assert not offenders, "presets override run-level keys:\n" + "\n".join(offenders)


def test_benchmark_preset_sets_no_worker_key_it_cannot_act_on() -> None:
    found = [
        path for path in presets() if path.relative_to(CONFIG_DIR).parts[0] == "benchmark"
    ]
    assert found, "the preset walk found no benchmark preset"

    offenders: list[str] = []
    for path in found:
        # The raw YAML is read rather than a loaded config, so an absent key
        # cannot be masked by the schema default of 1.
        raw = yaml.safe_load(path.read_text()) or {}
        for section in ("transcription", "evaluation"):
            if "max_workers" in (raw.get(section) or {}):
                offenders.append(
                    f"{path.relative_to(CONFIG_DIR)}: {section}.max_workers="
                    f"{raw[section]['max_workers']}"
                )

    # ``transcription.max_workers`` is read by ``sonitra transcribe`` and
    # ``evaluation.max_workers`` by ``sonitra evaluate``; ``sonitra benchmark``
    # reads neither, so a value here only invites the reader to believe it
    # parallelises the study.
    assert not offenders, "inert worker keys in benchmark presets:\n" + "\n".join(offenders)


def test_every_expanded_condition_resolves() -> None:
    found = presets()
    assert found, "the preset walk under config/ found nothing"

    offenders: list[str] = []
    expanded_count = 0
    for path in found:
        cfg = load_config(path)
        conditions = expand_conditions(cfg.benchmark)
        expanded_count += len(conditions)
        for condition in conditions:
            # An override can fail as an unknown key, an out-of-range index or a
            # rejected value, so every failure mode is collected as an offender.
            try:
                apply_overrides(cfg, condition.overrides)
            except Exception as exc:
                offenders.append(f"{path.relative_to(CONFIG_DIR)}: {condition.name}: {exc}")

    assert not offenders, "conditions do not resolve against their preset:\n" + "\n".join(
        offenders
    )
    assert expanded_count, "no preset expanded into a condition"