"""Invariants the shipped paper experiment presets must satisfy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sonitra.benchmark.conditions import Condition, apply_overrides, expand_conditions
from sonitra.config import InputType, PipelineConfig, load_config
from sonitra.effects import TuningOffsetConfig
from tests.presets._presets import (
    PAPER_DIR,
    PaperFamily,
    paper_family,
    paper_presets,
    presets,
    tuning_smoke_pairs,
)

# The file-name grammar the checks below derive their subject from: the split
# markers, the input-type suffixes and the prefixes the families are named by.
_MAESTRO_MARKER = "_maestro"
_TEST_MARKER = f"{_MAESTRO_MARKER}_test_"
_TRAIN_MARKER = f"{_MAESTRO_MARKER}_train_probe_"
_AUDIO_SUFFIX = "_audio"
_MIDI_SUFFIX = "_midi"
# A parameter study always runs on the recorded audio, so its name carries the
# same suffix as the split preset it was copied from.
_TEST_SUFFIX = f"{_TEST_MARKER}audio"
_PIANO_PREFIX = "piano_only"
_SWEEP_PREFIX = "transcriber_params_"

# Half a semitone: at or beyond it a note is as close to one named pitch as to
# its neighbour, which is where the chain builder warns.
_ACCURACY_GUARD_CENTS = 50.0

# A signed level, or the return leg after it. The sign is written out because a
# "+" would not survive Condition.slug, which every condition name goes through
# to name the directory its results land in.
_TUNE_LEVEL = re.compile(r"tune=(?:rt)?(-?\d+(?:\.\d+)?)c$")

# The shape a variant entry name follows: a short model tag, then the knob the
# entry declares it changes and the level it declares it at.
_VARIANT_NAME = re.compile(r"[a-z]+_([a-z0-9]+)=.+")


def _tuning_families() -> list[PaperFamily]:
    """The families whose presets carry a tuning sweep.

    A parameter study varies one model's settings instead, so it carries no
    tuning contract and is excluded rather than swept in by accident.
    """
    return sorted(set(paper_family(path) for path in paper_presets()) - {"sweep"})


def _tuning_family_files(family: PaperFamily) -> list[Path]:
    """Every shipped preset of one family."""
    if family == "sweep":
        raise ValueError("a parameter study carries no tuning contract")
    return [path for path in paper_presets() if paper_family(path) == family]


def _twin(path: Path) -> Path:
    """The preset a paper experiment preset varies against.

    A train probe and its split preset differ only in which files they select,
    and a parameter study differs from the split preset it was copied from. The
    twin is derived from the file name, so a renamed preset fails here instead
    of quietly dropping the comparison it exists to make.
    """
    stem = path.stem
    if stem.startswith(_SWEEP_PREFIX):
        if not stem.endswith(_TEST_SUFFIX):
            raise ValueError(f"no twin rule for parameter study: {path.name}")
        twin_stem = _PIANO_PREFIX + _TEST_SUFFIX
    elif _TEST_MARKER in stem:
        twin_stem = stem.replace(_TEST_MARKER, _TRAIN_MARKER)
    elif _TRAIN_MARKER in stem:
        twin_stem = stem.replace(_TRAIN_MARKER, _TEST_MARKER)
    else:
        raise ValueError(f"no twin rule for preset: {path.name}")
    twin = PAPER_DIR / f"{twin_stem}.yaml"
    if not twin.is_file():
        raise FileNotFoundError(f"{path.name} has no twin preset at {twin.name}")
    return twin


def _tuning_conditions(cfg: PipelineConfig) -> list[Condition]:
    """The trailing block of conditions that sweep the tuning offset."""
    block: list[Condition] = []
    for declared in reversed(cfg.benchmark.conditions):
        if "tune=" not in declared.name:
            break
        block.append(Condition(name=declared.name, overrides=dict(declared.overrides)))
    return list(reversed(block))


def _expected_slots(name: str) -> tuple[float, float | None]:
    """The cents a tuning condition name claims for the two leading slots.

    A plain level drives the instrument's own offset and leaves the return leg
    off; the round trip repeats the level in the opposite direction, so the
    artifacts of a shift that size can be heard on the way back.
    """
    match = _TUNE_LEVEL.search(name)
    if match is None:
        raise ValueError(f"condition name carries no signed cent level: {name!r}")
    level = float(match.group(1))
    if f"tune=rt{match.group(1)}c" in name:
        return level, -level
    return level, None


# The knob each variant label declares it changes. The vocabulary records the
# naming convention the entry names follow, not the values behind them.
_SWEEP_KNOBS: dict[str, frozenset[str]] = {
    "onset": frozenset({"onset_threshold"}),
    "frame": frozenset({"frame_threshold"}),
    "minlen": frozenset({"minimum_note_length_ms"}),
    "melodia": frozenset({"melodia_trick"}),
    "offset": frozenset({"offset_threshold"}),
    "mpe": frozenset({"mpe_threshold"}),
    "stride": frozenset({"n_stride"}),
    "head": frozenset({"output"}),
    "seg": frozenset({"segment_size_sec", "segment_hop_sec"}),
    "hop": frozenset({"segment_size_sec", "segment_hop_sec"}),
}


def _declared_knobs(name: str) -> frozenset[str]:
    """The config keys a variant entry declares it changes.

    The values are free: the name only claims which setting is under study.
    """
    match = _VARIANT_NAME.fullmatch(name)
    if match is None or match.group(1) not in _SWEEP_KNOBS:
        raise ValueError(f"variant name declares no known knob: {name!r}")
    return _SWEEP_KNOBS[match.group(1)]


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


def test_maestro_split_file_name_matches_its_filter() -> None:
    """A split segment in the name is a claim about the file selection.

    The parameter studies carry the same segment and satisfy the same filter, so
    they are part of this glob on purpose: narrowing it would have to be a
    deliberate change rather than a quiet one.
    """
    found = sorted(PAPER_DIR.glob(f"*{_TEST_MARKER}*.yaml"))
    assert found, f"no preset under {PAPER_DIR} carries {_TEST_MARKER!r}"

    offenders: list[str] = []
    for path in found:
        cfg = load_config(path)
        if cfg.io.where != {"split": ["test"]}:
            offenders.append(f"{path.name}: io.where={cfg.io.where}")
        if cfg.io.sample is not None:
            offenders.append(f"{path.name}: io.sample={cfg.io.sample}")
    assert not offenders, "file name and file selection disagree:\n" + "\n".join(offenders)


def test_paper_file_suffix_matches_its_input_type() -> None:
    """An input-type suffix in the name is a claim about how the run is fed.

    The suffix says how the run is fed, not which corpus it reads, so it covers
    every preset that ends in one, the guitar corpora included. A preset ending
    in neither claims nothing by name and is not covered here.
    """
    expected: dict[str, InputType] = {
        _AUDIO_SUFFIX: InputType.AUDIO,
        _MIDI_SUFFIX: InputType.MIDI,
    }

    offenders: list[str] = []
    checked: set[str] = set()
    for path in paper_presets():
        for suffix, input_type in expected.items():
            if not path.stem.endswith(suffix):
                continue
            checked.add(suffix)
            actual = load_config(path).render_pipeline.input_type
            if actual is not input_type:
                offenders.append(
                    f"{path.name}: name ends in {suffix!r} but input_type is {actual.value}"
                )
    assert not offenders, "file suffix and input type disagree:\n" + "\n".join(offenders)
    missing = sorted(set(expected) - checked)
    assert not missing, f"no paper experiment preset ends in {missing}"


def test_maestro_files_share_one_dataset_block() -> None:
    """Everything read from a MAESTRO corpus is declared in the same place."""
    found = [path for path in paper_presets() if _MAESTRO_MARKER in path.stem]
    assert found, f"no preset under {PAPER_DIR} names the MAESTRO corpus"

    blocks: dict[Path, tuple[Any, Any, Any]] = {}
    for path in found:
        cfg = load_config(path)
        blocks[path] = (cfg.io.dataset, cfg.io.metadata_csv, cfg.io.join_column)
    distinct = set(blocks.values())
    assert len(distinct) == 1, (
        "the MAESTRO presets declare more than one dataset block: "
        + ", ".join(f"{path.name}: {block}" for path, block in blocks.items())
    )


def test_paper_presets_never_override_the_benchmark_directory() -> None:
    """The run directory belongs to the caller, not to a shipped preset."""
    found = paper_presets()
    assert found, f"no preset discovered in {PAPER_DIR}"

    offenders: list[str] = []
    for path in found:
        benchmark_dir = load_config(path).benchmark.benchmark_dir
        if benchmark_dir is not None:
            offenders.append(f"{path.name}: benchmark_dir={benchmark_dir}")
    assert not offenders, "presets pin the benchmark run directory:\n" + "\n".join(offenders)


def test_maestro_split_presets_share_one_transcription_block() -> None:
    """The split presets score the same models, so they transcribe the same way.

    hFT-Transformer is the paper's own model, so the split presets score it from
    the MAESTRO checkpoint on CUDA next to the two reference models; "second" is
    the time-axis decoder head, the set the model reports in its paper, and
    n_stride 0 keeps the plain whole-window pass. TensorFlow reserves most GPU
    memory unless growth is on, which would starve TransKun in the same process
    even with basic_pitch on CPU. One block behind all of them is what keeps the
    audio, MIDI and probe runs comparable in the table they end up in.
    """
    found = sorted(PAPER_DIR.glob(f"{_PIANO_PREFIX}{_MAESTRO_MARKER}*.yaml"))
    assert found, f"no preset under {PAPER_DIR} names the MAESTRO split"

    blocks = {
        path: load_config(path).transcription.model_dump(mode="json") for path in found
    }
    reference_path = found[0]
    reference = blocks[reference_path]
    offenders = [
        f"{path.name} differs from {reference_path.name} in "
        f"{sorted(_differing_keys(reference, block))}"
        for path, block in blocks.items()
        if _differing_keys(reference, block)
    ]
    assert not offenders, "the split presets transcribe differently:\n" + "\n".join(offenders)


def test_maestro_test_files_differ_from_the_base_only_by_filter_and_transcription() -> None:
    """A split preset is its base preset plus a corpus, a filter and the models."""
    found = sorted(PAPER_DIR.glob(f"{_PIANO_PREFIX}{_TEST_MARKER}*.yaml"))
    assert found, f"no preset under {PAPER_DIR} carries {_TEST_MARKER!r}"

    shared = {
        "io.dataset",
        "io.metadata_csv",
        "transcription.numeric_mode",
        "transcription.gpu_memory_growth",
        "transcription.transcribers",
    }
    offenders: list[str] = []
    for path in found:
        base_path = PAPER_DIR / f"{path.stem.split(_MAESTRO_MARKER)[0]}.yaml"
        base_cfg = load_config(base_path)
        base = _flatten(base_cfg.model_dump(mode="json"))
        cfg = load_config(path)
        if path.stem.endswith(_MIDI_SUFFIX):
            # MIDI input drops the recordings, and the study keeps one TransKun
            # model per worker on the GPU.
            extra = {"render_pipeline.input_type", "benchmark.max_workers"}
        else:
            # The audio preset also keeps already-written render files on rerun.
            extra = {"render_pipeline.resume"}
        diff = _differing_keys(base, _flatten(cfg.model_dump(mode="json")))
        outside_filter = {key for key in diff if not key.startswith("io.where.")}
        if outside_filter != shared | extra:
            offenders.append(
                f"{path.name}: differs from {base_path.name} in {sorted(outside_filter)}"
            )
        elif "io.where.split" not in diff:
            offenders.append(f"{path.name}: no split filter against {base_path.name}")
        else:
            # basic_pitch keeps the base settings apart from the device and the
            # batch: the base leaves batch_size unset (1 on a GPU), while the
            # split presets run it on the CPU with an explicit 16.
            base_bp = base_cfg.transcription.transcribers[0].model_dump(mode="json")
            bp = cfg.transcription.transcribers[0].model_dump(mode="json")
            if {k for k in base_bp if base_bp[k] != bp[k]} != {"device", "batch_size"}:
                offenders.append(f"{path.name}: basic_pitch drifted beyond device and batch")
    assert not offenders, "the split presets drifted from their base:\n" + "\n".join(offenders)


def test_train_probe_file_name_matches_its_filter() -> None:
    """A probe is named for the split it samples, so it filters on that split."""
    found = sorted(PAPER_DIR.glob(f"*{_TRAIN_MARKER}*.yaml"))
    assert found, f"no preset under {PAPER_DIR} carries {_TRAIN_MARKER!r}"

    offenders: list[str] = []
    for path in found:
        cfg = load_config(path)
        if cfg.io.where != {"split": ["train"]}:
            offenders.append(f"{path.name}: io.where={cfg.io.where}")
        if cfg.io.sample is None:
            offenders.append(f"{path.name}: no io.sample, so the probe runs the whole split")
    assert not offenders, "probe name and probe filter disagree:\n" + "\n".join(offenders)


def test_train_probes_share_one_sample() -> None:
    """The probes are one sample of the split, measured on audio and on MIDI."""
    found = sorted(PAPER_DIR.glob(f"*{_TRAIN_MARKER}*.yaml"))
    assert len(found) > 1, f"expected more than one probe under {PAPER_DIR}, found {len(found)}"

    samples: dict[Path, tuple[int, int]] = {}
    for path in found:
        sample = load_config(path).io.sample
        assert sample is not None, f"{path.name} declares no io.sample"
        samples[path] = (sample.n, sample.seed)
    distinct = set(samples.values())
    assert len(distinct) == 1, (
        "the probes sample differently, so they measure different files: "
        + ", ".join(f"{path.name}: {sample}" for path, sample in samples.items())
    )


def test_train_probes_differ_from_their_twin_only_by_the_io_filter() -> None:
    """A probe is its split preset with the sample switched on."""
    found = sorted(PAPER_DIR.glob(f"*{_TRAIN_MARKER}*.yaml"))
    assert found, f"no preset under {PAPER_DIR} carries {_TRAIN_MARKER!r}"

    offenders: list[str] = []
    for path in found:
        twin = _twin(path)
        probe = _flatten(load_config(path).model_dump(mode="json"))
        twin_flat = _flatten(load_config(twin).model_dump(mode="json"))
        # Flattened, the sample becomes io.sample.n / io.sample.seed.
        diff = _differing_keys(twin_flat, probe)
        if diff != {"io.where.split", "io.sample.n", "io.sample.seed"}:
            offenders.append(f"{path.name} differs from {twin.name} in {sorted(diff)}")
    assert not offenders, "the probes drifted from their twin:\n" + "\n".join(offenders)


def test_sweep_files_differ_from_their_twin_only_by_transcribers_and_conditions() -> None:
    """A parameter study is its split preset with one model unrolled.

    The study runs the baseline condition only: transcriber settings are fixed
    when the model is built, so a study cannot express them as condition
    overrides and varies the transcriber list instead.
    """
    found = sorted(PAPER_DIR.glob(f"{_SWEEP_PREFIX}*.yaml"))
    assert found, f"no parameter study under {PAPER_DIR}"

    expected = {
        "transcription.transcribers",
        "benchmark.conditions",
        "benchmark.max_workers",
    }
    offenders: list[str] = []
    for path in found:
        twin = _twin(path)
        twin_flat = _flatten(load_config(twin).model_dump(mode="json"))
        study = _flatten(load_config(path).model_dump(mode="json"))
        diff = _differing_keys(twin_flat, study)
        if diff != expected:
            offenders.append(f"{path.name} differs from {twin.name} in {sorted(diff)}")
    assert not offenders, "the parameter studies drifted from their twin:\n" + "\n".join(offenders)


def test_sweep_files_run_only_the_baseline() -> None:
    """A study measures a knob across transcribers, not across conditions."""
    found = sorted(PAPER_DIR.glob(f"{_SWEEP_PREFIX}*.yaml"))
    assert found, f"no parameter study under {PAPER_DIR}"

    offenders: list[str] = []
    for path in found:
        cfg = load_config(path)
        names = [condition.name for condition in expand_conditions(cfg.benchmark)]
        if names != [cfg.benchmark.baseline_name]:
            offenders.append(f"{path.name}: runs {names}")
    assert not offenders, "the parameter studies run more than the baseline:\n" + "\n".join(
        offenders
    )


def test_sweep_default_entry_matches_the_twin_entry() -> None:
    """The first entry of a study is the entry the study was copied from."""
    found = sorted(PAPER_DIR.glob(f"{_SWEEP_PREFIX}*.yaml"))
    assert found, f"no parameter study under {PAPER_DIR}"

    offenders: list[str] = []
    for path in found:
        twin_path = _twin(path)
        twin = load_config(twin_path)
        entries = load_config(path).transcription.transcribers
        if not entries:
            offenders.append(f"{path.name}: declares no transcriber")
            continue
        default = entries[0]
        source = next(
            (entry for entry in twin.transcription.transcribers if entry.type == default.type),
            None,
        )
        if source is None:
            offenders.append(f"{path.name}: no {default.type} entry in {twin_path.name}")
            continue
        expected = {k: v for k, v in source.model_dump(mode="json").items() if k != "name"}
        actual = {k: v for k, v in default.model_dump(mode="json").items() if k != "name"}
        differing = _differing_keys(expected, actual)
        if differing:
            offenders.append(
                f"{path.name}: {default.name} differs from {twin_path.name} in {sorted(differing)}"
            )
    assert not offenders, "the study defaults drifted from their twin:\n" + "\n".join(offenders)


def test_sweep_variant_names_declare_the_knob_they_change() -> None:
    """A variant entry changes exactly the setting its name declares."""
    found = sorted(PAPER_DIR.glob(f"{_SWEEP_PREFIX}*.yaml"))
    assert found, f"no parameter study under {PAPER_DIR}"

    offenders: list[str] = []
    for path in found:
        # The file name carries the model under study, so every entry in it has
        # to be that model.
        declared_type = path.stem.removeprefix(_SWEEP_PREFIX).removesuffix(_TEST_SUFFIX)
        entries = load_config(path).transcription.transcribers
        if not entries:
            offenders.append(f"{path.name}: declares no transcriber")
            continue
        base = {k: v for k, v in entries[0].model_dump(mode="json").items() if k != "name"}
        seen: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            offender = f"{path.name} ({entry.name})"
            knobs = {k: v for k, v in entry.model_dump(mode="json").items() if k != "name"}
            if entry.type != declared_type:
                offenders.append(f"{offender}: type={entry.type}, expected {declared_type}")
            elif index > 0:
                try:
                    declared_knobs = _declared_knobs(entry.name)
                except ValueError as exc:
                    offenders.append(f"{offender}: {exc}")
                else:
                    if _differing_keys(base, knobs) != declared_knobs:
                        offenders.append(
                            f"{offender}: changes {sorted(_differing_keys(base, knobs))}, "
                            f"declares {sorted(declared_knobs)}"
                        )
            if any(knobs == earlier for earlier in seen):
                offenders.append(f"{offender}: repeats an earlier entry")
            seen.append(knobs)
    assert not offenders, "the variant names do not describe the variants:\n" + "\n".join(
        offenders
    )


def test_sweep_variant_names_are_path_safe() -> None:
    """A variant name becomes a directory, so it may not need rewriting."""
    found = sorted(PAPER_DIR.glob(f"{_SWEEP_PREFIX}*.yaml"))
    assert found, f"no parameter study under {PAPER_DIR}"

    offenders: list[str] = []
    for path in found:
        for entry in load_config(path).transcription.transcribers:
            if re.fullmatch(r"[a-z]+_[a-z0-9.=]+", entry.name) is None:
                offenders.append(f"{path.name}: {entry.name}")
    assert not offenders, "variant names are not path-safe:\n" + "\n".join(offenders)


def test_sweep_file_stems_stay_out_of_the_model_cards() -> None:
    """A study stem must not claim a model card of its own.

    The baselines export script owns the generated block in the model cards
    document and picks the runs to publish by file-name prefix, so a parameter
    study named after one of those prefixes would be written up as a model.
    """
    import importlib.util

    found = sorted(PAPER_DIR.glob(f"{_SWEEP_PREFIX}*.yaml"))
    assert found, f"no parameter study under {PAPER_DIR}"

    script = PAPER_DIR.parents[2] / "scripts" / "export_model_baselines.py"
    spec = importlib.util.spec_from_file_location("export_model_baselines", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefixes = [prefix.strip() for prefix in module.DEFAULT_RUN_PREFIXES.split(",")]

    offenders: list[str] = []
    for path in found:
        claimed = [prefix for prefix in prefixes if path.stem.startswith(prefix)]
        if claimed:
            offenders.append(f"{path.name} starts with {claimed}")
    assert not offenders, "the parameter studies claim a model card:\n" + "\n".join(offenders)


def test_paper_tuning_slots_lead_disabled_and_equal() -> None:
    """The tuning offset leads the chain in two adjacent slots.

    Slot 0 is the instrument's own offset: what is out of tune is the instrument
    itself, so the shift belongs ahead of the amp, the cabinet and the room.
    Slot 1 is the return leg, which the round-trip control alone switches on to
    expose the processing artifacts at the largest offset. Both ship disabled
    and at zero cents, so a preset that declares neither stays unprocessed.
    """
    checked = 0
    offenders: list[str] = []
    for family in _tuning_families():
        files = _tuning_family_files(family)
        assert files, f"no paper experiment preset in the {family!r} family"
        slots: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for path in files:
            effects = load_config(path).pedalboard.effects
            if len(effects) < 2:
                offenders.append(f"{path.name}: declares {len(effects)} effects, no tuning pair")
                continue
            for index in (0, 1):
                slot = effects[index]
                if not isinstance(slot, TuningOffsetConfig):
                    offenders.append(f"{path.name}: slot {index} is {type(slot).__name__}")
                elif slot.enabled is not False or slot.cents != 0.0:
                    offenders.append(
                        f"{path.name}: slot {index} is enabled={slot.enabled}, "
                        f"cents={slot.cents}"
                    )
            slots.append(
                (effects[0].model_dump(mode="json"), effects[1].model_dump(mode="json"))
            )
            checked += 1
        # One declaration per family: the conditions address the chain by index,
        # so two files that declare a leading slot differently cannot share a
        # condition list.
        if any(leading != slots[0][0] for leading, _ in slots):
            offenders.append(f"{family}: slot 0 is declared differently across the family")
        if any(following != slots[0][1] for _, following in slots):
            offenders.append(f"{family}: slot 1 is declared differently across the family")
    assert not offenders, "the leading tuning slots are not as declared:\n" + "\n".join(offenders)
    assert checked, "no preset declared the leading tuning slots"


def test_paper_effect_layout_matches_within_a_family() -> None:
    """Every condition addresses the chain by index, so a family shares a layout."""
    checked = 0
    offenders: list[str] = []
    for family in _tuning_families():
        files = _tuning_family_files(family)
        assert files, f"no paper experiment preset in the {family!r} family"
        layouts = {
            path: tuple(effect.type for effect in load_config(path).pedalboard.effects)
            for path in files
        }
        reference_path = files[0]
        reference = layouts[reference_path]
        assert reference, f"the {family!r} family declares no effect at all"
        for path, layout in layouts.items():
            checked += 1
            if layout != reference:
                offenders.append(
                    f"{path.name} chains {list(layout)}, "
                    f"{reference_path.name} chains {list(reference)}"
                )
    assert not offenders, "the effect chain differs inside a family:\n" + "\n".join(offenders)
    assert checked, "no preset declared an effect chain"


def test_tuning_condition_names_match_their_cents() -> None:
    """A tuning condition name says how far the instrument is shifted.

    The ±50 ¢ levels sit exactly half a semitone out, where a note is equally
    close to two named pitches. They probe which side of that tie a model lands
    on, and the tie-break is decided by the recording's own tuning rather than
    by the model's robustness, so they are probed deliberately rather than as a
    further step of the sweep. The round trip repeats the largest offset and
    returns, which exposes the processing artifacts at that size.
    """
    checked = 0
    offenders: list[str] = []
    for family in _tuning_families():
        files = _tuning_family_files(family)
        assert files, f"no paper experiment preset in the {family!r} family"
        suffixes: list[tuple[Path, list[str]]] = []
        for path in files:
            cfg = load_config(path)
            rel = path.relative_to(PAPER_DIR)
            declared = [condition.name for condition in cfg.benchmark.conditions]
            tuning = _tuning_conditions(cfg)
            if not tuning:
                offenders.append(f"{rel}: declares no tune= condition")
                continue
            names = [condition.name for condition in tuning]
            if any(name in names for name in declared[: len(declared) - len(tuning)]):
                offenders.append(f"{rel}: a tune= condition is not in the trailing block")
            levels: list[str] = []
            prefixes: set[str] = set()
            for condition in tuning:
                checked += 1
                prefixes.add(condition.name.split("tune=", 1)[0])
                levels.append(condition.name.split("tune=", 1)[1])
                try:
                    instrument_cents, return_cents = _expected_slots(condition.name)
                except ValueError as exc:
                    offenders.append(f"{rel}: {condition.name}: {exc}")
                    continue
                # The condition is resolved before it is read, so the check sees
                # the chain the run builds and not only the override dict.
                effects = apply_overrides(cfg, condition.overrides).pedalboard.effects
                if len(effects) < 2:
                    offenders.append(
                        f"{rel}: {condition.name}: the chain has {len(effects)} slots"
                    )
                    continue
                leading, following = effects[0], effects[1]
                if leading.cents != instrument_cents:
                    offenders.append(
                        f"{rel}: {condition.name}: the instrument is shifted by "
                        f"{leading.cents}c, not {instrument_cents}c"
                    )
                if return_cents is None:
                    if following.enabled is not False:
                        offenders.append(
                            f"{rel}: {condition.name}: the return leg is on at "
                            f"{following.cents}c without a round trip"
                        )
                elif following.enabled is not True or following.cents != return_cents:
                    offenders.append(
                        f"{rel}: {condition.name}: the return leg is enabled="
                        f"{following.enabled} at {following.cents}c, not {return_cents}c"
                    )
                # A clean signal: every slot the offset does not use is off.
                still_on = [
                    index for index, effect in enumerate(effects[2:], start=2) if effect.enabled
                ]
                if still_on:
                    offenders.append(f"{rel}: {condition.name}: slots {still_on} are on beside it")
            if len(prefixes) > 1:
                offenders.append(f"{rel}: the condition names carry prefixes {sorted(prefixes)}")
            suffixes.append((path, levels))
        # The same levels in the same order across the family, because the two
        # families are read side by side; the two families may differ.
        for path, levels in suffixes:
            if levels != suffixes[0][1]:
                offenders.append(
                    f"{path.relative_to(PAPER_DIR)} sweeps {levels}, "
                    f"{suffixes[0][0].relative_to(PAPER_DIR)} sweeps {suffixes[0][1]}"
                )
    assert not offenders, "the tuning conditions disagree with their names:\n" + "\n".join(
        offenders
    )
    assert checked, "no preset declared a tuning condition"


def test_non_tuning_conditions_leave_the_tuning_slots_off() -> None:
    """Only a tuning condition may move the instrument's own tuning."""
    checked = 0
    offenders: list[str] = []
    for family in _tuning_families():
        files = _tuning_family_files(family)
        assert files, f"no paper experiment preset in the {family!r} family"
        for path in files:
            rel = path.relative_to(PAPER_DIR)
            for condition in load_config(path).benchmark.conditions:
                if "tune=" in condition.name:
                    continue
                checked += 1
                for override in condition.overrides:
                    if override.startswith(("pedalboard.effects.0.", "pedalboard.effects.1.")):
                        offenders.append(f"{rel}: {condition.name}: {override}")
    assert not offenders, "a non-tuning condition moves the tuning slots:\n" + "\n".join(offenders)
    assert checked, "no preset declared a condition besides the tuning ones"


def test_tuning_smoke_preset_is_a_subset_of_its_twin() -> None:
    """A smoke run measures the same chain over a subset of the real conditions."""
    pairs = tuning_smoke_pairs()
    assert pairs, "no tuning smoke preset paired with a paper experiment preset"

    offenders: list[str] = []
    for smoke, twin_path in pairs:
        cfg = load_config(smoke)
        twin = load_config(twin_path)
        smoke_effects = [effect.model_dump(mode="json") for effect in cfg.pedalboard.effects]
        twin_effects = [effect.model_dump(mode="json") for effect in twin.pedalboard.effects]
        if smoke_effects != twin_effects:
            offenders.append(f"{smoke.name}: a different effects chain from {twin_path.name}")
        declared = {
            (condition.name, tuple(sorted(condition.overrides.items())))
            for condition in twin.benchmark.conditions
        }
        for condition in cfg.benchmark.conditions:
            if (condition.name, tuple(sorted(condition.overrides.items()))) not in declared:
                offenders.append(f"{smoke.name}: {condition.name} is not in {twin_path.name}")
        # MIDI input synthesises from the reference, so the run needs no
        # recordings.
        if cfg.render_pipeline.input_type is not InputType.MIDI:
            offenders.append(f"{smoke.name}: input_type={cfg.render_pipeline.input_type.value}")
        expanded = expand_conditions(cfg.benchmark)
        if not expanded or expanded[0].name != cfg.benchmark.baseline_name:
            offenders.append(f"{smoke.name}: does not expand from the baseline first")
    assert not offenders, "the smoke runs drifted from their twin:\n" + "\n".join(offenders)


def test_tuning_levels_stay_within_the_accuracy_guard() -> None:
    """A preset asks for an offset the renderer is prepared to warn about.

    Half a semitone is that boundary: at or beyond it a note is as close to one
    named pitch as to its neighbour, so the chain builder warns that notes may
    score as the neighbouring pitch. The boundary is the builder's contract, not
    a number a preset happens to use, and a preset outside it would ask the
    renderer to score a shift it has already said it cannot.
    """
    checked = 0
    offenders: list[str] = []
    for path in presets():
        for condition in load_config(path).benchmark.conditions:
            if "tune=" not in condition.name:
                continue
            try:
                instrument_cents, return_cents = _expected_slots(condition.name)
            except ValueError as exc:
                offenders.append(f"{path.name}: {condition.name}: {exc}")
                continue
            checked += 1
            levels = [instrument_cents] + ([] if return_cents is None else [return_cents])
            for level in levels:
                if abs(level) > _ACCURACY_GUARD_CENTS:
                    offenders.append(
                        f"{path.name}: {condition.name}: {level}c is outside the "
                        f"+/-{_ACCURACY_GUARD_CENTS}c the renderer guards"
                    )
    assert not offenders, "presets ask for an uncovered offset:\n" + "\n".join(offenders)
    assert checked, "no preset declared a tuning condition"