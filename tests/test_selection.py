from __future__ import annotations

import csv
import hashlib
import importlib.util
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import sonitra.selection as selection
from sonitra.config import PipelineConfig, SelectionSample
from sonitra.corpus import pair_audio_to_reference
from sonitra.selection import (
    SelectionError,
    load_selection_metadata,
    resolve_metadata_csv,
    sample_units,
    select_audio,
    select_references,
    split_notice,
)

DATASET = "maestro-v3"

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "export_regression_table.py"


def _base_payload(
    corpus_root: Path,
    *,
    dataset: str | None = DATASET,
    selection: dict[str, Any] | None = None,
    file_naming: str = "{stem}",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
            "corpus_root": str(corpus_root),
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": file_naming,
            "dataset": dataset,
        },
    }
    if selection is not None:
        payload["io"].update(selection)
    return payload


def _cfg(
    corpus_root: Path,
    *,
    dataset: str | None = DATASET,
    selection: dict[str, Any] | None = None,
    file_naming: str = "{stem}",
) -> PipelineConfig:
    return PipelineConfig.model_validate(
        _base_payload(
            corpus_root,
            dataset=dataset,
            selection=selection,
            file_naming=file_naming,
        )
    )


def _where_selection(
    where: dict[str, Any],
    *,
    metadata_csv: str = "meta.csv",
) -> dict[str, Any]:
    return {"metadata_csv": metadata_csv, "where": where}


def _touch(root: Path, dataset: str, section: str, name: str) -> Path:
    path = root / dataset / section / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _write_metadata(
    root: Path,
    rows: list[dict[str, str]],
    *,
    dataset: str = DATASET,
    name: str = "meta.csv",
) -> Path:
    path = root / dataset / "metadata" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


_SPLIT_ROWS = [
    {"midi_filename": "train_a.mid", "split": "train"},
    {"midi_filename": "train_b.mid", "split": "train"},
    {"midi_filename": "train_c.mid", "split": "train"},
    {"midi_filename": "val_a.mid", "split": "validation"},
    {"midi_filename": "test_a.mid", "split": "test"},
    {"midi_filename": "test_b.mid", "split": "test"},
]


def _reference_fixture(root: Path) -> tuple[list[Path], Path]:
    _write_metadata(root, _SPLIT_ROWS)
    refs = sorted(_touch(root, DATASET, "midi", row["midi_filename"]) for row in _SPLIT_ROWS)
    return refs, root / DATASET / "midi"


def _legacy_apply_subset(
    files: list[Path], limit: int | None, seed: int | None
) -> list[Path]:
    if limit is None or limit >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, limit))


def _load_export_regression_table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_regression_table", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Metadata and validation ─────────────────────────────────────────


def test_resolve_metadata_csv_relative_to_dataset_metadata_dir(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "a.mid", "split": "test"}])
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    resolved = resolve_metadata_csv(cfg)
    assert resolved == root / DATASET / "metadata" / "meta.csv"
    assert "corpus" not in resolved.parts


def test_metadata_csv_resolves_under_io_dataset(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(
        root,
        [{"midi_filename": "a.mid", "split": "test"}],
        dataset="gaps",
    )
    cfg = _cfg(
        root,
        dataset="gaps",
        selection=_where_selection({"split": ["test"]}),
    )
    assert resolve_metadata_csv(cfg) == root / "gaps" / "metadata" / "meta.csv"


def test_provenance_dataset_from_io(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs, midi_dir = _reference_fixture(root)

    sampled = select_references(_cfg(root, selection={"sample": {"n": 2}}), refs)
    sampled_block = sampled.provenance(unit_root=midi_dir)
    assert sampled_block["configured"] is True
    assert sampled_block["dataset"] == DATASET

    plain = select_references(_cfg(root, selection=None), refs)
    assert plain.dataset is None
    plain_block = plain.provenance(unit_root=midi_dir)
    assert list(plain_block) == ["configured", "unit", "counts", "files_sha256"]

    configured_keys = list(
        select_references(
            _cfg(root, selection=_where_selection({"split": ["test"]})), refs
        ).provenance(unit_root=midi_dir)
    )
    assert list(sampled_block) == configured_keys


def test_resolve_metadata_csv_absolute_path_kept(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    other = tmp_path / "elsewhere" / "pinned.csv"
    other.parent.mkdir(parents=True)
    other.write_text("midi_filename,split\n", encoding="utf-8")
    cfg = _cfg(
        root,
        selection=_where_selection({"split": ["test"]}, metadata_csv=str(other)),
    )
    assert resolve_metadata_csv(cfg) == other


def test_repo_relative_metadata_csv_gives_E2_with_bare_filename_hint(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    cfg = _cfg(
        root,
        selection=_where_selection(
            {"split": ["test"]},
            metadata_csv=f"corpus/{DATASET}/metadata/meta.csv",
        ),
    )
    with pytest.raises(SelectionError, match=r"metadata CSV not found.*bare filename"):
        resolve_metadata_csv(cfg)


def test_missing_csv_raises_E2(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    with pytest.raises(SelectionError, match=r"metadata CSV not found"):
        resolve_metadata_csv(cfg)


def test_missing_join_column_raises_E3(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"path": "a.mid", "split": "test"}])
    refs = [_touch(root, DATASET, "midi", "a.mid")]
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    with pytest.raises(
        SelectionError, match=r"column 'midi_filename' not found.*\(columns: "
    ):
        select_references(cfg, refs)


def test_missing_where_column_raises_E4_listing_columns(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "a.mid", "split": "test"}])
    refs = [_touch(root, DATASET, "midi", "a.mid")]
    cfg = _cfg(root, selection=_where_selection({"year": ["2018"]}))
    with pytest.raises(
        SelectionError,
        match=r"column 'year' not found.*\(columns: midi_filename, split\)",
    ):
        select_references(cfg, refs)


def test_unknown_value_raises_E5_listing_sorted_nonempty_values(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    rows = [
        {"midi_filename": "2018/a.midi", "split": "test"},
        {"midi_filename": "2018/b.midi", "split": ""},
        {"midi_filename": "2018/c.midi", "split": "train"},
        {"midi_filename": "2018/d.midi", "split": "validation"},
    ]
    _write_metadata(root, rows)
    refs = sorted(
        _touch(root, DATASET, "midi", name)
        for name in ("a.mid", "b.mid", "c.mid", "d.mid")
    )
    cfg = _cfg(root, selection=_where_selection({"split": ["Test"]}))
    with pytest.raises(
        SelectionError, match=r"'Test'.*available: test, train, validation"
    ):
        select_references(cfg, refs)


def test_duplicate_keys_agreeing_on_where_columns_ok(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    csv_path = _write_metadata(
        root,
        [
            {"midi_filename": "2018/foo.midi", "split": "test", "year": "2018"},
            {"midi_filename": "2019/foo.midi", "split": "test", "year": "2019"},
        ],
    )
    refs = [_touch(root, DATASET, "midi", "foo.mid")]
    metadata = load_selection_metadata(csv_path, "midi_filename", {"split": ["test"]})
    assert metadata["foo"]["year"] == "2018"

    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, refs)
    assert result.units == refs


def test_duplicate_keys_conflicting_raise_E6(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    csv_path = _write_metadata(
        root,
        [
            {"midi_filename": "2018/foo.midi", "split": "test"},
            {"midi_filename": "2019/foo.midi", "split": "train"},
        ],
    )
    with pytest.raises(
        SelectionError,
        match=r"metadata key 'foo' appears 2 times with different 'split' values",
    ):
        load_selection_metadata(csv_path, "midi_filename", {"split": ["test"]})


# ── Reference selection ─────────────────────────────────────────────


def test_select_references_filters_by_split(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs, _ = _reference_fixture(root)
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, refs)

    assert {p.stem for p in result.units} == {"test_a", "test_b"}
    assert list(result.counts) == [
        "discovered",
        "unmatched",
        "excluded_by_where",
        "unpaired_audio",
        "selected_before_sample",
        "selected",
    ]
    assert result.counts == {
        "discovered": 6,
        "unmatched": 0,
        "excluded_by_where": 4,
        "unpaired_audio": 0,
        "selected_before_sample": 2,
        "selected": 2,
    }
    assert result.by_value == {"split": {"test": 2}}
    assert result.summary_line() == "selection: maestro-v3 split=test -> 2/6 files"


def test_where_is_and_across_columns_or_within_list(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    rows = [
        {"midi_filename": "m1.midi", "split": "test", "year": "2018"},
        {"midi_filename": "m2.midi", "split": "validation", "year": "2018"},
        {"midi_filename": "m3.midi", "split": "test", "year": "2019"},
        {"midi_filename": "m4.midi", "split": "validation", "year": "2019"},
        {"midi_filename": "m5.midi", "split": "train", "year": "2018"},
        {"midi_filename": "m6.midi", "split": "test", "year": "2018"},
    ]
    _write_metadata(root, rows)
    refs = sorted(_touch(root, DATASET, "midi", f"m{i}.mid") for i in range(1, 7))
    cfg = _cfg(
        root,
        selection=_where_selection({"split": ["test", "validation"], "year": ["2018"]}),
    )
    result = select_references(cfg, refs)
    assert {p.stem for p in result.units} == {"m1", "m2", "m6"}


def test_unmatched_reference_excluded_and_reported(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "matched.mid", "split": "test"}])
    refs = sorted(
        _touch(root, DATASET, "midi", name)
        for name in (
            "matched.mid",
            "a-fine-aligned.mid",
            "b-fine-aligned.mid",
            "c-fine-aligned.mid",
        )
    )
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, refs)

    assert [p.stem for p in result.unmatched] == [
        "a-fine-aligned",
        "b-fine-aligned",
        "c-fine-aligned",
    ]
    assert result.counts["unmatched"] == 3
    assert {p.stem for p in result.units} == {"matched"}


def test_join_key_is_stem_of_join_column(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "2018/foo.midi", "split": "test"}])
    nested = _touch(root, DATASET, "midi", "2018/foo.midi")
    flat = _touch(root, DATASET, "midi", "foo.mid")
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, sorted([nested, flat]))

    assert result.units == sorted([nested, flat])
    assert result.counts["unmatched"] == 0


def test_no_selection_returns_all_with_configured_false(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs, midi_dir = _reference_fixture(root)
    cfg = _cfg(root, selection=None)
    result = select_references(cfg, refs)

    assert result.units == list(refs)
    relative = sorted(p.relative_to(midi_dir).as_posix() for p in refs)
    expected_hash = hashlib.sha256("\n".join(relative).encode("utf-8")).hexdigest()
    assert result.provenance(unit_root=midi_dir) == {
        "configured": False,
        "unit": "reference_midi",
        "counts": {"discovered": 6, "selected": 6},
        "files_sha256": expected_hash,
    }
    assert result.summary_line() == "selection: none -> 6/6 files"


def test_empty_result_raises_E7(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(
        root,
        [
            {"midi_filename": "a.mid", "split": "train"},
            {"midi_filename": "b.mid", "split": "train"},
            {"midi_filename": "ghost.mid", "split": "test"},
        ],
    )
    refs = [
        _touch(root, DATASET, "midi", "a.mid"),
        _touch(root, DATASET, "midi", "b.mid"),
    ]
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    with pytest.raises(SelectionError, match=r"selection matched no files"):
        select_references(cfg, refs)


# ── Audio selection ─────────────────────────────────────────────────


def test_select_audio_pairs_against_full_reference_list(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(
        root,
        [
            {"midi_filename": "piece_A.mid", "split": "train"},
            {"midi_filename": "piece_B.mid", "split": "test"},
        ],
    )
    ref_a = _touch(root, DATASET, "midi", "piece_A.mid")
    ref_b = _touch(root, DATASET, "midi", "piece_B.mid")
    rec_a = _touch(root, DATASET, "recordings", "piece_A_mic.wav")
    rec_b = _touch(root, DATASET, "recordings", "piece_B_mic.wav")

    assert pair_audio_to_reference([rec_a], [ref_a, ref_b]).mapping == {rec_a: ref_a}
    assert pair_audio_to_reference([rec_a], [ref_b]).mapping == {rec_a: ref_b}

    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_audio(cfg, [rec_a, rec_b], [ref_a, ref_b], unit_kind="recording")

    assert result.units == [rec_b]
    assert rec_a not in result.units
    assert result.unpaired_audio == []
    assert result.counts["selected_before_sample"] == 1
    assert result.by_value == {"split": {"test": 1}}


def test_select_audio_without_where_returns_all_audio_even_with_no_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data-root"
    audio = sorted(
        _touch(root, DATASET, "recordings", name) for name in ("c.wav", "a.wav", "b.wav")
    )

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "pair_audio_to_reference must not be called without a where filter"
        )

    monkeypatch.setattr(selection, "pair_audio_to_reference", _explode)

    sample_only = _cfg(root, selection={"sample": {"n": 2}})
    sampled = select_audio(sample_only, audio, [], unit_kind="recording")
    assert sampled.units == sample_units(audio, SelectionSample(n=2))
    assert sampled.references == []

    plain = _cfg(root, selection=None)
    assert select_audio(plain, audio, [], unit_kind="recording").units == audio


def test_select_audio_file_naming_without_stem_prefix_raises_E8(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs = [_touch(root, DATASET, "midi", "a.mid")]
    audio = [_touch(root, DATASET, "audio", "a.wav")]
    cfg = _cfg(
        root,
        selection=_where_selection({"split": ["test"]}),
        file_naming="render_{stem}",
    )
    with pytest.raises(
        SelectionError, match=r"needs io\.file_naming to start with '\{stem\}'"
    ):
        select_audio(cfg, audio, refs, unit_kind="audio")


def test_select_audio_unpaired_audio_excluded_and_reported(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "paired_x.mid", "split": "test"}])
    refs = [_touch(root, DATASET, "midi", "paired_x.mid")]
    paired = _touch(root, DATASET, "recordings", "paired_x_mic.wav")
    orphan = _touch(root, DATASET, "recordings", "orphan.wav")
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))

    result = select_audio(cfg, [paired, orphan], refs, unit_kind="recording")

    assert result.units == [paired]
    assert result.unpaired_audio == [orphan]
    assert result.counts["unpaired_audio"] == 1
    assert result.counts["selected_before_sample"] == 1


def test_select_audio_always_pair_without_where_drops_unpaired(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    ref_a = _touch(root, DATASET, "midi", "piece_A.mid")
    ref_b = _touch(root, DATASET, "midi", "piece_B.mid")
    rec_a = _touch(root, DATASET, "recordings", "piece_A_mic.wav")
    rec_b = _touch(root, DATASET, "recordings", "piece_B_mic.wav")
    orphan = _touch(root, DATASET, "recordings", "orphan.wav")

    cfg = _cfg(root, selection=None)
    result = select_audio(
        cfg,
        [rec_a, rec_b, orphan],
        [ref_a, ref_b],
        unit_kind="recording",
        always_pair=True,
    )

    assert result.units == [rec_a, rec_b]
    assert result.unpaired_audio == [orphan]
    assert result.references == [ref_a, ref_b]
    assert result.counts["unpaired_audio"] == 1
    assert result.counts["excluded_by_where"] == 0
    assert result.unmatched == []


def test_select_audio_guitarset_mic_and_mix_both_follow_reference(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(
        root,
        [
            {"midi_filename": "piece_A.mid", "split": "test"},
            {"midi_filename": "piece_B.mid", "split": "train"},
        ],
    )
    ref_a = _touch(root, DATASET, "midi", "piece_A.mid")
    ref_b = _touch(root, DATASET, "midi", "piece_B.mid")
    audio = [
        _touch(root, DATASET, "recordings", "piece_A_mic.wav"),
        _touch(root, DATASET, "recordings", "piece_A_mix.wav"),
        _touch(root, DATASET, "recordings", "piece_B_mic.wav"),
        _touch(root, DATASET, "recordings", "piece_B_mix.wav"),
    ]
    mic_a, mix_a, mic_b, mix_b = audio
    refs = [ref_a, ref_b]

    test_cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    test_result = select_audio(test_cfg, audio, refs, unit_kind="recording")
    assert set(test_result.units) == {mic_a, mix_a}

    train_cfg = _cfg(root, selection=_where_selection({"split": ["train"]}))
    train_result = select_audio(train_cfg, audio, refs, unit_kind="recording")
    assert set(train_result.units) == {mic_b, mix_b}


# ── Sampling ────────────────────────────────────────────────────────


def test_sample_units_matches_legacy_apply_subset_for_same_seed() -> None:
    files = sorted(Path(f"f{index:02d}.mid") for index in range(10))
    for seed in (0, 1, 123):
        for n in (1, 2, 5):
            assert sample_units(files, SelectionSample(n=n, seed=seed)) == _legacy_apply_subset(
                files, n, seed
            )


def test_sample_units_default_seed_zero_is_deterministic() -> None:
    files = sorted(Path(f"f{index:02d}.mid") for index in range(10))
    first = sample_units(files, SelectionSample(n=3))
    second = sample_units(files, SelectionSample(n=3))
    explicit = sample_units(files, SelectionSample(n=3, seed=0))
    assert first == second == explicit
    assert len(first) == 3


def test_sample_units_n_ge_len_returns_all_sorted() -> None:
    files = [Path("b.mid"), Path("a.mid"), Path("c.mid")]
    expected = sorted(files)
    assert sample_units(files, SelectionSample(n=3)) == expected
    assert sample_units(files, SelectionSample(n=8)) == expected


def test_sample_applies_after_filter(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    rows = [
        {"midi_filename": "test_a.mid", "split": "test"},
        {"midi_filename": "test_b.mid", "split": "test"},
        {"midi_filename": "test_c.mid", "split": "test"},
        {"midi_filename": "train_a.mid", "split": "train"},
        {"midi_filename": "train_b.mid", "split": "train"},
        {"midi_filename": "train_c.mid", "split": "train"},
    ]
    _write_metadata(root, rows)
    refs = sorted(_touch(root, DATASET, "midi", row["midi_filename"]) for row in rows)
    for seed in range(51):
        selection_payload: dict[str, Any] = {
            **_where_selection({"split": ["test"]}),
            "sample": {"n": 2, "seed": seed},
        }
        cfg = _cfg(root, selection=selection_payload)
        result = select_references(cfg, refs)
        assert len(result.units) == 2
        assert all(p.stem.startswith("test_") for p in result.units)
        assert result.counts["selected_before_sample"] == 3
        assert result.counts["selected"] == 2


# ── Notice and provenance ───────────────────────────────────────────


def test_split_notice_when_metadata_has_split_column_and_no_selection(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "a.mid", "split": "test"}])
    cfg = _cfg(root, selection=None)
    assert split_notice(cfg, 6) == (
        "note: maestro-v3 metadata labels splits (meta.csv: split); "
        "no selection set, using all 6 files"
    )


def test_no_notice_when_selection_set(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "a.mid", "split": "test"}])
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    assert split_notice(cfg, 6) is None


def test_no_notice_without_metadata_dir(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    (root / DATASET / "midi").mkdir(parents=True)
    cfg = _cfg(root, selection=None)
    assert split_notice(cfg, 6) is None


def test_no_notice_when_no_split_column(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    _write_metadata(root, [{"midi_filename": "a.mid", "year": "2018"}])
    cfg = _cfg(root, selection=None)
    assert split_notice(cfg, 6) is None


def test_provenance_block_shape_configured(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs, midi_dir = _reference_fixture(root)
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, refs)

    provenance = result.provenance(unit_root=midi_dir)
    assert list(provenance) == [
        "configured",
        "dataset",
        "metadata_csv",
        "metadata_sha256",
        "join_column",
        "where",
        "sample",
        "unit",
        "counts",
        "by_value",
        "files_sha256",
    ]
    csv_path = root / DATASET / "metadata" / "meta.csv"
    assert provenance["configured"] is True
    assert provenance["dataset"] == DATASET
    assert provenance["metadata_csv"] == str(csv_path)
    assert provenance["metadata_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert provenance["join_column"] == "midi_filename"
    assert provenance["where"] == {"split": ["test"]}
    assert provenance["sample"] is None
    assert provenance["unit"] == "reference_midi"
    assert provenance["counts"] == {
        "discovered": 6,
        "unmatched": 0,
        "excluded_by_where": 4,
        "unpaired_audio": 0,
        "selected_before_sample": 2,
        "selected": 2,
    }
    assert provenance["by_value"] == {"split": {"test": 2}}
    assert provenance["files_sha256"] == hashlib.sha256(
        "test_a.mid\ntest_b.mid".encode("utf-8")
    ).hexdigest()


def test_provenance_files_sha256_uses_relative_posix_sorted_paths(tmp_path: Path) -> None:
    hashes: list[str] = []
    for name in ("root_a", "root_b"):
        root = tmp_path / name
        refs, midi_dir = _reference_fixture(root)
        cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
        result = select_references(cfg, refs)
        hashes.append(result.provenance(unit_root=midi_dir)["files_sha256"])

    assert hashes[0] == hashes[1]
    expected = hashlib.sha256(
        "\n".join(sorted(["test_a.mid", "test_b.mid"])).encode("utf-8")
    ).hexdigest()
    assert hashes[0] == expected


def test_provenance_metadata_sha256_matches_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs, _ = _reference_fixture(root)
    csv_path = root / DATASET / "metadata" / "meta.csv"
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, refs)
    assert result.metadata_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()


def test_summary_line_text(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    refs, _ = _reference_fixture(root)
    configured: dict[str, Any] = {
        **_where_selection({"split": ["test"]}),
        "sample": {"n": 1, "seed": 7},
    }
    result = select_references(_cfg(root, selection=configured), refs)
    assert result.summary_line() == (
        "selection: maestro-v3 split=test sample n=1 seed=7 -> 1/6 files"
    )

    plain = select_references(_cfg(root, selection=None), refs)
    assert plain.summary_line() == "selection: none -> 6/6 files"


# ── Gate ────────────────────────────────────────────────────────────


def test_join_semantics_match_export_regression_table(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    csv_path = _write_metadata(
        root,
        [
            {"midi_filename": "2018/foo.midi", "split": "test"},
            {"midi_filename": "bar.mid", "split": "train"},
            {"midi_filename": "baz.midi", "split": "Test"},
        ],
    )
    refs = sorted(
        [
            _touch(root, DATASET, "midi", "2018/foo.midi"),
            _touch(root, DATASET, "midi", "bar.mid"),
            _touch(root, DATASET, "midi", "baz.mid"),
        ]
    )
    cfg = _cfg(root, selection=_where_selection({"split": ["test"]}))
    result = select_references(cfg, refs)

    ert = _load_export_regression_table()
    metadata = ert.load_metadata_join(csv_path, "midi_filename")
    export_rows = [
        {"song": key, "meta.split": row["split"]}
        for key, row in sorted(metadata.items())
    ]
    export_stems = {
        row["song"] for row in ert.filter_rows_by_split(export_rows, "split", {"test"})
    }

    assert export_stems == {p.stem for p in result.units} == {"foo"}
