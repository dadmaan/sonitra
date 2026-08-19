from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest
import yaml

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "export_regression_table.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_regression_table", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ert() -> ModuleType:
    return _load_module()


def _minimal_pedalboard_config() -> dict:
    """Just enough of a resolved PipelineConfig to exercise pedalboard.effects."""
    return {
        "pedalboard": {
            "effects": [
                {"type": "HighpassFilter", "enabled": True, "cutoff_frequency_hz": 100.0},
                {"type": "HighpassFilter", "enabled": True, "cutoff_frequency_hz": 100.0},
                {"type": "LowpassFilter", "enabled": True, "cutoff_frequency_hz": 8000.0},
                {"type": "LowpassFilter", "enabled": True, "cutoff_frequency_hz": 8000.0},
                {
                    "type": "PeakFilter",
                    "enabled": False,
                    "cutoff_frequency_hz": 2500.0,
                    "gain_db": 3.0,
                    "q": 1.0,
                },
                {"type": "PeakFilter", "enabled": False, "cutoff_frequency_hz": 80.0, "gain_db": 2.0, "q": 1.0},
                {"type": "Distortion", "enabled": False, "drive_db": 4.0},
                {
                    "type": "Compressor",
                    "enabled": False,
                    "threshold_db": -18.0,
                    "ratio": 4.0,
                    "attack_ms": 5.0,
                    "release_ms": 100.0,
                },
            ]
        }
    }


def test_load_effect_types_parses_pedalboard_effects(ert: ModuleType, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_pedalboard_config()))

    effect_types = ert.load_effect_types(config_path)

    assert effect_types == {
        0: "HighpassFilter",
        1: "HighpassFilter",
        2: "LowpassFilter",
        3: "LowpassFilter",
        4: "PeakFilter",
        5: "PeakFilter",
        6: "Distortion",
        7: "Compressor",
    }


def test_load_effect_types_missing_file_returns_empty(ert: ModuleType, tmp_path: Path) -> None:
    assert ert.load_effect_types(tmp_path / "does-not-exist.yaml") == {}


def test_rename_override_key_known_effect_type(ert: ModuleType) -> None:
    effect_types = {0: "HighpassFilter", 7: "Compressor"}
    assert (
        ert.rename_override_key("pedalboard.effects.0.cutoff_frequency_hz", effect_types)
        == "override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"
    )
    assert (
        ert.rename_override_key("pedalboard.effects.7.ratio", effect_types)
        == "override.pedalboard.effects.7_Compressor.ratio"
    )


def test_rename_override_key_unknown_effect_type_falls_back(ert: ModuleType) -> None:
    # No effect_types available (e.g. no config.yaml snapshot) -> raw dotted path.
    assert (
        ert.rename_override_key("pedalboard.effects.0.cutoff_frequency_hz", {})
        == "override.pedalboard.effects.0.cutoff_frequency_hz"
    )


def test_rename_override_key_non_pedalboard_path_passes_through(ert: ModuleType) -> None:
    effect_types = {0: "HighpassFilter"}
    assert (
        ert.rename_override_key("render_pipeline.sample_rate", effect_types)
        == "override.render_pipeline.sample_rate"
    )


def _make_record(ert: ModuleType, **kwargs):
    from sonitra.benchmark.results import BenchmarkRecord

    defaults = dict(
        condition="baseline",
        transcriber="basic_pitch",
        midi_path="corpus/maestro/midi/2004/song_01.midi",
        audio_path="corpus/maestro/audio/song_01.wav",
        status="succeeded",
        metrics={},
        overrides={},
    )
    defaults.update(kwargs)
    return BenchmarkRecord(**defaults)


def test_build_rows_baseline_has_no_override_columns(ert: ModuleType) -> None:
    record = _make_record(
        ert,
        metrics={"note.onset_f1": 0.82},
    )
    rows = ert.build_rows([record], effect_types={})

    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == "baseline"
    assert row["transcriber"] == "basic_pitch"
    assert row["song"] == "song_01"
    assert row["midi_path"] == "corpus/maestro/midi/2004/song_01.midi"
    assert row["status"] == "succeeded"
    assert row["note.onset_f1"] == pytest.approx(0.82)
    assert not any(key.startswith("override.") for key in row)


def test_build_rows_flattens_overrides_with_effect_type_labels(ert: ModuleType) -> None:
    record = _make_record(
        ert,
        condition="shellac_bandlimit_mild",
        metrics={"note.onset_f1": float("nan"), "frame.f1": 0.65},
        overrides={
            "pedalboard.effects.0.cutoff_frequency_hz": 51.4,
            "pedalboard.effects.4.gain_db": 3.0,
            "pedalboard.effects.7.ratio": 6.0,
        },
    )
    effect_types = {0: "HighpassFilter", 4: "PeakFilter", 7: "Compressor"}

    rows = ert.build_rows([record], effect_types=effect_types)

    row = rows[0]
    assert row["override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"] == 51.4
    assert row["override.pedalboard.effects.4_PeakFilter.gain_db"] == 3.0
    assert row["override.pedalboard.effects.7_Compressor.ratio"] == 6.0
    # NaN metrics become "" (matches the JSONL->CSV convention documented in CLAUDE.md)
    assert row["note.onset_f1"] == ""
    assert row["frame.f1"] == pytest.approx(0.65)


def test_write_csv_union_of_keys(ert: ModuleType, tmp_path: Path) -> None:
    rows = [
        {"condition": "baseline", "note.onset_f1": 0.9},
        {"condition": "shellac_bandlimit_mild", "note.onset_f1": 0.8, "override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz": 51.4},
    ]
    out_path = tmp_path / "regression_table.csv"

    ert.write_csv(rows, out_path)

    with out_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert set(reader.fieldnames) == {
        "condition", "note.onset_f1", "override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"
    }
    assert written[0]["override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"] == ""
    assert written[1]["override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"] == "51.4"


def test_main_end_to_end(ert: ModuleType, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_pedalboard_config()))

    results_path = work_dir / "benchmark_results.jsonl"
    records = [
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/maestro/midi/song_01.midi",
            "audio_path": "a.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.9},
            "overrides": {},
        },
        {
            "condition": "shellac_bandlimit_mild",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/maestro/midi/song_01.midi",
            "audio_path": "b.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.85},
            "overrides": {"pedalboard.effects.0.cutoff_frequency_hz": 51.4},
        },
    ]
    with results_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    out_path = work_dir / "regression_table.csv"
    exit_code = ert.main(["--work-dir", str(work_dir)])

    assert exit_code == 0
    assert out_path.exists()
    with out_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_condition = {row["condition"]: row for row in rows}
    assert (
        by_condition["shellac_bandlimit_mild"][
            "override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"
        ]
        == "51.4"
    )
    assert by_condition["baseline"][
        "override.pedalboard.effects.0_HighpassFilter.cutoff_frequency_hz"
    ] == ""


def _write_metadata_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_metadata_join_indexes_by_basename_of_join_column(ert: ModuleType, tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        csv_path,
        [
            {
                "canonical_composer": "Alban Berg",
                "canonical_title": "Sonata Op. 1",
                "midi_filename": "2018/MIDI-Unprocessed_Chamber3.midi",
            }
        ],
        fieldnames=["canonical_composer", "canonical_title", "midi_filename"],
    )

    index = ert.load_metadata_join(csv_path, "midi_filename")

    assert index == {
        "MIDI-Unprocessed_Chamber3": {
            "canonical_composer": "Alban Berg",
            "canonical_title": "Sonata Op. 1",
            "midi_filename": "2018/MIDI-Unprocessed_Chamber3.midi",
        }
    }


def test_load_metadata_join_missing_file_returns_empty(ert: ModuleType, tmp_path: Path) -> None:
    assert ert.load_metadata_join(tmp_path / "does-not-exist.csv", "midi_filename") == {}


def test_load_metadata_join_unknown_join_column_raises(ert: ModuleType, tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_csv(csv_path, [{"a": "1"}], fieldnames=["a"])

    with pytest.raises(ValueError, match="not_a_column"):
        ert.load_metadata_join(csv_path, "not_a_column")


def test_load_metadata_join_duplicate_key_keeps_first_and_warns(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        csv_path,
        [
            {"midi_filename": "2018/song.midi", "canonical_composer": "First"},
            {"midi_filename": "2019/song.midi", "canonical_composer": "Second"},
        ],
        fieldnames=["midi_filename", "canonical_composer"],
    )

    index = ert.load_metadata_join(csv_path, "midi_filename")

    assert index["song"]["canonical_composer"] == "First"
    assert "duplicate" in capsys.readouterr().err.lower()


def test_build_rows_joins_metadata_for_matched_song(ert: ModuleType) -> None:
    record = _make_record(
        ert,
        midi_path="corpus/maestro/midi/2018/song_01.midi",
        metrics={"note.onset_f1": 0.9},
    )
    metadata = {"song_01": {"canonical_composer": "Bach", "canonical_title": "Foo Piece"}}

    rows = ert.build_rows([record], effect_types={}, metadata=metadata)

    row = rows[0]
    assert row["meta.canonical_composer"] == "Bach"
    assert row["meta.canonical_title"] == "Foo Piece"


def test_build_rows_no_meta_keys_for_unmatched_song(ert: ModuleType) -> None:
    record = _make_record(
        ert,
        midi_path="corpus/maestro/midi/2018/unknown_song.midi",
        metrics={"note.onset_f1": 0.9},
    )
    metadata = {"song_01": {"canonical_composer": "Bach"}}

    rows = ert.build_rows([record], effect_types={}, metadata=metadata)

    assert not any(key.startswith("meta.") for key in rows[0])


def test_main_end_to_end_with_metadata_join(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    results_path = work_dir / "benchmark_results.jsonl"
    records = [
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/maestro/midi/song_01.midi",
            "audio_path": "a.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.9},
            "overrides": {},
        },
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/maestro/midi/song_02.midi",
            "audio_path": "c.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.7},
            "overrides": {},
        },
    ]
    with results_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    metadata_path = work_dir / "metadata.csv"
    _write_metadata_csv(
        metadata_path,
        [{"midi_filename": "song_01.midi", "canonical_composer": "Bach"}],
        fieldnames=["midi_filename", "canonical_composer"],
    )

    out_path = work_dir / "regression_table.csv"
    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path),
            "--metadata-join-column", "midi_filename",
        ]
    )

    assert exit_code == 0
    with out_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_song = {row["song"]: row for row in rows}
    assert by_song["song_01"]["meta.canonical_composer"] == "Bach"
    assert by_song["song_02"]["meta.canonical_composer"] == ""

    stderr = capsys.readouterr().err
    assert "1/2 songs had no metadata match" in stderr
