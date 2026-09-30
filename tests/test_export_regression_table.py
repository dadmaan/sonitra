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


def test_build_rows_includes_recording_and_source_path_when_present(ert: ModuleType) -> None:
    record = _make_record(
        ert,
        midi_path="corpus/guitarset/midi/00_BN1-129-Eb_comp.mid",
        source_path="corpus/guitarset/recordings/00_BN1-129-Eb_comp_mic.wav",
        metrics={"note.onset_f1": 0.9},
    )

    rows = ert.build_rows([record], effect_types={})

    row = rows[0]
    assert row["song"] == "00_BN1-129-Eb_comp"
    assert row["recording"] == "00_BN1-129-Eb_comp_mic"
    assert row["source_path"] == "corpus/guitarset/recordings/00_BN1-129-Eb_comp_mic.wav"


def test_build_rows_omits_recording_and_source_path_when_none(ert: ModuleType) -> None:
    record = _make_record(ert, source_path=None, metrics={"note.onset_f1": 0.9})

    rows = ert.build_rows([record], effect_types={})

    row = rows[0]
    assert "recording" not in row
    assert "source_path" not in row


def test_write_csv_omits_recording_columns_for_midi_mode(ert: ModuleType, tmp_path: Path) -> None:
    record = _make_record(ert, source_path=None)
    rows = ert.build_rows([record], effect_types={})
    out_path = tmp_path / "regression_table.csv"

    ert.write_csv(rows, out_path)

    with out_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "recording" not in reader.fieldnames
        assert "source_path" not in reader.fieldnames


def test_write_csv_includes_recording_columns_in_identity_order(
    ert: ModuleType, tmp_path: Path
) -> None:
    record = _make_record(
        ert,
        midi_path="corpus/guitarset/midi/00_BN1-129-Eb_comp.mid",
        source_path="corpus/guitarset/recordings/00_BN1-129-Eb_comp_mic.wav",
    )
    rows = ert.build_rows([record], effect_types={})
    out_path = tmp_path / "regression_table.csv"

    ert.write_csv(rows, out_path)

    with out_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        identity = [column for column in reader.fieldnames if column in ert._IDENTITY_COLUMNS]
        assert identity == [
            "condition",
            "transcriber",
            "song",
            "recording",
            "midi_path",
            "source_path",
            "status",
        ]


def test_build_rows_distinguishes_recordings_sharing_midi_path(ert: ModuleType) -> None:
    midi_path = "corpus/guitarset/midi/00_BN1-129-Eb_comp.mid"
    mic = _make_record(
        ert,
        midi_path=midi_path,
        source_path="corpus/guitarset/recordings/00_BN1-129-Eb_comp_mic.wav",
    )
    mix = _make_record(
        ert,
        midi_path=midi_path,
        source_path="corpus/guitarset/recordings/00_BN1-129-Eb_comp_mix.wav",
    )

    rows = ert.build_rows([mic, mix], effect_types={})

    assert len(rows) == 2
    assert rows[0]["song"] == rows[1]["song"] == "00_BN1-129-Eb_comp"
    assert rows[0]["midi_path"] == rows[1]["midi_path"] == midi_path
    assert rows[0]["recording"] == "00_BN1-129-Eb_comp_mic"
    assert rows[1]["recording"] == "00_BN1-129-Eb_comp_mix"
    assert rows[0]["recording"] != rows[1]["recording"]
    assert rows[0]["source_path"] != rows[1]["source_path"]


def test_build_rows_metadata_join_keys_on_song_for_both_recordings(ert: ModuleType) -> None:
    midi_path = "corpus/guitarset/midi/00_BN1-129-Eb_comp.mid"
    mic = _make_record(
        ert,
        midi_path=midi_path,
        source_path="corpus/guitarset/recordings/00_BN1-129-Eb_comp_mic.wav",
    )
    mix = _make_record(
        ert,
        midi_path=midi_path,
        source_path="corpus/guitarset/recordings/00_BN1-129-Eb_comp_mix.wav",
    )
    metadata = {"00_BN1-129-Eb_comp": {"midi_filename": "00_BN1-129-Eb_comp.mid", "style": "BN"}}

    rows = ert.build_rows([mic, mix], effect_types={}, metadata=metadata)

    assert len(rows) == 2
    for row in rows:
        assert row["meta.midi_filename"] == "00_BN1-129-Eb_comp.mid"
        assert row["meta.style"] == "BN"


# ---------------------------------------------------------------------------
# token-prefix metadata join (MusicNet)
# ---------------------------------------------------------------------------


def test_build_rows_token_prefix_joins_1727_schubert_to_1727(ert: ModuleType) -> None:
    """1727_schubert_op114_2 joins to id 1727 in token-prefix mode."""
    record = _make_record(
        ert,
        midi_path="corpus/musicnet/midi/1727_schubert_op114_2.mid",
        metrics={"note.onset_f1": 0.9},
    )
    metadata = {"1727": {"id": "1727", "composer": "Schubert", "composition": "Op114"}}

    rows = ert.build_rows([record], effect_types={}, metadata=metadata, metadata_match="token-prefix")

    assert rows[0]["meta.id"] == "1727"
    assert rows[0]["meta.composer"] == "Schubert"


def test_build_rows_default_exact_does_not_join_prefix(ert: ModuleType) -> None:
    """Default (exact) mode stays byte-identical: prefix does not join."""
    record = _make_record(
        ert,
        midi_path="corpus/musicnet/midi/1727_schubert_op114_2.mid",
        metrics={"note.onset_f1": 0.9},
    )
    metadata = {"1727": {"id": "1727", "composer": "Schubert"}}

    rows_default = ert.build_rows([record], effect_types={}, metadata=metadata)
    rows_exact = ert.build_rows([record], effect_types={}, metadata=metadata, metadata_match="exact")

    assert not any(k.startswith("meta.") for k in rows_default[0])
    assert not any(k.startswith("meta.") for k in rows_exact[0])


def test_build_rows_exact_beats_prefix_in_token_prefix_mode(ert: ModuleType) -> None:
    """If metadata has both 1727 and 1727_schubert_op114_2, song 1727 hits exact."""
    record = _make_record(
        ert,
        midi_path="corpus/musicnet/midi/1727.mid",
        metrics={"note.onset_f1": 0.9},
    )
    metadata = {
        "1727": {"id": "1727", "composer": "A"},
        "1727_schubert_op114_2": {"id": "1727_schubert_op114_2", "composer": "B"},
    }

    rows = ert.build_rows([record], effect_types={}, metadata=metadata, metadata_match="token-prefix")

    # Without exact-first, k=1 would see both candidates sharing prefix 1727
    # and would be ambiguous -> no match.  Exact-first must win.
    assert rows[0]["meta.composer"] == "A"
    assert rows[0]["meta.id"] == "1727"


def test_build_rows_token_prefix_ambiguous_is_unmatched(ert: ModuleType) -> None:
    """Ambiguous prefix means no match and no meta columns."""
    record = _make_record(
        ert,
        midi_path="corpus/musicnet/midi/1727_schubert_op114_2.mid",
        metrics={"note.onset_f1": 0.9},
    )
    # Both share prefix 1727 at k=1, so 1727_schubert_op114_2 is ambiguous.
    metadata = {
        "1727_a": {"id": "1727_a"},
        "1727_b": {"id": "1727_b"},
    }

    rows = ert.build_rows([record], effect_types={}, metadata=metadata, metadata_match="token-prefix")

    assert not any(k.startswith("meta.") for k in rows[0])


def test_build_rows_token_prefix_cached_per_song(ert: ModuleType) -> None:
    """Two rows with same song share the cached prefix lookup (no error)."""
    metadata = {"1727": {"id": "1727"}}
    records = [
        _make_record(ert, midi_path="corpus/musicnet/midi/1727_schubert_op114_2.mid"),
        _make_record(ert, midi_path="corpus/musicnet/midi/1727_schubert_op114_2.mid"),
    ]

    rows = ert.build_rows(records, effect_types={}, metadata=metadata, metadata_match="token-prefix")

    assert len(rows) == 2
    for row in rows:
        assert row["meta.id"] == "1727"


def test_main_token_prefix_join_and_unmatched_count(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """main with --metadata-match token-prefix: prefix match joins, ambiguous counted."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    results_path = work_dir / "benchmark_results.jsonl"
    records = [
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/musicnet/midi/1727_schubert_op114_2.mid",
            "audio_path": "a.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.9},
            "overrides": {},
        },
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/musicnet/midi/9999_unknown_piece.mid",
            "audio_path": "b.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.7},
            "overrides": {},
        },
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/musicnet/midi/ambiguous_song.mid",
            "audio_path": "c.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.5},
            "overrides": {},
        },
    ]
    with results_path.open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    metadata_path = work_dir / "metadata.csv"
    _write_metadata_csv(
        metadata_path,
        [
            {"id": "1727", "composer": "Schubert"},
            {"id": "1727_a", "composer": "A"},
            {"id": "1727_b", "composer": "B"},
        ],
        fieldnames=["id", "composer"],
    )

    # In this setup, 1727_schubert_op114_2 would be ambiguous if candidates were
    # 1727_a/1727_b, but we have distinct 1727 as well.  For the
    # ambiguous_song test we need candidates that force ambiguity for that song.
    # ambiguous_song splits as ["ambiguous","song"]; make two metadata keys
    # sharing that prefix.
    # Recreate with appropriate keys: 1727 for first, and ambiguous_a/b for third.
    metadata_path2 = work_dir / "metadata2.csv"
    _write_metadata_csv(
        metadata_path2,
        [
            {"id": "1727", "composer": "Schubert"},
            {"id": "ambiguous_a", "composer": "A"},
            {"id": "ambiguous_b", "composer": "B"},
        ],
        fieldnames=["id", "composer"],
    )

    out_path = work_dir / "regression_table.csv"
    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path2),
            "--metadata-join-column", "id",
            "--metadata-match", "token-prefix",
        ]
    )

    assert exit_code == 0
    with out_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_song = {row["song"]: row for row in rows}
    # 1727_schubert_op114_2 -> 1727 via prefix
    assert by_song["1727_schubert_op114_2"]["meta.composer"] == "Schubert"
    # ambiguous_song -> ambiguous (both ambiguous_a/b share prefix) -> no meta
    assert by_song["ambiguous_song"]["meta.composer"] == ""
    # unknown -> no match
    assert by_song["9999_unknown_piece"]["meta.composer"] == ""

    stderr = capsys.readouterr().err
    # 2 unmatched out of 3 (ambiguous + unknown), prefix match is NOT counted
    assert "2/3 songs had no metadata match" in stderr
    # In token-prefix mode there should be no hint
    assert "--metadata-match token-prefix" not in stderr


def test_main_exact_mode_warning_suggests_token_prefix(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    results_path = work_dir / "benchmark_results.jsonl"
    records = [
        {
            "condition": "baseline",
            "transcriber": "basic_pitch",
            "midi_path": "corpus/musicnet/midi/1727_schubert_op114_2.mid",
            "audio_path": "a.wav",
            "status": "succeeded",
            "metrics": {"note.onset_f1": 0.9},
            "overrides": {},
        },
    ]
    with results_path.open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    metadata_path = work_dir / "metadata.csv"
    _write_metadata_csv(
        metadata_path,
        [{"id": "1727", "composer": "Schubert"}],
        fieldnames=["id", "composer"],
    )

    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path),
            "--metadata-join-column", "id",
            # default exact
        ]
    )

    assert exit_code == 0
    stderr = capsys.readouterr().err
    assert "1/1 songs had no metadata match" in stderr
    assert "--metadata-match token-prefix" in stderr


def test_main_token_prefix_is_byte_identical_without_flag(
    ert: ModuleType, tmp_path: Path
) -> None:
    """Exports without --metadata-match stay byte-identical to before."""
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
    ]
    with results_path.open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    metadata_path = work_dir / "metadata.csv"
    _write_metadata_csv(
        metadata_path,
        [{"midi_filename": "song_01.midi", "composer": "Bach"}],
        fieldnames=["midi_filename", "composer"],
    )

    # Exact (default) run
    ert.main(["--work-dir", str(work_dir), "--metadata-csv", str(metadata_path)])
    with (work_dir / "regression_table.csv").open() as f:
        content_default = f.read()

    # Explicit exact should be identical
    ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path),
            "--metadata-match", "exact",
        ]
    )
    with (work_dir / "regression_table.csv").open() as f:
        content_exact = f.read()

    assert content_default == content_exact


# --- --split filter -----------------------------------------------------------


def _write_split_fixture(work_dir: Path) -> Path:
    """Three songs (train/test/validation) plus one song missing from the metadata."""
    results_path = work_dir / "benchmark_results.jsonl"
    songs = ["song_train", "song_test", "song_val", "song_unlisted"]
    with results_path.open("w") as handle:
        for condition in ("baseline", "reverb"):
            for song in songs:
                handle.write(
                    json.dumps(
                        {
                            "condition": condition,
                            "transcriber": "basic_pitch",
                            "midi_path": f"corpus/maestro/midi/2018/{song}.midi",
                            "audio_path": f"{condition}/{song}.wav",
                            "status": "succeeded",
                            "metrics": {"note.onset_f1": 0.8},
                            "overrides": {},
                        }
                    )
                    + "\n"
                )
    metadata_path = work_dir / "metadata.csv"
    _write_metadata_csv(
        metadata_path,
        [
            {"midi_filename": "2018/song_train.midi", "split": "train"},
            {"midi_filename": "2018/song_test.midi", "split": "test"},
            {"midi_filename": "2018/song_val.midi", "split": "validation"},
        ],
        fieldnames=["midi_filename", "split"],
    )
    return metadata_path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_filter_rows_by_split_keeps_only_requested_values(ert: ModuleType) -> None:
    rows = [
        {"song": "a", "meta.split": "train"},
        {"song": "b", "meta.split": "test"},
        {"song": "c"},  # no metadata match -> split unknown
    ]

    kept = ert.filter_rows_by_split(rows, "split", {"test"})

    assert [row["song"] for row in kept] == ["b"]


def test_main_split_writes_only_test_rows_to_split_named_default_output(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    metadata_path = _write_split_fixture(work_dir)

    exit_code = ert.main(
        ["--work-dir", str(work_dir), "--metadata-csv", str(metadata_path), "--split", "test"]
    )

    assert exit_code == 0
    # The unfiltered default table is never written, so a split export can't clobber it.
    assert not (work_dir / "regression_table.csv").exists()
    rows = _read_csv_rows(work_dir / "regression_table_split-test.csv")
    assert {row["song"] for row in rows} == {"song_test"}
    assert {row["condition"] for row in rows} == {"baseline", "reverb"}
    assert all(row["meta.split"] == "test" for row in rows)
    out = capsys.readouterr().out
    assert "wrote 2 rows across 2 conditions" in out


def test_main_split_accepts_multiple_values(ert: ModuleType, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    metadata_path = _write_split_fixture(work_dir)

    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path),
            "--split", "validation",
            "--split", "test",
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(work_dir / "regression_table_split-test+validation.csv")
    assert {row["song"] for row in rows} == {"song_test", "song_val"}


def test_main_split_honours_explicit_output(ert: ModuleType, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    metadata_path = _write_split_fixture(work_dir)
    out_path = tmp_path / "custom.csv"

    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path),
            "--split", "test",
            "--output", str(out_path),
        ]
    )

    assert exit_code == 0
    assert {row["song"] for row in _read_csv_rows(out_path)} == {"song_test"}


def test_main_split_reports_rows_dropped_for_missing_metadata(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    metadata_path = _write_split_fixture(work_dir)

    ert.main(["--work-dir", str(work_dir), "--metadata-csv", str(metadata_path), "--split", "test"])

    err = capsys.readouterr().err
    assert "kept 2/8 rows" in err
    assert "2 rows (1 songs) have no metadata match" in err


def test_main_split_uses_custom_split_column(ert: ModuleType, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_split_fixture(work_dir)
    metadata_path = work_dir / "metadata_fold.csv"
    _write_metadata_csv(
        metadata_path,
        [
            {"midi_filename": "song_train.midi", "fold": "dev"},
            {"midi_filename": "song_test.midi", "fold": "eval"},
        ],
        fieldnames=["midi_filename", "fold"],
    )

    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(metadata_path),
            "--split-column", "fold",
            "--split", "eval",
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(work_dir / "regression_table_split-eval.csv")
    assert {row["song"] for row in rows} == {"song_test"}


def test_main_split_without_metadata_csv_is_a_usage_error(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_split_fixture(work_dir)

    with pytest.raises(SystemExit) as excinfo:
        ert.main(["--work-dir", str(work_dir), "--split", "test"])

    assert excinfo.value.code == 2
    assert "--split requires --metadata-csv" in capsys.readouterr().err


def test_main_split_missing_metadata_file_fails_without_writing(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_split_fixture(work_dir)

    exit_code = ert.main(
        [
            "--work-dir", str(work_dir),
            "--metadata-csv", str(work_dir / "nope.csv"),
            "--split", "test",
        ]
    )

    assert exit_code == 1
    assert "nope.csv" in capsys.readouterr().err
    assert not list(work_dir.glob("regression_table*.csv"))


def test_main_split_column_absent_fails_without_writing(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_split_fixture(work_dir)
    metadata_path = work_dir / "no_split.csv"
    _write_metadata_csv(
        metadata_path,
        [{"midi_filename": "song_test.midi", "composer": "Bach"}],
        fieldnames=["midi_filename", "composer"],
    )

    exit_code = ert.main(
        ["--work-dir", str(work_dir), "--metadata-csv", str(metadata_path), "--split", "test"]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "'split'" in err and "composer" in err
    assert not list(work_dir.glob("regression_table*.csv"))


def test_main_split_unknown_value_fails_and_lists_available(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    metadata_path = _write_split_fixture(work_dir)

    exit_code = ert.main(
        ["--work-dir", str(work_dir), "--metadata-csv", str(metadata_path), "--split", "Test"]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "'Test'" in err
    assert "test, train, validation" in err
    assert not list(work_dir.glob("regression_table*.csv"))


def test_main_split_with_no_matching_rows_fails_without_writing(
    ert: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid split value that no run file belongs to (e.g. a train-only smoke run)."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_split_fixture(work_dir)
    metadata_path = work_dir / "metadata_extra.csv"
    _write_metadata_csv(
        metadata_path,
        [
            {"midi_filename": "song_train.midi", "split": "train"},
            {"midi_filename": "other.midi", "split": "test"},
        ],
        fieldnames=["midi_filename", "split"],
    )

    exit_code = ert.main(
        ["--work-dir", str(work_dir), "--metadata-csv", str(metadata_path), "--split", "test"]
    )

    assert exit_code == 1
    assert "no rows" in capsys.readouterr().err
    assert not list(work_dir.glob("regression_table*.csv"))
