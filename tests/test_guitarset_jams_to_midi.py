"""Tests for scripts/guitarset_jams_to_midi.py.

JAMS documents are built inline by the local ``_make_jams`` helper (the
per-file-helper convention used in ``test_export_regression_table.py``), so
each test can vary the JAMS shape: list vs dict-of-arrays ``data`` layouts,
interleaved namespaces, float pitches, skipped notes, and unisons.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

from sonitra.midi_reader import parse_midi

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "guitarset_jams_to_midi.py"
)

EXPECTED_COLUMNS = [
    "midi_filename",
    "player_id",
    "style",
    "progression",
    "tempo_bpm",
    "key",
    "mode",
    "duration",
    "tempo_measured",
    "key_mode",
    "n_notes",
    "n_strings_used",
]


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "guitarset_jams_to_midi", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gjm() -> ModuleType:
    return _load_module()


def _note_block(
    data_source: str,
    obs: Any,
    namespace: str = "note_midi",
) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "namespace": namespace,
        "annotation_metadata": {"data_source": data_source},
    }
    if obs is not None:
        block["data"] = obs
    return block


def _list_data(notes: List[tuple]) -> List[Dict[str, Any]]:
    out = []
    for item in notes:
        time, duration, value = item[0], item[1], item[2]
        confidence = item[3] if len(item) > 3 else 1.0
        out.append(
            {
                "time": time,
                "duration": duration,
                "value": value,
                "confidence": confidence,
            }
        )
    return out


def _dict_data(notes: List[tuple]) -> Dict[str, List[Any]]:
    return {
        "time": [n[0] for n in notes],
        "duration": [n[1] for n in notes],
        "value": [n[2] for n in notes],
        "confidence": [(n[3] if len(n) > 3 else 1.0) for n in notes],
    }


def _make_jams(
    annotations_dir: Path,
    stem: str = "00_BN1-129-Eb_comp",
    string_notes: Optional[Dict[str, Any]] = None,
    layout: str = "list",
    string_order: Optional[List[str]] = None,
    interleave_contours: bool = True,
    unknown_namespace: bool = False,
    duration: Optional[float] = 30.0,
    tempo_value: Optional[float] = 129.0,
    key_mode_value: Optional[str] = "major",
) -> Path:
    """Write a minimal GuitarSet-like JAMS file and return its path.

    Args:
        string_notes: data_source -> list of (time, duration, value[, confidence])
            tuples; ``None`` means the block carries no ``data`` key at all and
            ``[]`` means an empty block. Defaults to one note on strings 0/1.
        layout: ``"list"`` (list-of-observations) or ``"dict"`` (dict-of-arrays).
        string_order: emission order of the per-string blocks (default: sorted).
        interleave_contours: insert a ``pitch_contour`` dict-layout block after
            each ``note_midi`` block, mirroring real GuitarSet files.
        unknown_namespace: append an unrelated namespace block (must be ignored).
        duration: ``file_metadata.duration``; ``None`` omits it.
        tempo_value / key_mode_value: first-value annotations; ``None`` omits.
    """
    if string_notes is None:
        string_notes = {"0": [(0.0, 0.5, 60.2)], "1": [(0.5, 0.5, 64.0)]}
    annotations: List[Dict[str, Any]] = []
    for source in string_order or sorted(string_notes):
        notes = string_notes[source]
        if notes is None:
            annotations.append(_note_block(source, None))
        elif layout == "dict":
            annotations.append(_note_block(source, _dict_data(notes)))
        else:
            annotations.append(_note_block(source, _list_data(notes)))
        if interleave_contours:
            annotations.append(
                {
                    "namespace": "pitch_contour",
                    "annotation_metadata": {"data_source": source},
                    "data": {
                        "time": [0.0, 0.01],
                        "duration": [0.01, 0.01],
                        # 100.x would be a bogus note if contours were parsed.
                        "value": [100.1, 100.2],
                        "confidence": [0.9, 0.9],
                    },
                }
            )
    if unknown_namespace:
        annotations.append(
            {
                "namespace": "chord",
                "annotation_metadata": {},
                "data": [
                    {"time": 0.0, "duration": 1.0, "value": 60.0, "confidence": 1.0}
                ],
            }
        )
    if tempo_value is not None:
        annotations.append(
            {
                "namespace": "tempo",
                "annotation_metadata": {},
                "data": [
                    {
                        "time": 0.0,
                        "duration": duration or 0.0,
                        "value": tempo_value,
                        "confidence": 1.0,
                    }
                ],
            }
        )
    if key_mode_value is not None:
        annotations.append(
            {
                "namespace": "key_mode",
                "annotation_metadata": {},
                "data": [
                    {
                        "time": 0.0,
                        "duration": duration or 0.0,
                        "value": key_mode_value,
                        "confidence": 1.0,
                    }
                ],
            }
        )
    doc: Dict[str, Any] = {"annotations": annotations, "file_metadata": {}}
    if duration is not None:
        doc["file_metadata"] = {"duration": duration}
    annotations_dir.mkdir(parents=True, exist_ok=True)
    path = annotations_dir / f"{stem}.jams"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _paths(tmp_path: Path):
    annotations_dir = tmp_path / "annotations"
    midi_dir = tmp_path / "midi"
    csv_path = tmp_path / "metadata" / "guitarset.csv"
    return annotations_dir, midi_dir, csv_path


def _run(gjm: ModuleType, annotations_dir: Path, midi_dir: Path, csv_path: Path, *extra: str) -> int:
    return gjm.main(
        [
            "--annotations", str(annotations_dir),
            "--output-midi", str(midi_dir),
            "--output-metadata", str(csv_path),
            *extra,
        ]
    )


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert list(reader.fieldnames or []) == EXPECTED_COLUMNS
        return rows


def _provenance(csv_path: Path) -> Dict[str, Any]:
    prov_path = Path(str(csv_path) + ".provenance.json")
    assert prov_path.exists(), f"missing provenance sidecar: {prov_path}"
    return json.loads(prov_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# data layouts
# ---------------------------------------------------------------------------


def test_list_layout_note_midi_parsed(gjm: ModuleType, tmp_path: Path) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={"0": [(0.0, 0.5, 60.2)], "1": [(0.5, 0.5, 64.0)]},
        layout="list",
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert sorted(n["pitch"] for n in notes) == [60, 64]
    assert {n["velocity"] for n in notes} == {100}
    meta = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid", return_meta=True)
    assert meta["bpm"] == pytest.approx(120.0)
    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["n_notes"] == "2"
    assert rows[0]["n_strings_used"] == "2"


def test_dict_of_arrays_layout_tolerated(gjm: ModuleType, tmp_path: Path) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={"2": [(0.0, 0.5, 67.0576)], "4": [(1.0, 0.25, 72.9)]},
        layout="dict",
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert sorted(n["pitch"] for n in notes) == [67, 73]
    assert _read_rows(csv_path)[0]["n_strings_used"] == "2"


def test_blocks_selected_by_data_source_not_position(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    # Highest string index emitted first, contours interleaved, plus an unknown
    # namespace whose value (60.0) must not leak into the note list.
    _make_jams(
        ann,
        string_notes={"0": [(0.0, 0.5, 55.0)], "5": [(0.0, 0.5, 76.0)]},
        string_order=["5", "0"],
        interleave_contours=True,
        unknown_namespace=True,
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert sorted(n["pitch"] for n in notes) == [55, 76]
    assert _read_rows(csv_path)[0]["n_notes"] == "2"


def test_float_pitch_rounds_to_nearest_semitone(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={
            "0": [(0.0, 0.5, 67.0576)],  # mirdata fixture value -> 67
            "1": [(0.5, 0.5, 60.49)],
            "2": [(1.0, 0.5, 60.51)],
        },
        interleave_contours=False,
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    by_start = {round(n["start_sec"], 3): n["pitch"] for n in notes}
    assert by_start == {0.0: 67, 0.5: 60, 1.0: 61}


# ---------------------------------------------------------------------------
# guards + counting
# ---------------------------------------------------------------------------


def test_out_of_range_pitch_skipped_and_counted(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={
            "0": [(0.0, 0.5, 60.0), (0.5, 0.5, -3.2), (1.0, 0.5, 200.0)],
            "1": [(1.5, 0.5, 127.4)],  # boundary: rounds to 127, kept
        },
        interleave_contours=False,
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert sorted(n["pitch"] for n in notes) == [60, 127]
    prov = _provenance(csv_path)
    assert prov["skipped"]["out_of_range_pitch"] == 2
    assert prov["files"][0]["skipped"]["out_of_range_pitch"] == 2
    assert _read_rows(csv_path)[0]["n_notes"] == "2"


def test_non_positive_duration_skipped_and_counted(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={
            "0": [(0.0, 0.5, 60.0), (0.5, 0.0, 62.0), (1.0, -0.5, 64.0)],
        },
        interleave_contours=False,
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert [n["pitch"] for n in notes] == [60]
    prov = _provenance(csv_path)
    assert prov["skipped"]["non_positive_duration"] == 2


def test_missing_data_null_confidence_and_empty_blocks_tolerated(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={
            "0": [(0.0, 0.5, 60.0, None)],  # null confidence
            "1": [],  # empty block
            "2": None,  # missing data key entirely
        },
        interleave_contours=False,
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert [n["pitch"] for n in notes] == [60]
    assert _read_rows(csv_path)[0]["n_strings_used"] == "1"


# ---------------------------------------------------------------------------
# unisons
# ---------------------------------------------------------------------------


def _unison_strings() -> Dict[str, Any]:
    return {
        "0": [(1.0, 0.5, 64.0)],
        "1": [(1.02, 0.5, 64.4)],  # same semitone, 20 ms later -> unison
        "2": [(5.0, 0.5, 63.6)],  # same pitch class, far away -> not a unison
    }


def test_unisons_counted_but_kept_by_default(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann, string_notes=_unison_strings(), interleave_contours=False)

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert len(notes) == 3  # reported, not removed
    prov = _provenance(csv_path)
    assert prov["unisons_detected"] == 1
    assert prov["unisons_removed"] == 0
    assert prov["files"][0]["unisons_detected"] == 1


def test_dedupe_unisons_removes_duplicates_keeping_earliest(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann, string_notes=_unison_strings(), interleave_contours=False)

    assert _run(gjm, ann, midi_dir, csv_path, "--dedupe-unisons") == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert sorted(round(n["start_sec"], 3) for n in notes) == [1.0, 5.0]
    prov = _provenance(csv_path)
    assert prov["unisons_detected"] == 1
    assert prov["unisons_removed"] == 1
    assert _read_rows(csv_path)[0]["n_notes"] == "2"


def test_constant_velocity_written(gjm: ModuleType, tmp_path: Path) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        string_notes={"0": [(0.0, 0.5, 60.0)], "3": [(0.5, 0.5, 65.0)]},
        interleave_contours=False,
    )

    assert _run(gjm, ann, midi_dir, csv_path, "--velocity", "77") == 0

    notes = parse_midi(midi_dir / "00_BN1-129-Eb_comp.mid")
    assert {n["velocity"] for n in notes} == {77}


# ---------------------------------------------------------------------------
# metadata CSV
# ---------------------------------------------------------------------------


def test_metadata_csv_columns_and_style_codes(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    cases = [
        ("00_BN1-129-Eb_comp", "00", "Bossa Nova", "1", "129", "Eb", "comp"),
        ("01_SS2-95-C_solo", "01", "Singer-Songwriter", "2", "95", "C", "solo"),
        ("02_Funk1-100-A_comp", "02", "Funk", "1", "100", "A", "comp"),
        ("03_Jazz3-120-Bb_solo", "03", "Jazz", "3", "120", "Bb", "solo"),
        ("04_Rock2-140-C#_comp", "04", "Rock", "2", "140", "C#", "comp"),
    ]
    for stem, *_ in cases:
        _make_jams(ann, stem=stem, duration=27.5)

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    rows = _read_rows(csv_path)
    assert len(rows) == 5
    by_name = {row["midi_filename"]: row for row in rows}
    for stem, player, style, prog, tempo, key, mode in cases:
        row = by_name[stem]
        assert row["player_id"] == player
        assert row["style"] == style
        assert row["progression"] == prog
        assert row["tempo_bpm"] == tempo
        assert row["key"] == key
        assert row["mode"] == mode
        assert row["duration"] == "27.5"
        # Join column must resolve to export_regression_table's `song` key.
        assert Path(row["midi_filename"]).stem == stem


def test_unknown_style_code_kept_raw(gjm: ModuleType, tmp_path: Path) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann, stem="00_Polka1-100-C_comp")

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    assert _read_rows(csv_path)[0]["style"] == "Polka"


def test_duration_from_file_metadata_and_missing_tolerated(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann, stem="00_BN1-129-Eb_comp", duration=27.5)
    _make_jams(ann, stem="01_SS2-95-C_solo", duration=None)

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    by_name = {row["midi_filename"]: row for row in _read_rows(csv_path)}
    assert by_name["00_BN1-129-Eb_comp"]["duration"] == "27.5"
    assert by_name["01_SS2-95-C_solo"]["duration"] == ""


def test_tempo_measured_and_key_mode_columns(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann, stem="00_BN1-129-Eb_comp", tempo_value=128.5, key_mode_value="minor"
    )
    _make_jams(
        ann, stem="01_SS2-95-C_solo", tempo_value=None, key_mode_value=None
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    by_name = {row["midi_filename"]: row for row in _read_rows(csv_path)}
    assert by_name["00_BN1-129-Eb_comp"]["tempo_measured"] == "128.5"
    assert by_name["00_BN1-129-Eb_comp"]["key_mode"] == "minor"
    assert by_name["01_SS2-95-C_solo"]["tempo_measured"] == ""
    assert by_name["01_SS2-95-C_solo"]["key_mode"] == ""


# ---------------------------------------------------------------------------
# provenance / CLI behaviour
# ---------------------------------------------------------------------------


def test_provenance_json_contents(gjm: ModuleType, tmp_path: Path) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(
        ann,
        stem="00_BN1-129-Eb_comp",
        string_notes={"0": [(0.0, 0.5, 60.0), (0.5, 0.5, 200.0)]},
        interleave_contours=False,
        duration=27.5,
    )
    _make_jams(
        ann,
        stem="01_SS2-95-C_solo",
        string_notes={"1": [(0.0, 0.5, 62.0)]},
        interleave_contours=False,
    )

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    prov = _provenance(csv_path)
    assert "--annotations" in prov["argv"]
    assert prov["jams_read"] == 2
    assert prov["midi_written"] == 2
    assert prov["n_notes"] == 2
    assert prov["skipped"]["out_of_range_pitch"] == 1
    assert prov["unisons_detected"] == 0
    assert prov["unisons_removed"] == 0
    assert len(prov["files"]) == 2
    first = next(f for f in prov["files"] if f["filename"] == "00_BN1-129-Eb_comp")
    assert first["n_notes"] == 1
    assert first["skipped"]["out_of_range_pitch"] == 1
    assert first["strings_used"] == ["0"]
    assert first["duration"] == "27.5"
    assert first["status"] == "ok"
    assert first["error"] == ""


def test_dry_run_writes_nothing(
    gjm: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann)

    assert _run(gjm, ann, midi_dir, csv_path, "--dry-run") == 0

    assert list(midi_dir.glob("*.mid")) == []
    assert not csv_path.exists()
    assert not Path(str(csv_path) + ".provenance.json").exists()
    assert "dry" in capsys.readouterr().out.lower()


def test_output_never_clobbers_input_guard(
    gjm: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann)

    rc = gjm.main(
        [
            "--annotations", str(ann),
            "--output-midi", str(ann),
            "--output-metadata", str(csv_path),
        ]
    )
    assert rc == 1
    assert "refus" in capsys.readouterr().err.lower()

    rc = gjm.main(
        [
            "--annotations", str(ann),
            "--output-midi", str(midi_dir),
            "--output-metadata", str(ann / "guitarset.csv"),
        ]
    )
    assert rc == 1
    assert "refus" in capsys.readouterr().err.lower()


def test_overwrite_skips_existing_by_default(
    gjm: ModuleType, tmp_path: Path
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann)

    assert _run(gjm, ann, midi_dir, csv_path) == 0
    assert _provenance(csv_path)["midi_written"] == 1

    assert _run(gjm, ann, midi_dir, csv_path) == 0
    prov = _provenance(csv_path)
    assert prov["midi_written"] == 0
    assert prov["files"][0]["status"] == "skipped"

    assert _run(gjm, ann, midi_dir, csv_path, "--overwrite") == 0
    assert _provenance(csv_path)["midi_written"] == 1


def test_fail_soft_continues_after_bad_file(
    gjm: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann, stem="00_BN1-129-Eb_comp")
    (ann / "01_SS2-95-C_solo.jams").write_text("{ not valid json", encoding="utf-8")

    rc = _run(gjm, ann, midi_dir, csv_path)

    assert rc == 1
    assert (midi_dir / "00_BN1-129-Eb_comp.mid").exists()
    prov = _provenance(csv_path)
    bad = next(f for f in prov["files"] if f["filename"] == "01_SS2-95-C_solo")
    assert bad["status"] == "error"
    assert bad["error"] != ""
    assert "01_SS2-95-C_solo" in capsys.readouterr().err


def test_summary_to_stdout_and_per_file_records_to_stderr(
    gjm: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ann, midi_dir, csv_path = _paths(tmp_path)
    _make_jams(ann)

    assert _run(gjm, ann, midi_dir, csv_path) == 0

    captured = capsys.readouterr()
    assert "wrote" in captured.out.lower()
    assert "00_BN1-129-Eb_comp" in captured.err
