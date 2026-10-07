"""Tests for the SMD metadata script."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import mido
import pytest

from sonitra.selection import load_selection_metadata

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "smd_metadata.py"
EXPORT_SCRIPT_PATH = REPO / "scripts" / "export_regression_table.py"

PIANO = "smd-piano-v2"
SYNTH = "smd-synth-v1"

_TICKS_PER_BEAT = 480
_TEMPO_US_PER_BEAT = 500_000  # 120 BPM

_BACH = "Bach_Fugue_001_20081107-SMD"
_RACH = "Rachmaninov_Prelude_002_20090916-SMD"
_CHOPIN = "Chopin_Nocturne_003_20100611-SMD"
_LISZT = "Liszt_Hungarian_004_20110315-SMD"
_UNPARSED = "Untitled_Take_003_20100611"

_DEFAULT_NOTES: List[Tuple[float, float, int]] = [(0.0, 0.5, 60), (1.0, 0.5, 64)]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def smd() -> ModuleType:
    return _load_module("smd_metadata", SCRIPT_PATH)


def _synth(stem: str) -> str:
    return f"{stem}-synth"


def _write_midi(
    path: Path, notes: Optional[List[Tuple[float, float, int]]] = None
) -> Path:
    """Write a type-1 MIDI file; *notes* are ``(start_sec, duration_sec, pitch)``."""
    if notes is None:
        notes = list(_DEFAULT_NOTES)
    events: List[Tuple[float, int, int]] = []
    for start, duration, pitch in notes:
        events.append((start, 1, pitch))
        events.append((start + duration, 0, pitch))
    events.sort()
    midi = mido.MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage("set_tempo", tempo=_TEMPO_US_PER_BEAT, time=0)
    )
    last_tick = 0
    for time_sec, is_on, pitch in events:
        tick = int(
            round(mido.second2tick(time_sec, _TICKS_PER_BEAT, _TEMPO_US_PER_BEAT))
        )
        delta = tick - last_tick
        last_tick = tick
        if is_on:
            track.append(
                mido.Message("note_on", note=pitch, velocity=80, time=delta)
            )
        else:
            track.append(
                mido.Message("note_off", note=pitch, velocity=0, time=delta)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    return path


def _make_dataset(
    root: Path,
    dataset: str,
    stems: List[str],
    notes: Optional[List[Tuple[float, float, int]]] = None,
) -> Path:
    midi_dir = root / dataset / "midi"
    for stem in stems:
        _write_midi(midi_dir / f"{stem}.mid", notes)
    return midi_dir


def _run(smd: ModuleType, root: Path, *extra: str) -> int:
    return smd.main(["--corpus-root", str(root), *extra])


def _csv_path(root: Path, dataset: str) -> Path:
    return root / dataset / "metadata" / f"{dataset}.csv"


def _provenance(csv_path: Path) -> Dict[str, Any]:
    prov_path = Path(str(csv_path) + ".provenance.json")
    assert prov_path.exists(), f"missing provenance sidecar: {prov_path}"
    return json.loads(prov_path.read_text(encoding="utf-8"))


def _read_rows(smd: ModuleType, csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert list(reader.fieldnames or []) == list(smd._METADATA_COLUMNS)
        return rows


def _by_id(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["performance_id"]: row for row in rows}


def _midi_snapshot(root: Path) -> Dict[str, Tuple[bytes, int]]:
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*.mid"))
    }


# ---------------------------------------------------------------------------
# linked CSVs and filename fields
# ---------------------------------------------------------------------------


def test_both_sets_write_linked_csvs(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH, _RACH])
    _make_dataset(root, SYNTH, [_synth(_BACH), _synth(_RACH)])

    assert _run(smd, root) == 0

    piano_csv = _csv_path(root, PIANO)
    synth_csv = _csv_path(root, SYNTH)
    assert piano_csv.exists()
    assert synth_csv.exists()

    piano_rows = _read_rows(smd, piano_csv)
    synth_rows = _read_rows(smd, synth_csv)
    assert len(piano_rows) == 2
    assert len(synth_rows) == 2
    piano_by_id = _by_id(piano_rows)
    synth_by_id = _by_id(synth_rows)
    expected_ids = {"Bach_Fugue_001_20081107", "Rachmaninov_Prelude_002_20090916"}
    assert set(piano_by_id) == expected_ids
    assert set(synth_by_id) == expected_ids

    bach_piano = piano_by_id["Bach_Fugue_001_20081107"]
    assert bach_piano["midi_filename"] == _BACH
    assert bach_piano["variant"] == "piano"
    assert bach_piano["composer"] == "Bach"
    assert bach_piano["work"] == "Fugue"
    assert bach_piano["performer_id"] == "001"
    assert bach_piano["recording_date"] == "2008-11-07"
    assert bach_piano["n_notes"] == "2"
    assert bach_piano["duration_sec"] == "1.5"
    assert bach_piano["smd_piano_stem"] == _BACH
    assert bach_piano["smd_synth_stem"] == _synth(_BACH)

    bach_synth = synth_by_id["Bach_Fugue_001_20081107"]
    assert bach_synth["midi_filename"] == _synth(_BACH)
    assert bach_synth["variant"] == "synth"
    assert bach_synth["composer"] == "Bach"
    assert bach_synth["smd_piano_stem"] == _BACH
    assert bach_synth["smd_synth_stem"] == _synth(_BACH)

    prov = _provenance(piano_csv)
    assert prov["matched"] == 2
    assert prov["piano_only"] == []
    assert prov["synth_only"] == []
    assert prov["n_errors"] == 0
    assert _provenance(synth_csv) == prov
    assert "wrote 2 rows to" in capsys.readouterr().out


def test_composer_alias_applied_and_recorded(smd: ModuleType, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_RACH])
    _make_dataset(root, SYNTH, [_synth(_RACH)])

    assert _run(smd, root) == 0

    piano_row = _read_rows(smd, _csv_path(root, PIANO))[0]
    assert piano_row["midi_filename"] == _RACH
    assert "Rachmaninov" in piano_row["midi_filename"]
    assert piano_row["composer"] == "Rachmaninoff"

    synth_row = _read_rows(smd, _csv_path(root, SYNTH))[0]
    assert synth_row["composer"] == "Rachmaninoff"

    prov = _provenance(_csv_path(root, PIANO))
    assert prov["composer_aliases"] == {"Rachmaninov": "Rachmaninoff"}


def test_unmatched_performance_is_warned_and_blank(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH, _CHOPIN])
    _make_dataset(root, SYNTH, [_synth(_BACH), _synth(_LISZT)])

    assert _run(smd, root) == 0

    err = capsys.readouterr().err
    assert "Chopin_Nocturne_003_20100611" in err
    assert "Liszt_Hungarian_004_20110315" in err

    piano_by_id = _by_id(_read_rows(smd, _csv_path(root, PIANO)))
    assert piano_by_id["Chopin_Nocturne_003_20100611"]["smd_synth_stem"] == ""
    assert piano_by_id["Bach_Fugue_001_20081107"]["smd_synth_stem"] == _synth(_BACH)

    synth_by_id = _by_id(_read_rows(smd, _csv_path(root, SYNTH)))
    assert synth_by_id["Liszt_Hungarian_004_20110315"]["smd_piano_stem"] == ""
    assert synth_by_id["Bach_Fugue_001_20081107"]["smd_piano_stem"] == _BACH

    prov = _provenance(_csv_path(root, PIANO))
    assert prov["matched"] == 1
    assert prov["piano_only"] == ["Chopin_Nocturne_003_20100611"]
    assert prov["synth_only"] == ["Liszt_Hungarian_004_20110315"]


def test_only_one_dataset_present(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH])

    assert _run(smd, root) == 0

    piano_csv = _csv_path(root, PIANO)
    assert piano_csv.exists()
    assert not _csv_path(root, SYNTH).exists()

    rows = _read_rows(smd, piano_csv)
    assert rows and all(row["smd_synth_stem"] == "" for row in rows)

    err = capsys.readouterr().err
    assert SYNTH in err
    assert "warning" in err.lower()

    prov = _provenance(piano_csv)
    assert prov["datasets"]["synth"]["n_files"] == 0
    assert prov["datasets"]["synth"]["output_metadata"] is None


def test_no_dataset_present_fails_without_writing(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"

    assert _run(smd, root) == 1

    err = capsys.readouterr().err
    assert "error" in err.lower()
    assert not (root / PIANO / "metadata").exists()
    assert not (root / SYNTH / "metadata").exists()
    assert list(root.rglob("*.csv")) == []


def test_dry_run_writes_nothing(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH])
    _make_dataset(root, SYNTH, [_synth(_BACH)])

    assert _run(smd, root, "--dry-run") == 0

    out = capsys.readouterr().out
    assert (
        "dry run: 1 piano + 1 synth MIDI files, 1 matched, "
        "0 piano-only, 0 synth-only, 0 errors" in out
    )
    assert not (root / PIANO / "metadata").exists()
    assert not (root / SYNTH / "metadata").exists()
    assert list(root.rglob("*.csv")) == []
    assert list(root.rglob("*.json")) == []


def test_unparseable_stem_kept_with_blank_fields(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH, _UNPARSED])
    _make_dataset(root, SYNTH, [_synth(_BACH)])

    assert _run(smd, root) == 0

    rows = _by_id(_read_rows(smd, _csv_path(root, PIANO)))
    row = rows[_UNPARSED]
    assert row["midi_filename"] == _UNPARSED
    assert row["performance_id"] == _UNPARSED
    assert row["composer"] == ""
    assert row["work"] == ""
    assert row["performer_id"] == ""
    assert row["recording_date"] == ""

    prov = _provenance(_csv_path(root, PIANO))
    assert prov["unparsed"] == [_UNPARSED]
    assert _UNPARSED in capsys.readouterr().err


def test_corrupt_midi_recorded_and_exit_1(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH])
    (root / PIANO / "midi" / "Bad_Take_004_20110315-SMD.mid").write_bytes(
        b"not midi"
    )
    _make_dataset(root, SYNTH, [_synth(_BACH)])

    assert _run(smd, root) == 1

    piano_csv = _csv_path(root, PIANO)
    assert piano_csv.exists()
    assert _csv_path(root, SYNTH).exists()
    rows = _by_id(_read_rows(smd, piano_csv))
    bad = rows["Bad_Take_004_20110315"]
    assert bad["n_notes"] == ""
    assert bad["duration_sec"] == ""

    prov = _provenance(piano_csv)
    assert prov["n_errors"] == 1
    bad_record = next(
        f for f in prov["files"] if f["filename"] == "Bad_Take_004_20110315-SMD"
    )
    assert bad_record["status"] == "error"
    assert bad_record["error"] != ""
    assert "Bad_Take" in capsys.readouterr().err


def test_duplicate_performance_id_fails(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    midi_dir = root / PIANO / "midi"
    _write_midi(midi_dir / f"{_BACH}.mid")
    _write_midi(midi_dir / "sub" / f"{_BACH}.mid")
    _make_dataset(root, SYNTH, [_synth(_BACH)])

    assert _run(smd, root) == 1

    err = capsys.readouterr().err
    assert "duplicate" in err.lower()
    assert str(midi_dir / f"{_BACH}.mid") in err
    assert str(midi_dir / "sub" / f"{_BACH}.mid") in err
    assert not (root / PIANO / "metadata").exists()
    assert not (root / SYNTH / "metadata").exists()


# ---------------------------------------------------------------------------
# CLI and reproducibility
# ---------------------------------------------------------------------------


def test_same_dataset_names_rejected(
    smd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        smd.main(
            [
                "--corpus-root", str(tmp_path),
                "--piano-dataset", "x",
                "--synth-dataset", "x",
            ]
        )
    assert excinfo.value.code == 2
    assert "differ" in capsys.readouterr().err


def test_rerun_is_idempotent_and_leaves_inputs_untouched(
    smd: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH, _RACH])
    _make_dataset(root, SYNTH, [_synth(_BACH), _synth(_RACH)])

    assert _run(smd, root) == 0
    piano_csv = _csv_path(root, PIANO)
    synth_csv = _csv_path(root, SYNTH)
    before = {
        "piano_csv": piano_csv.read_bytes(),
        "synth_csv": synth_csv.read_bytes(),
        "piano_prov": _provenance(piano_csv).copy(),
        "midi": _midi_snapshot(root),
    }

    assert _run(smd, root) == 0

    assert piano_csv.read_bytes() == before["piano_csv"]
    assert synth_csv.read_bytes() == before["synth_csv"]
    assert _provenance(piano_csv) == before["piano_prov"]
    assert _midi_snapshot(root) == before["midi"]


def test_csv_joins_with_export_and_selection(
    smd: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    _make_dataset(root, PIANO, [_BACH, _RACH])
    _make_dataset(root, SYNTH, [_synth(_BACH), _synth(_RACH)])

    assert _run(smd, root) == 0

    export = _load_module("export_regression_table", EXPORT_SCRIPT_PATH)
    piano_csv = _csv_path(root, PIANO)
    index = export.load_metadata_join(piano_csv, "midi_filename")
    midi_stems = {path.stem for path in (root / PIANO / "midi").glob("*.mid")}
    assert set(index) == midi_stems

    selected = load_selection_metadata(
        piano_csv, "midi_filename", {"composer": ["Bach"]}
    )
    assert set(selected) == midi_stems
    assert selected[_BACH]["composer"] == "Bach"
