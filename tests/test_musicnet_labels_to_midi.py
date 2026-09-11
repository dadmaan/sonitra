"""Tests for the MusicNet label-CSV-to-MIDI converter."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

from sonitra.midi_reader import parse_midi

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "musicnet_labels_to_midi.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("musicnet_labels_to_midi", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mlm() -> ModuleType:
    return _load_module()


def _make_metadata(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "composer", "composition", "movement", "ensemble", "source", "transcriber", "catalog_name", "seconds"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_label_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["start_time", "end_time", "instrument", "note", "start_beat", "end_beat", "note_value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _default_paths(tmp_path: Path):
    corpus_root = tmp_path / "corpus"
    labels_dir = corpus_root / "musicnet" / "annotations" / "labels" / "musicnet" / "train_labels"
    # we will not create default dirs here; caller decides
    source_meta = corpus_root / "musicnet" / "metadata" / "musicnet_metadata.csv"
    score_midi = corpus_root / "musicnet" / "annotations" / "score_midi"
    output_midi = corpus_root / "musicnet" / "midi"
    output_meta = corpus_root / "musicnet" / "metadata" / "musicnet.csv"
    return corpus_root, labels_dir, source_meta, score_midi, output_midi, output_meta


def _run(mlm: ModuleType, *args: str) -> int:
    return mlm.main(list(args))


# ---------------------------------------------------------------------------
# sample-exact round trip
# ---------------------------------------------------------------------------

def test_sample_exact_round_trip(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    # Create train_labels with specific samples
    import random
    random.seed(0)
    samples = [random.randint(0, 200_000_000) for _ in range(20)]
    rows = [{"start_time": s, "end_time": s + 1000, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"} for s in samples]
    _make_label_csv(labels_dir / "1.csv", rows)
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    midi = corpus_root / "musicnet" / "midi" / "1.mid"
    assert midi.exists()
    notes = parse_midi(midi)
    assert len(notes) == len(samples)
    samples_sorted = sorted(samples)
    notes_sorted = sorted(notes, key=lambda n: n["start_sec"])
    for s, n in zip(samples_sorted, notes_sorted):
        assert round(n["start_sec"] * 44100) == s
        assert round((n["start_sec"] + n["duration_sec"]) * 44100) == s + 1000


def test_one_channel_and_program_per_instrument(mlm: ModuleType, tmp_path: Path) -> None:
    import mido

    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    rows = [
        {"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},
        {"start_time": 44100, "end_time": 88200, "instrument": 41, "note": 64, "start_beat": 1, "end_beat": 2, "note_value": "quarter"},
        {"start_time": 88200, "end_time": 132300, "instrument": 71, "note": 67, "start_beat": 2, "end_beat": 3, "note_value": "quarter"},
    ]
    _make_label_csv(labels_dir / "1.csv", rows)
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    midi = corpus_root / "musicnet" / "midi" / "1.mid"
    mf = mido.MidiFile(midi)
    # programs sorted: 0,40,70 -> channels 0,1,2
    prog_changes = [(m.program, m.channel) for m in mf.tracks[0] if m.type == "program_change"]
    assert prog_changes == [(0, 0), (40, 1), (70, 2)]
    # check note channels
    note_channels = {m.note: m.channel for m in mf.tracks[0] if m.type == "note_on"}
    assert note_channels[60] == 0
    assert note_channels[64] == 1
    assert note_channels[67] == 2


def test_skip_reasons_counted(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    rows = [
        {"start_time": "bad", "end_time": 1000, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},  # invalid_value
        {"start_time": 0, "end_time": 0, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},  # non_positive_duration
        {"start_time": 0, "end_time": 1000, "instrument": 1, "note": 200, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},  # out_of_range_pitch
        {"start_time": 0, "end_time": 1000, "instrument": 200, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},  # invalid_instrument
        {"start_time": 0, "end_time": 1000, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},  # valid
    ]
    _make_label_csv(labels_dir / "1.csv", rows)
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    prov_path = corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json"
    prov = json.loads(prov_path.read_text())
    skipped = prov["files"][0]["skipped"]
    assert skipped["invalid_value"] == 1
    assert skipped["non_positive_duration"] == 1
    assert skipped["out_of_range_pitch"] == 1
    assert skipped["invalid_instrument"] == 1
    notes = parse_midi(corpus_root / "musicnet" / "midi" / "1.mid")
    assert len(notes) == 1


def test_unisons_counted_and_kept_and_dedupe_keeps_earliest(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    # Unison: same pitch 60, 0.02 sec apart (882 samples), different instrument
    rows = [
        {"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},
        {"start_time": 882, "end_time": 44982, "instrument": 7, "note": 60, "start_beat": 0.02, "end_beat": 1.02, "note_value": "quarter"},
        {"start_time": 44100, "end_time": 88200, "instrument": 1, "note": 62, "start_beat": 1, "end_beat": 2, "note_value": "quarter"},
    ]
    _make_label_csv(labels_dir / "1.csv", rows)
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    prov = json.loads((corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").read_text())
    assert prov["files"][0]["unisons_detected"] == 1
    assert prov["files"][0]["unisons_removed"] == 0
    notes = parse_midi(corpus_root / "musicnet" / "midi" / "1.mid")
    assert len(notes) == 3
    # Dedupe
    corpus_root2 = tmp_path / "corpus2"
    # copy metadata and labels
    import shutil
    (tmp_path / "corpus2").mkdir()
    shutil.copytree(corpus_root / "musicnet", corpus_root2 / "musicnet")
    # Need to recreate? Simpler: run with dedupe and overwrite on original
    assert _run(mlm, "--corpus-root", str(corpus_root), "--dedupe-unisons", "--overwrite") == 0
    prov2 = json.loads((corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").read_text())
    assert prov2["files"][0]["unisons_detected"] == 1
    assert prov2["files"][0]["unisons_removed"] == 1
    notes2 = parse_midi(corpus_root / "musicnet" / "midi" / "1.mid")
    assert len(notes2) == 2
    # Earliest kept should be at 0.0
    assert min(n["start_sec"] for n in notes2) == pytest.approx(0.0, abs=0.01)


def test_same_channel_overlaps_counted(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    rows = [
        {"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},
        {"start_time": 22050, "end_time": 66150, "instrument": 1, "note": 60, "start_beat": 0.5, "end_beat": 1.5, "note_value": "quarter"},  # overlap same program/pitch
        {"start_time": 0, "end_time": 44100, "instrument": 41, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"},  # same pitch different program -> unison but not same-channel overlap
    ]
    _make_label_csv(labels_dir / "1.csv", rows)
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    prov = json.loads((corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").read_text())
    assert prov["files"][0]["same_channel_overlaps"] == 1
    assert prov["files"][0]["unisons_detected"] == 1  # cross-instrument same pitch within 0 sec? Actually 0 vs 0 same pitch different instrument at same onset -> unison
    # The two notes at 0.0 same pitch diff instrument -> unison 1, plus maybe second note at 0.5 also same pitch diff? But count is 1
    # Ensure overlaps not counted for cross-instrument
    # Overlaps should be 1 as above


def test_train_test_split_from_directory_name(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    # Create both train and test
    for split in ["train", "test"]:
        d = corpus_root / "musicnet" / "annotations" / "labels" / "musicnet" / f"{split}_labels"
        d.mkdir(parents=True)
        with (d / f"{split}1.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["start_time", "end_time", "instrument", "note"])
            w.writeheader()
            w.writerow({"start_time": 0, "end_time": 1000, "instrument": 1, "note": 60})
    source_meta = corpus_root / "musicnet" / "metadata" / "musicnet_metadata.csv"
    _make_metadata(source_meta, [{"id": "train1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}, {"id": "test1", "composer": "Mozart", "composition": "Sonata", "movement": "1", "ensemble": "Violin", "source": "src", "transcriber": "tr", "catalog_name": "K", "seconds": "10"}])
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    rows = list(csv.DictReader((corpus_root / "musicnet" / "metadata" / "musicnet.csv").open()))
    by_id = {r["id"]: r for r in rows}
    assert by_id["train1"]["split"] == "train"
    assert by_id["test1"]["split"] == "test"


def test_metadata_columns_and_source_join(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, score_midi, _, _ = _default_paths(tmp_path)
    # Create source metadata with known values
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "srcVal", "transcriber": "transVal", "catalog_name": "BWV 846", "seconds": "123"}])
    # Create score midi mapping
    score_midi.mkdir(parents=True)
    from sonitra.midi_writer import write_midi
    (score_midi / "Comp").mkdir(parents=True)
    write_midi([{"pitch": 60, "velocity": 100, "start_sec": 0, "duration_sec": 1}], score_midi / "Comp" / "1_song.mid")
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    csv_path = corpus_root / "musicnet" / "metadata" / "musicnet.csv"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    r = rows[0]
    # Check required columns
    for col in ["midi_filename", "id", "split", "composer", "composition", "movement", "ensemble", "source", "transcriber", "catalog_name", "seconds", "score_midi_filename", "n_notes", "n_programs", "programs", "labels_end_sec", "unisons_detected", "same_channel_overlaps"]:
        assert col in r, f"missing {col}"
    assert r["midi_filename"] == "1"
    assert r["id"] == "1"
    assert r["composer"] == "Bach"
    assert r["score_midi_filename"] == "Comp/1_song.mid"
    assert r["n_notes"] == "1"
    assert r["programs"] == "0"


def test_provenance_contents(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    assert _run(mlm, "--corpus-root", str(corpus_root), "--velocity", "100") == 0
    prov_path = corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert "argv" in prov
    assert "inputs_sha256" in prov
    assert "files" in prov
    assert prov["sample_rate"] == 44100
    assert prov["velocity"] == 100
    assert prov["ticks_per_beat"] == 22050
    assert prov["tempo_bpm"] == 120.0
    assert len(prov["files"]) == 1
    assert prov["files"][0]["filename"] == "1"


def test_dry_run_writes_nothing(mlm: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus_root, labels_dir, source_meta, _, output_midi, output_meta = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    assert _run(mlm, "--corpus-root", str(corpus_root), "--dry-run") == 0
    assert not (output_midi / "1.mid").exists()
    assert not output_meta.exists()
    assert not Path(str(output_meta) + ".provenance.json").exists()
    # summary goes to stdout, per-file to stderr
    captured = capsys.readouterr()
    assert "dry run" in captured.out.lower()


def test_never_clobber_guard(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, score_midi, output_midi, output_meta = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    # output-midi inside labels
    assert _run(mlm, "--corpus-root", str(corpus_root), "--output-midi", str(labels_dir / "midi")) == 1
    # output-metadata inside labels
    assert _run(mlm, "--corpus-root", str(corpus_root), "--output-metadata", str(labels_dir / "out.csv")) == 1
    # output-metadata same as source
    assert _run(mlm, "--corpus-root", str(corpus_root), "--output-metadata", str(source_meta)) == 1


def test_refusal_while_score_midi_present_and_replace_deletes_identical_and_moves_rest(mlm: ModuleType, tmp_path: Path) -> None:
    from sonitra.midi_writer import write_midi

    corpus_root = tmp_path / "corpus"
    labels_dir = corpus_root / "musicnet" / "annotations" / "labels" / "musicnet" / "train_labels"
    labels_dir.mkdir(parents=True)
    source_meta = corpus_root / "musicnet" / "metadata" / "musicnet_metadata.csv"
    _make_metadata(source_meta, [{"id": "2104", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    _make_label_csv(labels_dir / "2104.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    midi_dir = corpus_root / "musicnet" / "midi"
    score_dir = corpus_root / "musicnet" / "annotations" / "score_midi"
    # Create score tree inside midi (musicnet-midi)
    score_tree = midi_dir / "musicnet_midis" / "Composer" / "2104_prelude.mid"
    score_tree.parent.mkdir(parents=True)
    write_midi([{"pitch": 60, "velocity": 100, "start_sec": 0, "duration_sec": 1}], score_tree)
    # Create identical copy in score_midi for deletion case
    identical = score_dir / "musicnet_midis" / "Composer" / "2104_prelude.mid"
    identical.parent.mkdir(parents=True)
    identical.write_bytes(score_tree.read_bytes())
    # Without replace should refuse
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 1
    # With replace, identical should be deleted, then conversion succeeds
    assert _run(mlm, "--corpus-root", str(corpus_root), "--replace-score-midi") == 0
    assert not (midi_dir / "musicnet_midis" / "Composer" / "2104_prelude.mid").exists()
    assert identical.exists()
    assert (midi_dir / "2104.mid").exists()
    # Now test moving non-identical
    # Reset: create new score tree file with different content, and score file missing
    # Clean midi and score
    import shutil
    shutil.rmtree(midi_dir)
    shutil.rmtree(score_dir)
    midi_dir.mkdir(parents=True)
    score_dir.mkdir(parents=True)
    _make_label_csv(labels_dir / "2104.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    # Recreate musicnet-midi tree with different content than score_midi (score has different pitch)
    write_midi([{"pitch": 60, "velocity": 100, "start_sec": 0, "duration_sec": 1}], midi_dir / "musicnet_midis" / "Composer" / "2300_song.mid")
    # Need label for 2300 as well or else that file will be ignored? Create label for 2300 to make conversion attempt
    _make_label_csv(labels_dir / "2300.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    _make_metadata(source_meta, [{"id": "2104", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}, {"id": "2300", "composer": "Mozart", "composition": "Sonata", "movement": "1", "ensemble": "Violin", "source": "src", "transcriber": "tr", "catalog_name": "K", "seconds": "10"}])
    # Ensure score_dir doesn't have that file, so it will be moved
    assert _run(mlm, "--corpus-root", str(corpus_root), "--replace-score-midi") == 0
    assert not (midi_dir / "musicnet_midis").exists()
    assert (score_dir / "musicnet_midis" / "Composer" / "2300_song.mid").exists()
    # Now test differing copy refuses
    # Put back a score tree file and create a differing copy in score_dir
    write_midi([{"pitch": 60, "velocity": 100, "start_sec": 0, "duration_sec": 1}], midi_dir / "musicnet_midis" / "Composer" / "3000_song.mid")
    differing = score_dir / "musicnet_midis" / "Composer" / "3000_song.mid"
    differing.parent.mkdir(parents=True, exist_ok=True)
    write_midi([{"pitch": 62, "velocity": 100, "start_sec": 0, "duration_sec": 1}], differing)
    _make_label_csv(labels_dir / "3000.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    _make_metadata(source_meta, [{"id": "3000", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    assert _run(mlm, "--corpus-root", str(corpus_root), "--replace-score-midi") == 1
    # Should not have deleted
    assert (midi_dir / "musicnet_midis" / "Composer" / "3000_song.mid").exists()


def test_overwrite_behaviour(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    midi_path = corpus_root / "musicnet" / "midi" / "1.mid"
    mtime1 = midi_path.stat().st_mtime
    # Without overwrite, should skip
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 0
    prov = json.loads((corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").read_text())
    assert prov["files"][0]["status"] == "skipped"
    assert midi_path.stat().st_mtime == mtime1
    # With overwrite, should rewrite
    assert _run(mlm, "--corpus-root", str(corpus_root), "--overwrite") == 0
    prov2 = json.loads((corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").read_text())
    assert prov2["files"][0]["status"] == "ok"


def test_fail_soft_on_bad_file(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}, {"id": "2", "composer": "Mozart", "composition": "Sonata", "movement": "1", "ensemble": "Violin", "source": "src", "transcriber": "tr", "catalog_name": "K", "seconds": "10"}])
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    (labels_dir / "2.csv").write_text("not valid csv {", encoding="utf-8")
    assert _run(mlm, "--corpus-root", str(corpus_root)) == 1
    assert (corpus_root / "musicnet" / "midi" / "1.mid").exists()
    prov = json.loads((corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").read_text())
    bad = next(f for f in prov["files"] if f["filename"] == "2")
    assert bad["status"] == "error"
    assert bad["error"] != ""


def test_invalid_sample_rate_rejected(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root, labels_dir, source_meta, _, _, _ = _default_paths(tmp_path)
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    _make_label_csv(labels_dir / "1.csv", [{"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60, "start_beat": 0, "end_beat": 1, "note_value": "quarter"}])
    assert _run(mlm, "--corpus-root", str(corpus_root), "--sample-rate", "44101") == 1
    assert _run(mlm, "--corpus-root", str(corpus_root), "--sample-rate", "65535") == 1
    assert _run(mlm, "--corpus-root", str(corpus_root), "--sample-rate", "65536") == 1
    assert _run(mlm, "--corpus-root", str(corpus_root), "--sample-rate", "44100") == 0


def test_labels_accepts_v030_layout(mlm: ModuleType, tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    # v0.3.0 layout: labels under metadata/musicnet
    labels_root = corpus_root / "musicnet" / "metadata"
    labels_root.mkdir(parents=True)
    train_labels = labels_root / "musicnet" / "train_labels"
    train_labels.mkdir(parents=True)
    with (train_labels / "1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_time", "end_time", "instrument", "note"])
        w.writeheader()
        w.writerow({"start_time": 0, "end_time": 44100, "instrument": 1, "note": 60})
    source_meta = corpus_root / "musicnet" / "metadata" / "musicnet_metadata.csv"
    _make_metadata(source_meta, [{"id": "1", "composer": "Bach", "composition": "Prelude", "movement": "1", "ensemble": "Piano", "source": "src", "transcriber": "tr", "catalog_name": "BWV", "seconds": "10"}])
    # Use default output-metadata which is sibling to labels_root/musicnet, not inside a *_labels dir
    # If labels_root is metadata, output is metadata/musicnet.csv which is inside labels_root;
    # our guard would refuse, so we explicitly set labels to metadata/musicnet (the actual parent of train_labels)
    # Test both: labels as metadata/musicnet should work with default output
    assert _run(mlm, "--corpus-root", str(corpus_root), "--labels", str(labels_root / "musicnet")) == 0
    assert (corpus_root / "musicnet" / "midi" / "1.mid").exists()
    # Also test labels as metadata (broader) with explicit output outside
    import shutil
    shutil.rmtree(corpus_root / "musicnet" / "midi")
    (corpus_root / "musicnet" / "midi").mkdir(parents=True)
    # Need to clean previous metadata output
    (corpus_root / "musicnet" / "metadata" / "musicnet.csv").unlink(missing_ok=True)
    (corpus_root / "musicnet" / "metadata" / "musicnet.csv.provenance.json").unlink(missing_ok=True)
    out_meta = tmp_path / "out" / "musicnet.csv"
    assert _run(mlm, "--corpus-root", str(corpus_root), "--labels", str(labels_root), "--output-metadata", str(out_meta)) == 0
    assert (corpus_root / "musicnet" / "midi" / "1.mid").exists()
    assert out_meta.exists()
