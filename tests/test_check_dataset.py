from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import mido
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_dataset.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cd() -> ModuleType:
    return _load_module()


def _midi(
    path: Path,
    *,
    bpm: float = 120,
    programs: tuple[int, ...] = (),
    notes: tuple[int, ...] = (60, 64),
    drum: bool = False,
) -> Path:
    """Write a small single-track MIDI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    for program in programs:
        track.append(mido.Message("program_change", program=program, time=0))
    channel = 9 if drum else 0
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=90, channel=channel, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, channel=channel, time=480))
    midi.save(path)
    return path


def _audio(path: Path) -> Path:
    """Create a recording placeholder; the checker never opens audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _dataset(tmp_path: Path, midi: dict[str, dict] | list[str], recordings: list[str] = ()) -> Path:
    root = tmp_path / "corpus" / "mine"
    specs = midi if isinstance(midi, dict) else {name: {} for name in midi}
    for name, kwargs in specs.items():
        _midi(root / "midi" / name, **kwargs)
    for name in recordings:
        _audio(root / "recordings" / name)
    (root / "midi").mkdir(parents=True, exist_ok=True)
    return root


def _codes(report) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for finding in report.findings:
        grouped.setdefault(finding.code, []).append(finding)
    return grouped


def _names(findings) -> list[str]:
    return sorted(Path(finding.path).name for finding in findings)


# --- input type and a clean dataset -------------------------------------------------


def test_clean_audio_dataset_has_no_findings(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(
        tmp_path,
        ["set1/song1.mid", "song2.mid"],
        ["song1_take1.wav", "song1_take2.flac", "sub/song2_live.mp3"],
    )

    report = cd.check_dataset(root)

    assert report.input_type == "audio"
    assert report.findings == []
    assert report.pairing is not None and len(report.pairing.mapping) == 3
    assert report.exit_code == 0


def test_input_type_is_midi_when_there_are_no_recordings(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"])

    report = cd.check_dataset(root)

    assert report.input_type == "midi"
    assert report.pairing is None
    assert report.findings == []


def test_input_type_can_be_forced(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["song1.wav"])

    report = cd.check_dataset(root, input_type="midi")

    assert report.input_type == "midi"
    assert report.pairing is None


# --- pairing ------------------------------------------------------------------------


def test_recording_with_no_matching_midi_is_an_error(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["song1_take1.wav", "other.wav"])

    report = cd.check_dataset(root)

    assert _names(_codes(report)["unpaired"]) == ["other.wav"]
    assert report.exit_code == 1


def test_duplicate_midi_names_are_ambiguous_errors_in_audio_mode(
    cd: ModuleType, tmp_path: Path
) -> None:
    root = _dataset(tmp_path, ["a/intro.mid", "b/intro.mid"], ["intro.wav"])

    report = cd.check_dataset(root)
    codes = _codes(report)

    (ambiguous,) = codes["ambiguous"]
    assert Path(ambiguous.path).name == "intro.wav"
    assert "a/intro.mid" in ambiguous.detail and "b/intro.mid" in ambiguous.detail
    assert _names(codes["duplicate-name"]) == ["intro.mid", "intro.mid"]
    assert all(f.level == "error" for f in codes["duplicate-name"])


def test_duplicate_midi_names_are_only_a_warning_in_midi_mode(
    cd: ModuleType, tmp_path: Path
) -> None:
    root = _dataset(tmp_path, ["a/intro.mid", "b/intro.mid"])

    report = cd.check_dataset(root)

    assert {f.level for f in _codes(report)["duplicate-name"]} == {"warning"}
    assert report.exit_code == 0


def test_midi_name_that_prefixes_another_can_never_pair(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["piece_1.mid", "piece_1_arr.mid", "piece_10.mid"], ["piece_10.wav"])

    report = cd.check_dataset(root)

    assert _names(_codes(report)["name-clash"]) == ["piece_1.mid"]
    assert _codes(report)["name-clash"][0].level == "error"


def test_name_clash_is_not_checked_in_midi_mode(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["piece_1.mid", "piece_1_arr.mid"])

    report = cd.check_dataset(root)

    assert "name-clash" not in _codes(report)


def test_midi_with_no_recording_is_a_note(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid", "song2.mid"], ["song1.wav"])

    report = cd.check_dataset(root)

    (unused,) = _codes(report)["unused-midi"]
    assert Path(unused.path).name == "song2.mid"
    assert unused.level == "note"
    assert report.exit_code == 0


# --- MIDI contents ------------------------------------------------------------------


def test_off_tempo_midi_yields_no_finding_in_midi_mode(
    cd: ModuleType, tmp_path: Path
) -> None:
    root = _dataset(tmp_path, {"slow.mid": {"bpm": 90}, "ok.mid": {}})

    report = cd.check_dataset(root)

    assert "tempo" not in _codes(report)
    # off-tempo files must not be reported as errors or warnings
    assert all(f.code != "tempo" for f in report.findings)


def test_bpm_flag_no_longer_accepted(cd: ModuleType, tmp_path: Path) -> None:
    # --bpm was removed: parser must reject it
    with pytest.raises(SystemExit):
        cd.main(["--dataset", "mine", "--bpm", "120", "--corpus-root", str(tmp_path / "corpus")])
    # programmatic bpm kwarg was also removed
    root = _dataset(tmp_path, {"slow.mid": {"bpm": 90}})
    with pytest.raises(TypeError):
        cd.check_dataset(root, bpm=90)  # type: ignore[call-arg]


def test_several_programs_warn_in_midi_mode_only(cd: ModuleType, tmp_path: Path) -> None:
    specs = {"multi.mid": {"programs": (0, 24)}, "single.mid": {"programs": (24,)}}
    midi_root = _dataset(tmp_path / "m", specs)
    audio_root = _dataset(tmp_path / "a", specs, ["multi.wav", "single.wav"])

    midi_report = cd.check_dataset(midi_root)
    audio_report = cd.check_dataset(audio_root)

    (programs,) = _codes(midi_report)["programs"]
    assert Path(programs.path).name == "multi.mid"
    assert programs.level == "warning"
    assert "programs" not in _codes(audio_report)


def test_drum_channel_notes_warn_in_both_modes(cd: ModuleType, tmp_path: Path) -> None:
    specs = {"beat.mid": {"drum": True}}
    for root in (_dataset(tmp_path / "m", specs), _dataset(tmp_path / "a", specs, ["beat.wav"])):
        (drums,) = _codes(cd.check_dataset(root))["drums"]
        assert drums.level == "warning"


def test_empty_and_unreadable_midi_are_errors(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, {"empty.mid": {"notes": ()}, "ok.mid": {}})
    (root / "midi" / "broken.mid").write_bytes(b"not a midi file")

    report = cd.check_dataset(root)
    codes = _codes(report)

    assert _names(codes["no-notes"]) == ["empty.mid"]
    assert _names(codes["unreadable"]) == ["broken.mid"]
    assert report.exit_code == 1


def test_no_midi_files_is_an_error(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, [])

    report = cd.check_dataset(root)

    assert "no-midi" in _codes(report)
    assert report.exit_code == 1


# --- folders ------------------------------------------------------------------------


def test_symlinked_folder_inside_midi_is_reported(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"])
    elsewhere = tmp_path / "elsewhere"
    _midi(elsewhere / "hidden.mid")
    os.symlink(elsewhere, root / "midi" / "linked")

    report = cd.check_dataset(root)

    (link,) = _codes(report)["symlink"]
    assert Path(link.path).name == "linked"
    assert link.level == "warning"


def test_files_sonitra_cannot_read_in_recordings_are_reported(
    cd: ModuleType, tmp_path: Path
) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["song1.wav", "song1_phone.m4a", ".DS_Store"])

    report = cd.check_dataset(root)

    assert _names(_codes(report)["ignored"]) == ["song1_phone.m4a"]


def test_recordings_in_audio_folder_get_a_hint(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"])
    _audio(root / "audio" / "song1.wav")

    report = cd.check_dataset(root, input_type="audio")
    codes = _codes(report)

    assert "no-recordings" in codes
    assert "audio-dir" in codes


# --- rename proposals ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("midi", "recording", "expected"),
    [
        ("song1.mid", "Song1.wav", "song1.wav"),
        ("song1.mid", "song1-take1.wav", "song1_take1.wav"),
        ("song_1.mid", "Song 1 - Live.flac", "song_1_Live.flac"),
        ("sub/song1.mid", "deep/SONG1.take2.wav", "song1_take2.wav"),
    ],
)
def test_safe_rename_for_case_and_separator_differences(
    cd: ModuleType, tmp_path: Path, midi: str, recording: str, expected: str
) -> None:
    root = _dataset(tmp_path, [midi], [recording])

    report = cd.check_dataset(root)

    (rename,) = report.renames
    assert rename.recording == Path(recording)
    assert rename.new_name == expected
    assert rename.pairs_with == Path(midi)


def test_no_safe_rename_when_two_midi_files_could_match(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song.mid", "song_1.mid"], ["Song 1 live.wav"])

    report = cd.check_dataset(root)

    assert report.renames == []


def test_no_safe_rename_when_target_name_exists(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["Song1-take1.wav"])
    (root / "recordings" / "song1_take1.wav").write_text("not audio")

    report = cd.check_dataset(root)

    assert all(r.new_name != "song1_take1.wav" for r in report.renames)


def test_no_safe_rename_when_two_recordings_would_collide(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["Song1.wav", "SONG1.wav"])

    report = cd.check_dataset(root)

    assert report.renames == []


def test_near_miss_gets_a_suggestion_but_no_rename(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["sogn1.wav"])

    report = cd.check_dataset(root)

    assert report.renames == []
    (unpaired,) = _codes(report)["unpaired"]
    assert "song1.mid" in unpaired.detail


# --- plan files ---------------------------------------------------------------------


def _rename_dataset(tmp_path: Path) -> Path:
    return _dataset(tmp_path, ["song1.mid", "song2.mid"], ["Song1.wav", "sub/song2-live.wav"])


def test_plan_round_trips_and_refuses_to_overwrite(cd: ModuleType, tmp_path: Path) -> None:
    report = cd.check_dataset(_rename_dataset(tmp_path))
    plan = tmp_path / "plan.csv"

    cd.write_plan(report.renames, plan)

    with plan.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["recording"] for row in rows] == ["Song1.wav", "sub/song2-live.wav"]
    assert set(rows[0]) == {"recording", "new_name", "pairs_with", "reason"}
    assert cd.read_plan(plan) == report.renames
    with pytest.raises(cd.PlanError):
        cd.write_plan(report.renames, plan)


def test_apply_renames_writes_undo_and_undo_restores(cd: ModuleType, tmp_path: Path) -> None:
    root = _rename_dataset(tmp_path)
    recordings = root / "recordings"
    plan = tmp_path / "plan.csv"
    cd.write_plan(cd.check_dataset(root).renames, plan)

    undo = cd.apply_plan(plan, recordings)

    assert undo == tmp_path / "plan.undo.csv"
    assert sorted(p.name for p in recordings.rglob("*.wav")) == ["song1.wav", "song2_live.wav"]
    assert cd.check_dataset(root).exit_code == 0

    cd.apply_plan(undo, recordings)

    assert sorted(p.name for p in recordings.rglob("*.wav")) == ["Song1.wav", "song2-live.wav"]


@pytest.mark.parametrize(
    ("recording", "new_name", "problem"),
    [
        ("missing.wav", "song1.wav", "does not exist"),
        ("Song1.wav", "sub/song1.wav", "file name"),
        ("../escape.wav", "song1.wav", "outside"),
        ("Song1.wav", "song1.flac", "ending"),
        ("Song1.wav", "taken.wav", "already exists"),
    ],
)
def test_apply_validates_every_row_before_renaming_anything(
    cd: ModuleType, tmp_path: Path, recording: str, new_name: str, problem: str
) -> None:
    root = _rename_dataset(tmp_path)
    recordings = root / "recordings"
    _audio(recordings / "taken.wav")
    _audio(tmp_path / "corpus" / "mine" / "escape.wav")
    good = cd.Rename(Path("sub/song2-live.wav"), "song2_live.wav", Path("song2.mid"), "test")
    bad = cd.Rename(Path(recording), new_name, Path("song1.mid"), "test")
    plan = tmp_path / "plan.csv"
    cd.write_plan([good, bad], plan)

    with pytest.raises(cd.PlanError, match=problem):
        cd.apply_plan(plan, recordings)

    assert (recordings / "sub" / "song2-live.wav").exists()
    assert not (tmp_path / "plan.undo.csv").exists()


def test_apply_rejects_two_rows_with_the_same_target(cd: ModuleType, tmp_path: Path) -> None:
    root = _dataset(tmp_path, ["song1.mid"], ["a.wav", "b.wav"])
    plan = tmp_path / "plan.csv"
    cd.write_plan(
        [
            cd.Rename(Path("a.wav"), "song1.wav", Path("song1.mid"), "test"),
            cd.Rename(Path("b.wav"), "song1.wav", Path("song1.mid"), "test"),
        ],
        plan,
    )

    with pytest.raises(cd.PlanError, match="more than once"):
        cd.apply_plan(plan, root / "recordings")


def test_apply_refuses_when_undo_file_exists(cd: ModuleType, tmp_path: Path) -> None:
    root = _rename_dataset(tmp_path)
    plan = tmp_path / "plan.csv"
    cd.write_plan(cd.check_dataset(root).renames, plan)
    (tmp_path / "plan.undo.csv").write_text("keep me")

    with pytest.raises(cd.PlanError, match="undo"):
        cd.apply_plan(plan, root / "recordings")

    assert (root / "recordings" / "Song1.wav").exists()


# --- command line -------------------------------------------------------------------


def test_main_reports_and_exits_1_on_errors(
    cd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _rename_dataset(tmp_path)

    code = cd.main(["--dataset", "mine", "--corpus-root", str(tmp_path / "corpus")])

    out = capsys.readouterr().out
    assert code == 1
    assert "audio-input" in out
    assert "Song1.wav" in out and "song1.wav" in out
    assert "--plan" in out


def test_main_exits_0_for_a_clean_dataset(
    cd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path, ["song1.mid"], ["song1_take1.wav"])

    code = cd.main(["--dataset", "mine", "--corpus-root", str(tmp_path / "corpus")])

    assert code == 0
    assert "1 of 1 recording" in capsys.readouterr().out


def test_main_plan_then_apply(
    cd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _rename_dataset(tmp_path)
    args = ["--dataset", "mine", "--corpus-root", str(tmp_path / "corpus")]
    plan = tmp_path / "plan.csv"

    assert cd.main([*args, "--plan", str(plan)]) == 1
    assert plan.exists()
    assert (root / "recordings" / "Song1.wav").exists()

    assert cd.main([*args, "--apply", str(plan)]) == 0

    out = capsys.readouterr().out
    assert (root / "recordings" / "song1.wav").exists()
    assert "plan.undo.csv" in out


def test_main_plan_writes_nothing_without_renames(cd: ModuleType, tmp_path: Path) -> None:
    _dataset(tmp_path, ["song1.mid"], ["song1.wav"])
    plan = tmp_path / "plan.csv"

    cd.main(["--dataset", "mine", "--corpus-root", str(tmp_path / "corpus"), "--plan", str(plan)])

    assert not plan.exists()


def test_main_missing_dataset_folder(
    cd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cd.main(["--dataset", "nope", "--corpus-root", str(tmp_path)])

    assert code == 1
    assert "nope" in capsys.readouterr().err


def test_main_plan_and_apply_are_exclusive(cd: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cd.main(["--dataset", "x", "--plan", "a.csv", "--apply", "b.csv"])
    assert excinfo.value.code == 2


def test_long_lists_are_cut_short_unless_verbose(
    cd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _dataset(tmp_path, ["song1.mid"], ["song1.wav", *[f"stray{i:02d}.wav" for i in range(15)]])
    args = ["--dataset", "mine", "--corpus-root", str(tmp_path / "corpus")]

    cd.main(args)
    short = capsys.readouterr().out
    cd.main([*args, "--verbose"])
    full = capsys.readouterr().out

    assert "stray14.wav" not in short and "and 5 more" in short
    assert "stray14.wav" in full
