from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "download_datasets.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def dd() -> ModuleType:
    return _load_module()


# ── DATASETS registry sanity ──────────────────────────────────────────────────


def test_maestro_v3_has_extract_map(dd: ModuleType) -> None:
    spec = dd.DATASETS["maestro-v3"]
    assert spec["extract_map"]
    for prefix, subdir in spec["extract_map"]:
        assert isinstance(prefix, str) and prefix
        assert isinstance(subdir, str) and subdir
    assert spec["url"]
    assert spec["corpus_subdir"]


def test_bsed_has_extract_map(dd: ModuleType) -> None:
    spec = dd.DATASETS["bsed"]
    assert spec["extract_map"]
    for prefix, subdir in spec["extract_map"]:
        assert isinstance(prefix, str) and prefix
        assert isinstance(subdir, str) and subdir
    assert spec["url"]
    assert spec["corpus_subdir"]


def test_bsed_extract_map_has_midi_and_recordings_targets(dd: ModuleType) -> None:
    spec = dd.DATASETS["bsed"]
    targets = {subdir for _, subdir in spec["extract_map"]}
    assert "midi" in targets
    assert "recordings" in targets
    # recordings must not collide with the pipeline's own rendered-audio dir name
    assert "audio" not in targets


# ── _target_dirs ────────────────────────────────────────────────────────────


def test_target_dirs_returns_one_path_per_distinct_subdir(dd: ModuleType, tmp_path: Path) -> None:
    spec = {
        "corpus_subdir": "example",
        "extract_map": [("A/", "midi"), ("B/", "recordings")],
    }
    dirs = dd._target_dirs(tmp_path, spec)
    assert sorted(d.name for d in dirs) == ["midi", "recordings"]
    assert all(d.parent == tmp_path / "example" for d in dirs)


# ── _is_already_present ──────────────────────────────────────────────────────


def test_is_already_present_false_when_no_dirs_exist(dd: ModuleType, tmp_path: Path) -> None:
    spec = {
        "corpus_subdir": "example",
        "extract_map": [("A/", "midi"), ("B/", "recordings")],
    }
    assert dd._is_already_present(tmp_path, spec) is False


def test_is_already_present_false_when_partially_populated(dd: ModuleType, tmp_path: Path) -> None:
    spec = {
        "corpus_subdir": "example",
        "extract_map": [("A/", "midi"), ("B/", "recordings")],
    }
    midi_dir = tmp_path / "example" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    # recordings/ deliberately left missing
    assert dd._is_already_present(tmp_path, spec) is False


def test_is_already_present_true_when_all_dirs_populated(dd: ModuleType, tmp_path: Path) -> None:
    spec = {
        "corpus_subdir": "example",
        "extract_map": [("A/", "midi"), ("B/", "recordings")],
    }
    midi_dir = tmp_path / "example" / "midi"
    recordings_dir = tmp_path / "example" / "recordings"
    midi_dir.mkdir(parents=True)
    recordings_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    (recordings_dir / "a.wav").write_bytes(b"x")
    assert dd._is_already_present(tmp_path, spec) is True


# ── _download_and_extract ────────────────────────────────────────────────────


def _make_fixture_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Root/A/x.mid", b"midi-bytes")
        zf.writestr("Root/B/y.wav", b"wav-bytes")
        zf.writestr("Root/C/z.pdf", b"pdf-bytes")  # matches no extract_map prefix


def test_download_and_extract_routes_members_by_prefix_and_skips_unmatched(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(fixture_zip.read_bytes())
        if reporthook is not None:
            reporthook(1, len(fixture_zip.read_bytes()), len(fixture_zip.read_bytes()))
        return filename, None

    monkeypatch.setattr(dd, "urlretrieve", fake_urlretrieve)

    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "url": "https://example.invalid/fixture.zip",
        "corpus_subdir": "fixture",
        "extract_map": [("Root/A/", "midi"), ("Root/B/", "recordings")],
    }

    n_extracted = dd._download_and_extract("fixture", spec, output_dir)

    assert n_extracted == 2
    assert (output_dir / "fixture" / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert (output_dir / "fixture" / "recordings" / "y.wav").read_bytes() == b"wav-bytes"
    # unmatched member must not be extracted anywhere under corpus_subdir
    fixture_root = output_dir / "fixture"
    extracted_files = sorted(p.relative_to(fixture_root) for p in fixture_root.rglob("*") if p.is_file())
    assert extracted_files == [Path("midi/x.mid"), Path("recordings/y.wav")]


# ── _print_list ───────────────────────────────────────────────────────────────


def test_print_list_runs_and_shows_all_dataset_keys(
    dd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dd._print_list(tmp_path)
    out = capsys.readouterr().out
    for key in dd.DATASETS:
        assert key in out


# ── _parse_selection ────────────────────────────────────────────────────────


def test_parse_selection_returns_ordered_unique_keys(dd: ModuleType) -> None:
    keys = ["maestro-v3", "bsed"]
    assert dd._parse_selection("2,1", keys) == ["bsed", "maestro-v3"]
    # duplicates are collapsed but order is preserved
    assert dd._parse_selection("2,2,1", keys) == ["bsed", "maestro-v3"]
    assert dd._parse_selection(" 1 , 2 ", keys) == ["maestro-v3", "bsed"]


def test_parse_selection_all_q_and_empty(dd: ModuleType) -> None:
    keys = ["maestro-v3", "bsed"]
    assert dd._parse_selection("all", keys) == keys
    assert dd._parse_selection("ALL", keys) == keys
    assert dd._parse_selection("q", keys) == []
    assert dd._parse_selection("", keys) == []
    assert dd._parse_selection("   ", keys) == []


@pytest.mark.parametrize("response", ["0", "3", "-1", "a", "1,,2", "1,", ",1", "foo,bar", "1,0"])
def test_parse_selection_rejects_invalid_input(
    dd: ModuleType, response: str
) -> None:
    keys = ["maestro-v3", "bsed"]
    with pytest.raises(ValueError):
        dd._parse_selection(response, keys)


# ── rich helpers (table + fallback behaviour) ───────────────────────────────


def test_print_table_renders_dataset_rows_and_status_badges(
    dd: ModuleType, tmp_path: Path
) -> None:
    from io import StringIO

    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=120)
    dd._print_table(console, tmp_path)
    rendered = console.file.getvalue()

    for index, key in enumerate(dd.DATASETS, start=1):
        assert str(index) in rendered
        assert key in rendered
    assert "missing" in rendered  # fresh tmp_path: nothing downloaded yet

    # Pre-populate one target dir -> that row flips to "present"
    spec = dd.DATASETS["maestro-v3"]
    target_dir = tmp_path / spec["corpus_subdir"] / "midi"
    target_dir.mkdir(parents=True)
    (target_dir / "x.mid").write_bytes(b"x")

    console2 = dd._RichConsole(file=StringIO(), force_terminal=True, width=120)
    dd._print_table(console2, tmp_path)
    assert "present" in console2.file.getvalue()


def test_use_rich_output_false_when_rich_missing(
    dd: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dd, "_HAS_RICH", False)
    assert dd._use_rich_output() is False
    assert dd._can_interact() is False


def test_main_no_args_errors_without_rich_or_tty(
    dd: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dd, "_HAS_RICH", False)
    monkeypatch.setattr("sys.argv", ["download_datasets.py"])
    assert dd.main() == 1


# ── _DownloadDisplay + rich download path ───────────────────────────────────


def test_rich_download_path_renders_done(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from io import StringIO

    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(zip_bytes)
        if reporthook is not None:
            reporthook(1, 1024, 2048)
        return filename, None

    monkeypatch.setattr(dd, "urlretrieve", fake_urlretrieve)

    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "url": "https://example.invalid/fixture.zip",
        "corpus_subdir": "fixture",
        "extract_map": [("Root/A/", "midi"), ("Root/B/", "recordings")],
        "size_mb": 1,
    }
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    with display:
        status, n_files, error = dd._download_one(
            "fixture", spec, output_dir, display=display, task_id=display.tasks[0]
        )

    assert status == "done"
    assert error is None
    assert n_files == 2
    assert (output_dir / "fixture" / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert (output_dir / "fixture" / "recordings" / "y.wav").read_bytes() == b"wav-bytes"
    rendered = console.file.getvalue()
    assert "Fixture Dataset" in rendered
    assert "done" in rendered


def test_rich_download_path_skips_when_already_present(
    dd: ModuleType, tmp_path: Path
) -> None:
    from io import StringIO

    spec = {
        "name": "Fixture Dataset",
        "url": "https://example.invalid/fixture.zip",
        "corpus_subdir": "fixture",
        "extract_map": [("Root/A/", "midi")],
        "size_mb": 1,
    }
    midi_dir = tmp_path / "corpus" / "fixture" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "x.mid").write_bytes(b"x")

    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    with display:
        status, n_files, error = dd._download_one(
            "fixture", spec, tmp_path / "corpus", display=display, task_id=display.tasks[0]
        )

    assert status == "skip"
    assert n_files == 0
    assert error is None
    assert "skip" in console.file.getvalue()


def test_rich_download_path_reports_errors(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from io import StringIO

    def failing_urlretrieve(url, filename, reporthook=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(dd, "urlretrieve", failing_urlretrieve)

    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "url": "https://example.invalid/fixture.zip",
        "corpus_subdir": "fixture",
        "extract_map": [("Root/A/", "midi")],
        "size_mb": 1,
    }
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    with display:
        status, n_files, error = dd._download_one(
            "fixture", spec, output_dir, display=display, task_id=display.tasks[0]
        )

    assert status == "error"
    assert n_files == 0
    assert error == "boom"
    assert "error" in console.file.getvalue()
