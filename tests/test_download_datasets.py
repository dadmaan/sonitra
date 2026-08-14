from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
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


def _assert_valid_extract_map(dd: ModuleType, extract_map) -> None:
    for prefix, patterns, subdir in extract_map:
        assert isinstance(prefix, str)
        assert patterns is None or isinstance(patterns, frozenset)
        assert isinstance(subdir, str) and subdir


def _assert_valid_source(dd: ModuleType, source: dict) -> None:
    assert source["url"]
    assert source["kind"] in {"zip", "targz", "file"}
    if source["kind"] == "file":
        assert source["target_subdir"]
        assert source["filename"]
    else:
        assert source["extract_map"]
        _assert_valid_extract_map(dd, source["extract_map"])


# ── DATASETS registry sanity ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key",
    [
        "maestro-v3-midi",
        "maestro-v3-wav",
        "maestro-v3-full",
        "bsed",
        "musicnet",
        "e-gmd-midi",
        "e-gmd-full",
    ],
)
def test_every_registry_entry_has_valid_sources(dd: ModuleType, key: str) -> None:
    spec = dd.DATASETS[key]
    assert spec["corpus_subdir"]
    assert spec["sources"]
    for source in spec["sources"]:
        _assert_valid_source(dd, source)


def test_maestro_variants_share_corpus_subdir(dd: ModuleType) -> None:
    assert dd.DATASETS["maestro-v3-midi"]["corpus_subdir"] == "maestro-v3"
    assert dd.DATASETS["maestro-v3-wav"]["corpus_subdir"] == "maestro-v3"
    assert dd.DATASETS["maestro-v3-full"]["corpus_subdir"] == "maestro-v3"


def test_maestro_wav_and_full_share_the_same_full_archive_url(dd: ModuleType) -> None:
    # MAESTRO ships no audio-only archive: -wav downloads the same combined
    # zip as -full and discards the MIDI members during extraction.
    wav_url = dd.DATASETS["maestro-v3-wav"]["sources"][0]["url"]
    full_url = dd.DATASETS["maestro-v3-full"]["sources"][0]["url"]
    assert wav_url == full_url
    midi_only_url = dd.DATASETS["maestro-v3-midi"]["sources"][0]["url"]
    assert midi_only_url != full_url


def test_bsed_extract_map_has_midi_and_recordings_targets(dd: ModuleType) -> None:
    spec = dd.DATASETS["bsed"]
    targets = {target for source in spec["sources"] for _, _, target in source["extract_map"]}
    assert "midi" in targets
    assert "recordings" in targets
    # recordings must not collide with the pipeline's own rendered-audio dir name
    assert "audio" not in targets


def test_musicnet_has_three_sources_including_a_bare_file(dd: ModuleType) -> None:
    spec = dd.DATASETS["musicnet"]
    assert len(spec["sources"]) == 3
    kinds = [s["kind"] for s in spec["sources"]]
    assert kinds.count("targz") == 2
    assert kinds.count("file") == 1


def test_maps_was_deliberately_dropped(dd: ModuleType) -> None:
    # MAPS is gated behind a registration form with no scriptable direct
    # download URL, incompatible with this script's unattended-download model.
    assert "maps" not in dd.DATASETS


# ── _matches_patterns / _route_member ───────────────────────────────────────


def test_matches_patterns_none_matches_anything(dd: ModuleType) -> None:
    assert dd._matches_patterns("whatever.xyz", None) is True


def test_matches_patterns_by_extension(dd: ModuleType) -> None:
    assert dd._matches_patterns("a/b/foo.WAV", frozenset({".wav"})) is True
    assert dd._matches_patterns("a/b/foo.mid", frozenset({".wav"})) is False


def test_matches_patterns_by_exact_basename(dd: ModuleType) -> None:
    assert dd._matches_patterns("top/README", frozenset({"readme"})) is True
    assert dd._matches_patterns("top/README.md", frozenset({"readme"})) is False


def test_route_member_first_match_wins(dd: ModuleType) -> None:
    extract_map = [
        ("A/", frozenset({".mid"}), "midi"),
        ("A/", None, "metadata"),
    ]
    assert dd._route_member("A/x.mid", extract_map) == ("A/", "midi")
    assert dd._route_member("A/x.csv", extract_map) == ("A/", "metadata")
    assert dd._route_member("B/x.mid", extract_map) is None


# ── _dataset_size_mb / _all_target_subdirs ──────────────────────────────────


def test_dataset_size_mb_sums_sources(dd: ModuleType) -> None:
    spec = {"sources": [{"size_mb": 10}, {"size_mb": 5}, {}]}
    assert dd._dataset_size_mb(spec) == 15


def test_all_target_subdirs_unions_archive_and_file_sources(dd: ModuleType) -> None:
    spec = {
        "sources": [
            {"kind": "zip", "extract_map": [("A/", None, "midi"), ("B/", None, "recordings")]},
            {"kind": "file", "target_subdir": "metadata", "filename": "x.csv"},
        ]
    }
    assert dd._all_target_subdirs(spec) == ["metadata", "midi", "recordings"]


# ── _target_dirs ────────────────────────────────────────────────────────────


def test_target_dirs_returns_one_path_per_distinct_subdir(dd: ModuleType, tmp_path: Path) -> None:
    spec = {
        "corpus_subdir": "example",
        "sources": [{"kind": "zip", "extract_map": [("A/", None, "midi"), ("B/", None, "recordings")]}],
    }
    dirs = dd._target_dirs(tmp_path, spec)
    assert sorted(d.name for d in dirs) == ["midi", "recordings"]
    assert all(d.parent == tmp_path / "example" for d in dirs)


# ── _is_already_present ──────────────────────────────────────────────────────


def _two_dir_spec() -> dict:
    return {
        "corpus_subdir": "example",
        "sources": [{"kind": "zip", "extract_map": [("A/", None, "midi"), ("B/", None, "recordings")]}],
    }


def test_is_already_present_false_when_no_dirs_exist(dd: ModuleType, tmp_path: Path) -> None:
    assert dd._is_already_present(tmp_path, _two_dir_spec()) is False


def test_is_already_present_false_when_partially_populated(dd: ModuleType, tmp_path: Path) -> None:
    midi_dir = tmp_path / "example" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    # recordings/ deliberately left missing
    assert dd._is_already_present(tmp_path, _two_dir_spec()) is False


def test_is_already_present_true_when_all_dirs_populated(dd: ModuleType, tmp_path: Path) -> None:
    midi_dir = tmp_path / "example" / "midi"
    recordings_dir = tmp_path / "example" / "recordings"
    midi_dir.mkdir(parents=True)
    recordings_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    (recordings_dir / "a.wav").write_bytes(b"x")
    assert dd._is_already_present(tmp_path, _two_dir_spec()) is True


# ── _extract_archive ─────────────────────────────────────────────────────────


def _make_fixture_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Root/A/x.mid", b"midi-bytes")
        zf.writestr("Root/B/y.wav", b"wav-bytes")
        zf.writestr("Root/C/z.pdf", b"pdf-bytes")  # matches no extract_map rule


def _make_fixture_targz(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in [
            ("data/x.mid", b"midi-bytes"),
            ("data/y.wav", b"wav-bytes"),
            ("data/z.pdf", b"pdf-bytes"),  # matches no extract_map rule
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_extract_archive_zip_routes_by_prefix_and_skips_unmatched(
    dd: ModuleType, tmp_path: Path
) -> None:
    fixture_zip = tmp_path / "fixture.zip"
    _make_fixture_zip(fixture_zip)
    dataset_dir = tmp_path / "out"
    extract_map = [("Root/A/", None, "midi"), ("Root/B/", None, "recordings")]

    n = dd._extract_archive(str(fixture_zip), "zip", extract_map, dataset_dir)

    assert n == 2
    assert (dataset_dir / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert (dataset_dir / "recordings" / "y.wav").read_bytes() == b"wav-bytes"
    extracted_files = sorted(p.relative_to(dataset_dir) for p in dataset_dir.rglob("*") if p.is_file())
    assert extracted_files == [Path("midi/x.mid"), Path("recordings/y.wav")]


def test_extract_archive_targz_routes_by_extension(dd: ModuleType, tmp_path: Path) -> None:
    fixture_targz = tmp_path / "fixture.tar.gz"
    _make_fixture_targz(fixture_targz)
    dataset_dir = tmp_path / "out"
    extract_map = [("", frozenset({".mid"}), "midi"), ("", frozenset({".wav"}), "recordings")]

    n = dd._extract_archive(str(fixture_targz), "targz", extract_map, dataset_dir)

    assert n == 2
    assert (dataset_dir / "midi" / "data" / "x.mid").read_bytes() == b"midi-bytes"
    assert (dataset_dir / "recordings" / "data" / "y.wav").read_bytes() == b"wav-bytes"


def test_extract_archive_extension_routing_discards_unwanted_kind(
    dd: ModuleType, tmp_path: Path
) -> None:
    # Exercises the maestro-v3-wav case: an archive with interleaved midi/wav
    # under the same prefix, where only .wav should survive extraction.
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("root/2004/foo.midi", b"midi-bytes")
        zf.writestr("root/2004/foo.wav", b"wav-bytes")
    dataset_dir = tmp_path / "out"
    extract_map = [("root/", frozenset({".wav"}), "recordings")]

    n = dd._extract_archive(str(fixture_zip), "zip", extract_map, dataset_dir)

    assert n == 1
    assert (dataset_dir / "recordings" / "2004" / "foo.wav").exists()
    assert not (dataset_dir / "midi").exists()


# ── _download_and_extract ────────────────────────────────────────────────────


def test_download_and_extract_single_archive_source(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)

    def fake_urlretrieve(url, filename, reporthook=None):
        data = fixture_zip.read_bytes()
        Path(filename).write_bytes(data)
        if reporthook is not None:
            reporthook(1, len(data), len(data))
        return filename, None

    monkeypatch.setattr(dd, "urlretrieve", fake_urlretrieve)

    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/fixture.zip",
                "kind": "zip",
                "extract_map": [("Root/A/", None, "midi"), ("Root/B/", None, "recordings")],
                "size_mb": 1,
            }
        ],
    }

    n_extracted = dd._download_and_extract("fixture", spec, output_dir)

    assert n_extracted == 2
    assert (output_dir / "fixture" / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert (output_dir / "fixture" / "recordings" / "y.wav").read_bytes() == b"wav-bytes"


def test_download_and_extract_multiple_sources_including_bare_file(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_targz = tmp_path / "fixture_source.tar.gz"
    _make_fixture_targz(fixture_targz)

    def fake_urlretrieve(url, filename, reporthook=None):
        if url.endswith(".tar.gz"):
            data = fixture_targz.read_bytes()
        else:
            data = b"metadata-bytes"
        Path(filename).write_bytes(data)
        if reporthook is not None:
            reporthook(1, len(data), len(data))
        return filename, None

    monkeypatch.setattr(dd, "urlretrieve", fake_urlretrieve)

    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Multi-source Fixture",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/fixture.tar.gz",
                "kind": "targz",
                "extract_map": [
                    ("", frozenset({".mid"}), "midi"),
                    ("", frozenset({".wav"}), "recordings"),
                ],
                "size_mb": 1,
            },
            {
                "url": "https://example.invalid/meta.csv",
                "kind": "file",
                "target_subdir": "metadata",
                "filename": "meta.csv",
                "size_mb": 1,
            },
        ],
    }

    n_extracted = dd._download_and_extract("fixture", spec, output_dir)

    assert n_extracted == 3
    assert (output_dir / "fixture" / "metadata" / "meta.csv").read_bytes() == b"metadata-bytes"
    assert (output_dir / "fixture" / "midi" / "data" / "x.mid").exists()
    assert (output_dir / "fixture" / "recordings" / "data" / "y.wav").exists()


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
    keys = ["maestro-v3-midi", "bsed"]
    assert dd._parse_selection("2,1", keys) == ["bsed", "maestro-v3-midi"]
    # duplicates are collapsed but order is preserved
    assert dd._parse_selection("2,2,1", keys) == ["bsed", "maestro-v3-midi"]
    assert dd._parse_selection(" 1 , 2 ", keys) == ["maestro-v3-midi", "bsed"]


def test_parse_selection_all_q_and_empty(dd: ModuleType) -> None:
    keys = ["maestro-v3-midi", "bsed"]
    assert dd._parse_selection("all", keys) == keys
    assert dd._parse_selection("ALL", keys) == keys
    assert dd._parse_selection("q", keys) == []
    assert dd._parse_selection("", keys) == []
    assert dd._parse_selection("   ", keys) == []


@pytest.mark.parametrize("response", ["0", "3", "-1", "a", "1,,2", "1,", ",1", "foo,bar", "1,0"])
def test_parse_selection_rejects_invalid_input(
    dd: ModuleType, response: str
) -> None:
    keys = ["maestro-v3-midi", "bsed"]
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

    # Pre-populate every target dir -> that row flips to "present"
    spec = dd.DATASETS["bsed"]
    for subdir in dd._all_target_subdirs(spec):
        target_dir = tmp_path / spec["corpus_subdir"] / subdir
        target_dir.mkdir(parents=True)
        (target_dir / "x").write_bytes(b"x")

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


def _fixture_zip_spec(source_kwargs: dict | None = None) -> dict:
    return {
        "name": "Fixture Dataset",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/fixture.zip",
                "kind": "zip",
                "extract_map": [("Root/A/", None, "midi"), ("Root/B/", None, "recordings")],
                "size_mb": 1,
                **(source_kwargs or {}),
            }
        ],
    }


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
    spec = _fixture_zip_spec()
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

    spec = _fixture_zip_spec()
    midi_dir = tmp_path / "corpus" / "fixture" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "x.mid").write_bytes(b"x")
    # recordings/ also required by this spec's extract_map, so populate it too
    recordings_dir = tmp_path / "corpus" / "fixture" / "recordings"
    recordings_dir.mkdir(parents=True)
    (recordings_dir / "y.wav").write_bytes(b"x")

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
    spec = _fixture_zip_spec()
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    with display:
        status, n_files, error = dd._download_one(
            "fixture", spec, output_dir, display=display, task_id=display.tasks[0]
        )

    assert status == "error"
    assert n_files == 0
    assert error == "boom"
    assert "error" in console.file.getvalue()
