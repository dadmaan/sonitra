from __future__ import annotations

import hashlib
import importlib.util
import io
import queue
import sys
import tarfile
import urllib.error
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


def _two_source_spec() -> dict:
    return {
        "corpus_subdir": "example",
        "sources": [
            {"kind": "zip", "extract_map": [("A/", None, "midi")]},
            {"kind": "zip", "extract_map": [("B/", None, "recordings")]},
        ],
    }


def test_is_already_present_false_when_no_dirs_exist(dd: ModuleType, tmp_path: Path) -> None:
    assert dd._is_already_present("example", _two_dir_spec(), tmp_path) is False


def test_is_already_present_false_when_partially_populated(dd: ModuleType, tmp_path: Path) -> None:
    midi_dir = tmp_path / "example" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    # recordings/ deliberately left missing
    assert dd._is_already_present("example", _two_dir_spec(), tmp_path) is False


def test_is_already_present_true_when_all_dirs_populated(dd: ModuleType, tmp_path: Path) -> None:
    # Legacy fallback: no markers exist, but every target dir is non-empty
    # (corpora downloaded by the old script must not be re-downloaded).
    midi_dir = tmp_path / "example" / "midi"
    recordings_dir = tmp_path / "example" / "recordings"
    midi_dir.mkdir(parents=True)
    recordings_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    (recordings_dir / "a.wav").write_bytes(b"x")
    assert dd._is_already_present("example", _two_dir_spec(), tmp_path) is True


def test_is_already_present_true_when_all_markers_exist(dd: ModuleType, tmp_path: Path) -> None:
    spec = _two_dir_spec()
    dd._write_marker(tmp_path, "example", 0, {"url": "https://example.invalid/a.zip"})
    assert dd._is_already_present("example", spec, tmp_path) is True


def test_is_already_present_false_when_only_some_markers_exist(
    dd: ModuleType, tmp_path: Path
) -> None:
    spec = _two_source_spec()
    # Even with every target dir populated, a missing marker means the
    # dataset is only partially complete -> False.
    for subdir in dd._all_target_subdirs(spec):
        target_dir = tmp_path / "example" / subdir
        target_dir.mkdir(parents=True)
        (target_dir / "x").write_bytes(b"x")
    dd._write_marker(tmp_path, "example", 0, {"url": "https://example.invalid/a.zip"})
    assert dd._is_already_present("example", spec, tmp_path) is False


# ── markers / partial paths ──────────────────────────────────────────────────


def test_marker_path_is_deterministic_and_under_downloads(dd: ModuleType, tmp_path: Path) -> None:
    m1 = dd._marker_path(tmp_path, "ds", 0)
    m2 = dd._marker_path(tmp_path, "ds", 0)
    assert m1 == m2
    assert m1.parent == tmp_path / ".downloads"
    assert m1.name == "ds.0.ok"


def test_partial_path_is_deterministic_and_scoped_by_key_and_url(
    dd: ModuleType, tmp_path: Path
) -> None:
    source = {"url": "https://example.invalid/shared.zip"}
    p1 = dd._partial_path(tmp_path, "ds", 0, source)
    p2 = dd._partial_path(tmp_path, "ds", 0, source)
    assert p1 == p2
    assert p1.parent == tmp_path / ".downloads"
    digest = hashlib.sha256(source["url"].encode()).hexdigest()[:10]
    assert p1.name == f"ds.{digest}.part"
    # same URL under a different dataset key -> different partial
    # (e.g. maestro-v3-wav and maestro-v3-full share the full archive URL)
    p_other_key = dd._partial_path(tmp_path, "ds2", 0, source)
    assert p_other_key != p1
    assert p_other_key.name.startswith("ds2.")
    # different URL under the same key -> different partial
    p_other_url = dd._partial_path(tmp_path, "ds", 0, {"url": "https://example.invalid/other.zip"})
    assert p_other_url != p1


def test_write_marker_writes_source_url_plus_newline(dd: ModuleType, tmp_path: Path) -> None:
    source = {"url": "https://example.invalid/a.zip"}
    dd._write_marker(tmp_path, "ds", 0, source)
    assert dd._marker_path(tmp_path, "ds", 0).read_text() == source["url"] + "\n"


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


def test_extract_archive_is_atomic_when_a_member_is_corrupt(
    dd: ModuleType, tmp_path: Path
) -> None:
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("Root/A/x.mid", b"midi-bytes")
        zf.writestr("Root/B/y.wav", b"wav-bytes")

    # Flip a byte in the middle of the SECOND member's compressed payload so
    # zipfile raises BadZipFile on CRC mismatch at EOF of that member's stream.
    with zipfile.ZipFile(fixture_zip, "r") as zf:
        info = zf.getinfo("Root/B/y.wav")
        payload_offset = info.header_offset + 30 + len(info.filename)
    data = bytearray(fixture_zip.read_bytes())
    data[payload_offset + 2] ^= 0xFF
    fixture_zip.write_bytes(bytes(data))

    dataset_dir = tmp_path / "out"
    extract_map = [("Root/A/", None, "midi"), ("Root/B/", None, "recordings")]

    with pytest.raises(zipfile.BadZipFile):
        dd._extract_archive(str(fixture_zip), "zip", extract_map, dataset_dir)

    # First member reached its final path; the corrupt member is absent and
    # no .part files linger anywhere under the dataset dir.
    assert (dataset_dir / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert not (dataset_dir / "recordings" / "y.wav").exists()
    assert list(dataset_dir.rglob("*.part")) == []


# ── _download_and_extract ────────────────────────────────────────────────────


def test_download_and_extract_single_archive_source(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)

    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        data = fixture_zip.read_bytes()
        dest.write_bytes(data)
        if progress is not None:
            progress(len(data), len(data))
        return len(data)

    monkeypatch.setattr(dd, "_download_file", fake_download_file)

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

    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        if url.endswith(".tar.gz"):
            data = fixture_targz.read_bytes()
        else:
            data = b"metadata-bytes"
        dest.write_bytes(data)
        if progress is not None:
            progress(len(data), len(data))
        return len(data)

    monkeypatch.setattr(dd, "_download_file", fake_download_file)

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


def test_download_and_extract_archive_source_removes_partial_and_writes_marker(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()

    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)

    monkeypatch.setattr(dd, "_download_file", fake_download_file)

    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/fixture.zip",
                "kind": "zip",
                "extract_map": [("Root/A/", None, "midi")],
                "size_mb": 1,
            }
        ],
    }

    n_extracted = dd._download_and_extract("fixture", spec, output_dir)

    assert n_extracted == 1
    marker = dd._marker_path(output_dir, "fixture", 0)
    assert marker.exists()
    assert marker.read_text() == spec["sources"][0]["url"] + "\n"
    # the working partial is removed only after full success
    assert not dd._partial_path(output_dir, "fixture", 0, spec["sources"][0]).exists()


def test_download_and_extract_file_source_renames_part_to_final(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        dest.write_bytes(b"metadata-bytes")
        return len(b"metadata-bytes")

    monkeypatch.setattr(dd, "_download_file", fake_download_file)

    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/meta.csv",
                "kind": "file",
                "target_subdir": "metadata",
                "filename": "meta.csv",
                "size_mb": 1,
            }
        ],
    }

    n_extracted = dd._download_and_extract("fixture", spec, output_dir)

    assert n_extracted == 1
    assert (output_dir / "fixture" / "metadata" / "meta.csv").read_bytes() == b"metadata-bytes"
    assert dd._marker_path(output_dir, "fixture", 0).exists()


def test_download_and_extract_failure_keeps_partial_and_writes_no_marker(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        dest.write_bytes(b"partial-bytes")
        raise RuntimeError("boom")

    monkeypatch.setattr(dd, "_download_file", failing_download_file)

    output_dir = tmp_path / "corpus"
    spec = {
        "name": "Fixture Dataset",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/fixture.zip",
                "kind": "zip",
                "extract_map": [("Root/A/", None, "midi")],
                "size_mb": 1,
            }
        ],
    }

    with pytest.raises(RuntimeError, match="boom"):
        dd._download_and_extract("fixture", spec, output_dir)

    # partial is the resume foundation for the next run
    partial = dd._partial_path(output_dir, "fixture", 0, spec["sources"][0])
    assert partial.exists()
    assert partial.read_bytes() == b"partial-bytes"
    assert not dd._marker_path(output_dir, "fixture", 0).exists()


# ── _reset_download_state ────────────────────────────────────────────────────


def test_reset_download_state_removes_markers_partials_and_extracted_parts(
    dd: ModuleType, tmp_path: Path
) -> None:
    output_dir = tmp_path / "corpus"
    spec = {
        "corpus_subdir": "fixture",
        "sources": [
            {"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1},
            {"url": "https://example.invalid/b.zip", "kind": "zip", "extract_map": [("B/", None, "recordings")], "size_mb": 1},
        ],
    }

    dd._write_marker(output_dir, "fixture", 0, spec["sources"][0])
    dd._write_marker(output_dir, "fixture", 1, spec["sources"][1])
    partial_0 = dd._partial_path(output_dir, "fixture", 0, spec["sources"][0])
    partial_1 = dd._partial_path(output_dir, "fixture", 1, spec["sources"][1])
    partial_0.parent.mkdir(parents=True, exist_ok=True)
    partial_0.write_bytes(b"p0")
    partial_1.write_bytes(b"p1")
    extracted_part = output_dir / "fixture" / "midi" / "a.mid.part"
    extracted_part.parent.mkdir(parents=True)
    extracted_part.write_bytes(b"e")

    dd._reset_download_state(output_dir, "fixture", spec)

    assert not dd._marker_path(output_dir, "fixture", 0).exists()
    assert not dd._marker_path(output_dir, "fixture", 1).exists()
    assert not partial_0.exists()
    assert not partial_1.exists()
    assert not extracted_part.exists()


# ── _download_file_attempt ───────────────────────────────────────────────────


class _FakeResponse:
    """Minimal stand-in for http.client.HTTPResponse (status, headers, read)."""

    def __init__(self, status: int, headers: dict, chunks: list) -> None:
        self.status = status
        self.headers = headers
        self._chunks = list(chunks)

    def read(self, n: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_download_file_attempt_fresh_download(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"0123456789"
    monkeypatch.setattr(
        dd.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, {"Content-Length": str(len(data))}, [data]),
    )
    dest = tmp_path / "out.part"
    progress_calls: list = []

    n = dd._download_file_attempt(
        "https://example.invalid/x",
        dest,
        progress=lambda downloaded, total: progress_calls.append((downloaded, total)),
    )

    assert n == len(data)
    assert dest.read_bytes() == data
    assert progress_calls == [(len(data), len(data))]


def test_download_file_attempt_resumes_with_range_header(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"0123456789"
    dest = tmp_path / "out.part"
    dest.write_bytes(data[:5])  # pre-existing partial
    seen_ranges: list = []

    def fake_urlopen(req, timeout=None):
        seen_ranges.append(req.get_header("Range"))
        return _FakeResponse(206, {"Content-Range": "bytes 5-9/10"}, [data[5:]])

    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

    n = dd._download_file_attempt("https://example.invalid/x", dest, progress=None)

    assert n == 10
    assert dest.read_bytes() == data  # appended to the resumed prefix
    assert seen_ranges == ["bytes=5-"]


def test_download_file_attempt_restarts_when_server_ignores_range(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"abcdefghij"  # different content than the stale prefix
    dest = tmp_path / "out.part"
    dest.write_bytes(b"01234")

    monkeypatch.setattr(
        dd.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, {"Content-Length": str(len(data))}, [data]),
    )

    n = dd._download_file_attempt("https://example.invalid/x", dest, progress=None)

    assert n == len(data)
    assert dest.read_bytes() == data  # truncated and rewritten, not appended


def test_download_file_attempt_416_matching_partial_is_complete(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out.part"
    dest.write_bytes(b"0123456789")  # 10 bytes == server total

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.invalid/x",
            416,
            "Range Not Satisfiable",
            {"Content-Range": "bytes */10"},
            None,
        )

    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

    n = dd._download_file_attempt("https://example.invalid/x", dest, progress=None)

    assert n == 10
    assert dest.read_bytes() == b"0123456789"  # untouched


def test_download_file_attempt_416_larger_partial_raises_and_keeps_file(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out.part"
    dest.write_bytes(b"0123456789ab")  # 12 bytes > server's 10

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.invalid/x",
            416,
            "Range Not Satisfiable",
            {"Content-Range": "bytes */10"},
            None,
        )

    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="larger than the server's file"):
        dd._download_file_attempt("https://example.invalid/x", dest, progress=None)
    assert dest.read_bytes() == b"0123456789ab"  # partial kept on disk


def test_download_file_attempt_raises_on_incomplete_download(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dd.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, {"Content-Length": "10"}, [b"0123"]),
    )

    with pytest.raises(RuntimeError, match="incomplete download"):
        dd._download_file_attempt("https://example.invalid/x", tmp_path / "out.part", progress=None)


# ── _download_file retry loop ────────────────────────────────────────────────


def test_download_file_retries_after_transient_error(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}
    sleeps: list = []

    def fake_attempt(url, dest, *, progress=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.URLError("boom")
        dest.write_bytes(b"ok")
        return 2

    monkeypatch.setattr(dd, "_download_file_attempt", fake_attempt)
    monkeypatch.setattr(dd.time, "sleep", lambda seconds: sleeps.append(seconds))

    n = dd._download_file(
        "https://example.invalid/x",
        tmp_path / "out.part",
        name="x",
        output_dir=tmp_path,
        key="k",
        index=0,
    )

    assert n == 2
    assert attempts["n"] == 2
    assert sleeps == [3]  # first backoff only


def test_download_file_exhausts_retries_and_keeps_partial(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}
    sleeps: list = []

    def fake_attempt(url, dest, *, progress=None):
        attempts["n"] += 1
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(dd, "_download_file_attempt", fake_attempt)
    monkeypatch.setattr(dd.time, "sleep", lambda seconds: sleeps.append(seconds))

    dest = tmp_path / "out.part"
    dest.write_bytes(b"partial-bytes")

    with pytest.raises(RuntimeError, match="failed after 4 attempts"):
        dd._download_file(
            "https://example.invalid/x", dest, name="x", output_dir=tmp_path, key="k", index=0
        )

    assert attempts["n"] == 4
    assert sleeps == [3, 10, 30]
    assert dest.read_bytes() == b"partial-bytes"  # partial is never deleted


def test_download_file_raises_immediately_on_client_error(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def fake_attempt(url, dest, *, progress=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(dd, "_download_file_attempt", fake_attempt)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        dd._download_file(
            "https://example.invalid/x",
            tmp_path / "out.part",
            name="x",
            output_dir=tmp_path,
            key="k",
            index=0,
        )

    assert attempts["n"] == 1  # no retry for permanent client errors


def test_download_file_retries_on_429(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def fake_attempt(url, dest, *, progress=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
        dest.write_bytes(b"ok")
        return 2

    monkeypatch.setattr(dd, "_download_file_attempt", fake_attempt)
    monkeypatch.setattr(dd.time, "sleep", lambda seconds: None)

    n = dd._download_file(
        "https://example.invalid/x",
        tmp_path / "out.part",
        name="x",
        output_dir=tmp_path,
        key="k",
        index=0,
    )

    assert n == 2
    assert attempts["n"] == 2


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

    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        dest.write_bytes(zip_bytes)
        if progress is not None:
            progress(1024, 2048)
        return len(zip_bytes)

    monkeypatch.setattr(dd, "_download_file", fake_download_file)

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

    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(dd, "_download_file", failing_download_file)

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


def test_slot_worker_prints_error_to_stderr(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from io import StringIO

    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(dd, "_download_file", failing_download_file)
    spec = _fixture_zip_spec()
    monkeypatch.setattr(dd, "DATASETS", {"fixture": spec})

    # NOT force_terminal: rich's Live only redirects sys.stderr through the
    # console when console.is_terminal is true, so a plain console lets the
    # worker's stderr print reach capsys.
    console = dd._RichConsole(file=StringIO(), width=100)
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    work_queue: "queue.Queue[str]" = queue.Queue()
    work_queue.put("fixture")
    with display:
        failed = dd._slot_worker(0, display.tasks[0], work_queue, display, tmp_path / "corpus")

    assert failed is True
    err = capsys.readouterr().err
    assert "[error]" in err
    assert "boom" in err


def test_run_rich_reports_final_failure_summary(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from io import StringIO

    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(dd, "_download_file", failing_download_file)
    spec = _fixture_zip_spec()
    monkeypatch.setattr(dd, "DATASETS", {"fixture": spec})

    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)

    any_failure = dd._run_rich(["fixture"], tmp_path / "corpus", 1, display)

    assert any_failure is True
    assert display.done_count() == 0
    assert display.skipped_count() == 0
    assert display.failed_count() == 1
    assert "Fixture Dataset: boom" in display.errors()
    err = capsys.readouterr().err
    assert "download finished: 0 done, 0 skipped, 1 failed" in err


# ── _download_one (force / skip) ─────────────────────────────────────────────


def test_download_one_force_resets_state_and_redownloads(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()
    spec = _fixture_zip_spec()
    output_dir = tmp_path / "corpus"

    # Simulate a previously-completed dataset that --force must reset.
    dd._write_marker(output_dir, "fixture", 0, spec["sources"][0])
    stale_partial = dd._partial_path(output_dir, "fixture", 0, spec["sources"][0])
    stale_partial.parent.mkdir(parents=True, exist_ok=True)
    stale_partial.write_bytes(b"stale")

    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)  # part of _download_file's contract
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)

    monkeypatch.setattr(dd, "_download_file", fake_download_file)

    status, n_files, error = dd._download_one("fixture", spec, output_dir, force=True)

    assert status == "done"
    assert n_files == 2
    assert error is None
    assert not stale_partial.exists()
    assert dd._marker_path(output_dir, "fixture", 0).exists()
    assert (output_dir / "fixture" / "midi" / "x.mid").read_bytes() == b"midi-bytes"


def test_download_one_skips_when_markers_present(dd: ModuleType, tmp_path: Path) -> None:
    spec = _fixture_zip_spec()
    output_dir = tmp_path / "corpus"
    dd._write_marker(output_dir, "fixture", 0, spec["sources"][0])

    status, n_files, error = dd._download_one("fixture", spec, output_dir)

    assert status == "skip"
    assert n_files == 0
    assert error is None


# ── preflight (disk space) ───────────────────────────────────────────────────


class _FakeDiskUsage:
    def __init__(self, free: int) -> None:
        self.free = free


def test_check_disk_space_returns_none_when_ample_space(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda path: _FakeDiskUsage(10**12))
    spec = {"sources": [{"size_mb": 1}]}
    assert dd._check_disk_space(tmp_path, [spec]) is None


def test_check_disk_space_reports_when_free_space_insufficient(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda path: _FakeDiskUsage(1_000_000))
    spec = {"sources": [{"size_mb": 1000}]}
    msg = dd._check_disk_space(tmp_path, [spec])
    assert msg is not None
    assert "not enough free space" in msg


def test_check_disk_space_probes_nearest_existing_ancestor(
    dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list = []
    monkeypatch.setattr(
        dd.shutil,
        "disk_usage",
        lambda path: probed.append(path) or _FakeDiskUsage(10**12),
    )
    spec = {"sources": [{"size_mb": 1}]}
    missing = tmp_path / "a" / "b"  # neither a nor b exists
    assert dd._check_disk_space(missing, [spec]) is None
    assert probed == [tmp_path]


def test_main_aborts_on_preflight_failure(
    dd: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        dd, "_check_disk_space", lambda output_dir, specs: "not enough free space on X"
    )
    monkeypatch.setattr("sys.argv", ["download_datasets.py", "bsed"])
    assert dd.main() == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "not enough free space" in err


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_parse_args_force_flag(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["download_datasets.py", "bsed", "--force"])
    args = dd._parse_args()
    assert args.force is True
    assert args.dataset == "bsed"

    monkeypatch.setattr("sys.argv", ["download_datasets.py", "bsed"])
    assert dd._parse_args().force is False
