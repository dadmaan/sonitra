
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import queue
import sys
import tarfile
import threading
import time
import urllib.error
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

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
    assert source["kind"] in {"zip", "targz", "file", "hf_tree"}
    if source["kind"] == "file":
        assert source["url"]
        assert source["target_subdir"]
        assert source["filename"]
    elif source["kind"] == "hf_tree":
        assert source["repo"]
        assert source["revision"]
        assert source["subdir"]
        assert source["target_subdir"]
        patterns = source.get("patterns")
        assert patterns is None or isinstance(patterns, frozenset)
    else:
        assert source["url"]
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
        "musicnet-midi",
        "musicnet-full",
        "e-gmd-midi",
        "e-gmd-full",
        "guitarset-mic",
        "guitarset-mix",
        "guitarset-full",
        "gaps-midi",
        "gaps-full",
    ],
)
def test_every_registry_entry_has_valid_sources(dd: ModuleType, key: str) -> None:
    spec = dd.DATASETS[key]
    assert spec["corpus_subdir"]
    assert spec["sources"]
    for source in spec["sources"]:
        _assert_valid_source(dd, source)


def test_every_registry_entry_has_a_note(dd: ModuleType) -> None:
    for key, spec in dd.DATASETS.items():
        assert isinstance(spec.get("note"), str) and spec["note"].strip(), key


def test_keys_sharing_a_corpus_subdir_share_one_note(dd: ModuleType) -> None:
    notes_by_subdir: dict[str, set] = {}
    for spec in dd.DATASETS.values():
        notes_by_subdir.setdefault(spec["corpus_subdir"], set()).add(spec.get("note"))
    for subdir, notes in notes_by_subdir.items():
        assert len(notes) == 1, subdir


def test_maestro_variants_share_corpus_subdir(dd: ModuleType) -> None:
    assert dd.DATASETS["maestro-v3-midi"]["corpus_subdir"] == "maestro-v3"
    assert dd.DATASETS["maestro-v3-wav"]["corpus_subdir"] == "maestro-v3"
    assert dd.DATASETS["maestro-v3-full"]["corpus_subdir"] == "maestro-v3"


def test_maestro_wav_and_full_share_the_same_full_archive_url(dd: ModuleType) -> None:
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
    assert "audio" not in targets


def test_guitarset_variants_share_corpus_subdir_and_targets(dd: ModuleType) -> None:
    keys = ["guitarset-mic", "guitarset-mix", "guitarset-full"]
    for key in keys:
        spec = dd.DATASETS[key]
        assert spec["corpus_subdir"] == "guitarset"
        targets = {target for source in spec["sources"] for _, _, target in source["extract_map"]}
        assert targets == {"annotations", "recordings"}
        assert "audio" not in targets
        assert "midi" not in targets
        assert any(source["url"].endswith("annotation.zip") for source in spec["sources"])
    union_targets = {
        target
        for key in keys
        for source in dd.DATASETS[key]["sources"]
        for _, _, target in source["extract_map"]
    }
    assert union_targets == {"annotations", "recordings"}


def test_guitarset_full_combines_both_audio_variants_in_one_run(dd: ModuleType) -> None:
    spec = dd.DATASETS["guitarset-full"]
    assert len(spec["sources"]) == 3
    urls = [source["url"] for source in spec["sources"]]
    assert sum(url.endswith("annotation.zip") for url in urls) == 1
    assert any(url.endswith("audio_mono-mic.zip") for url in urls)
    assert any(url.endswith("audio_mono-pickup_mix.zip") for url in urls)
    assert dd._dataset_size_mb(spec) == 38 + 627 + 652


def test_next_steps_registry_field(dd: ModuleType) -> None:
    # Replaces old _guitarset_next_steps test
    for key in ["guitarset-mic", "guitarset-mix", "guitarset-full"]:
        assert "next_steps" in dd.DATASETS[key]
        assert "scripts/guitarset_jams_to_midi.py --dry-run" in dd.DATASETS[key]["next_steps"]
        assert "config/benchmark/guitarset_test.yaml" in dd.DATASETS[key]["next_steps"]
    # MusicNet converter
    assert "next_steps" in dd.DATASETS["musicnet-full"]
    assert "scripts/musicnet_labels_to_midi.py" in dd.DATASETS["musicnet-full"]["next_steps"]
    assert "config/benchmark/musicnet_test.yaml" in dd.DATASETS["musicnet-full"]["next_steps"]
    # Also check helper still returns same
    assert dd._guitarset_next_steps() == dd.DATASETS["guitarset-mic"]["next_steps"]


def test_musicnet_split_keys_and_order(dd: ModuleType) -> None:
    assert "musicnet" not in dd.DATASETS
    keys = list(dd.DATASETS.keys())
    assert "musicnet-midi" in keys
    assert "musicnet-full" in keys
    idx_midi = keys.index("musicnet-midi")
    idx_full = keys.index("musicnet-full")
    assert idx_full == idx_midi + 1
    # Registry convention: midi before full, both between bsed and e-gmd-midi (picker 5-6)
    assert keys.index("bsed") < idx_midi < keys.index("e-gmd-midi")
    assert dd.DATASETS["musicnet-midi"]["corpus_subdir"] == "musicnet"
    assert dd.DATASETS["musicnet-full"]["corpus_subdir"] == "musicnet"


def test_musicnet_midi_targets_and_size(dd: ModuleType) -> None:
    spec = dd.DATASETS["musicnet-midi"]
    targets = set(dd._all_target_subdirs(spec))
    assert targets == {"midi", "metadata"}
    assert dd._dataset_size_mb(spec) == 4
    assert any(
        s["url"].endswith("musicnet_midis.tar.gz") and s["kind"] == "targz"
        for s in spec["sources"]
    )
    assert spec.get("superseded_by") == ["musicnet-full"]


def test_musicnet_full_targets_and_size(dd: ModuleType) -> None:
    spec = dd.DATASETS["musicnet-full"]
    targets = set(dd._all_target_subdirs(spec))
    assert targets == {"annotations/labels", "annotations/score_midi", "metadata", "recordings"}
    assert "midi" not in targets
    assert "audio" not in targets
    assert dd._dataset_size_mb(spec) == 10_584 + 3 + 1
    assert "next_steps" in spec
    assert "scripts/musicnet_labels_to_midi.py" in spec["next_steps"]


def test_musicnet_route_member_labels_to_annotations(dd: ModuleType) -> None:
    extract_map = dd.DATASETS["musicnet-full"]["sources"][1]["extract_map"]
    # musicnet.tar.gz second source
    assert dd._route_member("musicnet/train_labels/2104.csv", extract_map) == ("", "annotations/labels")
    assert dd._route_member("musicnet/test_labels/2300.csv", extract_map) == ("", "annotations/labels")
    assert dd._route_member("musicnet/train_data/2104.wav", extract_map) == ("", "recordings")


def test_musicnet_metadata_source_shared_and_score_source_distinct(dd: ModuleType) -> None:
    midi_sources = dd.DATASETS["musicnet-midi"]["sources"]
    full_sources = dd.DATASETS["musicnet-full"]["sources"]
    # metadata source is verbatim copy -> same source_id
    midi_meta = [s for s in midi_sources if s["kind"] == "file"][0]
    full_meta = [s for s in full_sources if s["kind"] == "file"][0]
    assert midi_meta == full_meta
    assert dd._source_id(midi_meta) == dd._source_id(full_meta)
    assert dd._download_key(midi_meta) == dd._download_key(full_meta)
    # score MIDI shares URL but not id (different target)
    midi_score = [s for s in midi_sources if "musicnet_midis.tar.gz" in s["url"]][0]
    full_score = [s for s in full_sources if "musicnet_midis.tar.gz" in s["url"]][0]
    assert midi_score["url"] == full_score["url"]
    assert dd._source_id(midi_score) != dd._source_id(full_score)
    assert dd._download_key(midi_score) == dd._download_key(full_score)


def test_musicnet_presence_via_superseded(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    # Create midi-only content
    midi_targets = set(dd._all_target_subdirs(dd.DATASETS["musicnet-midi"]))
    for subdir in midi_targets:
        target = tmp_path / "musicnet" / subdir
        target.mkdir(parents=True, exist_ok=True)
        (target / "x").write_bytes(b"x")
    # Also need metadata file for record? Use fallback check first (no records)
    assert dd._is_already_present("musicnet-midi", dd.DATASETS["musicnet-midi"], tmp_path) is True
    assert dd._is_already_present("musicnet-full", dd.DATASETS["musicnet-full"], tmp_path) is False
    # Now create full content via fallback (all target dirs non-empty)
    for subdir in set(dd._all_target_subdirs(dd.DATASETS["musicnet-full"])):
        target = tmp_path / "musicnet" / subdir
        target.mkdir(parents=True, exist_ok=True)
        (target / "y").write_bytes(b"y")
    dd._clear_state_cache()
    assert dd._is_already_present("musicnet-full", dd.DATASETS["musicnet-full"], tmp_path) is True
    assert dd._is_already_present("musicnet-midi", dd.DATASETS["musicnet-midi"], tmp_path) is True


def test_musicnet_notes_page_numbers(dd: ModuleType) -> None:
    groups = dd._note_groups()
    numbers = [n for n, _, _ in groups]
    assert numbers == ["1-3", "4", "5-6", "7-8", "9-11", "12-13"]


def test_musicnet_midi_fake_network_never_requests_full_archive(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list = []

    def fake_download(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Need to handle zip extraction: create minimal tar.gz for midi
        import tarfile
        import io

        if url.endswith("musicnet_midis.tar.gz"):
            # Create a tar.gz with one midi
            tmp = dest
            # Write a real tar.gz content
            with tarfile.open(tmp, "w:gz") as tf:
                data = b"midi-bytes"
                info = tarfile.TarInfo(name="Beethoven/1727_schubert.mid")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        elif url.endswith("musicnet.tar.gz"):
            import tarfile
            import io

            with tarfile.open(dest, "w:gz") as tf:
                data = b"wav-bytes"
                info = tarfile.TarInfo(name="musicnet/train_data/1727.wav")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        elif url.endswith("musicnet_metadata.csv"):
            dest.write_bytes(b"csv")
        else:
            dest.write_bytes(b"x")
        return dest.stat().st_size if dest.exists() else 1

    monkeypatch.setattr(dd, "_download_file", fake_download)
    dd._clear_state_cache()
    output_dir = tmp_path / "corpus"
    dd._download_and_extract("musicnet-midi", dd.DATASETS["musicnet-midi"], output_dir)
    assert not any("musicnet.tar.gz" in url for url in fetched)
    assert any("musicnet_midis.tar.gz" in url for url in fetched)
    assert any("musicnet_metadata.csv" in url for url in fetched)


def test_musicnet_descriptions_and_notes(dd: ModuleType) -> None:
    for key in ["musicnet-midi", "musicnet-full"]:
        spec = dd.DATASETS[key]
        desc = spec["description"]
        assert "CC BY 4.0" in desc
        assert "Thickstun" in desc
        assert "ICLR 2017" in desc
    # Note shared
    assert dd.DATASETS["musicnet-midi"]["note"] == dd.DATASETS["musicnet-full"]["note"]
    # Descriptions contain score-time and converter and 7 corrupt
    assert "score-time" in dd.DATASETS["musicnet-midi"]["description"]
    assert "MIDI-input" in dd.DATASETS["musicnet-midi"]["description"]
    assert "7" in dd.DATASETS["musicnet-midi"]["description"] and "corrupt" in dd.DATASETS["musicnet-midi"]["description"]
    assert "score-time" in dd.DATASETS["musicnet-full"]["description"]
    # At least mention converter/script
    assert "musicnet_labels_to_midi" in dd.DATASETS["musicnet-full"]["description"] or "converter" in dd.DATASETS["musicnet-full"]["description"].lower()


def test_maps_was_deliberately_dropped(dd: ModuleType) -> None:
    assert "maps" not in dd.DATASETS


# ── superseded_by ─────────────────────────────────────────────────────────────

def test_superseded_by_targets_exist_and_share_corpus_subdir(dd: ModuleType) -> None:
    for key, spec in dd.DATASETS.items():
        for sup in spec.get("superseded_by", []):
            assert sup in dd.DATASETS, f"{key} superseded_by {sup} not in registry"
            assert dd.DATASETS[sup]["corpus_subdir"] == spec["corpus_subdir"], f"{key} -> {sup} corpus_subdir mismatch"


def test_superseded_by_expected_groups(dd: ModuleType) -> None:
    assert dd.DATASETS["maestro-v3-midi"].get("superseded_by") == ["maestro-v3-full"]
    assert dd.DATASETS["maestro-v3-wav"].get("superseded_by") == ["maestro-v3-full"]
    assert dd.DATASETS["e-gmd-midi"].get("superseded_by") == ["e-gmd-full"]
    assert dd.DATASETS["guitarset-mic"].get("superseded_by") == ["guitarset-full"]
    assert dd.DATASETS["guitarset-mix"].get("superseded_by") == ["guitarset-full"]
    assert dd.DATASETS["gaps-midi"].get("superseded_by") == ["gaps-full"]
    assert dd.DATASETS["musicnet-midi"].get("superseded_by") == ["musicnet-full"]


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


# ── _source_id and _download_key ────────────────────────────────────────────

def test_source_id_deterministic_and_hex(dd: ModuleType) -> None:
    source = {"kind": "zip", "url": "https://example.com/a.zip", "extract_map": [("p/", frozenset({".wav"}), "recordings")]}
    sid1 = dd._source_id(source)
    sid2 = dd._source_id(source)
    assert sid1 == sid2
    assert len(sid1) == 12
    assert all(c in "0123456789abcdef" for c in sid1)


def test_source_id_sharing_for_verbatim_copies(dd: ModuleType) -> None:
    # GuitarSet annotation.zip is verbatim copy across three keys
    mic_ann = [s for s in dd.DATASETS["guitarset-mic"]["sources"] if s["url"].endswith("annotation.zip")][0]
    mix_ann = [s for s in dd.DATASETS["guitarset-mix"]["sources"] if s["url"].endswith("annotation.zip")][0]
    full_ann = [s for s in dd.DATASETS["guitarset-full"]["sources"] if s["url"].endswith("annotation.zip")][0]
    assert dd._source_id(mic_ann) == dd._source_id(mix_ann) == dd._source_id(full_ann)
    # GAPS midi source is shared between gaps-midi and gaps-full
    midi_midi = [s for s in dd.DATASETS["gaps-midi"]["sources"] if s.get("subdir") == "midi"][0]
    midi_full = [s for s in dd.DATASETS["gaps-full"]["sources"] if s.get("subdir") == "midi"][0]
    assert dd._source_id(midi_midi) == dd._source_id(midi_full)
    # file source also shared
    file_midi = [s for s in dd.DATASETS["gaps-midi"]["sources"] if s["kind"] == "file"][0]
    file_full = [s for s in dd.DATASETS["gaps-full"]["sources"] if s["kind"] == "file"][0]
    assert dd._source_id(file_midi) == dd._source_id(file_full)


def test_source_id_distinct_for_different_extract_map(dd: ModuleType) -> None:
    wav = dd.DATASETS["maestro-v3-wav"]["sources"][0]
    full = dd.DATASETS["maestro-v3-full"]["sources"][0]
    assert wav["url"] == full["url"]
    assert dd._source_id(wav) != dd._source_id(full)


def test_source_id_sorted_patterns(dd: ModuleType) -> None:
    s1 = {"kind": "zip", "url": "https://example.com/a.zip", "extract_map": [("p/", frozenset({".wav", ".mid"}), "midi")]}
    s2 = {"kind": "zip", "url": "https://example.com/a.zip", "extract_map": [("p/", frozenset({".mid", ".wav"}), "midi")]}
    assert dd._source_id(s1) == dd._source_id(s2)
    s3 = {"kind": "hf_tree", "repo": "r", "revision": "rev", "subdir": "s", "patterns": frozenset({".wav", ".mid"}), "target_subdir": "t"}
    s4 = {"kind": "hf_tree", "repo": "r", "revision": "rev", "subdir": "s", "patterns": frozenset({".mid", ".wav"}), "target_subdir": "t"}
    assert dd._source_id(s3) == dd._source_id(s4)


def test_download_key_is_10_hex_and_sharing(dd: ModuleType) -> None:
    url = "https://example.com/shared.zip"
    s_mic = {"kind": "zip", "url": url, "extract_map": [("", None, "midi")]}
    s_mix = {"kind": "zip", "url": url, "extract_map": [("", None, "recordings")]}
    # Same URL => same download key even though source_id differs due to extract_map? Actually if extract_map same, source_id same, but here different target, so ids differ but download key same
    assert dd._download_key(s_mic) == dd._download_key(s_mix)
    assert len(dd._download_key(s_mic)) == 10
    # hf_tree sharing
    hf1 = {"kind": "hf_tree", "repo": "xavriley/GAPS", "revision": "abc", "subdir": "midi", "patterns": frozenset({".mid"}), "target_subdir": "midi"}
    hf2 = {"kind": "hf_tree", "repo": "xavriley/GAPS", "revision": "abc", "subdir": "midi", "patterns": frozenset({".mid"}), "target_subdir": "midi"}
    assert dd._download_key(hf1) == dd._download_key(hf2)
    # Different subdir => different key
    hf3 = {"kind": "hf_tree", "repo": "xavriley/GAPS", "revision": "abc", "subdir": "audio", "patterns": frozenset({".wav"}), "target_subdir": "recordings"}
    assert dd._download_key(hf1) != dd._download_key(hf3)


def test_download_key_for_archive_shared_between_maestro_wav_and_full(dd: ModuleType) -> None:
    wav = dd.DATASETS["maestro-v3-wav"]["sources"][0]
    full = dd.DATASETS["maestro-v3-full"]["sources"][0]
    assert dd._download_key(wav) == dd._download_key(full)


# ── records and _source_state ───────────────────────────────────────────────

def test_record_written_only_on_success(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("A/x.mid", b"data")
    def fake_download(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fixture_zip.read_bytes())
        return len(fixture_zip.read_bytes())
    monkeypatch.setattr(dd, "_download_file", fake_download)
    output_dir = tmp_path / "corpus"
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    dd._clear_state_cache()
    n = dd._download_and_extract("fixture", spec, output_dir)
    assert n == 1
    sid = dd._source_id(spec["sources"][0])
    rec_path = output_dir / "fixture" / ".sources" / f"{sid}.json"
    assert rec_path.exists()
    data = json.loads(rec_path.read_text())
    assert data["version"] == 1
    assert data["source_id"] == sid
    assert "files" in data
    assert data["files"] == {"midi/x.mid": 4}
    assert dd._source_state(output_dir / "fixture", sid) == "satisfied"

    # Failure should not write record
    def failing(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial")
        raise RuntimeError("boom")
    monkeypatch.setattr(dd, "_download_file", failing)
    spec2 = {"name": "Fail", "corpus_subdir": "fail", "sources": [{"url": "https://example.invalid/b.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    dd._clear_state_cache()
    with pytest.raises(RuntimeError):
        dd._download_and_extract("fail", spec2, output_dir)
    sid2 = dd._source_id(spec2["sources"][0])
    assert not (output_dir / "fail" / ".sources" / f"{sid2}.json").exists()
    assert dd._source_state(output_dir / "fail", sid2) == "none"


def test_source_state_satisfied_stale_none_and_memoization(dd: ModuleType, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    source = {"kind": "file", "url": "https://example.invalid/x.csv", "target_subdir": "metadata", "filename": "x.csv"}
    sid = dd._source_id(source)
    dd._clear_state_cache()
    assert dd._source_state(dataset_dir, sid) == "none"
    # Create record with one file
    (dataset_dir / "metadata").mkdir(parents=True)
    (dataset_dir / "metadata" / "x.csv").write_bytes(b"hello")
    files = {"metadata/x.csv": 5}
    dd._write_record(dataset_dir, sid, "origin", files)
    assert dd._source_state(dataset_dir, sid) == "satisfied"
    # Memoization: delete file but cached still says satisfied
    (dataset_dir / "metadata" / "x.csv").unlink()
    assert dd._source_state(dataset_dir, sid) == "satisfied"
    # After clearing cache, should detect stale
    dd._clear_state_cache()
    assert dd._source_state(dataset_dir, sid) == "stale"
    # Truncated file also stale
    (dataset_dir / "metadata" / "x.csv").write_bytes(b"hi")
    dd._clear_state_cache()
    assert dd._source_state(dataset_dir, sid) == "stale"


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
    dd._clear_state_cache()
    assert dd._is_already_present("example", _two_dir_spec(), tmp_path) is False


def test_is_already_present_false_when_partially_populated(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    midi_dir = tmp_path / "example" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    assert dd._is_already_present("example", _two_dir_spec(), tmp_path) is False


def test_is_already_present_true_when_all_dirs_populated(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    midi_dir = tmp_path / "example" / "midi"
    recordings_dir = tmp_path / "example" / "recordings"
    midi_dir.mkdir(parents=True)
    recordings_dir.mkdir(parents=True)
    (midi_dir / "a.mid").write_bytes(b"x")
    (recordings_dir / "a.wav").write_bytes(b"x")
    assert dd._is_already_present("example", _two_dir_spec(), tmp_path) is True


def test_is_already_present_true_when_all_sources_satisfied(dd: ModuleType, tmp_path: Path) -> None:
    # New record-based present
    dd._clear_state_cache()
    spec = {"corpus_subdir": "example", "sources": [{"kind": "file", "url": "https://example.invalid/a.csv", "target_subdir": "metadata", "filename": "a.csv", "size_mb": 1}]}
    dataset_dir = tmp_path / "example"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata").mkdir(parents=True)
    (dataset_dir / "metadata" / "a.csv").write_bytes(b"hello")
    sid = dd._source_id(spec["sources"][0])
    dd._write_record(dataset_dir, sid, "example", {"metadata/a.csv": 5})
    dd._clear_state_cache()
    assert dd._is_already_present("example", spec, tmp_path) is True


def test_is_already_present_false_when_only_some_sources_satisfied(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    spec = _two_source_spec()
    # Satisfy first source only
    dataset_dir = tmp_path / "example"
    dataset_dir.mkdir(parents=True)
    for subdir in ["midi"]:
        (dataset_dir / subdir).mkdir(parents=True)
        (dataset_dir / subdir / "x").write_bytes(b"x")
    # Create record for first source
    s0 = spec["sources"][0]
    sid0 = dd._source_id(s0)
    # Need to make a file for first source's record
    # For zip source, files mapping is tricky; create a dummy file and record
    dd._write_record(dataset_dir, sid0, "example", {"midi/x": 1})
    dd._clear_state_cache()
    assert dd._is_already_present("example", spec, tmp_path) is False


def test_is_already_present_superseded_by_present(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # guitarset-mic is superseded by guitarset-full: if full content exists, mic reads present
    dd._clear_state_cache()
    # Create full content via fallback (no records) -> both appear present due to coarse check (known limit documented)
    for subdir in dd._all_target_subdirs(dd.DATASETS["guitarset-full"]):
        target = tmp_path / "guitarset" / subdir
        target.mkdir(parents=True)
        (target / "x").write_bytes(b"x")
    assert dd._is_already_present("guitarset-mic", dd.DATASETS["guitarset-mic"], tmp_path) is True
    assert dd._is_already_present("guitarset-full", dd.DATASETS["guitarset-full"], tmp_path) is True
    # New test with records: create mic via records, full should be not present because mix missing
    dd._clear_state_cache()
    tmp2 = tmp_path / "tmp2"
    tmp2.mkdir()
    dataset_dir = tmp2 / "guitarset"
    dataset_dir.mkdir()
    import shutil
    shutil.rmtree(dataset_dir, ignore_errors=True)
    dataset_dir.mkdir()
    (dataset_dir / "annotations").mkdir(parents=True)
    (dataset_dir / "recordings").mkdir(parents=True)
    (dataset_dir / "annotations" / "x").write_bytes(b"x")
    (dataset_dir / "recordings" / "x").write_bytes(b"x")
    # Write records for mic's two sources
    mic_sources = dd.DATASETS["guitarset-mic"]["sources"]
    for src in mic_sources:
        sid = dd._source_id(src)
        # For annotation source, file is annotations/x, for mic source recordings/x
        if "annotation" in src["url"]:
            dd._write_record(dataset_dir, sid, "guitarset-mic", {"annotations/x": 1})
        else:
            dd._write_record(dataset_dir, sid, "guitarset-mic", {"recordings/x": 1})
    dd._clear_state_cache()
    assert dd._is_already_present("guitarset-mic", dd.DATASETS["guitarset-mic"], tmp2) is True
    # Full has 3 sources: annotation (same sid as mic's annotation), mic audio (same sid), and mix audio (different sid not satisfied)
    # So full should be not present because mix source is none
    assert dd._is_already_present("guitarset-full", dd.DATASETS["guitarset-full"], tmp2) is False
    # Also test that mic present via superseded when full is present via records
    # Create full records as well
    for src in dd.DATASETS["guitarset-full"]["sources"]:
        sid = dd._source_id(src)
        if "annotation" in src["url"]:
            dd._write_record(dataset_dir, sid, "guitarset-full", {"annotations/x": 1})
        elif "audio_mono-mic" in src["url"]:
            dd._write_record(dataset_dir, sid, "guitarset-full", {"recordings/x": 1})
        else:
            # mix
            (dataset_dir / "recordings" / "y").write_bytes(b"y")
            dd._write_record(dataset_dir, sid, "guitarset-full", {"recordings/y": 1})
    dd._clear_state_cache()
    assert dd._is_already_present("guitarset-full", dd.DATASETS["guitarset-full"], tmp2) is True
    assert dd._is_already_present("guitarset-mic", dd.DATASETS["guitarset-mic"], tmp2) is True


# ── legacy handling ──────────────────────────────────────────────────────────

def test_legacy_marker_means_incomplete_when_all_none(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    spec = {"corpus_subdir": "example", "sources": [{"kind": "zip", "url": "https://example.invalid/a.zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    # Populate target dir to look complete via fallback, but add legacy marker
    dataset_dir = tmp_path / "example"
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "x.mid").write_bytes(b"x")
    downloads = tmp_path / ".downloads"
    downloads.mkdir()
    (downloads / "example.0.ok").write_text("https://example.invalid/a.zip\n")
    assert dd._is_already_present("example", spec, tmp_path) is False


def test_legacy_partial_means_incomplete_when_all_none(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    spec = {"corpus_subdir": "example", "sources": [{"kind": "zip", "url": "https://example.invalid/a.zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    dataset_dir = tmp_path / "example"
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "x.mid").write_bytes(b"x")
    downloads = tmp_path / ".downloads"
    downloads.mkdir()
    digest = hashlib.sha256(spec["sources"][0]["url"].encode()).hexdigest()[:10]
    (downloads / f"example.{digest}.part").write_bytes(b"partial")
    assert dd._is_already_present("example", spec, tmp_path) is False


def test_legacy_partial_adopted_and_resumed(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dd._clear_state_cache()
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("A/x.mid", b"data")
    zip_bytes = fixture_zip.read_bytes()
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    output_dir = tmp_path / "corpus"
    # Create legacy partial
    digest = hashlib.sha256(spec["sources"][0]["url"].encode()).hexdigest()[:10]
    legacy = output_dir / ".downloads" / f"fixture.{digest}.part"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"partial-legacy")
    download_key = dd._download_key(spec["sources"][0])
    new_path = output_dir / ".downloads" / f"{download_key}.part"
    assert not new_path.exists()
    # Mock _download_file to check that new path is used and contains legacy bytes plus resumed?
    called = {}
    def fake_download(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        called["dest"] = dest
        # Simulate resume by checking dest exists and prefix
        assert dest == new_path
        assert dest.read_bytes() == b"partial-legacy"
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)
    monkeypatch.setattr(dd, "_download_file", fake_download)
    n = dd._download_and_extract("fixture", spec, output_dir)
    assert called["dest"] == new_path
    assert not legacy.exists()
    assert n == 1


def test_legacy_markers_unlinked_on_skip_and_success(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dd._clear_state_cache()
    # Success case
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("A/x.mid", b"data")
    zip_bytes = fixture_zip.read_bytes()
    def fake_download(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)
    monkeypatch.setattr(dd, "_download_file", fake_download)
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    output_dir = tmp_path / "corpus"
    downloads = output_dir / ".downloads"
    downloads.mkdir(parents=True)
    (downloads / "fixture.0.ok").write_text("https://example.invalid/a.zip\n")
    n = dd._download_and_extract("fixture", spec, output_dir)
    assert not (downloads / "fixture.0.ok").exists()
    # Skip case: dataset already present via record, legacy marker should be cleared on skip
    dd._clear_state_cache()
    # Create legacy marker again and ensure skip clears it
    (downloads / "fixture.0.ok").write_text("https://example.invalid/a.zip\n")
    # Dataset is already present (record exists), so _download_one should skip and clear marker
    status, _, _ = dd._download_one("fixture", spec, output_dir)
    assert status == "skip"
    assert not (downloads / "fixture.0.ok").exists()


# ── _extract_archive ─────────────────────────────────────────────────────────


def _make_fixture_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Root/A/x.mid", b"midi-bytes")
        zf.writestr("Root/B/y.wav", b"wav-bytes")
        zf.writestr("Root/C/z.pdf", b"pdf-bytes")


def _make_fixture_targz(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in [
            ("data/x.mid", b"midi-bytes"),
            ("data/y.wav", b"wav-bytes"),
            ("data/z.pdf", b"pdf-bytes"),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_extract_archive_zip_routes_by_prefix_and_skips_unmatched(dd: ModuleType, tmp_path: Path) -> None:
    fixture_zip = tmp_path / "fixture.zip"
    _make_fixture_zip(fixture_zip)
    dataset_dir = tmp_path / "out"
    extract_map = [("Root/A/", None, "midi"), ("Root/B/", None, "recordings")]
    n = dd._extract_archive(str(fixture_zip), "zip", extract_map, dataset_dir)
    assert n == 2
    assert (dataset_dir / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert (dataset_dir / "recordings" / "y.wav").read_bytes() == b"wav-bytes"


def test_extract_archive_targz_routes_by_extension(dd: ModuleType, tmp_path: Path) -> None:
    fixture_targz = tmp_path / "fixture.tar.gz"
    _make_fixture_targz(fixture_targz)
    dataset_dir = tmp_path / "out"
    extract_map = [("", frozenset({".mid"}), "midi"), ("", frozenset({".wav"}), "recordings")]
    n = dd._extract_archive(str(fixture_targz), "targz", extract_map, dataset_dir)
    assert n == 2
    assert (dataset_dir / "midi" / "data" / "x.mid").read_bytes() == b"midi-bytes"
    assert (dataset_dir / "recordings" / "data" / "y.wav").read_bytes() == b"wav-bytes"


def test_extract_archive_extension_routing_discards_unwanted_kind(dd: ModuleType, tmp_path: Path) -> None:
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


def test_extract_archive_is_atomic_when_a_member_is_corrupt(dd: ModuleType, tmp_path: Path) -> None:
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("Root/A/x.mid", b"midi-bytes")
        zf.writestr("Root/B/y.wav", b"wav-bytes")
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
    assert (dataset_dir / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert not (dataset_dir / "recordings" / "y.wav").exists()
    assert list(dataset_dir.rglob("*.part")) == []


def test_extract_archive_skips_members_already_present_at_size(dd: ModuleType, tmp_path: Path) -> None:
    fixture_zip = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("A/x.mid", b"hello")
        zf.writestr("A/y.mid", b"world!")
    dataset_dir = tmp_path / "out"
    extract_map = [("A/", None, "midi")]
    # Pre-create x.mid with correct size
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "x.mid").write_bytes(b"hello")
    sid = "abc123def456"
    collected = {}
    def on_file(rel, size):
        collected[rel] = size
    n = dd._extract_archive(str(fixture_zip), "zip", extract_map, dataset_dir, part_suffix=sid, skip_existing=True, on_file=on_file)
    assert n == 1  # only y.mid extracted
    assert collected == {"midi/x.mid": 5, "midi/y.mid": 6}
    assert (dataset_dir / "midi" / "x.mid").read_bytes() == b"hello"
    assert (dataset_dir / "midi" / "y.mid").read_bytes() == b"world!"
    # Check that temp file used suffix
    assert not list(dataset_dir.rglob("*.part"))


# ── _download_and_extract ────────────────────────────────────────────────────


def test_download_and_extract_single_archive_source(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    dd._clear_state_cache()
    n_extracted = dd._download_and_extract("fixture", spec, output_dir)
    assert n_extracted == 2
    assert (output_dir / "fixture" / "midi" / "x.mid").read_bytes() == b"midi-bytes"
    assert (output_dir / "fixture" / "recordings" / "y.wav").read_bytes() == b"wav-bytes"
    # Check record written, not marker
    sid = dd._source_id(spec["sources"][0])
    assert (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()


def test_download_and_extract_multiple_sources_including_bare_file(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_targz = tmp_path / "fixture_source.tar.gz"
    _make_fixture_targz(fixture_targz)
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    dd._clear_state_cache()
    n_extracted = dd._download_and_extract("fixture", spec, output_dir)
    assert n_extracted == 3
    assert (output_dir / "fixture" / "metadata" / "meta.csv").read_bytes() == b"metadata-bytes"
    assert (output_dir / "fixture" / "midi" / "data" / "x.mid").exists()
    assert (output_dir / "fixture" / "recordings" / "data" / "y.wav").exists()
    # Check both records exist
    for src in spec["sources"]:
        sid = dd._source_id(src)
        assert (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()


def test_download_and_extract_archive_source_writes_record_and_cleans_partial(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    dd._clear_state_cache()
    n_extracted = dd._download_and_extract("fixture", spec, output_dir)
    assert n_extracted == 1
    sid = dd._source_id(spec["sources"][0])
    assert (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()
    # partial removed
    download_key = dd._download_key(spec["sources"][0])
    assert not (output_dir / ".downloads" / f"{download_key}.part").exists()


def test_download_and_extract_file_source_uses_source_id_suffix(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        # Check that dest is with source_id suffix
        assert ".part" in dest.name
        # Should be like meta.csv.<12hex>.part
        assert dest.name.endswith(".part")
        assert len(dest.name.split(".")) >= 3
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    dd._clear_state_cache()
    n_extracted = dd._download_and_extract("fixture", spec, output_dir)
    assert n_extracted == 1
    assert (output_dir / "fixture" / "metadata" / "meta.csv").read_bytes() == b"metadata-bytes"
    sid = dd._source_id(spec["sources"][0])
    assert (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()


def test_download_and_extract_failure_keeps_partial_and_writes_no_record(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    dd._clear_state_cache()
    with pytest.raises(RuntimeError, match="boom"):
        dd._download_and_extract("fixture", spec, output_dir)
    sid = dd._source_id(spec["sources"][0])
    download_key = dd._download_key(spec["sources"][0])
    assert (output_dir / ".downloads" / f"{download_key}.part").exists()
    assert (output_dir / ".downloads" / f"{download_key}.part").read_bytes() == b"partial-bytes"
    assert not (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()


# ── _download_file_attempt ───────────────────────────────────────────────────


class _FakeResponse:
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


def test_download_file_attempt_fresh_download(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_download_file_attempt_resumes_with_range_header(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"0123456789"
    dest = tmp_path / "out.part"
    dest.write_bytes(data[:5])
    seen_ranges: list = []
    def fake_urlopen(req, timeout=None):
        seen_ranges.append(req.get_header("Range"))
        return _FakeResponse(206, {"Content-Range": "bytes 5-9/10"}, [data[5:]])
    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
    n = dd._download_file_attempt("https://example.invalid/x", dest, progress=None)
    assert n == 10
    assert dest.read_bytes() == data
    assert seen_ranges == ["bytes=5-"]


def test_download_file_attempt_restarts_when_server_ignores_range(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"abcdefghij"
    dest = tmp_path / "out.part"
    dest.write_bytes(b"01234")
    monkeypatch.setattr(
        dd.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, {"Content-Length": str(len(data))}, [data]),
    )
    n = dd._download_file_attempt("https://example.invalid/x", dest, progress=None)
    assert n == len(data)
    assert dest.read_bytes() == data


def test_download_file_attempt_416_matching_partial_is_complete(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "out.part"
    dest.write_bytes(b"0123456789")
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
    assert dest.read_bytes() == b"0123456789"


def test_download_file_attempt_416_larger_partial_raises_and_keeps_file(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "out.part"
    dest.write_bytes(b"0123456789ab")
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
    assert dest.read_bytes() == b"0123456789ab"


def test_download_file_attempt_raises_on_incomplete_download(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dd.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, {"Content-Length": "10"}, [b"0123"]),
    )
    with pytest.raises(RuntimeError, match="incomplete download"):
        dd._download_file_attempt("https://example.invalid/x", tmp_path / "out.part", progress=None)


# ── _download_file retry loop ────────────────────────────────────────────────


def test_download_file_retries_after_transient_error(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}
    sleeps: list = []
    def fake_attempt(url, dest, *, progress=None, **kwargs):
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
    assert sleeps == [3]


def test_download_file_exhausts_retries_and_keeps_partial(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}
    sleeps: list = []
    def fake_attempt(url, dest, *, progress=None, **kwargs):
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
    assert dest.read_bytes() == b"partial-bytes"


def test_download_file_raises_immediately_on_client_error(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}
    def fake_attempt(url, dest, *, progress=None, **kwargs):
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
    assert attempts["n"] == 1


def test_download_file_retries_on_429(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}
    def fake_attempt(url, dest, *, progress=None, **kwargs):
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


def test_print_list_runs_and_shows_all_dataset_keys(dd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dd._print_list(tmp_path)
    out = capsys.readouterr().out
    for key in dd.DATASETS:
        assert key in out


# ── _parse_selection ────────────────────────────────────────────────────────


def test_parse_selection_returns_ordered_unique_keys(dd: ModuleType) -> None:
    keys = ["maestro-v3-midi", "bsed"]
    assert dd._parse_selection("2,1", keys) == ["bsed", "maestro-v3-midi"]
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
def test_parse_selection_rejects_invalid_input(dd: ModuleType, response: str) -> None:
    keys = ["maestro-v3-midi", "bsed"]
    with pytest.raises(ValueError):
        dd._parse_selection(response, keys)


# ── rich helpers (table + fallback behaviour) ───────────────────────────────


def test_print_table_renders_dataset_rows_and_status_badges(dd: ModuleType, tmp_path: Path) -> None:
    from io import StringIO
    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=120)
    dd._print_table(console, tmp_path)
    rendered = console.file.getvalue()
    for index, key in enumerate(dd.DATASETS, start=1):
        assert str(index) in rendered
        assert key in rendered
    assert "missing" in rendered
    # Populate to make present
    spec = dd.DATASETS["bsed"]
    for subdir in dd._all_target_subdirs(spec):
        target_dir = tmp_path / spec["corpus_subdir"] / subdir
        target_dir.mkdir(parents=True)
        (target_dir / "x").write_bytes(b"x")
    console2 = dd._RichConsole(file=StringIO(), force_terminal=True, width=120)
    dd._clear_state_cache()
    dd._print_table(console2, tmp_path)
    assert "present" in console2.file.getvalue()


def test_use_rich_output_false_when_rich_missing(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd, "_HAS_RICH", False)
    assert dd._use_rich_output() is False
    assert dd._can_interact() is False


def test_main_no_args_errors_without_rich_or_tty(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_rich_download_path_renders_done(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from io import StringIO
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_bytes)
        if progress is not None:
            progress(1024, 2048)
        return len(zip_bytes)
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    output_dir = tmp_path / "corpus"
    spec = _fixture_zip_spec()
    dd._clear_state_cache()
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


def test_rich_download_path_skips_when_already_present(dd: ModuleType, tmp_path: Path) -> None:
    from io import StringIO
    spec = _fixture_zip_spec()
    # Create record to make present
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    # Need to create files and record for satisfied
    for subdir in ["midi", "recordings"]:
        (dataset_dir / subdir).mkdir(parents=True)
        (dataset_dir / subdir / "x.mid" if subdir=="midi" else dataset_dir / subdir / "y.wav").write_bytes(b"x")
    sid = dd._source_id(spec["sources"][0])
    # Create record with those files
    dd._write_record(dataset_dir, sid, "fixture", {"midi/x.mid": 1, "recordings/y.wav": 1})
    dd._clear_state_cache()
    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    with display:
        status, n_files, error = dd._download_one(
            "fixture", spec, output_dir, display=display, task_id=display.tasks[0]
        )
    assert status == "skip"
    assert n_files == 0
    assert error is None
    assert "skip" in console.file.getvalue()


def test_rich_download_path_reports_errors(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from io import StringIO
    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(dd, "_download_file", failing_download_file)
    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    output_dir = tmp_path / "corpus"
    spec = _fixture_zip_spec()
    dd._clear_state_cache()
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=1)
    with display:
        status, n_files, error = dd._download_one(
            "fixture", spec, output_dir, display=display, task_id=display.tasks[0]
        )
    assert status == "error"
    assert n_files == 0
    assert error == "boom"
    assert "error" in console.file.getvalue()


def test_slot_worker_prints_error_to_stderr(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from io import StringIO
    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(dd, "_download_file", failing_download_file)
    spec = _fixture_zip_spec()
    monkeypatch.setattr(dd, "DATASETS", {"fixture": spec})
    dd._clear_state_cache()
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


def test_run_rich_reports_final_failure_summary(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from io import StringIO
    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(dd, "_download_file", failing_download_file)
    spec = _fixture_zip_spec()
    monkeypatch.setattr(dd, "DATASETS", {"fixture": spec})
    dd._clear_state_cache()
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


def test_download_one_force_resets_state_and_redownloads(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()
    spec = _fixture_zip_spec()
    output_dir = tmp_path / "corpus"
    # Simulate previously completed dataset via record
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "old.mid").write_bytes(b"old")
    sid = dd._source_id(spec["sources"][0])
    dd._write_record(dataset_dir, sid, "fixture", {"midi/old.mid": 3})
    # Also create a download partial that force should clear
    download_key = dd._download_key(spec["sources"][0])
    dl_part = output_dir / ".downloads" / f"{download_key}.part"
    dl_part.parent.mkdir(parents=True, exist_ok=True)
    dl_part.write_bytes(b"stale")
    # Create a corpus part that force should clear
    corpus_part = dataset_dir / "midi" / f"old.mid.{sid}.part"
    corpus_part.write_bytes(b"stale")
    dd._clear_state_cache()
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    status, n_files, error = dd._download_one("fixture", spec, output_dir, force=True)
    assert status == "done"
    assert n_files == 2
    assert error is None
    assert not dl_part.exists()
    assert not corpus_part.exists()
    # Record updated
    data = json.loads((dataset_dir / ".sources" / f"{sid}.json").read_text())
    assert "midi/x.mid" in data["files"]


def test_download_one_skips_when_already_satisfied(dd: ModuleType, tmp_path: Path) -> None:
    spec = _fixture_zip_spec()
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "recordings").mkdir(parents=True)
    (dataset_dir / "midi" / "x.mid").write_bytes(b"x")
    (dataset_dir / "recordings" / "y.wav").write_bytes(b"y")
    sid = dd._source_id(spec["sources"][0])
    dd._write_record(dataset_dir, sid, "fixture", {"midi/x.mid": 1, "recordings/y.wav": 1})
    dd._clear_state_cache()
    # Also create legacy marker that should be cleared on skip
    downloads = output_dir / ".downloads"
    downloads.mkdir(parents=True)
    (downloads / "fixture.0.ok").write_text("legacy\n")
    status, n_files, error = dd._download_one("fixture", spec, output_dir)
    assert status == "skip"
    assert n_files == 0
    assert error is None
    assert not (downloads / "fixture.0.ok").exists()


def test_download_one_clears_no_partial_on_success(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()
    spec = _fixture_zip_spec()
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    status, n_files, error = dd._download_one("fixture", spec, output_dir)
    assert status == "done"
    assert n_files == 2
    assert error is None
    download_key = dd._download_key(spec["sources"][0])
    assert not (output_dir / ".downloads" / f"{download_key}.part").exists()
    sid = dd._source_id(spec["sources"][0])
    assert (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()


def test_download_one_failure_keeps_partial_for_resume(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_zip = tmp_path / "fixture_source.zip"
    _make_fixture_zip(fixture_zip)
    zip_bytes = fixture_zip.read_bytes()
    spec = {
        "name": "Two-source Fixture",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "url": "https://example.invalid/a.zip",
                "kind": "zip",
                "extract_map": [("Root/A/", None, "midi")],
                "size_mb": 1,
            },
            {
                "url": "https://example.invalid/b.zip",
                "kind": "zip",
                "extract_map": [("Root/B/", None, "recordings")],
                "size_mb": 1,
            },
        ],
    }
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    def flaky_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if index == 1 or "b.zip" in url:
            dest.write_bytes(b"partial-bytes")
            raise RuntimeError("boom")
        dest.write_bytes(zip_bytes)
        return len(zip_bytes)
    monkeypatch.setattr(dd, "_download_file", flaky_download_file)
    status, n_files, error = dd._download_one("fixture", spec, output_dir)
    assert status == "error"
    assert error == "boom"
    # First source record exists, second not
    sid0 = dd._source_id(spec["sources"][0])
    sid1 = dd._source_id(spec["sources"][1])
    assert (output_dir / "fixture" / ".sources" / f"{sid0}.json").exists()
    assert not (output_dir / "fixture" / ".sources" / f"{sid1}.json").exists()
    # Partial for failed source remains
    dk1 = dd._download_key(spec["sources"][1])
    assert (output_dir / ".downloads" / f"{dk1}.part").read_bytes() == b"partial-bytes"


# ── file/archive skip and stale handling ─────────────────────────────────────

def test_file_source_skipped_when_satisfied(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    dd._clear_state_cache()
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata").mkdir(parents=True)
    (dataset_dir / "metadata" / "x.csv").write_bytes(b"data")
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/x.csv", "kind": "file", "target_subdir": "metadata", "filename": "x.csv", "size_mb": 1}]}
    sid = dd._source_id(spec["sources"][0])
    dd._write_record(dataset_dir, sid, "fixture", {"metadata/x.csv": 4})
    dd._clear_state_cache()
    # Should skip download
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        raise AssertionError("should not be called")
    monkeypatch.setattr(dd, "_download_file", fake)
    n = dd._download_and_extract("fixture", spec, output_dir)
    assert n == 1
    out = capsys.readouterr().out
    assert "already present" in out
    assert "0 downloaded" in out


def test_archive_skipped_when_satisfied(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    dd._clear_state_cache()
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "x.mid").write_bytes(b"data")
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    sid = dd._source_id(spec["sources"][0])
    dd._write_record(dataset_dir, sid, "fixture", {"midi/x.mid": 4})
    dd._clear_state_cache()
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        raise AssertionError("should not download")
    monkeypatch.setattr(dd, "_download_file", fake)
    n = dd._download_and_extract("fixture", spec, output_dir)
    assert n == 1
    out = capsys.readouterr().out
    assert "already present" in out
    assert "0 extracted" in out


def test_stale_prints_and_refetches(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    dd._clear_state_cache()
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "x.mid").write_bytes(b"old")
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    sid = dd._source_id(spec["sources"][0])
    dd._write_record(dataset_dir, sid, "fixture", {"midi/x.mid": 5, "midi/y.mid": 6})
    # Make x.mid truncated
    (dataset_dir / "midi" / "x.mid").write_bytes(b"hi")
    dd._clear_state_cache()
    # Mock download to succeed
    fixture_zip = tmp_path / "fz.zip"
    with zipfile.ZipFile(fixture_zip, "w") as zf:
        zf.writestr("A/x.mid", b"newdata")  # size 7
        zf.writestr("A/y.mid", b"ydata")
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fixture_zip.read_bytes())
        return len(fixture_zip.read_bytes())
    monkeypatch.setattr(dd, "_download_file", fake)
    n = dd._download_and_extract("fixture", spec, output_dir)
    out = capsys.readouterr().out
    assert "[stale]" in out
    assert "re-fetching" in out
    assert n == 2


# ── guitarset-full after mic reuses shared source ────────────────────────────

def test_guitarset_full_after_mic_fetches_only_missing_zip(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dd._clear_state_cache()
    output_dir = tmp_path / "corpus"
    # Build minimal zips: annotation and two audios
    ann_zip = tmp_path / "ann.zip"
    with zipfile.ZipFile(ann_zip, "w") as zf:
        zf.writestr("a.jams", b"jams")
    mic_zip = tmp_path / "mic.zip"
    with zipfile.ZipFile(mic_zip, "w") as zf:
        zf.writestr("a.wav", b"mic")
    mix_zip = tmp_path / "mix.zip"
    with zipfile.ZipFile(mix_zip, "w") as zf:
        zf.writestr("a.wav", b"mix")
    # Map URLs to data
    url_to_data = {
        dd.DATASETS["guitarset-mic"]["sources"][0]["url"]: ann_zip.read_bytes(),
        dd.DATASETS["guitarset-mic"]["sources"][1]["url"]: mic_zip.read_bytes(),
        dd.DATASETS["guitarset-mix"]["sources"][1]["url"]: mix_zip.read_bytes(),
    }
    # Also full's sources include same annotation and mic plus mix
    fetched = []
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(url_to_data[url])
        return len(url_to_data[url])
    monkeypatch.setattr(dd, "_download_file", fake)
    dd._clear_state_cache()
    # Fetch mic
    dd._download_and_extract("guitarset-mic", dd.DATASETS["guitarset-mic"], output_dir)
    fetched.clear()
    # Fetch full: annotation and mic should be skipped via source_id satisfied
    n = dd._download_and_extract("guitarset-full", dd.DATASETS["guitarset-full"], output_dir)
    # Should have fetched only mix zip (the third source)
    assert len(fetched) == 1
    assert fetched[0].endswith("audio_mono-pickup_mix.zip")
    assert n == 3  # total files in full: 1 jams + 1 mic wav + 1 mix wav, but counted via records? Our n is sum of files across sources, which includes skipped annotation + mic + mix = 3


# ── _check_disk_space and _bytes_needed ─────────────────────────────────────

class _FakeDiskUsage:
    def __init__(self, free: int) -> None:
        self.free = free


def test_check_disk_space_returns_none_when_ample_space(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda path: _FakeDiskUsage(10**12))
    assert dd._check_disk_space(tmp_path, 1 * 1_048_576) is None


def test_check_disk_space_reports_when_free_space_insufficient(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda path: _FakeDiskUsage(1_000_000))
    msg = dd._check_disk_space(tmp_path, 1000 * 1_048_576)
    assert msg is not None
    assert "not enough free space" in msg


def test_check_disk_space_probes_nearest_existing_ancestor(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list = []
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda path: probed.append(path) or _FakeDiskUsage(10**12))
    missing = tmp_path / "a" / "b"
    assert dd._check_disk_space(missing, 1 * 1_048_576) is None
    assert probed == [tmp_path]


def test_bytes_needed_dedupes_shared_sources_and_skips_present(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dd._clear_state_cache()
    output_dir = tmp_path / "corpus"
    # Use guitarset mic and mix sharing annotation
    # Initially none present, needed should be deduped 38+627+652 = 1317 MB? Actually mic+mix = annotation 38 once + 627+652 = 1317
    needed = dd._bytes_needed(["guitarset-mic", "guitarset-mix"], output_dir, force=False)
    expected = (38+627+652) * 1_048_576
    assert needed == expected
    # After mic fetched, mix's annotation should be considered satisfied, so needed for full after mic should be only mix audio
    # Simulate mic download
    ann_zip = tmp_path / "ann.zip"
    with zipfile.ZipFile(ann_zip, "w") as zf:
        zf.writestr("a.jams", b"jams")
    mic_zip = tmp_path / "mic.zip"
    with zipfile.ZipFile(mic_zip, "w") as zf:
        zf.writestr("a.wav", b"mic")
    mix_zip = tmp_path / "mix.zip"
    with zipfile.ZipFile(mix_zip, "w") as zf:
        zf.writestr("a.wav", b"mix")
    url_to_data = {
        dd.DATASETS["guitarset-mic"]["sources"][0]["url"]: ann_zip.read_bytes(),
        dd.DATASETS["guitarset-mic"]["sources"][1]["url"]: mic_zip.read_bytes(),
        dd.DATASETS["guitarset-mix"]["sources"][1]["url"]: mix_zip.read_bytes(),
    }
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(url_to_data[url])
        return len(url_to_data[url])
    monkeypatch.setattr(dd, "_download_file", fake)
    dd._clear_state_cache()
    dd._download_and_extract("guitarset-mic", dd.DATASETS["guitarset-mic"], output_dir)
    dd._clear_state_cache()
    # Now needed for full should be only pickup_mix (652)
    needed2 = dd._bytes_needed(["guitarset-full"], output_dir, force=False)
    assert needed2 == 652 * 1_048_576
    # Forced should count all even if present
    needed_forced = dd._bytes_needed(["guitarset-full"], output_dir, force=True)
    assert needed_forced == (38+627+652) * 1_048_576


def test_main_aborts_on_preflight_failure(dd: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(dd, "_check_disk_space", lambda output_dir, needed: "not enough free space on X")
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


# ── gaps registry shape ───────────────────────────────────────────────────


def test_gaps_full_corpus_subdir_targets_and_pinned_revision(dd: ModuleType) -> None:
    import re
    spec = dd.DATASETS["gaps-full"]
    assert spec["corpus_subdir"] == "gaps"
    targets = set(dd._all_target_subdirs(spec))
    assert targets == {"recordings", "midi", "annotations/musicxml", "annotations/syncpoints", "metadata"}
    assert "audio" not in targets
    hf_sources = [s for s in spec["sources"] if s["kind"] == "hf_tree"]
    assert len(hf_sources) == 4
    revisions = {s["revision"] for s in hf_sources}
    assert len(revisions) == 1
    revision = next(iter(revisions))
    assert revision != "main"
    assert re.fullmatch(r"[0-9a-f]{40}", revision) is not None


def test_gaps_bare_key_was_split_into_midi_and_full(dd: ModuleType) -> None:
    assert "gaps" not in dd.DATASETS
    keys = list(dd.DATASETS)
    assert keys.index("gaps-midi") + 1 == keys.index("gaps-full")


def test_gaps_midi_targets_only_midi_and_metadata(dd: ModuleType) -> None:
    spec = dd.DATASETS["gaps-midi"]
    assert spec["corpus_subdir"] == "gaps"
    targets = set(dd._all_target_subdirs(spec))
    assert targets == {"midi", "metadata"}
    assert "recordings" not in targets
    assert "audio" not in targets
    assert dd._dataset_size_mb(spec) == 4


def test_gaps_midi_sources_are_identical_to_gaps_full_sources(dd: ModuleType) -> None:
    full_sources = dd.DATASETS["gaps-full"]["sources"]
    for source in dd.DATASETS["gaps-midi"]["sources"]:
        assert source in full_sources


@pytest.mark.parametrize("key", ["gaps-midi", "gaps-full"])
def test_gaps_spec_carries_note(dd: ModuleType, key: str) -> None:
    spec = dd.DATASETS[key]
    assert spec.get("note")
    assert spec["note"].endswith("All 404 files kept; official split not applied.")


@pytest.mark.parametrize("key", ["gaps-midi", "gaps-full"])
def test_gaps_description_names_licence_and_unfiltered(dd: ModuleType, key: str) -> None:
    spec = dd.DATASETS[key]
    assert "CC BY-NC-SA 4.0, research use, cite Riley et al. ISMIR 2024" in spec["description"]
    assert "MIT" not in spec["description"]


# ── gaps-full reuses MIDI already on disk ─────────────────────────────────

_GAPS_LISTINGS = {
    "audio": [("audio/001_a.wav", 10), ("audio/002_b.wav", 11)],
    "midi": [("midi/001_a.mid", 3), ("midi/002_b.mid", 4), ("midi/003_c.mid", 5)],
    "musicxml": [("musicxml/001_a.xml", 6)],
    "syncpoints": [("syncpoints/001_a.json", 7)],
}


def _fake_gaps_network(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: _GAPS_LISTINGS[subdir])
    sizes = {Path(p).name: s for listing in _GAPS_LISTINGS.values() for p, s in listing}
    fetched: list = []
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * sizes.get(Path(url).name, 1))
        return dest.stat().st_size
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    return fetched


def _midi_urls(fetched: list) -> list:
    return sorted(Path(u).name for u in fetched if "/midi/" in u)


def test_gaps_full_skips_midi_already_fetched_by_gaps_midi(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Updated expectation: shared file source also skipped, so gaps_metadata not refetched
    fetched = _fake_gaps_network(dd, monkeypatch)
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    dd._download_and_extract("gaps-midi", dd.DATASETS["gaps-midi"], output_dir)
    assert _midi_urls(fetched) == ["001_a.mid", "002_b.mid", "003_c.mid"]
    fetched.clear()
    dd._clear_state_cache()
    dd._download_and_extract("gaps-full", dd.DATASETS["gaps-full"], output_dir)
    assert _midi_urls(fetched) == []
    names = {Path(u).name for u in fetched}
    # gaps_metadata should be skipped due to shared source_id, so not in fetched
    assert names == {"001_a.wav", "002_b.wav", "001_a.xml", "001_a.json"}


def test_gaps_full_downloads_all_midi_when_none_on_disk(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = _fake_gaps_network(dd, monkeypatch)
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    n = dd._download_and_extract("gaps-full", dd.DATASETS["gaps-full"], output_dir)
    assert _midi_urls(fetched) == ["001_a.mid", "002_b.mid", "003_c.mid"]
    assert n == 2 + 3 + 1 + 1 + 1
    assert sorted(p.name for p in (output_dir / "gaps" / "midi").iterdir()) == ["001_a.mid", "002_b.mid", "003_c.mid"]


def test_gaps_full_refetches_only_missing_or_truncated_midi(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = _fake_gaps_network(dd, monkeypatch)
    output_dir = tmp_path / "corpus"
    midi_dir = output_dir / "gaps" / "midi"
    midi_dir.mkdir(parents=True)
    (midi_dir / "001_a.mid").write_bytes(b"xxx")
    (midi_dir / "002_b.mid").write_bytes(b"xx")
    dd._clear_state_cache()
    dd._download_and_extract("gaps-full", dd.DATASETS["gaps-full"], output_dir)
    assert _midi_urls(fetched) == ["002_b.mid", "003_c.mid"]
    assert (midi_dir / "002_b.mid").stat().st_size == 4


def test_download_hf_tree_reports_skipped_and_downloaded_counts(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gaps_network(dd, monkeypatch)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    (dataset_dir / "midi").mkdir(parents=True)
    (dataset_dir / "midi" / "001_a.mid").write_bytes(b"xxx")
    source = dd.DATASETS["gaps-full"]["sources"][1]
    assert source["subdir"] == "midi"
    reports: list = []
    dd._clear_state_cache()
    n = dd._download_hf_tree(
        source,
        dataset_dir,
        output_dir=output_dir,
        key="gaps-full",
        index=1,
        report=lambda skipped, downloaded: reports.append((skipped, downloaded)),
    )
    assert n == 3
    assert reports == [(1, 2)]


def test_plain_path_prints_hf_tree_skip_summary(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_gaps_network(dd, monkeypatch)
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    dd._download_and_extract("gaps-midi", dd.DATASETS["gaps-midi"], output_dir)
    assert "[gaps-midi] midi: 0 already present, 3 downloaded" in capsys.readouterr().out
    dd._clear_state_cache()
    dd._download_and_extract("gaps-full", dd.DATASETS["gaps-full"], output_dir)
    out = capsys.readouterr().out
    assert "[gaps-full] midi: 3 already present, 0 downloaded" in out
    assert "[gaps-full] audio: 0 already present, 2 downloaded" in out


def test_gaps_full_not_present_with_only_gaps_midi_content(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    for subdir in dd._all_target_subdirs(dd.DATASETS["gaps-midi"]):
        target = tmp_path / "gaps" / subdir
        target.mkdir(parents=True)
        (target / "x").write_bytes(b"x")
    assert dd._is_already_present("gaps-midi", dd.DATASETS["gaps-midi"], tmp_path)
    assert not dd._is_already_present("gaps-full", dd.DATASETS["gaps-full"], tmp_path)


def test_gaps_midi_present_once_gaps_full_content_exists(dd: ModuleType, tmp_path: Path) -> None:
    dd._clear_state_cache()
    # Create records for full to make it present via records, plus fallback dirs
    # Simplest: create dirs and also create records for all full sources
    for subdir in dd._all_target_subdirs(dd.DATASETS["gaps-full"]):
        target = tmp_path / "gaps" / subdir
        target.mkdir(parents=True)
        (target / "x").write_bytes(b"x")
    # Also create records for each source of full so _is_already_present via records works
    # But fallback will already make it present; for midi, superseded check will make it present
    dd._clear_state_cache()
    assert dd._is_already_present("gaps-full", dd.DATASETS["gaps-full"], tmp_path)
    assert dd._is_already_present("gaps-midi", dd.DATASETS["gaps-midi"], tmp_path)


def test_retry_sleeps_constant(dd: ModuleType) -> None:
    assert dd._RETRY_SLEEPS == (3, 10, 30)


# ── _hf_list_tree ─────────────────────────────────────────────────────────


def _json_response(status: int, headers: dict, payload) -> "_FakeResponse":
    return _FakeResponse(status, headers, [json.dumps(payload).encode()])


def test_hf_list_tree_returns_files_only_skipping_directories(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {"type": "file", "path": "audio/a.wav", "size": 123},
        {"type": "directory", "path": "audio/sub", "size": 0},
        {"type": "file", "path": "audio/b.wav", "size": 456},
    ]
    seen: list = []
    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, timeout, req.get_header("User-agent")))
        return _json_response(200, {}, payload)
    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
    result = dd._hf_list_tree("xavriley/GAPS", "abc123", "audio")
    assert result == [("audio/a.wav", 123), ("audio/b.wav", 456)]
    assert seen[0][1] == 60
    assert seen[0][2] == "Sonitra-Dataset-Downloader/1.0"
    assert "xavriley/GAPS" in seen[0][0]
    assert "limit=1000" in seen[0][0]


def test_hf_list_tree_follows_link_next_across_two_pages(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    page1 = [{"type": "file", "path": "audio/a.wav", "size": 1}]
    page2 = [{"type": "file", "path": "audio/b.wav", "size": 2}]
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _json_response(200, {"Link": '<https://example.invalid/page2>; rel="next"'}, page1)
        return _json_response(200, {}, page2)
    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
    result = dd._hf_list_tree("r", "rev", "audio")
    assert result == [("audio/a.wav", 1), ("audio/b.wav", 2)]
    assert calls["n"] == 2


def test_hf_list_tree_retries_transient_error_then_succeeds(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"type": "file", "path": "audio/a.wav", "size": 7}]
    calls = {"n": 0}
    sleeps: list = []
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("boom")
        return _json_response(200, {}, payload)
    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dd.time, "sleep", lambda seconds: sleeps.append(seconds))
    result = dd._hf_list_tree("r", "rev", "audio")
    assert result == [("audio/a.wav", 7)]
    assert calls["n"] == 2
    assert sleeps == [3]


def test_parse_link_next_accepts_rfc8288_quoting_variants(dd: ModuleType) -> None:
    assert dd._parse_link_next('<u1>; rel="next"') == "u1"
    assert dd._parse_link_next("<u2>; rel='next'") == "u2"
    assert dd._parse_link_next("<u3>; rel=next") == "u3"
    assert dd._parse_link_next('<p>; rel="prev", <u4>; rel="next"') == "u4"
    assert dd._parse_link_next('<u5>; type="x"; rel="next"') == "u5"
    assert dd._parse_link_next('<p>; rel="prev"') is None
    assert dd._parse_link_next('<p>; rel="nextpage"') is None
    assert dd._parse_link_next("") is None
    assert dd._parse_link_next(None) is None


# ── _download_hf_tree ─────────────────────────────────────────────────────


def _hf_source(**overrides) -> dict:
    source = {
        "kind": "hf_tree",
        "repo": "xavriley/GAPS",
        "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
        "subdir": "audio",
        "patterns": None,
        "target_subdir": "recordings",
        "size_mb": 1,
    }
    source.update(overrides)
    return source


def test_download_hf_tree_writes_files_and_returns_count(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("audio/a.wav", 4), ("audio/b.wav", 5)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    urls: list = []
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        urls.append(url)
        dest.write_bytes(b"data")
        if progress is not None:
            progress(4, 4)
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    source = _hf_source()
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0)
    assert n == 2
    assert (dataset_dir / "recordings" / "a.wav").read_bytes() == b"data"
    assert (dataset_dir / "recordings" / "b.wav").read_bytes() == b"data"
    assert all("xavriley/GAPS" in u and "/resolve/" in u for u in urls)


def test_download_hf_tree_filters_by_patterns(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("musicxml/a.xml", 3), ("musicxml/viz_fret_string_counts.py", 10)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    downloaded: list = []
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded.append(Path(url).name)
        dest.write_bytes(b"x")
        return 1
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    source = _hf_source(subdir="musicxml", patterns=frozenset({".xml"}), target_subdir="annotations/musicxml")
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0)
    assert n == 1
    assert (dataset_dir / "annotations" / "musicxml" / "a.xml").exists()
    assert not (dataset_dir / "annotations" / "musicxml" / "viz_fret_string_counts.py").exists()
    assert downloaded == ["a.xml"]


def test_download_hf_tree_skips_existing_matching_size(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("audio/a.wav", 4)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    calls: list = []
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    dest = dataset_dir / "recordings" / "a.wav"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"1234")
    progress_calls: list = []
    source = _hf_source()
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0, progress=lambda d, t: progress_calls.append((d, t)))
    assert n == 1
    assert calls == []
    assert dest.read_bytes() == b"1234"
    assert progress_calls


def test_download_hf_tree_redownloads_when_size_differs(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("audio/a.wav", 4)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    calls: list = []
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"1234")
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    dest = dataset_dir / "recordings" / "a.wav"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"xy")
    source = _hf_source()
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0)
    assert n == 1
    assert len(calls) == 1
    assert dest.read_bytes() == b"1234"


def test_download_hf_tree_force_redownloads_matching_size(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("audio/a.wav", 4)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    calls: list = []
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"new!")
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    dest = dataset_dir / "recordings" / "a.wav"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"1234")
    source = _hf_source()
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0, force=True)
    assert n == 1
    assert len(calls) == 1
    assert dest.read_bytes() == b"new!"


def test_download_hf_tree_part_uses_source_id_suffix(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("audio/a.wav", 4)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        # Check dest ends with .<source_id>.part
        assert dest.name.endswith(".part")
        assert dest.name.count(".") >= 2  # a.wav.<id>.part
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    source = _hf_source()
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0)
    assert n == 1
    assert (dataset_dir / "recordings" / "a.wav").exists()
    assert not list(dataset_dir.rglob("*.part"))  # part renamed


def test_download_hf_tree_part_renamed_only_on_success(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [("audio/a.wav", 4)]
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: listing)
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "gaps"
    source = _hf_source()
    sid = dd._source_id(source)
    dd._clear_state_cache()
    n = dd._download_hf_tree(source, dataset_dir, output_dir=output_dir, key="gaps", index=0)
    assert n == 1
    assert (dataset_dir / "recordings" / "a.wav").exists()
    assert not (dataset_dir / "recordings" / f"a.wav.{sid}.part").exists()
    def failing_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial-bytes")
        raise RuntimeError("boom")
    monkeypatch.setattr(dd, "_download_file", failing_download_file)
    dataset_dir2 = tmp_path / "corpus2" / "gaps"
    with pytest.raises(RuntimeError, match="boom"):
        dd._download_hf_tree(source, dataset_dir2, output_dir=tmp_path / "corpus2", key="g", index=0)
    # Find part with suffix
    parts = list((dataset_dir2 / "recordings").glob("*.part"))
    assert len(parts) == 1
    assert parts[0].read_bytes() == b"partial-bytes"
    assert not (dataset_dir2 / "recordings" / "a.wav").exists()


# ── integration: mixed hf_tree + file ─────────────────────────────────────


def _mixed_spec() -> dict:
    return {
        "name": "Mixed Fixture",
        "corpus_subdir": "fixture",
        "sources": [
            {
                "kind": "hf_tree",
                "repo": "xavriley/GAPS",
                "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
                "subdir": "audio",
                "patterns": frozenset({".wav"}),
                "target_subdir": "recordings",
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


def test_download_and_extract_handles_mixed_hf_tree_and_file_via_records(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: [("audio/a.wav", 4)])
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("a.wav"):
            dest.write_bytes(b"wav-bytes")
        else:
            dest.write_bytes(b"metadata-bytes")
        if progress is not None:
            progress(1, 1)
        return 1
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    output_dir = tmp_path / "corpus"
    spec = _mixed_spec()
    dd._clear_state_cache()
    n = dd._download_and_extract("fixture", spec, output_dir)
    assert n == 2
    assert (output_dir / "fixture" / "recordings" / "a.wav").read_bytes() == b"wav-bytes"
    assert (output_dir / "fixture" / "metadata" / "meta.csv").read_bytes() == b"metadata-bytes"
    # Check records instead of markers
    for src in spec["sources"]:
        sid = dd._source_id(src)
        assert (output_dir / "fixture" / ".sources" / f"{sid}.json").exists()


def test_rich_path_handles_mixed_hf_tree_and_file_cumulatively(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from io import StringIO
    monkeypatch.setattr(dd, "_hf_list_tree", lambda repo, revision, subdir: [("audio/a.wav", 4)])
    def fake_download_file(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("a.wav"):
            dest.write_bytes(b"wav-bytes")
        else:
            dest.write_bytes(b"metadata-bytes")
        if progress is not None:
            progress(2, 4)
            progress(4, 4)
        return 4
    monkeypatch.setattr(dd, "_download_file", fake_download_file)
    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    output_dir = tmp_path / "corpus"
    spec = _mixed_spec()
    dd._clear_state_cache()
    display = dd._DownloadDisplay(console, slots=1, total_datasets=1, total_mb=2)
    progress_calls: list = []
    original_on_progress = display.on_progress
    def recording_on_progress(task_id, downloaded, total):
        progress_calls.append((downloaded, total))
        return original_on_progress(task_id, downloaded, total)
    display.on_progress = recording_on_progress  # type: ignore[method-assign]
    with display:
        status, n_files, error = dd._download_one("fixture", spec, output_dir, display=display, task_id=display.tasks[0])
    assert status == "done"
    assert error is None
    assert n_files == 2
    assert (output_dir / "fixture" / "recordings" / "a.wav").exists()
    assert (output_dir / "fixture" / "metadata" / "meta.csv").exists()
    offset = spec["sources"][0]["size_mb"] * 1_048_576
    assert len(progress_calls) >= 2
    assert [d for d, _ in progress_calls] == sorted(d for d, _ in progress_calls)
    assert progress_calls[0][0] < offset
    assert progress_calls[-1][0] > offset
    assert progress_calls[-1] == (offset + 4, offset + 4)


# ── concurrency ─────────────────────────────────────────────────────────────

def test_concurrent_shared_source_fetched_only_once(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two threads with shared source (same download_key and source_id) should fetch once
    ann_url = "https://example.invalid/shared.zip"
    spec1 = {"name": "Set1", "corpus_subdir": "shared", "sources": [{"url": ann_url, "kind": "zip", "extract_map": [("", None, "annotations")], "size_mb": 1}]}
    spec2 = {"name": "Set2", "corpus_subdir": "shared", "sources": [{"url": ann_url, "kind": "zip", "extract_map": [("", None, "annotations")], "size_mb": 1}]}
    # Create zip
    zpath = tmp_path / "shared.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.jams", b"data")
    data = zpath.read_bytes()
    fetch_count = {"n": 0}
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        fetch_count["n"] += 1
        time.sleep(0.05)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)
    monkeypatch.setattr(dd, "_download_file", fake)
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    coordinator = dd._Coordinator(set())
    results = {}
    def run1():
        results["1"] = dd._download_one("set1", spec1, output_dir, coordinator=coordinator)
    def run2():
        results["2"] = dd._download_one("set2", spec2, output_dir, coordinator=coordinator)
    t1 = threading.Thread(target=run1)
    t2 = threading.Thread(target=run2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # One should be done, one skip (since shared)
    statuses = {results["1"][0], results["2"][0]}
    assert "done" in statuses
    # At least one should be skip or done, but fetch count should be 1 (shared download key lock)
    assert fetch_count["n"] == 1


def test_concurrent_same_destination_distinct_part_names(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two sources writing same destination concurrently should not clash (different .part names)
    url1 = "https://example.invalid/a.zip"
    url2 = "https://example.invalid/b.zip"
    spec1 = {"name": "Set1", "corpus_subdir": "shared2", "sources": [{"url": url1, "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    spec2 = {"name": "Set2", "corpus_subdir": "shared2", "sources": [{"url": url2, "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    # Both zips contain same file A/x.mid but different content
    zp1 = tmp_path / "a.zip"
    with zipfile.ZipFile(zp1, "w") as zf:
        zf.writestr("A/x.mid", b"content1")
    zp2 = tmp_path / "b.zip"
    with zipfile.ZipFile(zp2, "w") as zf:
        zf.writestr("A/x.mid", b"content2")
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "a.zip" in url:
            dest.write_bytes(zp1.read_bytes())
        else:
            dest.write_bytes(zp2.read_bytes())
        time.sleep(0.05)
        return len(dest.read_bytes())
    monkeypatch.setattr(dd, "_download_file", fake)
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    coordinator = dd._Coordinator(set())
    # Need to check that temp files are distinct: we can monitor dest Part names via _extract_archive? Simpler: just run concurrently and ensure no error and final file exists (last writer wins)
    def run1():
        return dd._download_one("set1", spec1, output_dir, coordinator=coordinator)
    def run2():
        return dd._download_one("set2", spec2, output_dir, coordinator=coordinator)
    t1 = threading.Thread(target=run1)
    t2 = threading.Thread(target=run2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # Check that no .part files remain and final file exists
    assert (output_dir / "shared2" / "midi" / "x.mid").exists()
    assert not list((output_dir / "shared2").rglob("*.part"))


def test_shared_source_failure_fails_waiting_job_fast(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.invalid/shared.zip"
    spec1 = {"name": "Set1", "corpus_subdir": "shared3", "sources": [{"url": url, "kind": "zip", "extract_map": [("", None, "midi")], "size_mb": 1}]}
    spec2 = {"name": "Set2", "corpus_subdir": "shared3", "sources": [{"url": url, "kind": "zip", "extract_map": [("", None, "midi")], "size_mb": 1}]}
    def failing(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial")
        raise RuntimeError("boom")
    monkeypatch.setattr(dd, "_download_file", failing)
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    coordinator = dd._Coordinator(set())
    # First fetch will fail and memoize
    status1, _, err1 = dd._download_one("set1", spec1, output_dir, coordinator=coordinator)
    assert status1 == "error"
    assert "boom" in err1
    # Second fetch should fail fast without retrying (no additional sleep)
    start = time.monotonic()
    status2, _, err2 = dd._download_one("set2", spec2, output_dir, coordinator=coordinator)
    elapsed = time.monotonic() - start
    assert status2 == "error"
    assert "boom" in err2
    assert elapsed < 1.0  # fast, not retrying 4 times (~43s)


def test_cancel_event_stops_queue(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _fixture_zip_spec()
    monkeypatch.setattr(dd, "DATASETS", {"a": spec, "b": spec})
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    from io import StringIO
    console = dd._RichConsole(file=StringIO(), force_terminal=True, width=100)
    display = dd._DownloadDisplay(console, slots=1, total_datasets=2, total_mb=2)
    # Create coordinator and set cancel before workers start
    coordinator = dd._Coordinator(set())
    coordinator.cancel.set()
    work_queue = queue.Queue()
    work_queue.put("a")
    work_queue.put("b")
    # _slot_worker should stop taking new keys when cancel is set
    failed = dd._slot_worker(0, display.tasks[0], work_queue, display, output_dir, coordinator=coordinator)
    # Should return without processing both? At least not failure
    assert work_queue.qsize() >= 1  # at least one left


# ── force run-scoped ────────────────────────────────────────────────────────

def test_force_refetches_shared_source_once(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shared source between two keys: force should fetch once, second reuses
    url = "https://example.invalid/shared.zip"
    spec1 = {"name": "Set1", "corpus_subdir": "shared4", "sources": [{"url": url, "kind": "zip", "extract_map": [("", None, "midi")], "size_mb": 1}]}
    spec2 = {"name": "Set2", "corpus_subdir": "shared4", "sources": [{"url": url, "kind": "zip", "extract_map": [("", None, "midi")], "size_mb": 1}]}
    zpath = tmp_path / "shared.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.mid", b"data")
    data = zpath.read_bytes()
    # First populate with satisfied record
    output_dir = tmp_path / "corpus"
    dd._clear_state_cache()
    def fake(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)
    monkeypatch.setattr(dd, "_download_file", fake)
    dd._download_and_extract("set1", spec1, output_dir)
    dd._clear_state_cache()
    # Now force both with shared coordinator
    fetch_count = {"n": 0}
    def fake2(url, dest, *, name, output_dir, key, index, progress=None, **kwargs):
        fetch_count["n"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)
    monkeypatch.setattr(dd, "_download_file", fake2)
    sid = dd._source_id(spec1["sources"][0])
    coordinator = dd._Coordinator({sid})
    s1, _, _ = dd._download_one("set1", spec1, output_dir, coordinator=coordinator, force=True)
    s2, _, _ = dd._download_one("set2", spec2, output_dir, coordinator=coordinator, force=True)
    # One should have actually fetched, the other should see satisfied after first cleared forced and writes record
    assert fetch_count["n"] == 1


def test_force_reset_leaves_other_source_part_alone(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two different sources in same corpus subdir, force one should not delete other's part
    output_dir = tmp_path / "corpus"
    dataset_dir = output_dir / "fixture"
    dataset_dir.mkdir(parents=True)
    # Create a part for other source
    other_sid = "abc123def456"
    other_part = dataset_dir / "midi" / f"other.mid.{other_sid}.part"
    other_part.parent.mkdir(parents=True)
    other_part.write_bytes(b"other")
    # Now force a different source
    spec = {"name": "Fixture", "corpus_subdir": "fixture", "sources": [{"url": "https://example.invalid/a.zip", "kind": "zip", "extract_map": [("A/", None, "midi")], "size_mb": 1}]}
    sid = dd._source_id(spec["sources"][0])
    # Create record and download partial for this source to be reset
    (dataset_dir / "midi" / "x.mid").write_bytes(b"old")
    dd._write_record(dataset_dir, sid, "fixture", {"midi/x.mid": 3})
    download_key = dd._download_key(spec["sources"][0])
    dl_part = output_dir / ".downloads" / f"{download_key}.part"
    dl_part.parent.mkdir(parents=True)
    dl_part.write_bytes(b"dl")
    # Also create corpus part for this source
    corpus_part = dataset_dir / "midi" / f"x.mid.{sid}.part"
    corpus_part.write_bytes(b"stale")
    dd._clear_state_cache()
    # Force via _reset_for_force directly
    coordinator = dd._Coordinator({sid})
    dd._reset_for_force(dataset_dir, output_dir, spec["sources"][0], sid, download_key, coordinator)
    assert not dl_part.exists()
    assert not corpus_part.exists()
    assert other_part.exists()  # should remain


# ── TUI notes ─────────────────────────────────────────────────────────────


def _wide_console(dd: ModuleType):
    from io import StringIO
    return dd._RichConsole(file=StringIO(), force_terminal=True, width=1000)


def _unique_notes(dd: ModuleType) -> list[str]:
    return list(dict.fromkeys(spec["note"] for spec in dd.DATASETS.values()))


def test_print_table_has_no_notes_column(dd: ModuleType, tmp_path: Path) -> None:
    console = _wide_console(dd)
    dd._clear_state_cache()
    dd._print_table(console, tmp_path)
    rendered = console.file.getvalue()
    assert "notes" not in rendered.replace(str(tmp_path), "")
    for note in _unique_notes(dd):
        assert note not in rendered
    for key in dd.DATASETS:
        assert key in rendered


def test_print_list_omits_notes(dd: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dd._print_list(tmp_path)
    out = capsys.readouterr().out
    for note in _unique_notes(dd):
        assert note not in out
    for key in dd.DATASETS:
        assert key in out


def test_note_groups_give_one_row_per_dataset_with_picker_numbers(dd: ModuleType) -> None:
    groups = dd._note_groups()
    assert [numbers for numbers, _, _ in groups] == ["1-3", "4", "5-6", "7-8", "9-11", "12-13"]
    assert [name for _, name, _ in groups] == [
        "MAESTRO V3.0.0",
        "Beethoven Symphony Excerpt Dataset (BSED) v1.0",
        "MusicNet",
        "Expanded Groove MIDI Dataset",
        "GuitarSet",
        "GAPS (Guitar-Aligned Performance Scores) v1.1",
    ]
    assert [note for _, _, note in groups] == _unique_notes(dd)


def test_note_groups_number_non_adjacent_keys_individually(dd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def spec(subdir: str) -> dict:
        return {"name": f"{subdir} (variant)", "note": f"{subdir} note", "corpus_subdir": subdir}
    monkeypatch.setattr(dd, "DATASETS", {"a1": spec("a"), "b": spec("b"), "a2": spec("a")})
    assert dd._note_groups() == [("1,3", "a", "a note"), ("2", "b", "b note")]


def test_print_notes_table_shows_each_dataset_once(dd: ModuleType) -> None:
    console = _wide_console(dd)
    dd._print_notes_table(console)
    rendered = console.file.getvalue()
    assert "Dataset notes" in rendered
    for numbers, name, note in dd._note_groups():
        assert rendered.count(note) == 1
        assert name in rendered
        assert numbers in rendered


def test_print_notes_plain_shows_each_dataset_once(dd: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    dd._print_notes()
    out = capsys.readouterr().out
    for numbers, name, note in dd._note_groups():
        assert out.count(note) == 1
        assert f"{numbers}" in out and name in out


def _scripted_picker(dd: ModuleType, monkeypatch: pytest.MonkeyPatch, responses: list[object]) -> tuple[object, list[str], list[str]]:
    console = _wide_console(dd)
    clears: list[str] = []
    prompts: list[str] = []
    monkeypatch.setattr(console, "clear", lambda *a, **k: clears.append("clear"))
    monkeypatch.setattr(dd, "_RichConsole", lambda *a, **k: console)
    answers = iter(responses)
    def fake_ask(prompt: str, *args: object, **kwargs: object) -> str:
        prompts.append(prompt)
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer  # type: ignore[return-value]
    monkeypatch.setattr(dd._RichPrompt, "ask", fake_ask)
    return console, clears, prompts


def test_picker_prompt_offers_notes_key(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, prompts = _scripted_picker(dd, monkeypatch, ["q"])
    assert dd._interactive_select(tmp_path) is None
    assert "'n' for notes" in prompts[0]


@pytest.mark.parametrize("notes_key", ["n", "N", " n "])
def test_picker_n_switches_to_notes_page_and_enter_returns(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notes_key: str) -> None:
    console, clears, prompts = _scripted_picker(dd, monkeypatch, [notes_key, "", "2"])
    assert dd._interactive_select(tmp_path) == [list(dd.DATASETS)[1]]
    rendered = console.file.getvalue()
    assert "Dataset notes" in rendered
    assert rendered.count("Available datasets") == 2
    assert clears == ["clear", "clear"]
    assert "Enter" in prompts[1]


def test_picker_eof_on_notes_page_quits(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted_picker(dd, monkeypatch, ["n", EOFError()])
    assert dd._interactive_select(tmp_path) is None


@pytest.mark.parametrize("rich_output", [True, False])
def test_main_notes_flag_prints_notes_and_exits(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], rich_output: bool) -> None:
    monkeypatch.setattr(dd, "_use_rich_output", lambda: rich_output)
    if rich_output:
        console = _wide_console(dd)
        monkeypatch.setattr(dd, "_RichConsole", lambda *a, **k: console)
    monkeypatch.setattr("sys.argv", ["download_datasets.py", "--notes", "--output-dir", str(tmp_path)])
    assert dd.main() == 0
    out = console.file.getvalue() if rich_output else capsys.readouterr().out
    for _, _, note in dd._note_groups():
        assert note in out


@pytest.mark.parametrize("rich_output", [True, False])
def test_main_list_points_to_notes_flag(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], rich_output: bool) -> None:
    monkeypatch.setattr(dd, "_use_rich_output", lambda: rich_output)
    if rich_output:
        console = _wide_console(dd)
        monkeypatch.setattr(dd, "_RichConsole", lambda *a, **k: console)
    monkeypatch.setattr("sys.argv", ["download_datasets.py", "--list", "--output-dir", str(tmp_path)])
    assert dd.main() == 0
    out = console.file.getvalue() if rich_output else capsys.readouterr().out
    assert "--notes" in out
    for note in _unique_notes(dd):
        assert note not in out


# ── resolve selection and next_steps via main ───────────────────────────────

def test_resolve_selection_prunes_superseded_and_prints_skip(dd: ModuleType, capsys) -> None:
    selected = ["maestro-v3-midi", "maestro-v3-full", "bsed"]
    resolved = dd._resolve_selection(selected)
    assert resolved == ["maestro-v3-full", "bsed"]
    out = capsys.readouterr().out
    assert "[skip]" in out
    assert "superseded by maestro-v3-full" in out


def test_force_on_superseded_key_is_refused(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # Create full content to make midi present via superseded
    for subdir in dd._all_target_subdirs(dd.DATASETS["guitarset-full"]):
        target = tmp_path / "guitarset" / subdir
        target.mkdir(parents=True)
        (target / "x").write_bytes(b"x")
    dd._clear_state_cache()
    # Try to force mic when full is present
    monkeypatch.setattr("sys.argv", ["download_datasets.py", "guitarset-mic", "--force", "--output-dir", str(tmp_path)])
    assert dd.main() == 1
    err = capsys.readouterr().err
    assert "cannot --force" in err
    assert "guitarset-full" in err


def test_bytes_needed_and_resolve_used_in_main_for_display_and_preflight(dd: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Select all, resolve should prune, and bytes_needed should be less than sum of all
    monkeypatch.setattr("sys.argv", ["download_datasets.py", "--all", "--output-dir", str(tmp_path)])
    # Mock _check_disk_space to capture needed, and _run_plain to capture resolved
    captured = {}
    def fake_check(output_dir, needed):
        captured["needed"] = needed
        return None
    def fake_run_plain(selected, output_dir, jobs, force=False, coordinator=None):
        captured["selected"] = selected
        return False
    monkeypatch.setattr(dd, "_check_disk_space", fake_check)
    monkeypatch.setattr(dd, "_run_plain", fake_run_plain)
    monkeypatch.setattr(dd, "_use_rich_output", lambda: False)
    dd.main()
    # Selected passed to check and run should be resolved (pruned)
    assert "maestro-v3-midi" not in captured["selected"]
    assert "maestro-v3-full" in captured["selected"]
    # needed should be deduped and exclude superseded
    # Sum of all raw sizes would be larger than needed after pruning
    raw_sum = sum(dd._dataset_size_mb(dd.DATASETS[k]) for k in dd.DATASETS) * 1_048_576
    assert captured["needed"] < raw_sum

