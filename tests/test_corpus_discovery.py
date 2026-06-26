from __future__ import annotations

from pathlib import Path

import pytest

from sonitra.cli import _apply_subset, _discover_midi_files


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


def test_discover_finds_mid_in_flat_dir(tmp_path: Path) -> None:
    (tmp_path / "piece.mid").write_bytes(b"")
    result = _discover_midi_files(tmp_path)
    assert result == [tmp_path / "piece.mid"]


def test_discover_finds_midi_extension(tmp_path: Path) -> None:
    (tmp_path / "piece.midi").write_bytes(b"")
    result = _discover_midi_files(tmp_path)
    assert result == [tmp_path / "piece.midi"]


def test_discover_is_recursive(tmp_path: Path) -> None:
    subdir = tmp_path / "2004"
    subdir.mkdir()
    midi = subdir / "piece.mid"
    midi.write_bytes(b"")
    result = _discover_midi_files(tmp_path)
    assert result == [midi]


def test_discover_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "a.MID").write_bytes(b"")
    (tmp_path / "b.MIDI").write_bytes(b"")
    result = _discover_midi_files(tmp_path)
    assert len(result) == 2


def test_discover_ignores_non_midi(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "meta.csv").write_bytes(b"")
    result = _discover_midi_files(tmp_path)
    assert result == []


def test_discover_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "subdir.mid").mkdir()   # a directory named like a MIDI file
    result = _discover_midi_files(tmp_path)
    assert result == []


def test_discover_returns_sorted(tmp_path: Path) -> None:
    for name in ("c.mid", "a.mid", "b.mid"):
        (tmp_path / name).write_bytes(b"")
    result = _discover_midi_files(tmp_path)
    assert result == sorted(result)


def test_discover_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert _discover_midi_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Subset tests (pure; build a list of fake Paths, no filesystem needed)
# ---------------------------------------------------------------------------


def _fake_files(n: int) -> list[Path]:
    return [Path(f"file_{i:03d}.mid") for i in range(n)]


def test_apply_subset_none_limit_returns_all() -> None:
    files = _fake_files(5)
    assert _apply_subset(files, limit=None, seed=0) == files


def test_apply_subset_limit_equal_to_total_returns_all() -> None:
    files = _fake_files(5)
    assert _apply_subset(files, limit=5, seed=0) == files


def test_apply_subset_limit_above_total_returns_all() -> None:
    files = _fake_files(5)
    assert _apply_subset(files, limit=100, seed=0) == files


def test_apply_subset_limit_below_total_returns_exactly_limit() -> None:
    files = _fake_files(10)
    result = _apply_subset(files, limit=3, seed=42)
    assert len(result) == 3


def test_apply_subset_seed_is_deterministic() -> None:
    files = _fake_files(20)
    r1 = _apply_subset(files, limit=5, seed=7)
    r2 = _apply_subset(files, limit=5, seed=7)
    assert r1 == r2


def test_apply_subset_different_seeds_differ() -> None:
    files = _fake_files(20)
    r1 = _apply_subset(files, limit=5, seed=1)
    r2 = _apply_subset(files, limit=5, seed=999)
    # With 20 files choosing 5, collision probability is negligible
    assert r1 != r2


def test_apply_subset_result_is_sorted() -> None:
    files = _fake_files(20)
    result = _apply_subset(files, limit=7, seed=13)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# CLI smoke tests (use CliRunner + real fixtures)
# ---------------------------------------------------------------------------

# Minimal pedalboard-only config with silence_threshold_rms=0.0 so that
# PedalboardSynth without a plugin (which renders zeros) passes quality gates.
# This mirrors the pattern used in test_benchmark_runner.py and is required
# because config_pedalboard_only.yaml sets silence_threshold_rms=0.001, which
# rejects the silence produced by a plugin-less pedalboard render.
_RENDER_SMOKE_CONFIG = """\
pipeline:
  rendering_mode: pedalboard_only
  sample_rate: 22050
  bit_depth: 16
  channels: 1
  duration_padding_sec: 0.5
  overwrite: true
  resume: false
  max_workers: 1
  log_level: INFO
io:
  corpus_root: ./corpus
  output_format: wav
  mp3_bitrate_kbps: 192
  file_naming: "{stem}"
pedalboard:
  enabled: true
  instrument:
    plugin_path: null
    preset_path: null
    reload_plugin_per_file: false
    silence_flush_sec: 0.0
  effects: []
normalisation:
  enabled: false
  mode: peak
  target_db: -1.0
  pre_effects: false
quality_gates:
  silence_threshold_rms: 0.0
  min_duration_sec: 0.0
  max_duration_deviation_sec: 0.0
  clip_threshold: 1.0
observability:
  write_manifest: false
  manifest_path: ./renders.jsonl
  write_failed_list: false
  emit_sse_events: false
"""


def test_render_discovers_nested_midi_via_cli(tmp_path: Path) -> None:
    """render command finds MIDI files in subdirectories via rglob."""
    import shutil
    from typer.testing import CliRunner
    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"
    nested = tmp_path / "2004"
    nested.mkdir()
    shutil.copy(fixtures / "test_c4.mid", nested / "test_c4.mid")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_RENDER_SMOKE_CONFIG)
    out_dir = tmp_path / "audio"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "render",
            "--corpus", str(tmp_path),
            "--output", str(out_dir),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"
    rendered = list(out_dir.rglob("*.wav"))
    assert len(rendered) == 1


def test_render_limit_constrains_output(tmp_path: Path) -> None:
    """--limit N produces at most N rendered files."""
    import shutil
    from typer.testing import CliRunner
    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"
    for i, src in enumerate(["test_c4.mid", "test_empty.mid", "test_polyphonic.mid"]):
        shutil.copy(fixtures / src, tmp_path / f"piece_{i}.mid")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_RENDER_SMOKE_CONFIG)
    out_dir = tmp_path / "audio"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "render",
            "--corpus", str(tmp_path),
            "--output", str(out_dir),
            "--config", str(config_path),
            "--limit", "1",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"
    rendered = list(out_dir.rglob("*.wav"))
    assert len(rendered) == 1
