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

# Minimal fluidsynth config used for CLI smoke tests. PedalboardSynth without a
# plugin now raises ValueError, so we use FluidSynth with the system SoundFont.
_RENDER_SMOKE_CONFIG = """\
pipeline:
  synth_backend: fluidsynth
  effects_chain: pedalboard
  bpm: 120
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
fluidsynth:
  soundfont_path: /usr/share/sounds/sf2/default-GM.sf2
pedalboard:
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


def test_evaluate_limit_constrains_output(tmp_path: Path) -> None:
    import shutil
    from typer.testing import CliRunner

    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"
    ref_dir = tmp_path / "ref"
    est_dir = tmp_path / "est"
    ref_dir.mkdir()
    est_dir.mkdir()
    for i in range(3):
        shutil.copy(fixtures / "test_c4.mid", ref_dir / f"piece_{i}.mid")
        shutil.copy(fixtures / "test_c4.mid", est_dir / f"piece_{i}.mid")

    out_jsonl = tmp_path / "results.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--reference", str(ref_dir),
            "--estimate", str(est_dir),
            "--output", str(out_jsonl),
            "--limit", "1",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, f"evaluate failed: {result.output}"
    lines = [line for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 1


def test_evaluate_limit_samples_only_files_with_estimates(tmp_path: Path) -> None:
    import shutil
    from typer.testing import CliRunner

    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"
    ref_dir = tmp_path / "ref"
    est_dir = tmp_path / "est"
    ref_dir.mkdir()
    est_dir.mkdir()
    # 3 references, but only 2 have matching estimates.
    for i in range(3):
        shutil.copy(fixtures / "test_c4.mid", ref_dir / f"piece_{i}.mid")
    for i in range(2):
        shutil.copy(fixtures / "test_c4.mid", est_dir / f"piece_{i}.mid")

    # limit 5 exceeds the 2 matchable files -> all 2 evaluated (not 3).
    out_jsonl = tmp_path / "results.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--reference", str(ref_dir),
            "--estimate", str(est_dir),
            "--output", str(out_jsonl),
            "--limit", "5",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, f"evaluate failed: {result.output}"
    lines = [line for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 2


def test_evaluate_dataset_with_config_unnamed_transcriber(tmp_path: Path) -> None:
    """evaluate --config + --dataset works when the transcriber omits `name`.

    Regression: the dataset+config branch read `transcribers[0].name` directly,
    which is None when the config entry has no `name` field, crashing with
    `PosixPath / None`. It must fall back to the backend `type` (basic_pitch),
    matching how the transcribe command names its output directories.
    """
    import shutil
    from typer.testing import CliRunner

    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"
    dataset = "maestro-v3"
    config_stem = "eval_cfg"
    corpus_root = tmp_path / "corpus"
    midi_dir = corpus_root / dataset / "midi"
    est_dir = corpus_root / dataset / "transcription" / config_stem / "basic_pitch"
    midi_dir.mkdir(parents=True)
    est_dir.mkdir(parents=True)
    shutil.copy(fixtures / "test_c4.mid", midi_dir / "piece_0.mid")
    shutil.copy(fixtures / "test_c4.mid", est_dir / "piece_0.mid")

    config_path = tmp_path / f"{config_stem}.yaml"
    config_path.write_text(
        f"""\
pipeline:
  synth_backend: fluidsynth
  effects_chain: pedalboard
  bpm: 120
  sample_rate: 22050
  bit_depth: 16
  channels: 1
  duration_padding_sec: 0.5
  overwrite: true
  resume: false
  max_workers: 1
  log_level: INFO
io:
  corpus_root: {corpus_root}
  output_format: wav
  mp3_bitrate_kbps: 192
  file_naming: "{{stem}}"
fluidsynth:
  soundfont_path: /usr/share/sounds/sf2/default-GM.sf2
transcription:
  transcribers:
    - type: basic_pitch
      enabled: true
"""
    )

    out_jsonl = tmp_path / "results.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--dataset", dataset,
            "--output", str(out_jsonl),
        ],
    )
    assert result.exit_code == 0, f"evaluate failed: {result.output}"
    lines = [line for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
