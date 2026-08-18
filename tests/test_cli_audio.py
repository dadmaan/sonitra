from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sonitra.benchmark import runner as runner_module
from sonitra.corpus import discover_audio_files, pair_audio_to_reference


@pytest.fixture(autouse=True)
def _reset_console_singleton():
    """Reset the process-wide Console singleton before *and* after each test.

    `sonitra.terminal.get_console()` pins its Console to `sys.stdout` on
    first use; CliRunner's isolation closes that stream when its `invoke()`
    context exits, so a Console cached from an earlier CLI test/invocation
    in this session raises `ValueError: I/O operation on closed file` on the
    next `console.print`. This is the pre-existing CliRunner flakiness noted
    in PLAN.md's verification notes. Reset before each test so this module's
    tests are not order-dependent on whichever CLI test ran first in the
    batch, and reset again afterward so this module's CliRunner usage does
    not leave a closed-stream singleton behind for unrelated test modules
    that call `get_console()` directly (outside a CliRunner context) later
    in the same session.
    """
    import sonitra.terminal as terminal_module

    terminal_module._console = None
    yield
    terminal_module._console = None


# ---------------------------------------------------------------------------
# Inline config templates (cf. tests/test_corpus_discovery.py's
# `_RENDER_SMOKE_CONFIG` and tests/test_cli.py's `_MINIMAL_CONFIG_TEMPLATE`).
# `synth_backend: fluidsynth` is used purely as a required enum value -- in
# `input_type: audio` mode the synth is never constructed (Phase 1), so no
# `fluidsynth:`/`dawdreamer:` section is needed.
# ---------------------------------------------------------------------------

_AUDIO_RENDER_CONFIG = """\
render_pipeline:
  synth_backend: fluidsynth
  effects_chain: none
  input_type: audio
  bpm: 120
  sample_rate: 44100
  bit_depth: 16
  channels: 2
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
  progress: false
"""

_AUDIO_BENCHMARK_CONFIG = """\
render_pipeline:
  synth_backend: fluidsynth
  effects_chain: none
  input_type: audio
  bpm: 120
  sample_rate: 44100
  bit_depth: 16
  channels: 2
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
observability:
  write_manifest: false
  progress: false
transcription:
  transcribers:
    - type: precomputed
      name: oracle
      midi_dir: {precomputed_dir}
"""

_AUDIO_TRANSCRIBE_EVALUATE_CONFIG = """\
render_pipeline:
  synth_backend: fluidsynth
  effects_chain: none
  input_type: audio
  bpm: 120
  sample_rate: 44100
  bit_depth: 16
  channels: 2
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
observability:
  write_manifest: false
  progress: false
transcription:
  transcribers:
    - type: precomputed
      name: precomputed
      midi_dir: {precomputed_dir}
"""


def _write_wav(path: Path, *, freq: float = 440.0, sample_rate: int = 44100) -> None:
    from sonitra.storage import write_wav

    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.25 * np.sin(2 * np.pi * freq * t)
    tone = np.stack([signal, signal])
    write_wav(tone, path, sample_rate=sample_rate, normalize=False)


# ---------------------------------------------------------------------------
# render command
# ---------------------------------------------------------------------------


def test_render_audio_mode_uses_recordings_source(audio_corpus_dir: Path) -> None:
    """No --corpus override: audio mode reads from paths.recordings, writes
    to paths.audio, and never touches the source recordings/ directory."""
    from typer.testing import CliRunner

    from sonitra.cli import app

    corpus_root = audio_corpus_dir.parent
    dataset = audio_corpus_dir.name
    recordings_dir = audio_corpus_dir / "recordings"
    before = sorted(p.name for p in recordings_dir.iterdir())

    config_path = audio_corpus_dir / "config.yaml"
    config_path.write_text(_AUDIO_RENDER_CONFIG.format(corpus_root=str(corpus_root)))

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["render", "--config", str(config_path), "--dataset", dataset],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"

    out_dir = audio_corpus_dir / "audio" / "config"
    rendered = sorted(out_dir.rglob("*.wav"))
    assert len(rendered) == 2

    after = sorted(p.name for p in recordings_dir.iterdir())
    assert after == before  # source recordings/ untouched


def test_render_audio_mode_corpus_override(audio_corpus_dir: Path, tmp_path: Path) -> None:
    """--corpus overrides the source directory even in audio mode."""
    from typer.testing import CliRunner

    from sonitra.cli import app

    alt_recordings = tmp_path / "alt_recordings"
    alt_recordings.mkdir()
    _write_wav(alt_recordings / "solo_performer.wav")

    corpus_root = audio_corpus_dir.parent
    dataset = audio_corpus_dir.name

    config_path = audio_corpus_dir / "config.yaml"
    config_path.write_text(_AUDIO_RENDER_CONFIG.format(corpus_root=str(corpus_root)))

    out_dir = tmp_path / "override_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "render",
            "--config", str(config_path),
            "--dataset", dataset,
            "--corpus", str(alt_recordings),
            "--output", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"

    rendered = list(out_dir.rglob("*.wav"))
    assert len(rendered) == 1
    assert rendered[0].stem == "solo_performer"


def test_render_audio_mode_empty_recordings_exit_1(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    recordings_dir = corpus_root / "recordings"
    recordings_dir.mkdir(parents=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_AUDIO_RENDER_CONFIG.format(corpus_root=str(corpus_root)))

    from typer.testing import CliRunner

    from sonitra.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["render", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "No audio files found" in result.output


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------


def test_benchmark_audio_mode_cli_discovers_and_subsets_consistently(
    audio_corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--limit subsets the recordings list first; midi_paths passed to
    run_benchmark must be exactly the deduplicated references paired to the
    *sampled* recordings, never an independently sampled list (PLAN §2.5.1)."""
    from sonitra.cli import _apply_subset

    recordings_dir = audio_corpus_dir / "recordings"
    midi_dir = audio_corpus_dir / "midi"

    # Fan-out: a second performer per reference, so subsetting order is
    # actually observable (a naive independent-sample of midi_paths could
    # otherwise coincidentally match).
    _write_wav(recordings_dir / "piece_0_performer2.wav", freq=450.0)
    _write_wav(recordings_dir / "piece_1_performer2.wav", freq=460.0)

    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()

    corpus_root = audio_corpus_dir.parent
    dataset = audio_corpus_dir.name

    config_path = audio_corpus_dir / "config.yaml"
    config_path.write_text(
        _AUDIO_BENCHMARK_CONFIG.format(
            corpus_root=str(corpus_root), precomputed_dir=str(precomputed_dir)
        )
    )

    captured: dict = {}

    def _fake_run_benchmark(midi_paths, work_dir, config, corpus_root=None, *, audio_paths=None, progress=None):
        captured["midi_paths"] = sorted(Path(p) for p in midi_paths)
        captured["audio_paths"] = sorted(Path(p) for p in audio_paths) if audio_paths is not None else None
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=[],
            summary=[],
            degradation=[],
            results_path=work_dir / "results.jsonl",
            summary_path=work_dir / "summary.json",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)

    from typer.testing import CliRunner

    from sonitra.cli import app

    seed = 0
    limit = 1
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", dataset,
            "--limit", str(limit),
            "--seed", str(seed),
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    assert captured["audio_paths"] is not None
    assert len(captured["audio_paths"]) == limit

    # Recompute the expected subset independently to assert the exact
    # subsetting order: recordings sampled first, references derived from
    # the sampled recordings via pair_audio_to_reference.
    all_recordings = discover_audio_files(recordings_dir)
    all_midi = sorted(midi_dir.glob("*.mid"))
    expected_recordings = _apply_subset(all_recordings, limit, seed)
    expected_pairing = pair_audio_to_reference(expected_recordings, all_midi)
    expected_midi = sorted(set(expected_pairing.mapping.values()))

    assert captured["audio_paths"] == sorted(expected_recordings)
    assert captured["midi_paths"] == expected_midi

    # The references list must be exactly the ones paired to the sampled
    # recordings -- not an independent subset of all references.
    assert set(captured["midi_paths"]).issubset(set(all_midi))
    assert len(captured["midi_paths"]) <= limit


def test_benchmark_timing_table_shows_condition_timing(
    audio_corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark command renders a dedicated per-condition timing table
    from result.timing; the summary table no longer carries timing columns.
    An all-NaN separate_seconds (separation disabled) must not produce a
    `separate (sec)` column."""
    import sys

    from rich.console import Console

    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()

    corpus_root = audio_corpus_dir.parent
    dataset = audio_corpus_dir.name

    config_path = audio_corpus_dir / "config.yaml"
    config_path.write_text(
        _AUDIO_BENCHMARK_CONFIG.format(
            corpus_root=str(corpus_root), precomputed_dir=str(precomputed_dir)
        )
    )

    # Widen the console so the timing table's headers render untruncated in
    # CliRunner's 80-char capture.
    monkeypatch.setattr(
        "sonitra.cli.get_console",
        lambda *args, **kwargs: Console(file=sys.stdout, width=200),
    )

    def _fake_run_benchmark(midi_paths, work_dir, config, corpus_root=None, *, audio_paths=None, progress=None):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=[],
            summary=[
                {
                    "condition": "baseline",
                    "transcriber": "basic_pitch",
                    "n_files": 2,
                    "n_succeeded": 2,
                    "note.f1": 0.9,
                }
            ],
            degradation=[],
            results_path=work_dir / "results.jsonl",
            summary_path=work_dir / "summary.json",
            elapsed_seconds=0.0,
            timing={
                "overall_seconds": 100.0,
                "host": {},
                "conditions": [
                    {
                        "condition": "baseline",
                        "wall_seconds": 30.2,
                        "render_seconds": 4.0,
                        "separate_seconds": float("nan"),
                        "transcribe_seconds": 6.5,
                        "evaluate_seconds": 1.2,
                    }
                ],
            },
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)

    from typer.testing import CliRunner

    from sonitra.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", dataset,
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    # Summary table: no timing columns anymore.
    assert "render s" not in result.output
    assert "transcribe s" not in result.output
    assert "eval s" not in result.output

    # Dedicated timing table.
    assert "Benchmark timing (seconds)" in result.output
    assert "wall (sec)" in result.output
    assert "render (sec)" in result.output
    assert "transcribe (sec)" in result.output
    assert "evaluate (sec)" in result.output
    assert "separate (sec)" not in result.output
    assert "30.2" in result.output
    assert "4.0" in result.output
    assert "6.5" in result.output
    assert "1.2" in result.output


def test_benchmark_timing_table_shows_separate_seconds_when_present(
    audio_corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-NaN separate_seconds in any timing condition adds the
    `separate (sec)` column."""
    import sys

    from rich.console import Console

    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()

    corpus_root = audio_corpus_dir.parent
    dataset = audio_corpus_dir.name

    config_path = audio_corpus_dir / "config.yaml"
    config_path.write_text(
        _AUDIO_BENCHMARK_CONFIG.format(
            corpus_root=str(corpus_root), precomputed_dir=str(precomputed_dir)
        )
    )

    # Widen the console so the timing table's headers render untruncated in
    # CliRunner's 80-char capture.
    monkeypatch.setattr(
        "sonitra.cli.get_console",
        lambda *args, **kwargs: Console(file=sys.stdout, width=200),
    )

    def _fake_run_benchmark(midi_paths, work_dir, config, corpus_root=None, *, audio_paths=None, progress=None):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=[],
            summary=[
                {
                    "condition": "baseline",
                    "transcriber": "basic_pitch",
                    "n_files": 2,
                    "n_succeeded": 2,
                    "note.f1": 0.9,
                }
            ],
            degradation=[],
            results_path=work_dir / "results.jsonl",
            summary_path=work_dir / "summary.json",
            elapsed_seconds=0.0,
            timing={
                "overall_seconds": 100.0,
                "host": {},
                "conditions": [
                    {
                        "condition": "baseline",
                        "wall_seconds": 30.2,
                        "render_seconds": 4.0,
                        "separate_seconds": 3.3,
                        "transcribe_seconds": 6.5,
                        "evaluate_seconds": 1.2,
                    }
                ],
            },
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)

    from typer.testing import CliRunner

    from sonitra.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", dataset,
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    assert "separate (sec)" in result.output
    assert "3.3" in result.output


def test_benchmark_timing_table_absent_without_timing(
    audio_corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BenchmarkResult(timing=None) renders no timing table and no crash;
    the summary table renders as before."""
    import sys

    from rich.console import Console

    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()

    corpus_root = audio_corpus_dir.parent
    dataset = audio_corpus_dir.name

    config_path = audio_corpus_dir / "config.yaml"
    config_path.write_text(
        _AUDIO_BENCHMARK_CONFIG.format(
            corpus_root=str(corpus_root), precomputed_dir=str(precomputed_dir)
        )
    )

    # Widen the console so the summary table's headers render untruncated in
    # CliRunner's 80-char capture.
    monkeypatch.setattr(
        "sonitra.cli.get_console",
        lambda *args, **kwargs: Console(file=sys.stdout, width=200),
    )

    def _fake_run_benchmark(midi_paths, work_dir, config, corpus_root=None, *, audio_paths=None, progress=None):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=[],
            summary=[
                {
                    "condition": "baseline",
                    "transcriber": "basic_pitch",
                    "n_files": 2,
                    "n_succeeded": 2,
                    "note.f1": 0.9,
                }
            ],
            degradation=[],
            results_path=work_dir / "results.jsonl",
            summary_path=work_dir / "summary.json",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)

    from typer.testing import CliRunner

    from sonitra.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", dataset,
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    assert "Benchmark timing (seconds)" not in result.output
    assert "Benchmark summary" in result.output
    assert "note.f1" in result.output


# ---------------------------------------------------------------------------
# transcribe / evaluate — no behavioural change in audio-mode configs
# ---------------------------------------------------------------------------


def test_transcribe_evaluate_commands_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from typer.testing import CliRunner

    import sonitra.terminal as terminal_module
    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"

    # transcribe --audio ... against an audio-mode config.
    audio_dir = tmp_path / "audio_in"
    audio_dir.mkdir()
    _write_wav(audio_dir / "clip.wav")

    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()
    shutil.copy(fixtures / "test_c4.mid", precomputed_dir / "clip.mid")

    corpus_root = tmp_path / "corpus"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _AUDIO_TRANSCRIBE_EVALUATE_CONFIG.format(
            corpus_root=str(corpus_root), precomputed_dir=str(precomputed_dir)
        )
    )

    transcribe_out = tmp_path / "transcribe_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio", str(audio_dir),
            "--output", str(transcribe_out),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, f"transcribe failed:\n{result.output}"
    assert (transcribe_out / "precomputed" / "clip.mid").exists()

    # evaluate --reference/--estimate ... independent of pipeline.input_type.
    ref_dir = tmp_path / "ref"
    est_dir = tmp_path / "est"
    ref_dir.mkdir()
    est_dir.mkdir()
    shutil.copy(fixtures / "test_c4.mid", ref_dir / "piece_0.mid")
    shutil.copy(fixtures / "test_c4.mid", est_dir / "piece_0.mid")

    out_jsonl = tmp_path / "eval_results.jsonl"
    # Reset the Console singleton between the two invokes in this test: it
    # is pinned to `sys.stdout` on first use (see `_reset_console_singleton`
    # above), and the transcribe invoke's CliRunner context already closed
    # that stream by the time evaluate runs.
    monkeypatch.setattr(terminal_module, "_console", None)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--reference", str(ref_dir),
            "--estimate", str(est_dir),
            "--config", str(config_path),
            "--output", str(out_jsonl),
        ],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
    lines = [line for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
