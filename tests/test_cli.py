from __future__ import annotations

import json
import os
import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from sonitra.cli import evaluate, init
from sonitra.config import SynthBackend, load_config
from sonitra.pipeline import run_pipeline
from sonitra.transcribe.base import NumericSettingsError, TranscriptionResult


def test_init_writes_working_basic_pitch_config(tmp_path: Path) -> None:
    path = tmp_path / "init.yaml"
    init(path)
    cfg = load_config(path)
    assert cfg.render_pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST
    assert cfg.normalisation.enabled is True
    assert any(t.type == "basic_pitch" and t.enabled for t in cfg.transcription.transcribers)


def test_init_config_renders(corpus_dir: Path, tmp_path: Path) -> None:
    init(tmp_path / "init.yaml")
    cfg = load_config(tmp_path / "init.yaml")
    # Keep observability enabled but redirect manifest files into tmp_path so
    # tests do not write into the working directory.
    cfg.observability.manifest_path = tmp_path / "renders.jsonl"
    result = run_pipeline(sorted(corpus_dir.glob("*.mid")), tmp_path / "audio", config=cfg)
    assert result.succeeded >= 2


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_init_config_transcribes(corpus_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig
    from sonitra.transcribe.protocol import make_transcriber

    init(tmp_path / "init.yaml")
    cfg = load_config(tmp_path / "init.yaml")
    cfg.observability.manifest_path = tmp_path / "renders.jsonl"
    audio_dir = tmp_path / "audio"
    run_pipeline([corpus_dir / "test_c4.mid"], audio_dir, config=cfg)
    wav = next(audio_dir.glob("*.wav"))
    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    result = transcriber.transcribe(wav)
    assert len(result.notes) > 0


def test_sonitra_console_script_is_registered() -> None:
    eps = entry_points(group="console_scripts")
    sonitra_eps = [ep for ep in eps if ep.name == "sonitra"]
    assert len(sonitra_eps) == 1
    assert sonitra_eps[0].value == "sonitra.cli:app"


# ---------------------------------------------------------------------------
# transcribe command tests — use PrecomputedTranscriber, no Basic Pitch inference
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG_TEMPLATE = """\
render_pipeline:
  synth_backend: dawdreamer_faust
  effects_chain: pedalboard
  bpm: 120
  sample_rate: 44100
  bit_depth: 24
  channels: 2
  duration_padding_sec: 2.0
  overwrite: true
  resume: false
  max_workers: 1
  log_level: INFO
io:
  corpus_root: ./corpus
  output_format: wav
  mp3_bitrate_kbps: 192
  file_naming: "{{stem}}"
dawdreamer:
  block_size: 512
  plugin_path: null
  preset_path: null
  faust_code: null
  clear_midi_between_renders: true
fluidsynth:
  soundfont_path: null
pedalboard:
  instrument:
    plugin_path: null
    preset_path: null
    reload_plugin_per_file: false
    silence_flush_sec: 0.5
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
separation:
  enabled: false
  backend: passthrough
  model: htdemucs
  device: cpu
  stem: null
  output_dir: stems
transcription:
  output_dir: transcriptions
  transcribers:
    - type: precomputed
      name: precomputed
      midi_dir: {midi_dir}
evaluation:
  note_metrics:
    enabled: true
    onset_tolerance_sec: 0.05
    offset_ratio: 0.2
    offset_min_tolerance_sec: 0.05
    velocity_tolerance: 0.1
  frame_metrics:
    enabled: true
    hop_sec: 0.01
  expressive_metrics:
    enabled: true
    harmony_window_sec: 2.0
  dtw:
    enabled: false
    frame_size: 4096
    hop_size: 2048
    max_frames: 4000
benchmark:
  results_path: benchmark_results.jsonl
  include_baseline: true
  baseline_name: baseline
  conditions: []
  sweeps: []
"""


class _StubTranscriber:
    """Minimal transcriber stub returning a fixed raw_outputs payload."""

    name = "stub"

    def __init__(self, raw_outputs):
        self._raw = raw_outputs

    def transcribe(self, audio_path):
        return TranscriptionResult(
            notes=[],
            transcriber=self.name,
            raw_outputs=self._raw,
            backend_type="basic_pitch",
        )


def test_transcribe_creates_midi_by_backend_name(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    wav_path = tmp_path / "test_c4.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    midi_path = tmp_path / "test_c4.mid"
    fixtures_midi = Path(__file__).parent / "fixtures" / "test_c4.mid"
    midi_path.write_bytes(fixtures_midi.read_bytes())

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", "--audio", str(tmp_path), "--output", str(out_dir), "--config", str(config_path)],
    )
    assert result.exit_code == 0, f"transcribe failed: {result.output}"
    assert (out_dir / "precomputed" / "test_c4.mid").exists()


def test_transcribe_limit_constrains_output(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    fixtures = Path(__file__).parent / "fixtures"
    for i in range(3):
        (tmp_path / f"piece_{i}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        (tmp_path / f"piece_{i}.mid").write_bytes(
            (fixtures / "test_c4.mid").read_bytes()
        )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(out_dir),
            "--config", str(config_path),
            "--limit", "1",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, f"transcribe failed: {result.output}"
    transcribed = list((out_dir / "precomputed").rglob("*.mid"))
    assert len(transcribed) == 1


def test_transcribe_empty_audio_dir_exits_nonzero(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    audio_dir = tmp_path / "empty_audio"
    audio_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", "--audio", str(audio_dir), "--output", str(out_dir), "--config", str(config_path)],
    )
    assert result.exit_code != 0


_TWO_TRANSCRIBERS_CONFIG_TEMPLATE = """\
render_pipeline:
  synth_backend: dawdreamer_faust
  effects_chain: pedalboard
  bpm: 120
  sample_rate: 44100
  bit_depth: 24
  channels: 2
  duration_padding_sec: 2.0
  overwrite: true
  resume: false
  max_workers: 1
  log_level: INFO
io:
  corpus_root: ./corpus
  output_format: wav
  mp3_bitrate_kbps: 192
  file_naming: "{{stem}}"
dawdreamer:
  block_size: 512
  plugin_path: null
  preset_path: null
  faust_code: null
  clear_midi_between_renders: true
fluidsynth:
  soundfont_path: null
pedalboard:
  instrument:
    plugin_path: null
    preset_path: null
    reload_plugin_per_file: false
    silence_flush_sec: 0.5
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
separation:
  enabled: false
  backend: passthrough
  model: htdemucs
  device: cpu
  stem: null
  output_dir: stems
transcription:
  output_dir: transcriptions
  transcribers:
    - type: precomputed
      name: keep_this
      midi_dir: {keep_midi_dir}
    - type: precomputed
      name: skip_this
      midi_dir: {skip_midi_dir}
evaluation:
  note_metrics:
    enabled: true
    onset_tolerance_sec: 0.05
    offset_ratio: 0.2
    offset_min_tolerance_sec: 0.05
    velocity_tolerance: 0.1
  frame_metrics:
    enabled: true
    hop_sec: 0.01
  expressive_metrics:
    enabled: true
    harmony_window_sec: 2.0
  dtw:
    enabled: false
    frame_size: 4096
    hop_size: 2048
    max_frames: 4000
benchmark:
  results_path: benchmark_results.jsonl
  include_baseline: true
  baseline_name: baseline
  conditions: []
  sweeps: []
"""


def test_transcribe_filter_by_name_selects_correct_backend(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    wav_path = tmp_path / "test_c4.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    keep_midi_dir = tmp_path / "keep_midi"
    keep_midi_dir.mkdir()
    fixtures_midi = Path(__file__).parent / "fixtures" / "test_c4.mid"
    (keep_midi_dir / "test_c4.mid").write_bytes(fixtures_midi.read_bytes())

    skip_midi_dir = tmp_path / "skip_midi"
    skip_midi_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _TWO_TRANSCRIBERS_CONFIG_TEMPLATE.format(
            keep_midi_dir=str(keep_midi_dir),
            skip_midi_dir=str(skip_midi_dir),
        )
    )

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(out_dir),
            "--config", str(config_path),
            "--transcriber", "keep_this",
        ],
    )
    assert result.exit_code == 0, f"transcribe failed: {result.output}"
    assert (out_dir / "keep_this" / "test_c4.mid").exists()
    assert not (out_dir / "skip_this").exists()


# ---------------------------------------------------------------------------
# init command output verification tests
# ---------------------------------------------------------------------------

from typer.testing import CliRunner
from sonitra.cli import app
from sonitra.config import SynthBackend, EffectsChain, load_config

runner = CliRunner()


def test_init_config_has_synth_backend(tmp_path) -> None:
    result = runner.invoke(app, ["init", "--config", str(tmp_path / "cfg.yaml")])
    assert result.exit_code == 0
    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.render_pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST


def test_init_config_has_effects_chain(tmp_path) -> None:
    result = runner.invoke(app, ["init", "--config", str(tmp_path / "cfg.yaml")])
    assert result.exit_code == 0
    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.render_pipeline.effects_chain == EffectsChain.NONE


def test_init_config_no_rendering_mode_in_yaml(tmp_path) -> None:
    runner.invoke(app, ["init", "--config", str(tmp_path / "cfg.yaml")])
    raw = (tmp_path / "cfg.yaml").read_text()
    assert "rendering_mode" not in raw


def test_init_config_has_fluidsynth_section_in_raw_yaml(tmp_path) -> None:
    """Verify the init command actually writes the fluidsynth: section."""
    runner.invoke(app, ["init", "--config", str(tmp_path / "cfg.yaml")])
    raw = (tmp_path / "cfg.yaml").read_text()
    assert "fluidsynth:" in raw
    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.fluidsynth.soundfont_path is None


def test_init_config_no_section_level_enabled_in_yaml(tmp_path) -> None:
    """Verify dawdreamer and pedalboard sections have no top-level 'enabled' field.

    Only section-level enabled: in dawdreamer/pedalboard is prohibited (H2 constraint).
    Per-effect enabled: and per-transcriber enabled: are legitimate.
    """
    runner.invoke(app, ["init", "--config", str(tmp_path / "cfg.yaml")])
    cfg = load_config(tmp_path / "cfg.yaml")
    # DawDreamerSection model has no 'enabled' field
    assert not hasattr(cfg.dawdreamer, "enabled")
    # PedalboardSection model has no 'enabled' field
    assert not hasattr(cfg.pedalboard, "enabled")
    # Verify in raw YAML too
    raw = (tmp_path / "cfg.yaml").read_text()
    # dawdreamer section — look for enabled at the section level (2-space indent)
    import re
    dawdreamer_block = re.search(r"dawdreamer:\n(  .*(?:\n|$))*", raw)
    if dawdreamer_block:
        assert "enabled:" not in dawdreamer_block.group()
    pedalboard_block = re.search(r"pedalboard:\n(  .*(?:\n|$))*", raw)
    if pedalboard_block:
        assert "enabled:" not in pedalboard_block.group()


# ---------------------------------------------------------------------------
# transcribe command — raw-outputs sidecar wiring (stubbed transcriber)
# ---------------------------------------------------------------------------

def test_transcribe_writes_raw_outputs_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import numpy as np

    from sonitra.cli import app
    from sonitra.transcribe import protocol

    wav_path = tmp_path / "test_c4.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))

    raw_outputs = {
        "onset": np.zeros((2, 88)),
        "contour": np.zeros((2, 264)),
        "note": np.zeros((2, 88)),
    }
    monkeypatch.setattr(
        protocol, "make_transcriber", lambda cfg: _StubTranscriber(raw_outputs)
    )

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(out_dir),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, f"transcribe failed: {result.output}"
    assert (out_dir / "stub" / "test_c4.mid").exists()
    assert (out_dir / "stub" / "test_c4.model_outputs.csv").exists()


def test_transcribe_sidecar_failure_does_not_fail_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import numpy as np

    import sonitra.midi_writer
    from sonitra.cli import app
    from sonitra.transcribe import protocol

    def _boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        sonitra.midi_writer, "write_raw_outputs", _boom, raising=False
    )

    wav_path = tmp_path / "test_c4.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))

    raw_outputs = {
        "onset": np.zeros((2, 88)),
        "contour": np.zeros((2, 264)),
        "note": np.zeros((2, 88)),
    }
    monkeypatch.setattr(
        protocol, "make_transcriber", lambda cfg: _StubTranscriber(raw_outputs)
    )

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(out_dir),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, f"transcribe failed: {result.output}"
    assert (out_dir / "stub" / "test_c4.mid").exists()


@pytest.mark.parametrize("command", ["render", "transcribe", "evaluate", "benchmark"])
def test_dataset_option_help_names_the_dataset_first_layout(command: str) -> None:
    import typer

    from sonitra.cli import app

    click_command = typer.main.get_command(app).commands[command]
    (option,) = [param for param in click_command.params if "--dataset" in param.opts]
    assert "corpus/{dataset}/" in option.help
    assert "corpus/midi/{dataset}" not in option.help


def _write_simple_midi(path: Path, pitch: int = 60) -> None:
    """Write a one-note MIDI file that parse_midi can read back."""
    from sonitra.midi_writer import write_midi

    write_midi(
        [{"pitch": pitch, "velocity": 64, "start_sec": 0.0, "duration_sec": 1.0}],
        path,
    )


def test_evaluate_is_fail_soft_on_unreadable_reference(tmp_path: Path) -> None:
    """One corrupt reference must not abort the whole evaluate run.

    parse_midi raises OSError on a file with no MTrk header. Batch loops are
    fail-soft per AGENT.md: the bad pair is recorded and the run continues.
    """
    from typer.testing import CliRunner

    from sonitra.cli import app

    reference = tmp_path / "reference"
    estimate = tmp_path / "estimate"
    reference.mkdir()
    estimate.mkdir()

    for stem in ("good_a", "good_b", "corrupt"):
        _write_simple_midi(estimate / f"{stem}.mid")
    _write_simple_midi(reference / "good_a.mid")
    _write_simple_midi(reference / "good_b.mid")
    (reference / "corrupt.mid").write_bytes(
        b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0GARBAGE"
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))

    output = tmp_path / "results.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--reference", str(reference),
            "--estimate", str(estimate),
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, f"evaluate aborted: {result.output}"

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    by_file = {row["file"]: row for row in rows}

    # Both readable pairs were scored despite the corrupt sibling.
    assert "good_a.mid" in by_file
    assert "good_b.mid" in by_file
    assert "error" not in by_file["good_a.mid"]

    # The failure is recorded rather than swallowed or fatal.
    assert "corrupt.mid" in by_file
    assert "error" in by_file["corrupt.mid"]


# ---------------------------------------------------------------------------
# Run directory and YAML io.dataset handling
# ---------------------------------------------------------------------------

_BENCH_STEM = "study"


@pytest.fixture
def _fresh_console():
    """Drop the cached Console so each CliRunner invocation gets a live stream."""
    import sonitra.terminal as terminal_module

    terminal_module._console = None
    yield
    terminal_module._console = None


def _squash(text: str) -> str:
    """Collapse whitespace: Rich wraps stderr at 80 columns under CliRunner."""
    return re.sub(r"\s+", " ", text)


def _write_cli_config(
    tmp_path: Path,
    *,
    dataset: str | None = None,
    benchmark_dir: str | None = None,
) -> Path:
    import yaml

    payload = yaml.safe_load(
        _MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path / "oracle"))
    )
    payload["io"]["corpus_root"] = str(tmp_path / "corpus")
    if dataset is not None:
        payload["io"]["dataset"] = dataset
    if benchmark_dir is not None:
        payload["benchmark"]["benchmark_dir"] = benchmark_dir
    path = tmp_path / f"{_BENCH_STEM}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _seed_midi(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a.mid").write_bytes(
        (Path(__file__).parent / "fixtures" / "test_c4.mid").read_bytes()
    )


def _capture_run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    from sonitra.benchmark import runner as runner_module

    captured: dict = {}

    def _fake_run_benchmark(
        midi_paths, work_dir, config, corpus_root=None, *,
        audio_paths=None, progress=None, selection=None,
    ):
        captured["work_dir"] = Path(work_dir)
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=[],
            summary=[],
            degradation=[],
            results_path=Path(work_dir) / "results.jsonl",
            summary_path=Path(work_dir) / "summary.json",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)
    return captured


def _run_benchmark_cli(config_path: Path, *extra: str):
    from typer.testing import CliRunner

    from sonitra.cli import app

    return CliRunner().invoke(app, ["benchmark", "--config", str(config_path), *extra])


def test_benchmark_yaml_dataset_without_flag_uses_corpus_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")

    result = _run_benchmark_cli(config_path)

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == (
        Path(str(tmp_path / "corpus")) / "mini" / "benchmark" / _BENCH_STEM
    )


def test_benchmark_dataset_flag_uses_corpus_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path)

    result = _run_benchmark_cli(config_path, "--dataset", "mini")

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == (
        Path(str(tmp_path / "corpus")) / "mini" / "benchmark" / _BENCH_STEM
    )


def test_benchmark_no_dataset_uses_cwd_benchmark_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "midi")
    config_path = _write_cli_config(tmp_path)

    result = _run_benchmark_cli(config_path)

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == Path("benchmark") / _BENCH_STEM


def test_benchmark_dir_key_used_as_full_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    fixed = tmp_path / "runs" / "fixed"
    config_path = _write_cli_config(tmp_path, dataset="mini", benchmark_dir=str(fixed))

    result = _run_benchmark_cli(config_path)

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == fixed


def test_workdir_flag_overrides_benchmark_dir_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(
        tmp_path, dataset="mini", benchmark_dir=str(tmp_path / "runs" / "fixed")
    )
    explicit = tmp_path / "explicit"

    result = _run_benchmark_cli(config_path, "--workdir", str(explicit))

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == explicit


@pytest.mark.parametrize("yaml_dataset", [None, "mini"])
def test_benchmark_dir_with_changed_dataset_exits_1(
    yaml_dataset: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fresh_console,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "gaps" / "midi")
    fixed = tmp_path / "runs" / "fixed"
    config_path = _write_cli_config(
        tmp_path, dataset=yaml_dataset, benchmark_dir=str(fixed)
    )

    result = _run_benchmark_cli(config_path, "--dataset", "gaps")

    assert result.exit_code == 1
    # Rich may fold the long path mid-word, so compare without any whitespace.
    expected = (
        f"error: benchmark.benchmark_dir is fixed ({fixed}) but --dataset 'gaps' "
        f"differs from io.dataset '{yaml_dataset}'; pass --workdir or edit "
        "benchmark_dir"
    )
    assert re.sub(r"\s+", "", result.stderr) == re.sub(r"\s+", "", expected)
    assert captured == {}
    assert not fixed.exists()
    assert not (tmp_path / "benchmark").exists()


def test_benchmark_dir_with_equal_dataset_flag_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    fixed = tmp_path / "runs" / "fixed"
    config_path = _write_cli_config(tmp_path, dataset="mini", benchmark_dir=str(fixed))

    result = _run_benchmark_cli(config_path, "--dataset", "mini")

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == fixed


def test_benchmark_dir_with_changed_dataset_and_workdir_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "gaps" / "midi")
    config_path = _write_cli_config(
        tmp_path, dataset="mini", benchmark_dir=str(tmp_path / "runs" / "fixed")
    )
    explicit = tmp_path / "explicit"

    result = _run_benchmark_cli(
        config_path, "--dataset", "gaps", "--workdir", str(explicit)
    )

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == explicit


def test_transcribe_yaml_dataset_defaults_audio_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "corpus"
    audio_dir = root / "mini" / "audio" / _BENCH_STEM
    audio_dir.mkdir(parents=True)
    (audio_dir / "a.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    _seed_midi(tmp_path / "oracle")
    config_path = _write_cli_config(tmp_path, dataset="mini")

    result = CliRunner().invoke(app, ["transcribe", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    expected = root / "mini" / "transcription" / _BENCH_STEM / "precomputed" / "a.mid"
    assert expected.exists()
    assert not (tmp_path / "transcriptions").exists()


def test_evaluate_yaml_dataset_defaults_reference_and_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "corpus"
    corpus_ds = root / "mini"
    (corpus_ds / "midi").mkdir(parents=True)
    (corpus_ds / "transcription" / _BENCH_STEM / "precomputed").mkdir(
        parents=True
    )
    _write_simple_midi(corpus_ds / "midi" / "a.mid")
    _write_simple_midi(
        corpus_ds / "transcription" / _BENCH_STEM / "precomputed" / "a.mid"
    )
    config_path = _write_cli_config(tmp_path, dataset="mini")
    output = tmp_path / "results.jsonl"

    result = CliRunner().invoke(
        app, ["evaluate", "--config", str(config_path), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["file"] for row in rows] == ["a.mid"]


def test_transcribe_explicit_paths_win_over_yaml_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    audio_dir = tmp_path / "given_audio"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    _seed_midi(tmp_path / "oracle")
    config_path = _write_cli_config(tmp_path, dataset="mini")
    out_dir = tmp_path / "given_out"

    result = CliRunner().invoke(
        app,
        [
            "transcribe", "--config", str(config_path),
            "--audio", str(audio_dir), "--output", str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "precomputed" / "a.mid").exists()
    assert not (tmp_path / "corpus" / "mini" / "transcription").exists()


def test_evaluate_explicit_paths_win_over_yaml_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    reference = tmp_path / "given_ref"
    estimate = tmp_path / "given_est"
    reference.mkdir()
    estimate.mkdir()
    _write_simple_midi(reference / "b.mid")
    _write_simple_midi(estimate / "b.mid")
    config_path = _write_cli_config(tmp_path, dataset="mini")
    output = tmp_path / "results.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "evaluate", "--config", str(config_path),
            "--reference", str(reference), "--estimate", str(estimate),
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["file"] for row in rows] == ["b.mid"]


def test_transcribe_no_dataset_anywhere_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    config_path = _write_cli_config(tmp_path)

    result = CliRunner().invoke(app, ["transcribe", "--config", str(config_path)])

    assert result.exit_code == 1
    assert (
        "--audio is required when no dataset is set (--dataset or io.dataset)"
        in _squash(result.output)
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            [],
            "--reference is required when no dataset is set "
            "(--dataset or io.dataset)",
        ),
        (
            ["--reference", "REF"],
            "--estimate is required when no dataset is set "
            "(--dataset or io.dataset)",
        ),
    ],
)
def test_evaluate_no_dataset_anywhere_errors(
    extra: list[str],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fresh_console,
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    config_path = _write_cli_config(tmp_path)
    args = [str(tmp_path) if a == "REF" else a for a in extra]

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config_path), *args])

    assert result.exit_code == 1
    assert message in _squash(result.output)


@pytest.mark.parametrize("command", ["render", "transcribe", "evaluate", "benchmark"])
def test_invalid_config_prints_error_without_traceback(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fresh_console,
) -> None:
    import yaml
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    config_path = _write_cli_config(tmp_path, dataset="mini")
    payload = yaml.safe_load(config_path.read_text())
    payload["io"]["sample"] = 2
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = CliRunner().invoke(app, [command, "--config", str(config_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    # Rich may fold a long tmp path mid-word, so compare with whitespace removed.
    stderr = re.sub(r"\s+", "", result.stderr)
    assert f"error:invalidconfig{config_path}:" in stderr
    assert "io.samplemustbeamappinglike{n:2,seed:0},ornull(got2)" in stderr
    assert "Traceback" not in result.output
    assert not (tmp_path / "corpus").exists()


@pytest.mark.parametrize("command", ["render", "transcribe", "evaluate", "benchmark"])
def test_missing_config_prints_error_without_traceback(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fresh_console,
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "nope.yaml"

    result = CliRunner().invoke(app, [command, "--config", str(missing)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"error:Confignotfound:{missing}" in re.sub(r"\s+", "", result.stderr)
    assert "Traceback" not in result.output


def test_benchmark_prints_failure_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from sonitra.benchmark import runner as runner_module
    from sonitra.benchmark.results import BenchmarkRecord

    monkeypatch.chdir(tmp_path)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")

    def _record(condition: str, name: str, error: str | None) -> BenchmarkRecord:
        return BenchmarkRecord(
            condition=condition,
            transcriber="oracle",
            midi_path=f"{name}.mid",
            audio_path=f"{name}.wav",
            status="failed" if error else "succeeded",
            error=error,
        )

    records = [
        _record("baseline", "a", "CUDA_ERROR_NOT_INITIALIZED"),
        _record("baseline", "b", "CUDA_ERROR_NOT_INITIALIZED"),
        _record("noisy", "c", "CUDA_ERROR_NOT_INITIALIZED"),
        _record("noisy", "d", "model file missing"),
        _record("noisy", "e", None),
    ]

    def _fake_run_benchmark(midi_paths, work_dir, config, *args, **kwargs):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=records,
            summary=[],
            degradation=[],
            results_path=Path(work_dir) / "results.jsonl",
            summary_path=Path(work_dir) / "summary.json",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)

    result = _run_benchmark_cli(config_path)

    text = _squash(result.output)
    assert result.exit_code == 1, result.output
    assert "Benchmark failures" in text
    assert "CUDA_ERROR_NOT_INITIALIZED" in text
    assert "model file missing" in text
    # the most frequent error comes first, with its count
    assert text.index("CUDA_ERROR_NOT_INITIALIZED") < text.index("model file missing")
    assert re.search(r"CUDA_ERROR_NOT_INITIALIZED\s*│?\s*3\b", text)


def test_benchmark_failure_summary_caps_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from sonitra.benchmark import runner as runner_module
    from sonitra.benchmark.results import BenchmarkRecord

    monkeypatch.chdir(tmp_path)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")
    records = [
        BenchmarkRecord(
            condition="baseline",
            transcriber="oracle",
            midi_path=f"{i}.mid",
            audio_path=f"{i}.wav",
            status="failed",
            error=f"error-{i}",
        )
        for i in range(7)
    ]

    def _fake_run_benchmark(midi_paths, work_dir, config, *args, **kwargs):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=records,
            summary=[],
            degradation=[],
            results_path=Path(work_dir) / "results.jsonl",
            summary_path=Path(work_dir) / "summary.json",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)

    result = _run_benchmark_cli(config_path)

    text = _squash(result.output)
    assert result.exit_code == 1
    assert "2 more distinct errors" in text
    assert "error-5" not in text


# ---------------------------------------------------------------------------
# Preflight / backend-unavailable errors exit cleanly (no traceback)
# ---------------------------------------------------------------------------

_PREFLIGHT_MESSAGE = (
    "transkun: transkun backend unavailable: no module named 'torch'. "
    "Install with `uv sync --extra transkun` [gpu]"
)


def test_benchmark_preflight_error_exits_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.benchmark import runner as runner_module
    from sonitra.cli import app
    from sonitra.transcribe.base import TranscriptionError

    monkeypatch.chdir(tmp_path)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")

    def _raise(*_args, **_kwargs):
        raise TranscriptionError(_PREFLIGHT_MESSAGE)

    monkeypatch.setattr(runner_module, "run_benchmark", _raise)

    result = CliRunner().invoke(app, ["benchmark", "--config", str(config_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "error:" in result.stderr
    assert "no module named 'torch'" in _squash(result.stderr)
    assert "`uv sync --extra transkun` [gpu]" in _squash(result.stderr)
    assert "Traceback" not in result.output
    assert "Traceback" not in result.stderr


def test_benchmark_preflight_real_path_exits_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    """The real ``_preflight_devices`` error is caught, not only a stubbed one."""
    from typer.testing import CliRunner

    from sonitra.benchmark import runner as runner_module
    from sonitra.cli import app
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.protocol import make_transcriber

    monkeypatch.chdir(tmp_path)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")

    real_factory = make_transcriber

    def _factory(cfg):
        backend = real_factory(cfg)

        def _validate() -> None:
            raise TranscriptionError("device 'cuda' unavailable")

        backend.validate_device = _validate
        return backend

    monkeypatch.setattr(runner_module, "make_transcriber", _factory)

    result = CliRunner().invoke(app, ["benchmark", "--config", str(config_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "error:" in result.stderr
    assert "device 'cuda' unavailable" in _squash(result.stderr)
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# Numeric settings come from the config, never from a process-wide export
# ---------------------------------------------------------------------------


def test_benchmark_cli_does_not_export_numeric_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    monkeypatch.chdir(tmp_path)
    _capture_run_benchmark(monkeypatch)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")

    result = _run_benchmark_cli(config_path)

    assert result.exit_code == 0, result.output
    assert "SONITRA_NUMERIC_MODE" not in os.environ
    assert "SONITRA_GPU_MEMORY_GROWTH" not in os.environ


def test_transcribe_cli_uses_config_numeric_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml
    from typer.testing import CliRunner

    from sonitra.cli import app
    from sonitra.transcribe import protocol
    from sonitra.transcribe.numerics import read_numeric_env

    payload = yaml.safe_load(
        _MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path / "oracle"))
    )
    payload["transcription"]["numeric_mode"] = "strict"
    payload["transcription"]["gpu_memory_growth"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    (tmp_path / "test_c4.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    monkeypatch.setenv("SONITRA_NUMERIC_MODE", "off")
    monkeypatch.setenv("SONITRA_GPU_MEMORY_GROWTH", "0")
    recorded: list[tuple[str, bool]] = []

    def factory(cfg):
        recorded.append(read_numeric_env())
        return _StubTranscriber(None)

    monkeypatch.setattr(protocol, "make_transcriber", factory)

    result = CliRunner().invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--config", str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded == [("strict", True)]


# ---------------------------------------------------------------------------
# A strict numeric failure aborts the transcribe run instead of failing files
# ---------------------------------------------------------------------------


class _StrictNumericStub:
    """Transcriber stub that rejects the run's numeric settings outright."""

    name = "stub"

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls = 0

    def apply_numeric_settings(self) -> tuple[str, ...]:
        raise NumericSettingsError("boom")

    def transcribe(self, audio_path):
        self.calls += 1
        if self.error is not None and self.calls >= 2:
            raise self.error
        return TranscriptionResult(notes=[], transcriber=self.name)


def test_transcribe_strict_failure_exits_1_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app
    from sonitra.transcribe import protocol

    (tmp_path / "test_c4.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))
    monkeypatch.setattr(protocol, "make_transcriber", lambda cfg: _StrictNumericStub())

    result = CliRunner().invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--config", str(config_path),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "boom" in _squash(result.stderr)
    assert not list((tmp_path / "out").rglob("*.mid"))


def test_transcribe_strict_failure_mid_run_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from sonitra.cli import app
    from sonitra.transcribe import protocol

    (tmp_path / "piece_0.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    (tmp_path / "piece_1.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG_TEMPLATE.format(midi_dir=str(tmp_path)))
    monkeypatch.setattr(
        protocol,
        "make_transcriber",
        lambda cfg: _StrictNumericStub(NumericSettingsError("boom")),
    )

    result = CliRunner().invoke(
        app,
        [
            "transcribe",
            "--audio", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--config", str(config_path),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "boom" in _squash(result.stderr)
    # A fatal numeric failure is not a per-file failure: the run must not report
    # an "N ok, M failed" tally for it.
    assert "failed" not in _squash(result.output)


def test_benchmark_strict_failure_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fresh_console
) -> None:
    from typer.testing import CliRunner

    from sonitra.benchmark import runner as runner_module
    from sonitra.cli import app

    monkeypatch.chdir(tmp_path)
    _seed_midi(tmp_path / "corpus" / "mini" / "midi")
    config_path = _write_cli_config(tmp_path, dataset="mini")
    monkeypatch.setattr(
        runner_module, "make_transcriber", lambda cfg: _StrictNumericStub()
    )

    result = CliRunner().invoke(app, ["benchmark", "--config", str(config_path)])

    assert result.exit_code == 1, result.output
    assert "boom" in _squash(result.stderr)
    assert "Traceback" not in result.output
