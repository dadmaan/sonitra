from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import pytest

from sonitra.cli import init
from sonitra.config import RenderingMode, load_config
from sonitra.pipeline import run_pipeline


def test_init_writes_working_basic_pitch_config(tmp_path: Path) -> None:
    path = tmp_path / "init.yaml"
    init(path)
    cfg = load_config(path)
    assert cfg.pipeline.rendering_mode == RenderingMode.DAWDREAMER_ONLY
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
pipeline:
  rendering_mode: dawdreamer_synth_pedalboard_fx
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
  enabled: true
  block_size: 512
  plugin_path: null
  preset_path: null
  soundfont_path: null
  bpm: 120
  faust_code: null
  clear_midi_between_renders: true
pedalboard:
  enabled: true
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
pipeline:
  rendering_mode: dawdreamer_synth_pedalboard_fx
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
  enabled: true
  block_size: 512
  plugin_path: null
  preset_path: null
  soundfont_path: null
  bpm: 120
  faust_code: null
  clear_midi_between_renders: true
pedalboard:
  enabled: true
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
