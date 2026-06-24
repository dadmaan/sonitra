import json
from pathlib import Path

import numpy as np
import pedalboard
import pytest

from sonitra.config import RenderingMode, default_config_path, load_config
from sonitra.pipeline import run_pipeline
from sonitra.synth.dawdreamer_synth import DawDreamerSynth
from sonitra.synth.pedalboard_synth import PedalboardSynth


# ── Mode routing ─────────────────────────────────────────────────────

def test_pipeline_uses_dawdreamer_synth_in_dawdreamer_only_mode(
    monkeypatch, tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pipeline.rendering_mode = RenderingMode.DAWDREAMER_ONLY
    spy = []
    orig = DawDreamerSynth.render
    monkeypatch.setattr(DawDreamerSynth, "render", lambda self, *a, **kw: spy.append("dd") or orig(self, *a, **kw))
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert "dd" in spy


def test_pipeline_uses_pedalboard_synth_in_pedalboard_only_mode(
    monkeypatch, tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_pedalboard_only.yaml"))
    spy = []
    orig = PedalboardSynth.render
    monkeypatch.setattr(PedalboardSynth, "render", lambda self, *a, **kw: spy.append("pb") or orig(self, *a, **kw))
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert "pb" in spy


def test_hybrid_mode_applies_effects_chain(monkeypatch, tmp_path, midi_fixture, config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    chain_calls = []
    orig_call = pedalboard.Pedalboard.__call__
    monkeypatch.setattr(
        pedalboard.Pedalboard,
        "__call__",
        lambda self, *a, **kw: chain_calls.append(True) or orig_call(self, *a, **kw),
    )
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert len(chain_calls) >= 1


# ── Normalisation ordering ────────────────────────────────────────────

def test_pre_effects_normalisation_applied_before_chain(
    monkeypatch, tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.normalisation.pre_effects = True
    call_order = []
    monkeypatch.setattr("sonitra.normaliser.normalise", lambda a, **kw: call_order.append("norm") or a)
    monkeypatch.setattr(
        "pedalboard.Pedalboard.__call__",
        lambda self, a, sr: call_order.append("fx") or a,
    )
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    norm_idx = call_order.index("norm")
    fx_idx = call_order.index("fx")
    assert norm_idx < fx_idx


# ── Quality gate integration ──────────────────────────────────────────

def test_silent_render_marked_failed_in_result(monkeypatch, tmp_path, midi_fixture, config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    monkeypatch.setattr(
        "sonitra.synth.dawdreamer_synth.DawDreamerSynth.render",
        lambda *a, **kw: np.zeros((2, 44100), dtype=np.float32),
    )
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert result.failed == 1
    assert result.log[0]["quality_flags"]["is_silent"] is True


# ── Manifest output ───────────────────────────────────────────────────

def test_manifest_file_written_after_run(tmp_path, midi_fixture, config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert (tmp_path / "renders.jsonl").exists()


def test_manifest_contains_correct_mode(tmp_path, midi_fixture, config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    line = json.loads((tmp_path / "renders.jsonl").read_text().strip())
    assert line["rendering_mode"] == "dawdreamer_synth_pedalboard_fx"


# ── Output formats ────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3"])

def test_output_format_config_controls_extension(tmp_path, midi_fixture, config_fixture, fmt):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.io.output_format = fmt
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    outputs = list(tmp_path.glob(f"*.{fmt}"))
    assert len(outputs) == 1


# ── max_workers constraint ────────────────────────────────────────────

def test_dawdreamer_mode_uses_single_worker(monkeypatch, tmp_path, midi_fixture, config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pipeline.rendering_mode = RenderingMode.DAWDREAMER_ONLY
    cfg.pipeline.max_workers = 8
    worker_counts = []

    monkeypatch.setattr(
        "sonitra.pipeline._get_worker_count",
        lambda cfg: worker_counts.append(cfg.pipeline.max_workers) or cfg.pipeline.max_workers,
    )
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert all(w == 1 for w in worker_counts)


# ── Default config regression ─────────────────────────────────────────

def test_default_config_peak_below_clip_threshold(corpus_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(default_config_path())
    result = run_pipeline([corpus_dir / "test_c4.mid"], tmp_path, config=cfg)
    assert result.succeeded == 1
    succeeded = [entry for entry in result.log if entry["status"] == "succeeded"]
    assert len(succeeded) == 1
    flags = succeeded[0]["quality_flags"]
    assert flags["is_clipped"] is False
    assert flags["peak"] < 1.0


# ── Vital VST3 end-to-end regression ──────────────────────────────────

def test_run_pipeline_dawdreamer_only_with_vital(vital_vst_path, midi_fixture, tmp_path):
    cfg = load_config("config/dawdreamer_vital.yaml")
    cfg.io.output_dir = tmp_path
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], out_dir=tmp_path, config=cfg)
    assert result.succeeded == 1
    assert result.failed == 0
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_run_pipeline_dawdreamer_vital_with_preset(vital_vst_path, midi_fixture, tmp_path):
    cfg = load_config("config/dawdreamer_vital_goodies.yaml")
    cfg.io.output_dir = tmp_path
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], out_dir=tmp_path, config=cfg)
    assert result.succeeded == 1
    assert result.failed == 0
    assert len(list(tmp_path.glob("*.wav"))) == 1


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_default_config_transcribes_with_basic_pitch(corpus_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig
    from sonitra.transcribe.protocol import make_transcriber

    cfg = load_config(default_config_path())
    audio_dir = tmp_path / "audio"
    run_pipeline([corpus_dir / "test_c4.mid"], audio_dir, config=cfg)
    wav = next(audio_dir.glob("*.wav"))
    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    transcription = transcriber.transcribe(wav)
    assert len(transcription.notes) > 0
