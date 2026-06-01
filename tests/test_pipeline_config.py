import json

import numpy as np
import pedalboard
import pytest

from midi_renderer.config import RenderingMode, load_config
from midi_renderer.pipeline import run_pipeline
from midi_renderer.synth.dawdreamer_synth import DawDreamerSynth
from midi_renderer.synth.pedalboard_synth import PedalboardSynth


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
    monkeypatch.setattr("midi_renderer.normaliser.normalise", lambda a, **kw: call_order.append("norm") or a)
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
        "midi_renderer.synth.dawdreamer_synth.DawDreamerSynth.render",
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
        "midi_renderer.pipeline._get_worker_count",
        lambda cfg: worker_counts.append(cfg.pipeline.max_workers) or cfg.pipeline.max_workers,
    )
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert all(w == 1 for w in worker_counts)
