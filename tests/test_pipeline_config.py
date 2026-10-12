import json
from pathlib import Path

import numpy as np
import pedalboard
import pytest
import soundfile as sf

from sonitra.config import (
    NormalisationSection,
    NormalisationStage,
    SynthBackend,
    load_config,
)
from sonitra.pipeline import run_pipeline
from sonitra.synth.dawdreamer_synth import DawDreamerSynth
from sonitra.synth.pedalboard_synth import PedalboardSynth


# ── Mode routing ─────────────────────────────────────────────────────

def test_pipeline_uses_dawdreamer_synth_in_dawdreamer_only_mode(
    monkeypatch, tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.render_pipeline.synth_backend = SynthBackend.DAWDREAMER_FAUST
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

def test_placement_stages_run_around_the_chain(
    monkeypatch, tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.normalisation = NormalisationSection(
        enabled=True,
        placement="both",
        pre=NormalisationStage(mode="rms", target_db=-18.0),
        post=NormalisationStage(mode="rms", target_db=-33.0),
    )
    call_order = []

    import sonitra.pipeline as pipeline_module

    original_normalise = pipeline_module.normalise_from_config

    def _spy_normalise(audio, cfg_arg, *, stage, call_log=None):
        call_order.append(stage)
        return original_normalise(audio, cfg_arg, stage=stage, call_log=call_log)

    monkeypatch.setattr(pipeline_module, "normalise_from_config", _spy_normalise)
    monkeypatch.setattr(
        "pedalboard.Pedalboard.__call__",
        lambda self, a, sr: call_order.append("fx") or a,
    )
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)

    assert call_order == ["pre", "fx", "post"]


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
    assert line["synth_backend"] == "dawdreamer_faust"


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
    cfg.render_pipeline.synth_backend = SynthBackend.DAWDREAMER_FAUST
    cfg.render_pipeline.max_workers = 8
    worker_counts = []

    monkeypatch.setattr(
        "sonitra.pipeline._get_worker_count",
        lambda cfg: worker_counts.append(cfg.render_pipeline.max_workers) or cfg.render_pipeline.max_workers,
    )
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert all(w == 1 for w in worker_counts)


# ── Peak normalisation regression ─────────────────────────────────────

def test_peak_normalised_config_stays_below_the_clip_threshold(
    midi_fixture, config_fixture, tmp_path: Path
) -> None:
    cfg = load_config(config_fixture("config_valid.yaml"))
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert result.succeeded == 1
    succeeded = [entry for entry in result.log if entry["status"] == "succeeded"]
    assert len(succeeded) == 1
    flags = succeeded[0]["quality_flags"]
    assert flags["is_clipped"] is False
    assert flags["peak"] < cfg.quality_gates.clip_threshold
    assert flags["peak_before_guard"] == pytest.approx(
        10 ** (cfg.normalisation.peak_ceiling_db / 20.0), abs=1e-4
    )
    assert flags["guard_gain_db"] == 0.0
    assert flags["guard_engaged"] is False


# ── Vital VST3 end-to-end regression ──────────────────────────────────

def test_run_pipeline_dawdreamer_only_with_vital(
    vital_vst_path, midi_fixture, config_fixture, tmp_path
):
    cfg = load_config(config_fixture("config_dawdreamer_vital.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], out_dir=tmp_path, config=cfg)
    assert result.succeeded == 1
    assert result.failed == 0
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_run_pipeline_dawdreamer_vital_with_preset(
    vital_vst_path, midi_fixture, config_fixture, tmp_path
):
    cfg = load_config(config_fixture("config_dawdreamer_vital_preset.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], out_dir=tmp_path, config=cfg)
    assert result.succeeded == 1
    assert result.failed == 0
    assert len(list(tmp_path.glob("*.wav"))) == 1


@pytest.mark.slow
def test_run_pipeline_pedalboard_only_with_vital(
    vital_vst_path, midi_fixture, config_fixture, tmp_path
):
    """Prove that Pedalboard can render Vital VST3 instrument end-to-end."""
    cfg = load_config(config_fixture("config_pedalboard_vital.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], out_dir=tmp_path, config=cfg)
    assert result.succeeded == 1
    assert result.failed == 0
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    # Bonus: verify audio is non-silent
    audio, _ = sf.read(str(wavs[0]))
    assert np.max(np.abs(audio)) > 0.001


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_fixture_config_transcribes_with_basic_pitch(
    midi_fixture, config_fixture, tmp_path: Path
) -> None:
    pytest.importorskip("basic_pitch")
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig
    from sonitra.transcribe.protocol import make_transcriber

    cfg = load_config(config_fixture("config_valid.yaml"))
    audio_dir = tmp_path / "audio"
    run_pipeline([midi_fixture("test_c4.mid")], audio_dir, config=cfg)
    wav = next(audio_dir.glob("*.wav"))
    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    transcription = transcriber.transcribe(wav)
    assert len(transcription.notes) > 0
