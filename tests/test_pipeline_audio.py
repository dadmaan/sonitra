from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pedalboard
import pytest

from sonitra.config import PipelineConfig, load_config
from sonitra.pipeline import run_pipeline
from sonitra.storage import read_audio, write_wav


def _tone(sample_rate: int = 44100, duration: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.25 * np.sin(2 * np.pi * freq * t)
    return np.stack([signal, signal])


def _write_tone_wav(path: Path, *, sample_rate: int = 44100) -> Path:
    # normalize=False: keep the tone's own (well below full-scale) peak so
    # the default quality-gate clip_threshold doesn't trip on test fixtures.
    return write_wav(_tone(sample_rate=sample_rate), path, sample_rate=sample_rate, normalize=False)


def _write_crest_wav(
    path: Path, *, crest_db: float = 25.0, sample_rate: int = 44100
) -> Path:
    """A quiet two-channel tone whose peak sits *crest_db* above its RMS."""
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    sine = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    audio = np.stack([sine, sine]).astype(np.float64)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = rms * 10.0 ** (crest_db / 20.0)
    audio[0, 0] = peak
    audio[1, 0] = -peak
    return write_wav(
        audio.astype(np.float32), path, sample_rate=sample_rate, normalize=False
    )


def _rms_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    return 20.0 * np.log10(float(np.sqrt(np.mean(arr**2))))


def _audio_config(
    tmp_path: Path,
    *,
    effects_chain: str = "none",
    pedalboard_effects: list[dict] | None = None,
    normalisation: dict | None = None,
    quality_gates: dict | None = None,
    manifest: bool = False,
    overwrite: bool = True,
    output_format: str = "wav",
) -> PipelineConfig:
    data: Dict[str, Any] = {
        "render_pipeline": {
            "synth_backend": "fluidsynth",
            "effects_chain": effects_chain,
            "input_type": "audio",
            "bpm": 120,
            "sample_rate": 44100,
            "bit_depth": 24,
            "channels": 2,
            "duration_padding_sec": 2.0,
            "overwrite": overwrite,
            "resume": True,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": {
            "corpus_root": str(tmp_path),
            "output_format": output_format,
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
        "quality_gates": quality_gates
        or {
            "silence_threshold_rms": 0.001,
            "min_duration_sec": 0.05,
            "max_duration_deviation_sec": 5.0,
            "clip_threshold": 0.999,
        },
    }
    if pedalboard_effects is not None:
        data["pedalboard"] = {"effects": pedalboard_effects}
    if normalisation is not None:
        data["normalisation"] = normalisation
    if manifest:
        data["observability"] = {
            "write_manifest": True,
            "manifest_path": str(tmp_path / "renders.jsonl"),
        }
    return PipelineConfig.model_validate(data)


# ── Synth bypass & sample rate ──────────────────────────────────────────


def test_audio_input_skips_synth(tmp_path, monkeypatch):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(tmp_path)

    def _boom(*_args, **_kwargs):
        raise AssertionError("make_synth must not be called in audio mode")

    monkeypatch.setattr("sonitra.source.make_synth", _boom)

    result = run_pipeline([wav], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    assert result.failed == 0


def test_audio_input_uses_source_sample_rate_not_config_rate(tmp_path, off_rate_wav):
    cfg = _audio_config(tmp_path)  # pipeline.sample_rate stays 44100
    out_dir = tmp_path / "out"

    result = run_pipeline([off_rate_wav], out_dir, config=cfg)

    assert result.succeeded == 1
    entry = result.log[0]
    output_path = Path(entry["output"])
    _audio, sr = read_audio(output_path)
    assert sr == 48000
    assert entry["quality_flags"]["duration_sec"] == pytest.approx(1.0, rel=1e-2)


# ── Effects chain ────────────────────────────────────────────────────────


def test_audio_input_applies_effects_chain(tmp_path, monkeypatch):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(
        tmp_path,
        effects_chain="pedalboard",
        pedalboard_effects=[
            {
                "type": "Gain",
                "gain_db": 3.0,
                "enabled": True,
            }
        ],
    )
    call_srs = []
    orig_call = pedalboard.Pedalboard.__call__
    monkeypatch.setattr(
        pedalboard.Pedalboard,
        "__call__",
        lambda self, audio, sr: call_srs.append(sr) or orig_call(self, audio, sr),
    )

    out_dir = tmp_path / "out"
    result = run_pipeline([wav], out_dir, config=cfg)

    assert result.succeeded == 1
    assert call_srs == [44100]
    assert len(list(out_dir.glob("*.wav"))) == 1


def test_audio_input_effects_chain_none_skips_chain(tmp_path, monkeypatch):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(
        tmp_path,
        effects_chain="none",
        pedalboard_effects=[{"type": "Gain", "gain_db": 3.0, "enabled": True}],
    )
    calls = []
    monkeypatch.setattr(
        pedalboard.Pedalboard,
        "__call__",
        lambda self, *a, **kw: calls.append(True),
    )

    result = run_pipeline([wav], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    assert calls == []


# ── Normalisation ─────────────────────────────────────────────────────────


def test_audio_input_applies_normalisation(tmp_path):
    quiet = 0.1 * _tone()
    quiet_path = write_wav(quiet, tmp_path / "quiet.wav", sample_rate=44100, normalize=False)
    cfg = _audio_config(
        tmp_path,
        normalisation={
            "enabled": True,
            "placement": "post",
            "post": {"mode": "peak", "target_db": -1.0},
        },
    )

    out_dir = tmp_path / "out"
    result = run_pipeline([quiet_path], out_dir, config=cfg)

    assert result.succeeded == 1
    output_path = Path(result.log[0]["output"])
    audio, _sr = read_audio(output_path)
    target_linear = 10 ** (-1.0 / 20.0)
    assert np.max(np.abs(audio)) == pytest.approx(target_linear, rel=0.05)
    assert result.log[0]["quality_flags"]["guard_engaged"] is False


def test_audio_input_pre_stage_guard_records_diagnostics(tmp_path):
    wav = _write_crest_wav(tmp_path / "crest.wav")
    cfg = _audio_config(
        tmp_path,
        normalisation={
            "enabled": True,
            "placement": "pre",
            "pre": {"mode": "rms", "target_db": -18.0},
        },
        manifest=True,
    )

    out_dir = tmp_path / "out"
    result = run_pipeline([wav], out_dir, config=cfg)

    assert result.succeeded == 1
    flags = result.log[0]["quality_flags"]
    assert flags["guard_engaged"] is True
    assert flags["guard_gain_db"] < 0.0
    assert flags["peak_before_guard"] > 1.0
    assert flags["pre_gain_db"] is not None
    assert flags["post_gain_db"] is None
    written, _sr = read_audio(Path(result.log[0]["output"]))
    ceiling = 10 ** (-1.0 / 20.0)
    assert np.max(np.abs(written)) <= ceiling * (1.0 + 1e-4)

    manifest_line = json.loads((tmp_path / "renders.jsonl").read_text().strip())
    assert manifest_line["quality_flags"]["guard_engaged"] is True


def test_audio_input_post_stage_guard_replaces_hard_clip(tmp_path):
    wav = _write_crest_wav(tmp_path / "crest.wav")
    cfg = _audio_config(
        tmp_path,
        normalisation={
            "enabled": True,
            "placement": "post",
            "post": {"mode": "rms", "target_db": -18.0},
        },
    )

    out_dir = tmp_path / "out"
    result = run_pipeline([wav], out_dir, config=cfg)

    assert result.succeeded == 1
    flags = result.log[0]["quality_flags"]
    assert flags["guard_engaged"] is True
    assert flags["peak_before_guard"] > 1.0
    written, _sr = read_audio(Path(result.log[0]["output"]))
    assert not np.any(np.abs(written) >= 1.0)
    assert np.max(np.abs(written)) == pytest.approx(10 ** (-1.0 / 20.0), abs=1e-4)


def test_audio_input_gain_cut_kept_with_pre_placement(tmp_path):
    wav = _write_crest_wav(tmp_path / "crest.wav")
    effects = [{"type": "Gain", "gain_db": -20.0, "enabled": True}]

    pre_cfg = _audio_config(
        tmp_path,
        effects_chain="pedalboard",
        pedalboard_effects=effects,
        normalisation={
            "enabled": True,
            "placement": "pre",
            "pre": {"mode": "rms", "target_db": -18.0},
        },
    )
    result_pre = run_pipeline([wav], tmp_path / "pre_out", config=pre_cfg)

    assert result_pre.succeeded == 1
    audio_pre, _sr = read_audio(Path(result_pre.log[0]["output"]))
    assert _rms_db(audio_pre) == pytest.approx(-38.0, abs=0.1)

    both_cfg = _audio_config(
        tmp_path,
        effects_chain="pedalboard",
        pedalboard_effects=effects,
        normalisation={
            "enabled": True,
            "placement": "both",
            "pre": {"mode": "rms", "target_db": -18.0},
            "post": {"mode": "rms", "target_db": -33.0},
        },
    )
    result_both = run_pipeline([wav], tmp_path / "both_out", config=both_cfg)

    assert result_both.succeeded == 1
    audio_both, _sr = read_audio(Path(result_both.log[0]["output"]))
    assert _rms_db(audio_both) == pytest.approx(-33.0, abs=0.1)
    assert _rms_db(audio_both) == pytest.approx(_rms_db(audio_pre) + 5.0, abs=0.1)


def test_quality_flags_carry_normalisation_keys_on_success_and_gate_failure(
    tmp_path, silent_wav
):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(
        tmp_path,
        normalisation={"enabled": True, "placement": "both"},
    )

    result = run_pipeline([wav, silent_wav], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    assert result.failed == 1
    flags_by_status = {entry["status"]: entry["quality_flags"] for entry in result.log}
    expected_keys = {
        "pre_gain_db",
        "post_gain_db",
        "peak_before_guard",
        "guard_gain_db",
        "guard_engaged",
        "duration_expected_sec",
        "duration_mismatch",
    }
    for flags in flags_by_status.values():
        assert expected_keys <= set(flags)
    assert flags_by_status["succeeded"]["duration_expected_sec"] is None
    assert flags_by_status["succeeded"]["duration_mismatch"] is False
    assert flags_by_status["failed"]["duration_expected_sec"] is None
    assert flags_by_status["failed"]["duration_mismatch"] is False


def test_tuning_offset_and_distortion_accept_samples_above_full_scale(
    tmp_path, monkeypatch
):
    from sonitra.effects.chain_builder import TuningOffsetStage

    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(
        tmp_path,
        effects_chain="pedalboard",
        pedalboard_effects=[
            {"type": "TuningOffset", "cents": 10.0, "enabled": True},
            {"type": "Distortion", "drive_db": 10.0, "enabled": True},
        ],
        normalisation={
            "enabled": True,
            "placement": "pre",
            "pre": {"mode": "peak", "target_db": 20.0 * np.log10(3.0)},
            "peak_ceiling_db": None,
        },
        quality_gates={
            "silence_threshold_rms": 0.001,
            "min_duration_sec": 0.05,
            "max_duration_deviation_sec": 5.0,
            "clip_threshold": 100.0,
        },
    )

    tuning_peaks: list[float] = []
    original_tuning_call = TuningOffsetStage.__call__

    def _spy_tuning(self, audio, sample_rate):
        tuning_peaks.append(float(np.max(np.abs(audio))))
        return original_tuning_call(self, audio, sample_rate)

    pedalboard_peaks: list[float] = []
    original_pedalboard_call = pedalboard.Pedalboard.__call__

    def _spy_pedalboard(self, audio, sample_rate):
        pedalboard_peaks.append(float(np.max(np.abs(audio))))
        return original_pedalboard_call(self, audio, sample_rate)

    monkeypatch.setattr(TuningOffsetStage, "__call__", _spy_tuning)
    monkeypatch.setattr(pedalboard.Pedalboard, "__call__", _spy_pedalboard)

    result = run_pipeline([wav], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    assert tuning_peaks and tuning_peaks[0] > 1.0
    assert pedalboard_peaks and pedalboard_peaks[0] > 1.0
    written, _sr = read_audio(Path(result.log[0]["output"]))
    assert np.all(np.isfinite(written))


# ── Quality gate ────────────────────────────────────────────────────────


def test_audio_input_quality_gate_failure_recorded(tmp_path, silent_wav):
    cfg = _audio_config(tmp_path)

    result = run_pipeline([silent_wav], tmp_path / "out", config=cfg)

    assert result.failed == 1
    assert result.succeeded == 0
    assert result.log[0]["quality_flags"]["is_silent"] is True
    assert result.log[0]["status"] == "failed"


# ── Fail-soft behaviour ───────────────────────────────────────────────────


def test_audio_input_missing_file_fails_softly(tmp_path):
    good = _write_tone_wav(tmp_path / "good.wav")
    missing = tmp_path / "missing.wav"
    cfg = _audio_config(tmp_path)

    result = run_pipeline([good, missing], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    assert result.failed == 1
    paths = {entry["midi"] for entry in result.log}
    assert paths == {str(good), str(missing)}


# ── Manifest ───────────────────────────────────────────────────────────────


def test_audio_input_manifest_records_source_path(tmp_path):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(tmp_path, manifest=True)

    result = run_pipeline([wav], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    lines = (tmp_path / "renders.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source_path"] == str(wav)
    assert entry["midi_path"] == ""
    assert entry["status"] == "done"


# ── Output paths ─────────────────────────────────────────────────────────


def test_audio_input_output_avoids_source_recordings(tmp_path, audio_corpus_dir):
    recordings_dir = audio_corpus_dir / "recordings"
    before = sorted(p.name for p in recordings_dir.iterdir())
    wavs = sorted(recordings_dir.glob("*.wav"))
    cfg = _audio_config(tmp_path)

    out_dir = tmp_path / "work_audio"
    result = run_pipeline(wavs, out_dir, config=cfg)

    assert result.succeeded == len(wavs)
    after = sorted(p.name for p in recordings_dir.iterdir())
    assert before == after
    for wav in wavs:
        assert (out_dir / f"{wav.stem}.wav").exists()


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3"])
def test_audio_input_output_format_respected(tmp_path, fmt):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(tmp_path, output_format=fmt)

    out_dir = tmp_path / "out"
    result = run_pipeline([wav], out_dir, config=cfg)

    assert result.succeeded == 1
    outputs = list(out_dir.glob(f"*.{fmt}"))
    assert len(outputs) == 1


def test_audio_input_skip_existing_output(tmp_path):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(tmp_path, overwrite=True)
    out_dir = tmp_path / "out"
    run_pipeline([wav], out_dir, config=cfg)

    cfg2 = _audio_config(tmp_path, overwrite=False)
    result = run_pipeline([wav], out_dir, config=cfg2)

    assert result.skipped == 1
    assert result.succeeded == 0


# ── Config regression tie-in ────────────────────────────────────────────


def test_audio_input_fluidsynth_config_without_soundfont_runs(tmp_path):
    wav = _write_tone_wav(tmp_path / "in.wav")
    cfg = _audio_config(tmp_path)  # fluidsynth backend, no fluidsynth section at all

    result = run_pipeline([wav], tmp_path / "out", config=cfg)

    assert result.succeeded == 1
    assert result.failed == 0


# ── Callback ────────────────────────────────────────────────────────────


def test_audio_input_callback_reports_each_file(tmp_path, audio_corpus_dir):
    wavs = sorted((audio_corpus_dir / "recordings").glob("*.wav"))
    cfg = _audio_config(tmp_path)

    entries: list[dict] = []
    result = run_pipeline(wavs, tmp_path / "out", config=cfg, on_file_done=entries.append)

    assert result.succeeded == len(wavs)
    assert len(entries) == len(wavs)
    assert entries == result.log
    assert {entry["midi"] for entry in entries} == {str(w) for w in wavs}


# ── Full MIDI regression (spot check; the real gate is running the
# existing test_pipeline.py / test_pipeline_config.py suites unmodified) ──


def test_midi_input_unaffected_by_source_refactor(tmp_path, midi_fixture, config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)
    assert result.succeeded == 1
    assert result.failed == 0
    entry = result.log[0]
    assert entry["midi"] == str(midi_fixture("test_c4.mid"))
    output_path = Path(entry["output"])
    _audio, sr = read_audio(output_path)
    assert sr == cfg.render_pipeline.sample_rate
