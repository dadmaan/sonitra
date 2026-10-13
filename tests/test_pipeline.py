import json
from pathlib import Path

import numpy as np
import pytest

from sonitra.config import PipelineConfig, load_config
from sonitra.pipeline import run_pipeline
from sonitra.storage import write_wav


def test_pipeline_processes_single_midi(tmp_path, midi_fixture, session_engine):
    result = run_pipeline(
        midi_paths=[midi_fixture("test_c4.mid")],
        out_dir=tmp_path,
        engine=session_engine,
    )
    assert result.succeeded == 1
    assert result.failed == 0


def test_pipeline_output_file_exists(tmp_path, midi_fixture, session_engine):
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_pipeline_batch_multiple_midis(tmp_path, midi_fixture, session_engine):
    midis = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]
    result = run_pipeline(midis, tmp_path, session_engine)
    assert result.succeeded == len(midis)


def test_pipeline_skips_existing_output(tmp_path, midi_fixture, session_engine):
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine, overwrite=False)
    assert result.skipped == 1


def test_pipeline_logs_failure_without_abort(tmp_path, session_engine, monkeypatch):
    def broken_render(*_args, **_kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("sonitra.renderer.render_notes_faust", broken_render)
    result = run_pipeline([Path("nonexistent.mid")], tmp_path, session_engine)
    assert result.failed == 1
    assert result.succeeded == 0


def test_pipeline_result_contains_timing(tmp_path, midi_fixture, session_engine):
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    assert result.elapsed_seconds > 0.0


def test_pipeline_result_is_serialisable(tmp_path, midi_fixture, session_engine):
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    import json

    json.dumps(result.to_dict())


def test_pipeline_config_path_reports_each_file(
    tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    midis = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]

    entries: list[dict] = []
    result = run_pipeline(midis, tmp_path, config=cfg, on_file_done=entries.append)

    assert result.succeeded == len(midis)
    assert result.failed == 0
    # one callback per input MIDI file, carrying the same per-file log entry
    assert len(entries) == len(midis)
    assert entries == result.log
    assert {entry["midi"] for entry in entries} == {str(m) for m in midis}
    assert {entry["status"] for entry in entries} == {"succeeded"}
    # per-file wall-clock timing is recorded on log entries and in the manifest
    assert all(entry["elapsed_seconds"] > 0.0 for entry in entries)
    lines = (tmp_path / "renders.jsonl").read_text().strip().split("\n")
    assert len(lines) == len(midis)
    for line in lines:
        manifest_entry = json.loads(line)
        assert manifest_entry["status"] == "done"
        assert manifest_entry["elapsed_seconds"] > 0.0


def test_pipeline_config_path_failed_entry_records_elapsed(
    tmp_path, midi_fixture, config_fixture, monkeypatch
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")

    class _BrokenSource:
        def load(self, *_args, **_kwargs):
            raise RuntimeError("simulated crash")

    monkeypatch.setattr("sonitra.pipeline.make_source", lambda _cfg: _BrokenSource())
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)

    assert result.failed == 1
    assert result.succeeded == 0
    entry = result.log[0]
    assert entry["status"] == "failed"
    assert entry["elapsed_seconds"] > 0.0
    manifest_entry = json.loads((tmp_path / "renders.jsonl").read_text().strip())
    assert manifest_entry["status"] == "failed"
    assert manifest_entry["elapsed_seconds"] > 0.0


def test_pipeline_config_path_skipped_entry_records_zero_elapsed(
    tmp_path, midi_fixture, config_fixture
):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.observability.manifest_path = str(tmp_path / "renders.jsonl")
    cfg.render_pipeline.overwrite = False
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)

    cfg.observability.manifest_path = str(tmp_path / "renders2.jsonl")
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, config=cfg)

    assert result.skipped == 1
    entry = result.log[0]
    assert entry["status"] == "skipped"
    assert entry["elapsed_seconds"] == 0.0


# ── Silence judged on the loaded audio ────────────────────────────────


class _StubSource:
    """Source returning one fixed array, so the test controls the input level."""

    def __init__(self, audio: np.ndarray, sample_rate: int = 44100) -> None:
        self._audio = audio
        self._sample_rate = sample_rate

    def load(self, _path):
        return self._audio, self._sample_rate

    def expected_duration_sec(self, _path):
        return None


def _normalising_cfg(
    tmp_path: Path,
    *,
    input_type: str = "midi",
    placement: str = "both",
    post_target_db: float = -33.0,
) -> PipelineConfig:
    normalisation = {
        "enabled": True,
        "placement": placement,
        "pre": {"mode": "peak", "target_db": -1.0},
        "peak_ceiling_db": -1.0,
    }
    if placement in ("post", "both"):
        normalisation["post"] = {"mode": "rms", "target_db": post_target_db}
    return PipelineConfig.model_validate(
        {
            "render_pipeline": {
                "synth_backend": "dawdreamer_faust",
                "effects_chain": "none",
                "input_type": input_type,
                "sample_rate": 44100,
                "bit_depth": 24,
                "channels": 2,
                "duration_padding_sec": 2.0,
                "overwrite": True,
                "resume": False,
                "max_workers": 1,
                "log_level": "INFO",
            },
            "io": {
                "corpus_root": str(tmp_path),
                "output_format": "wav",
                "mp3_bitrate_kbps": 192,
                "file_naming": "{stem}",
            },
            "normalisation": normalisation,
            "observability": {
                "write_manifest": True,
                "manifest_path": str(tmp_path / "renders.jsonl"),
            },
        }
    )


def _noise_at(rms_db: float, *, seconds: float = 30.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((2, int(44100 * seconds))).astype(np.float32)
    noise *= 10 ** (rms_db / 20.0) / float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    return noise


def _tone_at(rms_db: float, *, seconds: float = 1.0) -> np.ndarray:
    sample_rate = 44100
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    amplitude = np.sqrt(2.0) * 10 ** (rms_db / 20.0)
    tone = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([tone, tone])


def _manifest_entry(tmp_path: Path) -> dict:
    lines = (tmp_path / "renders.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    return json.loads(lines[0])


def test_pipeline_input_silence_rejects_inaudible_noise(tmp_path, monkeypatch):
    noise = _noise_at(-90.0)
    cfg = _normalising_cfg(tmp_path)
    monkeypatch.setattr("sonitra.pipeline.make_source", lambda _cfg: _StubSource(noise))

    result = run_pipeline([tmp_path / "stub.mid"], tmp_path, config=cfg)

    entry = _manifest_entry(tmp_path)
    assert entry["status"] == "failed"
    assert result.succeeded == 0
    assert result.failed == 1
    flags = entry["quality_flags"]
    assert flags["is_silent"] is True
    assert flags["input_rms"] == pytest.approx(3.16e-5, rel=0.05)
    assert flags["pre_gain_db"] == pytest.approx(74.0, abs=2.0)


def test_pipeline_input_silence_allows_low_post_target(tmp_path, monkeypatch):
    tone = _tone_at(-25.0)
    cfg = _normalising_cfg(tmp_path, post_target_db=-63.0)
    monkeypatch.setattr("sonitra.pipeline.make_source", lambda _cfg: _StubSource(tone))

    result = run_pipeline([tmp_path / "stub.mid"], tmp_path, config=cfg)

    entry = _manifest_entry(tmp_path)
    assert entry["status"] == "done"
    assert result.succeeded == 1
    flags = entry["quality_flags"]
    assert flags["is_silent"] is False
    assert flags["input_rms"] == pytest.approx(10 ** (-25 / 20), rel=0.01)


def test_pipeline_input_silence_still_catches_zero_output(tmp_path, monkeypatch):
    zeros = np.zeros((2, 44100), dtype=np.float32)
    cfg = _normalising_cfg(tmp_path)
    monkeypatch.setattr("sonitra.pipeline.make_source", lambda _cfg: _StubSource(zeros))

    result = run_pipeline([tmp_path / "stub.mid"], tmp_path, config=cfg)

    entry = _manifest_entry(tmp_path)
    assert entry["status"] == "failed"
    assert result.failed == 1
    flags = entry["quality_flags"]
    assert flags["is_silent"] is True


def test_pipeline_input_silence_records_input_levels_pre_only(tmp_path, monkeypatch):
    tone = _tone_at(-25.0)
    cfg = _normalising_cfg(tmp_path, placement="pre")
    monkeypatch.setattr("sonitra.pipeline.make_source", lambda _cfg: _StubSource(tone))

    result = run_pipeline([tmp_path / "stub.mid"], tmp_path, config=cfg)

    entry = _manifest_entry(tmp_path)
    assert entry["status"] == "done"
    assert result.succeeded == 1
    flags = entry["quality_flags"]
    assert flags["is_silent"] is False
    assert flags["input_rms"] == pytest.approx(10 ** (-25 / 20), rel=0.01)
    assert flags["input_peak"] == pytest.approx(
        np.sqrt(2.0) * 10 ** (-25 / 20), rel=0.01
    )
    assert flags["post_gain_db"] is None


def test_pipeline_input_silence_audio_mode_inaudible_recording(tmp_path):
    rng = np.random.default_rng(1)
    inaudible = (rng.standard_normal((2, 44100)) * 10 ** (-90.0 / 20.0)).astype(
        np.float32
    )
    recording = write_wav(
        inaudible, tmp_path / "recording.wav", sample_rate=44100, normalize=False
    )
    cfg = _normalising_cfg(tmp_path, input_type="audio")

    result = run_pipeline([recording], tmp_path / "out", config=cfg)

    entry = _manifest_entry(tmp_path)
    assert entry["status"] == "failed"
    assert result.failed == 1
    flags = entry["quality_flags"]
    assert flags["is_silent"] is True
