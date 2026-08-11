from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sonitra.benchmark import runner as runner_module
from sonitra.benchmark.runner import run_benchmark
from sonitra.config import PipelineConfig
from sonitra.midi_reader import parse_midi
from sonitra.separation.protocol import register_separator
from sonitra.storage import write_wav
from sonitra.transcribe.base import TranscriptionResult


@pytest.fixture
def benchmark_config(corpus_dir: Path) -> PipelineConfig:
    # Use FluidSynth with the system SoundFont so audio is non-silent.
    # The precomputed transcriber returns the original corpus MIDI,
    # so symbolic metrics must come out perfect.
    return PipelineConfig.model_validate(
        {
            "pipeline": {
                "synth_backend": "fluidsynth",
                "effects_chain": "pedalboard",
                "bpm": 120,
                "sample_rate": 22050,
                "bit_depth": 16,
                "channels": 1,
                "duration_padding_sec": 0.5,
                "overwrite": True,
                "resume": False,
                "max_workers": 1,
                "log_level": "INFO",
            },
            "io": {
                "corpus_root": str(corpus_dir),
                "output_format": "wav",
                "mp3_bitrate_kbps": 192,
                "file_naming": "{stem}",
            },
            "fluidsynth": {"soundfont_path": "/usr/share/sounds/sf2/default-GM.sf2"},
            "transcription": {
                "transcribers": [
                    {"type": "precomputed", "midi_dir": str(corpus_dir), "name": "oracle"}
                ]
            },
            "benchmark": {
                "sweeps": [
                    {"parameter": "pipeline.duration_padding_sec", "values": [1.0], "name": "padding"}
                ]
            },
        }
    )


def test_full_benchmark_run(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    midi_paths = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]
    result = run_benchmark(midi_paths, tmp_path, benchmark_config)

    # 2 conditions (baseline + padding=1.0) x 1 transcriber x 2 files
    assert len(result.records) == 4
    assert all(record.status == "succeeded" for record in result.records)
    for record in result.records:
        assert record.metrics["note.onset_f1"] == pytest.approx(1.0)
        assert record.metrics["frame.f1"] == pytest.approx(1.0)

    assert {record.condition for record in result.records} == {"baseline", "padding=1.0"}
    assert all(record.transcriber == "oracle" for record in result.records)

    # artefacts on disk
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert (tmp_path / "audio" / "baseline" / "test_c4.wav").exists()
    assert (tmp_path / "transcriptions" / "baseline" / "oracle" / "test_c4.mid").exists()

    payload = json.loads(result.summary_path.read_text())
    assert len(payload["summary"]) == 2
    assert len(payload["degradation"]) == 1
    assert payload["degradation"][0]["delta_note.onset_f1"] == pytest.approx(0.0)


def test_benchmark_with_dtw_resynthesis(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.evaluation.dtw.enabled = True
    benchmark_config.benchmark.sweeps = []
    result = run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)
    assert len(result.records) == 1
    assert "dtw.distance" in result.records[0].metrics


def test_benchmark_records_render_failures(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    missing = tmp_path / "missing.mid"
    result = run_benchmark(
        [midi_fixture("test_c4.mid"), missing], tmp_path, benchmark_config
    )
    by_midi = {record.midi_path: record for record in result.records}
    assert by_midi[str(missing)].status == "render_failed"
    assert by_midi[str(midi_fixture("test_c4.mid"))].status == "succeeded"


# ── Raw-outputs sidecar wiring (stubbed transcriber) ─────────────────

class _StubTranscriber:
    """Minimal transcriber stub returning reference notes + fixed raw outputs."""

    name = "oracle"

    def __init__(self, notes, raw_outputs):
        self._notes = notes
        self._raw = raw_outputs

    def transcribe(self, audio_path):
        return TranscriptionResult(
            notes=self._notes, transcriber=self.name, raw_outputs=self._raw
        )


def test_benchmark_writes_raw_outputs_sidecar(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid")]

    raw_outputs = {
        "onset": np.zeros((2, 88)),
        "contour": np.zeros((2, 264)),
        "note": np.zeros((2, 88)),
    }
    monkeypatch.setattr(
        runner_module,
        "make_transcriber",
        lambda cfg: _StubTranscriber(
            parse_midi(midi_fixture("test_c4.mid")), raw_outputs
        ),
    )

    result = run_benchmark(midi_paths, tmp_path, benchmark_config)

    assert result.records[0].status == "succeeded"
    assert (tmp_path / "transcriptions" / "baseline" / "oracle" / "test_c4.mid").exists()
    assert (
        tmp_path / "transcriptions" / "baseline" / "oracle" / "test_c4.model_outputs.csv"
    ).exists()


def test_benchmark_sidecar_failure_does_not_fail_evaluation(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_config.benchmark.sweeps = []

    import sonitra.midi_writer

    def _boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        sonitra.midi_writer, "write_raw_outputs", _boom, raising=False
    )

    raw_outputs = {
        "onset": np.zeros((2, 88)),
        "contour": np.zeros((2, 264)),
        "note": np.zeros((2, 88)),
    }
    monkeypatch.setattr(
        runner_module,
        "make_transcriber",
        lambda cfg: _StubTranscriber(
            parse_midi(midi_fixture("test_c4.mid")), raw_outputs
        ),
    )

    result = run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)

    record = result.records[0]
    assert record.status == "succeeded"
    assert record.metrics["note.onset_f1"] == pytest.approx(1.0)

    lines = result.results_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "succeeded"


def test_benchmark_requires_transcribers(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.transcription.transcribers = []
    with pytest.raises(ValueError, match="No enabled transcribers"):
        run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)


def test_benchmark_skips_disabled_transcribers(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.transcription.transcribers[0].enabled = False
    with pytest.raises(ValueError, match="No enabled transcribers"):
        run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)


def test_save_audio_true_keeps_condition_audio_dir(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    assert benchmark_config.benchmark.save_audio is True
    run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)
    assert (tmp_path / "audio" / "baseline" / "test_c4.wav").exists()


def test_save_audio_false_removes_condition_audio_dir(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    benchmark_config.benchmark.save_audio = False
    result = run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)

    assert not (tmp_path / "audio" / "baseline").exists()
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert (tmp_path / "transcriptions" / "baseline" / "oracle" / "test_c4.mid").exists()
    assert len(result.records) == 1
    assert result.records[0].status == "succeeded"
    assert result.records[0].metrics["note.onset_f1"] == pytest.approx(1.0)


class _FakeStemSeparator:
    name = "fake_stems"

    def separate(self, audio_path: Path, output_dir: Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem_path = output_dir / Path(audio_path).name
        write_wav(np.zeros((1, 100), dtype=np.float32), stem_path, sample_rate=22050)
        return {"mix": stem_path}


register_separator("fake_stems")(lambda cfg: _FakeStemSeparator())


def test_save_audio_false_removes_stems_dir(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    benchmark_config.benchmark.save_audio = False
    benchmark_config.separation.enabled = True
    benchmark_config.separation.backend = "fake_stems"

    result = run_benchmark([midi_fixture("test_c4.mid")], tmp_path, benchmark_config)

    assert not (tmp_path / "audio" / "baseline").exists()
    assert not (tmp_path / "stems" / "baseline").exists()
    assert result.records[0].status == "succeeded"


def test_save_audio_false_cleans_up_after_partial_failure(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    benchmark_config.benchmark.save_audio = False
    missing = tmp_path / "missing.mid"

    result = run_benchmark(
        [midi_fixture("test_c4.mid"), missing], tmp_path, benchmark_config
    )

    by_midi = {record.midi_path: record for record in result.records}
    assert by_midi[str(missing)].status == "render_failed"
    assert by_midi[str(midi_fixture("test_c4.mid"))].status == "succeeded"
    assert not (tmp_path / "audio" / "baseline").exists()


def test_resume_skips_completed_condition(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_config.benchmark.sweeps = []
    benchmark_config.benchmark.resume = True
    midi_paths = [midi_fixture("test_c4.mid")]

    first = run_benchmark(midi_paths, tmp_path, benchmark_config)
    assert len(first.records) == 1

    calls: list = []
    original = runner_module._evaluate_one
    monkeypatch.setattr(
        runner_module,
        "_evaluate_one",
        lambda *a, **kw: calls.append(a) or original(*a, **kw),
    )

    second = run_benchmark(midi_paths, tmp_path, benchmark_config)

    assert calls == []
    assert len(second.records) == 1
    assert second.records[0].status == "succeeded"


def test_resume_recomputes_missing_records_only(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]

    first = run_benchmark(midi_paths, tmp_path, benchmark_config)
    assert len(first.records) == 2

    # Simulate a crash before the second file's record was ever written.
    target = str(midi_fixture("test_polyphonic.mid"))
    lines = first.results_path.read_text().splitlines()
    kept = [line for line in lines if json.loads(line)["midi_path"] != target]
    assert len(kept) == 1
    first.results_path.write_text("\n".join(kept) + "\n")

    benchmark_config.benchmark.resume = True
    calls: list = []
    original = runner_module._evaluate_one
    monkeypatch.setattr(
        runner_module,
        "_evaluate_one",
        lambda *a, **kw: calls.append(a) or original(*a, **kw),
    )

    second = run_benchmark(midi_paths, tmp_path, benchmark_config)

    assert len(calls) == 1
    assert len(second.records) == 2
    by_midi = {record.midi_path: record for record in second.records}
    assert by_midi[target].status == "succeeded"


def test_resume_skips_fully_completed_condition_without_rendering(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_config.benchmark.sweeps = []
    benchmark_config.benchmark.save_audio = False
    midi_paths = [midi_fixture("test_c4.mid")]

    run_benchmark(midi_paths, tmp_path, benchmark_config)
    assert not (tmp_path / "audio" / "baseline").exists()

    benchmark_config.benchmark.resume = True
    calls: list = []
    original = runner_module.run_pipeline
    monkeypatch.setattr(
        runner_module,
        "run_pipeline",
        lambda *a, **kw: calls.append(a) or original(*a, **kw),
    )

    second = run_benchmark(midi_paths, tmp_path, benchmark_config)

    assert calls == []
    assert not (tmp_path / "audio" / "baseline").exists()
    assert len(second.records) == 1


def test_resume_treats_recorded_failures_as_done(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]

    first = run_benchmark(midi_paths, tmp_path, benchmark_config)
    target = str(midi_fixture("test_c4.mid"))

    lines = first.results_path.read_text().splitlines()
    rewritten = []
    for line in lines:
        record = json.loads(line)
        if record["midi_path"] == target:
            record["status"] = "failed"
            record["error"] = "synthetic failure"
        rewritten.append(json.dumps(record))
    first.results_path.write_text("\n".join(rewritten) + "\n")

    benchmark_config.benchmark.resume = True
    calls: list = []
    original = runner_module._evaluate_one
    monkeypatch.setattr(
        runner_module,
        "_evaluate_one",
        lambda *a, **kw: calls.append(a) or original(*a, **kw),
    )

    second = run_benchmark(midi_paths, tmp_path, benchmark_config)

    assert calls == []
    by_midi = {record.midi_path: record for record in second.records}
    assert by_midi[target].status == "failed"
    assert by_midi[target].error == "synthetic failure"


def test_resume_rejects_config_fingerprint_mismatch(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid")]
    run_benchmark(midi_paths, tmp_path, benchmark_config)

    benchmark_config.benchmark.resume = True
    benchmark_config.evaluation.note_metrics.onset_tolerance_sec = 0.5

    with pytest.raises(ValueError, match="fingerprint|config"):
        run_benchmark(midi_paths, tmp_path, benchmark_config)


def test_resume_false_starts_clean_in_reused_work_dir(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid")]

    run_benchmark(midi_paths, tmp_path, benchmark_config)
    second = run_benchmark(midi_paths, tmp_path, benchmark_config)

    lines = second.results_path.read_text().splitlines()
    assert len(lines) == 1
    assert len(second.records) == 1


def test_render_failed_write_guard_in_parallel_path(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.sweeps = []
    benchmark_config.benchmark.max_workers = 2
    missing = tmp_path / "missing.mid"

    result = run_benchmark(
        [midi_fixture("test_c4.mid"), missing], tmp_path, benchmark_config
    )

    by_midi = {record.midi_path: record for record in result.records}
    assert by_midi[str(missing)].status == "render_failed"
    assert by_midi[str(midi_fixture("test_c4.mid"))].status == "succeeded"
