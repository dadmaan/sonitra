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

    # summary/degradation rows are in declared condition order (baseline, padding=1.0)
    assert [row["condition"] for row in result.summary] == ["baseline", "padding=1.0"]
    assert [row["condition"] for row in result.degradation] == ["padding=1.0"]


class _RecordingProgress:
    """Minimal BenchmarkProgress fake recording every callback."""

    def __init__(self) -> None:
        self.started: list[tuple[str, dict, int, list[str]]] = []
        self.worker_events: list = []
        self.conditions_done: list[str] = []

    def on_condition_started(
        self, condition_name: str, overrides: dict, total_files: int, transcriber_names: list[str]
    ) -> None:
        self.started.append(
            (condition_name, dict(overrides), total_files, list(transcriber_names))
        )

    def on_worker_event(self, event) -> None:
        self.worker_events.append(event)

    def on_condition_done(self, condition_name: str) -> None:
        self.conditions_done.append(condition_name)


def test_benchmark_reports_progress(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    midi_paths = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]
    progress = _RecordingProgress()

    result = run_benchmark(midi_paths, tmp_path, benchmark_config, progress=progress)

    # 2 conditions (baseline + padding=1.0) x 1 transcriber x 2 files
    assert len(result.records) == 4

    # one on_condition_started per condition, with the right metadata
    assert {name for name, _, _, _ in progress.started} == {"baseline", "padding=1.0"}
    overrides_by_name = {name: overrides for name, overrides, _, _ in progress.started}
    assert overrides_by_name["baseline"] == {}
    assert overrides_by_name["padding=1.0"] == {"pipeline.duration_padding_sec": 1.0}
    for _, _, total_files, transcriber_names in progress.started:
        assert total_files == len(midi_paths)
        assert transcriber_names == ["oracle"]

    # one "start" + one "done" event per (condition, file, transcriber) record,
    # with matching fields and ok reflecting success
    record_keys = {(r.condition, r.midi_path, r.transcriber) for r in result.records}
    start_keys = {
        (e.condition, e.midi_path, e.transcriber)
        for e in progress.worker_events
        if e.status == "start"
    }
    done_keys = {
        (e.condition, e.midi_path, e.transcriber)
        for e in progress.worker_events
        if e.status == "done"
    }
    assert start_keys == record_keys
    assert done_keys == record_keys
    assert all(e.ok for e in progress.worker_events if e.status == "done")
    # every "start" event now carries stage="transcribe"
    assert all(
        e.stage == "transcribe" for e in progress.worker_events if e.status == "start"
    )

    # exactly one stage(render) event per condition actually processed, with
    # no transcriber/midi_path context yet (render runs once for the whole
    # condition, before any per-file work)
    render_events = [
        e for e in progress.worker_events if e.status == "stage" and e.stage == "render"
    ]
    assert len(render_events) == len(progress.started)
    assert all(e.transcriber == "" and e.midi_path == "" for e in render_events)

    # per-outcome event-count formula (not a flat multiplier): one
    # stage(render) per condition, plus one start+done pair per (condition,
    # file, transcriber) that isn't render_failed, plus a single done for any
    # that is. No separation is enabled in this fixture, so no
    # stage(separate) events.
    expected_transcribe_events = sum(
        1 if record.status == "render_failed" else 2 for record in result.records
    )
    assert len(progress.worker_events) == len(render_events) + expected_transcribe_events

    # one on_condition_done per condition, in the same order they started
    assert len(progress.conditions_done) == 2
    assert progress.conditions_done == [name for name, _, _, _ in progress.started]
    assert {name for name in progress.conditions_done} == {"baseline", "padding=1.0"}


def test_benchmark_reports_stage_events_in_serial_order(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    """Serial mode, one condition, no separation: the exact per-condition
    event sequence must be stage(render) -> start(f1) -> done(f1) ->
    start(f2) -> done(f2), in that order."""
    benchmark_config.benchmark.sweeps = []
    file1 = midi_fixture("test_c4.mid")
    file2 = midi_fixture("test_polyphonic.mid")
    progress = _RecordingProgress()

    result = run_benchmark([file1, file2], tmp_path, benchmark_config, progress=progress)

    assert len(result.records) == 2
    events = progress.worker_events
    assert [(e.status, e.stage) for e in events] == [
        ("stage", "render"),
        ("start", "transcribe"),
        ("done", ""),
        ("start", "transcribe"),
        ("done", ""),
    ]
    assert events[0].condition == "baseline"
    assert events[0].transcriber == ""
    assert events[0].midi_path == ""
    assert [e.midi_path for e in events[1:3]] == [str(file1), str(file1)]
    assert [e.midi_path for e in events[3:5]] == [str(file2), str(file2)]


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


class _NoisyTranscriber:
    """Transcriber stub that prints directly to stdout, like basic-pitch's
    bare ``print()`` call in ``predict()`` — used to verify serial-mode
    benchmark runs contain that output instead of corrupting an active Rich
    Live display."""

    name = "oracle"

    def __init__(self, notes) -> None:
        self._notes = notes

    def transcribe(self, audio_path):
        print("NOISY-TRANSCRIBER-OUTPUT-MARKER")
        return TranscriptionResult(notes=self._notes, transcriber=self.name)


def test_serial_mode_contains_backend_stdout_when_progress_active(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid")]
    monkeypatch.setattr(
        runner_module,
        "make_transcriber",
        lambda cfg: _NoisyTranscriber(parse_midi(midi_fixture("test_c4.mid"))),
    )

    result = run_benchmark(
        midi_paths, tmp_path, benchmark_config, progress=_RecordingProgress()
    )

    assert result.records[0].status == "succeeded"
    captured = capsys.readouterr()
    assert "NOISY-TRANSCRIBER-OUTPUT-MARKER" not in captured.out
    log_path = tmp_path / "logs" / "serial.log"
    assert log_path.exists()
    assert "NOISY-TRANSCRIBER-OUTPUT-MARKER" in log_path.read_text()


def test_serial_mode_does_not_suppress_stdout_without_progress(
    benchmark_config: PipelineConfig,
    midi_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_config.benchmark.sweeps = []
    midi_paths = [midi_fixture("test_c4.mid")]
    monkeypatch.setattr(
        runner_module,
        "make_transcriber",
        lambda cfg: _NoisyTranscriber(parse_midi(midi_fixture("test_c4.mid"))),
    )

    result = run_benchmark(midi_paths, tmp_path, benchmark_config, progress=None)

    assert result.records[0].status == "succeeded"
    captured = capsys.readouterr()
    assert "NOISY-TRANSCRIBER-OUTPUT-MARKER" in captured.out


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


def test_benchmark_reports_separate_stage_event(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    """Separation enabled: exactly one stage(separate) event per file (shared
    across that file's transcribers, since separator.separate() itself runs
    once per file), with the real midi_path and no transcriber context yet."""
    benchmark_config.benchmark.sweeps = []
    benchmark_config.separation.enabled = True
    benchmark_config.separation.backend = "fake_stems"
    file1 = midi_fixture("test_c4.mid")
    progress = _RecordingProgress()

    result = run_benchmark([file1], tmp_path, benchmark_config, progress=progress)

    assert len(result.records) == 1
    separate_events = [
        e for e in progress.worker_events if e.status == "stage" and e.stage == "separate"
    ]
    assert len(separate_events) == 1
    assert separate_events[0].midi_path == str(file1)
    assert separate_events[0].transcriber == ""

    assert [(e.status, e.stage) for e in progress.worker_events] == [
        ("stage", "render"),
        ("stage", "separate"),
        ("start", "transcribe"),
        ("done", ""),
    ]


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


def test_benchmark_reports_progress_in_parallel_path(
    benchmark_config: PipelineConfig, midi_fixture, tmp_path: Path
) -> None:
    benchmark_config.benchmark.max_workers = 2
    midi_paths = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]
    progress = _RecordingProgress()

    result = run_benchmark(midi_paths, tmp_path, benchmark_config, progress=progress)

    assert len(result.records) == 4
    assert {name for name, _, _, _ in progress.started} == {"baseline", "padding=1.0"}
    assert len(progress.conditions_done) == 2
    assert {name for name in progress.conditions_done} == {"baseline", "padding=1.0"}

    # events flow through the real queue + drainer thread; order is
    # nondeterministic, so compare by counts/sets, not order
    record_keys = {(r.condition, r.midi_path, r.transcriber) for r in result.records}
    start_keys = {
        (e.condition, e.midi_path, e.transcriber)
        for e in progress.worker_events
        if e.status == "start"
    }
    done_keys = {
        (e.condition, e.midi_path, e.transcriber)
        for e in progress.worker_events
        if e.status == "done"
    }
    assert start_keys == record_keys
    assert done_keys == record_keys
    assert all(e.ok for e in progress.worker_events if e.status == "done")
    assert all(
        e.stage == "transcribe" for e in progress.worker_events if e.status == "start"
    )

    # exactly one stage(render) event per condition, order-agnostic across
    # conditions (different conditions' events can interleave across
    # processes, so this stays count/set-based, not order-based)
    render_events = [
        e for e in progress.worker_events if e.status == "stage" and e.stage == "render"
    ]
    assert len(render_events) == len(progress.started)
    assert all(e.transcriber == "" and e.midi_path == "" for e in render_events)

    # per-outcome event-count formula (§ A6): render events + start/done
    # pairs (or a lone done for render_failed records); no separation here.
    expected_transcribe_events = sum(
        1 if record.status == "render_failed" else 2 for record in result.records
    )
    assert len(progress.worker_events) == len(render_events) + expected_transcribe_events

    # worker subprocesses wrote per-worker log files (may be empty, must exist)
    assert list((tmp_path / "logs").glob("worker-*.log"))

    # summary/degradation rows are in declared condition order regardless of
    # which subprocess happened to finish first
    assert [row["condition"] for row in result.summary] == ["baseline", "padding=1.0"]
    assert [row["condition"] for row in result.degradation] == ["padding=1.0"]
