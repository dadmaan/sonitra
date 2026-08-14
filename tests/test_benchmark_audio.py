from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sonitra.benchmark import runner as runner_module
from sonitra.benchmark.runner import run_benchmark
from sonitra.config import PipelineConfig
from sonitra.corpus import discover_audio_files
from sonitra.storage import write_wav

# ── Fixtures ──────────────────────────────────────────────────────────────
#
# Builds on the Phase-3 `audio_corpus_dir` fixture (tests/conftest.py), which
# already lays out:
#   midi/piece_0.mid, midi/piece_1.mid
#   recordings/piece_0_performer1.wav, recordings/piece_1_performer1.wav
# (piece_N_performerX pairs to piece_N.mid via the §2.3 token-prefix rule.)
#
# The `precomputed` transcriber looks up its own midi_dir *by audio stem*
# (PrecomputedTranscriber.transcribe), which is orthogonal to the §2.3
# pairing scheme (that pairs by reference filename tokens). So every
# recording used with the precomputed transcriber needs its own
# `<recording-stem>.mid` seeded into a dedicated lookup dir, distinct from
# the pairing reference in `midi/`.


@pytest.fixture
def precomputed_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "precomputed"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def audio_benchmark_config(tmp_path: Path, precomputed_dir: Path) -> PipelineConfig:
    # fluidsynth backend, no fluidsynth section: valid in audio mode (P1) --
    # the synth is never constructed for audio-mode input.
    return PipelineConfig.model_validate(
        {
            "render_pipeline": {
                "synth_backend": "fluidsynth",
                "effects_chain": "none",
                "input_type": "audio",
                "bpm": 120,
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2,
                "duration_padding_sec": 0.5,
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
            "transcription": {
                "transcribers": [
                    {"type": "precomputed", "midi_dir": str(precomputed_dir), "name": "oracle"}
                ]
            },
            "benchmark": {},
        }
    )


def _seed_precomputed(precomputed_dir: Path, audio_path: Path, reference_midi: Path) -> None:
    """Seed a precomputed-lookup MIDI, keyed by *audio_path*'s own stem."""
    shutil.copy(reference_midi, precomputed_dir / f"{audio_path.stem}.mid")


def _add_recording(
    recordings_dir: Path, name: str, *, freq: float = 440.0, sample_rate: int = 44100
) -> Path:
    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.25 * np.sin(2 * np.pi * freq * t)
    tone = np.stack([signal, signal])
    path = recordings_dir / name
    write_wav(tone, path, sample_rate=sample_rate, normalize=False)
    return path


class _RecordingProgress:
    """Minimal BenchmarkProgress fake recording every callback (mirrors
    tests/test_benchmark_runner.py's fixture of the same shape)."""

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


# ── Basic success path ───────────────────────────────────────────────────


def test_benchmark_audio_mode_succeeds_vs_reference_midi(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    perf2 = _add_recording(recordings_dir, "piece_0_performer2.wav", freq=523.25)

    _seed_precomputed(precomputed_dir, perf1, reference)
    _seed_precomputed(precomputed_dir, perf2, reference)

    result = run_benchmark(
        [reference], tmp_path / "work", audio_benchmark_config, audio_paths=[perf1, perf2]
    )

    assert len(result.records) == 2
    assert all(record.status == "succeeded" for record in result.records)
    for record in result.records:
        assert record.metrics["note.onset_f1"] == pytest.approx(1.0)
        assert record.midi_path == str(reference)
        assert record.audio_path.startswith(str(tmp_path / "work" / "audio"))
    assert {record.source_path for record in result.records} == {str(perf1), str(perf2)}


def test_benchmark_audio_mode_distinct_transcription_outputs(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    perf2 = _add_recording(recordings_dir, "piece_0_performer2.wav", freq=523.25)

    _seed_precomputed(precomputed_dir, perf1, reference)
    _seed_precomputed(precomputed_dir, perf2, reference)

    work_dir = tmp_path / "work"
    result = run_benchmark(
        [reference], work_dir, audio_benchmark_config, audio_paths=[perf1, perf2]
    )

    assert len(result.records) == 2
    out1 = work_dir / "transcriptions" / "baseline" / "oracle" / "piece_0_performer1.mid"
    out2 = work_dir / "transcriptions" / "baseline" / "oracle" / "piece_0_performer2.mid"
    assert out1.exists()
    assert out2.exists()
    # not both landing on the reference stem (the collision bug this phase fixes)
    collision = work_dir / "transcriptions" / "baseline" / "oracle" / "piece_0.mid"
    assert not collision.exists()


# ── DTW gating ───────────────────────────────────────────────────────────


def test_benchmark_audio_mode_skips_dtw(
    audio_benchmark_config: PipelineConfig,
    audio_corpus_dir: Path,
    precomputed_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_benchmark_config.evaluation.dtw.enabled = True
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    _seed_precomputed(precomputed_dir, perf1, reference)

    calls: list = []
    monkeypatch.setattr(
        runner_module,
        "_audio_metric_values",
        lambda *a, **kw: calls.append(a) or {},
    )

    result = run_benchmark(
        [reference], tmp_path / "work", audio_benchmark_config, audio_paths=[perf1]
    )

    assert len(result.records) == 1
    assert result.records[0].status == "succeeded"
    assert calls == []
    assert not any(key.startswith("dtw.") for key in result.records[0].metrics)


# ── Pairing edge cases ────────────────────────────────────────────────────


def test_benchmark_audio_mode_unpaired_recording_ignored(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    ref0 = midi_dir / "piece_0.mid"
    ref1 = midi_dir / "piece_1.mid"
    perf0 = recordings_dir / "piece_0_performer1.wav"
    perf1 = recordings_dir / "piece_1_performer1.wav"
    unrelated = _add_recording(recordings_dir, "unrelated_track.wav", freq=200.0)

    _seed_precomputed(precomputed_dir, perf0, ref0)
    _seed_precomputed(precomputed_dir, perf1, ref1)

    work_dir = tmp_path / "work"
    result = run_benchmark(
        [ref0, ref1],
        work_dir,
        audio_benchmark_config,
        audio_paths=[perf0, perf1, unrelated],
    )

    assert len(result.records) == 2
    assert str(unrelated) not in {record.source_path for record in result.records}
    # not rendered
    assert not (work_dir / "audio" / "baseline" / "unrelated_track.wav").exists()


def test_benchmark_audio_mode_missing_audio_skips_reference(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    ref0 = midi_dir / "piece_0.mid"
    ref1 = midi_dir / "piece_1.mid"
    perf0 = recordings_dir / "piece_0_performer1.wav"
    _seed_precomputed(precomputed_dir, perf0, ref0)

    result = run_benchmark(
        [ref0, ref1], tmp_path / "work", audio_benchmark_config, audio_paths=[perf0]
    )

    assert len(result.records) == 1
    assert result.records[0].midi_path == str(ref0)


def test_benchmark_audio_mode_reference_parse_failure(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    ref0 = midi_dir / "piece_0.mid"
    ref1 = midi_dir / "piece_1.mid"
    perf0 = recordings_dir / "piece_0_performer1.wav"
    perf1 = recordings_dir / "piece_1_performer1.wav"
    _seed_precomputed(precomputed_dir, perf0, ref0)
    _seed_precomputed(precomputed_dir, perf1, ref1)

    ref0.write_bytes(b"not a midi file")

    result = run_benchmark(
        [ref0, ref1], tmp_path / "work", audio_benchmark_config, audio_paths=[perf0, perf1]
    )

    assert len(result.records) == 2
    by_midi = {record.midi_path: record for record in result.records}
    assert by_midi[str(ref0)].status in {"failed", "render_failed"}
    assert by_midi[str(ref0)].error
    assert by_midi[str(ref1)].status == "succeeded"


# ── Resume ────────────────────────────────────────────────────────────────


def test_benchmark_audio_mode_resume_keyed_on_source_path(
    audio_benchmark_config: PipelineConfig,
    audio_corpus_dir: Path,
    precomputed_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_benchmark_config.benchmark.resume = True
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    perf2 = _add_recording(recordings_dir, "piece_0_performer2.wav", freq=523.25)
    _seed_precomputed(precomputed_dir, perf1, reference)
    _seed_precomputed(precomputed_dir, perf2, reference)

    work_dir = tmp_path / "work"
    first = run_benchmark(
        [reference], work_dir, audio_benchmark_config, audio_paths=[perf1, perf2]
    )
    assert len(first.records) == 2

    calls: list = []
    original = runner_module._evaluate_one
    monkeypatch.setattr(
        runner_module,
        "_evaluate_one",
        lambda *a, **kw: calls.append(a) or original(*a, **kw),
    )

    second = run_benchmark(
        [reference], work_dir, audio_benchmark_config, audio_paths=[perf1, perf2]
    )

    assert calls == []
    assert len(second.records) == 2
    assert {record.source_path for record in second.records} == {str(perf1), str(perf2)}


def test_benchmark_audio_mode_resume_fingerprint_flip(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    from sonitra.config import InputType

    audio_benchmark_config.benchmark.resume = True
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    _seed_precomputed(precomputed_dir, perf1, reference)

    work_dir = tmp_path / "work"
    run_benchmark([reference], work_dir, audio_benchmark_config, audio_paths=[perf1])

    audio_benchmark_config.render_pipeline.input_type = InputType.MIDI
    with pytest.raises(ValueError, match="fingerprint|config"):
        run_benchmark([reference], work_dir, audio_benchmark_config, audio_paths=[perf1])


# ── save_audio cleanup ────────────────────────────────────────────────────


def test_benchmark_audio_mode_save_audio_false_cleanup(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    audio_benchmark_config.benchmark.save_audio = False
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    _seed_precomputed(precomputed_dir, perf1, reference)

    before = sorted(p.name for p in recordings_dir.iterdir())
    work_dir = tmp_path / "work"
    result = run_benchmark(
        [reference], work_dir, audio_benchmark_config, audio_paths=[perf1]
    )

    assert not (work_dir / "audio" / "baseline").exists()
    assert sorted(p.name for p in recordings_dir.iterdir()) == before
    assert result.results_path.exists()
    assert result.summary_path.exists()


# ── Progress / event counting ─────────────────────────────────────────────


def test_benchmark_audio_mode_progress_event_counts(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    perf2 = _add_recording(recordings_dir, "piece_0_performer2.wav", freq=523.25)
    _seed_precomputed(precomputed_dir, perf1, reference)
    _seed_precomputed(precomputed_dir, perf2, reference)

    progress = _RecordingProgress()
    result = run_benchmark(
        [reference],
        tmp_path / "work",
        audio_benchmark_config,
        audio_paths=[perf1, perf2],
        progress=progress,
    )

    assert len(result.records) == 2
    # total_files reported against len(audio_paths) (2 recordings), not
    # len(midi_paths) (1 reference)
    for _, _, total_files, _ in progress.started:
        assert total_files == 2

    start_events = [e for e in progress.worker_events if e.status == "start"]
    done_events = [e for e in progress.worker_events if e.status == "done"]
    assert len(start_events) == 2
    assert len(done_events) == 2
    render_events = [
        e for e in progress.worker_events if e.status == "stage" and e.stage == "render"
    ]
    assert len(render_events) == 1


# ── Condition overrides re-render shared recordings ───────────────────────


def test_benchmark_audio_mode_condition_overrides(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    data = audio_benchmark_config.model_dump(mode="python")
    data["render_pipeline"]["effects_chain"] = "pedalboard"
    data["pedalboard"] = {"effects": [{"type": "Gain", "gain_db": 0.0, "enabled": True}]}
    data["benchmark"]["sweeps"] = [
        {"parameter": "pedalboard.effects.0.gain_db", "values": [3.0], "name": "gain"}
    ]
    audio_benchmark_config = PipelineConfig.model_validate(data)
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference = midi_dir / "piece_0.mid"
    perf1 = recordings_dir / "piece_0_performer1.wav"
    _seed_precomputed(precomputed_dir, perf1, reference)

    result = run_benchmark(
        [reference], tmp_path / "work", audio_benchmark_config, audio_paths=[perf1]
    )

    assert {record.condition for record in result.records} == {"baseline", "gain=3.0"}
    assert len(result.records) == 2
    assert all(record.source_path == str(perf1) for record in result.records)


# ── Parallel parity ────────────────────────────────────────────────────────


def test_benchmark_audio_mode_parallel(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, precomputed_dir: Path, tmp_path: Path
) -> None:
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    ref0 = midi_dir / "piece_0.mid"
    ref1 = midi_dir / "piece_1.mid"
    perf0 = recordings_dir / "piece_0_performer1.wav"
    perf1 = recordings_dir / "piece_1_performer1.wav"
    _seed_precomputed(precomputed_dir, perf0, ref0)
    _seed_precomputed(precomputed_dir, perf1, ref1)

    audio_benchmark_config.benchmark.max_workers = 2
    result = run_benchmark(
        [ref0, ref1],
        tmp_path / "work",
        audio_benchmark_config,
        audio_paths=[perf0, perf1],
    )

    assert len(result.records) == 2
    assert all(record.status == "succeeded" for record in result.records)
    assert {record.source_path for record in result.records} == {str(perf0), str(perf1)}


# ── Required audio_paths ────────────────────────────────────────────────────


def test_benchmark_audio_mode_requires_audio_paths(
    audio_benchmark_config: PipelineConfig, audio_corpus_dir: Path, tmp_path: Path
) -> None:
    reference = audio_corpus_dir / "midi" / "piece_0.mid"

    with pytest.raises(ValueError, match="audio_paths"):
        run_benchmark([reference], tmp_path / "work", audio_benchmark_config)
