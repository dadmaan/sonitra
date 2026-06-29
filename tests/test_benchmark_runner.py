from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonitra.benchmark.runner import run_benchmark
from sonitra.config import PipelineConfig


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
