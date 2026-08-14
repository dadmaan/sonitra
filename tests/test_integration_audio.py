"""Phase 7 — gated end-to-end integration tests for audio-input mode.

Exercises the audio-mode building blocks (``make_source``-backed
``run_pipeline``, a transcriber, and ``evaluate_notes``) wired together
directly, as an end user would, rather than through ``run_benchmark`` (which
already has its own dedicated coverage in ``tests/test_benchmark_audio.py``).

Marker note (PLAN.md §3 Phase 7): audio-mode tests need no
``skip_if_no_vst``/``integration`` marker -- no VST is involved anywhere in
this file. Only ``test_audio_mode_roundtrip_with_basic_pitch`` carries
``slow``, since it invokes the real basic-pitch/TensorFlow backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sonitra.config import PipelineConfig
from sonitra.evaluation.protocol import evaluate_notes, make_symbolic_metrics
from sonitra.evaluation.types import notes_from_dicts
from sonitra.midi_reader import parse_midi
from sonitra.midi_writer import write_transcription_outputs
from sonitra.pipeline import run_pipeline
from sonitra.storage import write_wav
from sonitra.transcribe.configs import PrecomputedTranscriberConfig
from sonitra.transcribe.protocol import make_transcriber


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


def _audio_pipeline_config(corpus_root: Path) -> PipelineConfig:
    # fluidsynth is a required enum value only -- the synth is never
    # constructed for `input_type: audio` (Phase 1), so no `fluidsynth:`
    # section is needed.
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
                "corpus_root": str(corpus_root),
                "output_format": "wav",
                "mp3_bitrate_kbps": 192,
                "file_naming": "{stem}",
            },
        }
    )


def test_benchmark_audio_mode_end_to_end_oracle(tmp_path: Path, audio_corpus_dir: Path) -> None:
    """Full audio corpus, with fan-out, through the real building blocks.

    ``run_pipeline`` (audio mode) renders every recording -> a precomputed
    ("oracle") transcriber returns a perfect transcription per recording ->
    ``evaluate_notes`` scores each against its paired reference MIDI. Two
    recordings (``piece_0_performer1``/``piece_0_performer2``) share the
    same reference (``piece_0.mid``), exercising the fan-out case; a third
    recording (``piece_1_performer1``) is paired to a distinct reference
    (``piece_1.mid``).
    """
    midi_dir = audio_corpus_dir / "midi"
    recordings_dir = audio_corpus_dir / "recordings"
    reference0 = midi_dir / "piece_0.mid"
    reference1 = midi_dir / "piece_1.mid"

    perf1 = recordings_dir / "piece_0_performer1.wav"
    perf2 = _add_recording(recordings_dir, "piece_0_performer2.wav", freq=523.25)
    perf3 = recordings_dir / "piece_1_performer1.wav"
    recordings = sorted([perf1, perf2, perf3])
    reference_by_recording = {perf1: reference0, perf2: reference0, perf3: reference1}

    cfg = _audio_pipeline_config(tmp_path)

    audio_out_dir = tmp_path / "rendered"
    render_result = run_pipeline(recordings, audio_out_dir, config=cfg)

    assert render_result.succeeded == len(recordings)
    assert render_result.failed == 0
    assert all(entry["status"] == "succeeded" for entry in render_result.log)

    # Seed a precomputed ("oracle") lookup dir: a perfect transcription for
    # each recording, keyed by the recording's own stem (orthogonal to the
    # §2.3 reference-pairing scheme -- see tests/test_benchmark_audio.py).
    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()
    import shutil

    for recording, reference in reference_by_recording.items():
        shutil.copy(reference, precomputed_dir / f"{recording.stem}.mid")

    transcriber = make_transcriber(PrecomputedTranscriberConfig(midi_dir=precomputed_dir))
    symbolic_metrics = make_symbolic_metrics(cfg.evaluation)

    transcription_out_dir = tmp_path / "transcriptions"
    transcription_out_dir.mkdir()
    output_paths: set[Path] = set()

    for recording in recordings:
        rendered_audio = audio_out_dir / f"{recording.stem}.wav"
        assert rendered_audio.exists()

        transcription = transcriber.transcribe(rendered_audio)
        estimate = notes_from_dicts(transcription.notes)

        reference_notes = notes_from_dicts(parse_midi(reference_by_recording[recording]))
        metrics = evaluate_notes(reference_notes, estimate, symbolic_metrics)
        assert metrics["note.onset_f1"] == pytest.approx(1.0)

        # Transcription output keyed by the *recording's* stem -- this is
        # the fix under test: fan-out recordings sharing one reference must
        # not collide on a single reference-stemmed output file.
        out_path = transcription_out_dir / f"{recording.stem}.mid"
        write_transcription_outputs(transcription, out_path)
        assert out_path.exists()
        output_paths.add(out_path)

    assert len(output_paths) == len(recordings)


@pytest.mark.slow
def test_audio_mode_roundtrip_with_basic_pitch(tmp_path: Path) -> None:
    """Real end-to-end roundtrip through the basic-pitch backend.

    Render a MIDI-derived tone WAV through the audio-mode pipeline, then
    transcribe the rendered output with the real basic-pitch model.
    """
    pytest.importorskip("basic_pitch")
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    tone = np.stack([signal, signal])
    source_wav = write_wav(
        tone, tmp_path / "source.wav", sample_rate=sample_rate, normalize=False
    )

    cfg = _audio_pipeline_config(tmp_path)
    audio_out_dir = tmp_path / "rendered"
    render_result = run_pipeline([source_wav], audio_out_dir, config=cfg)

    assert render_result.succeeded == 1
    rendered = audio_out_dir / f"{source_wav.stem}.wav"
    assert rendered.exists()

    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    transcription = transcriber.transcribe(rendered)

    assert len(transcription.notes) > 0
