from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sonitra.midi_reader import parse_midi
from sonitra.transcribe.base import TranscriptionError
from sonitra.transcribe.configs import (
    ExternalCommandTranscriberConfig,
    PrecomputedTranscriberConfig,
)
from sonitra.transcribe.protocol import make_transcriber
from sonitra.transcribe.external_command import ExternalCommandTranscriber
from sonitra.transcribe.precomputed import PrecomputedTranscriber


def test_precomputed_finds_midi_by_stem(midi_fixture, tmp_path: Path) -> None:
    fixtures_dir = midi_fixture("test_c4.mid").parent
    transcriber = PrecomputedTranscriber(midi_dir=fixtures_dir)
    result = transcriber.transcribe(tmp_path / "test_c4.wav")
    assert result.notes == parse_midi(midi_fixture("test_c4.mid"))
    assert result.transcriber == "precomputed"


def test_precomputed_missing_midi_raises(tmp_path: Path) -> None:
    transcriber = PrecomputedTranscriber(midi_dir=tmp_path)
    with pytest.raises(TranscriptionError, match="No precomputed MIDI"):
        transcriber.transcribe(tmp_path / "missing.wav")


def test_external_command_runs_tool(midi_fixture, tmp_path: Path) -> None:
    fixture = midi_fixture("test_c4.mid")
    script = tmp_path / "fake_amt.py"
    script.write_text(
        "import shutil, sys\n"
        f"shutil.copy({str(fixture)!r}, sys.argv[2])\n"
    )
    transcriber = ExternalCommandTranscriber(
        command=f"{sys.executable} {script} {{input}} {{output}}"
    )
    result = transcriber.transcribe(tmp_path / "test_c4.wav")
    assert result.notes == parse_midi(fixture)


def test_external_command_failure_raises(tmp_path: Path) -> None:
    transcriber = ExternalCommandTranscriber(
        command=f"{sys.executable} -c exit(3) {{input}} {{output}}"
    )
    with pytest.raises(TranscriptionError, match="failed"):
        transcriber.transcribe(tmp_path / "audio.wav")


def test_external_command_requires_placeholders() -> None:
    with pytest.raises(ValueError, match="placeholders"):
        ExternalCommandTranscriber(command="amt-tool run")


def test_external_command_no_output_raises(tmp_path: Path) -> None:
    transcriber = ExternalCommandTranscriber(
        command=f"{sys.executable} -c pass {{input}} {{output}}"
    )
    with pytest.raises(TranscriptionError, match="no output"):
        transcriber.transcribe(tmp_path / "audio.wav")


def test_factory_builds_from_config(tmp_path: Path) -> None:
    cfg = PrecomputedTranscriberConfig(midi_dir=tmp_path, name="klangio")
    transcriber = make_transcriber(cfg)
    assert isinstance(transcriber, PrecomputedTranscriber)
    assert transcriber.name == "klangio"


def test_factory_builds_external_command() -> None:
    cfg = ExternalCommandTranscriberConfig(command="tool {input} {output}")
    transcriber = make_transcriber(cfg)
    assert isinstance(transcriber, ExternalCommandTranscriber)


def test_basic_pitch_builder_defers_import() -> None:
    # building must not require basic-pitch; only transcribe() does
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    assert transcriber.name == "basic_pitch"
    if "basic_pitch" not in sys.modules:
        try:
            import basic_pitch  # noqa: F401
        except ImportError:
            with pytest.raises(TranscriptionError, match="not installed"):
                transcriber.transcribe("missing.wav")


def test_basic_pitch_docstring_reflects_core_dependency() -> None:
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    doc = BasicPitchTranscriber.__doc__ or ""
    assert "optional" not in doc.lower()
    assert "pip install sonitra" in doc


@pytest.mark.slow
def test_basic_pitch_transcribes_simple_sine_wave(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "a440.wav"
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))

    transcriber = BasicPitchTranscriber()
    result = transcriber.transcribe(audio_path)

    assert result.transcriber == "basic_pitch"
    assert len(result.notes) >= 1
    notes_near_440 = [n for n in result.notes if abs(n["pitch"] - 69) <= 1]
    assert notes_near_440, f"expected a note near A4, got {result.notes}"


def test_basic_pitch_satisfies_transcriber_protocol() -> None:
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig
    from sonitra.transcribe.protocol import TranscriberProtocol, make_transcriber

    t = make_transcriber(BasicPitchTranscriberConfig())
    assert isinstance(t, TranscriberProtocol)
    assert isinstance(t, BasicPitchTranscriber)


@pytest.mark.slow
def test_basic_pitch_chord_detects_multiple_pitches(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t) + 0.5 * np.sin(2.0 * np.pi * 554.37 * t)
    signal = (signal / np.max(np.abs(signal)) * 0.9).astype(np.float32)
    audio_path = tmp_path / "chord.wav"
    wavfile.write(audio_path, sample_rate, signal)

    transcriber = BasicPitchTranscriber()
    result = transcriber.transcribe(audio_path)

    distinct_pitches = {n["pitch"] for n in result.notes}
    assert len(distinct_pitches) >= 2, f"expected >= 2 distinct pitches, got {distinct_pitches}"

    notes_near_a4 = [n for n in result.notes if abs(n["pitch"] - 69) <= 1]
    notes_near_cs5 = [n for n in result.notes if abs(n["pitch"] - 73) <= 1]
    assert notes_near_a4, f"expected pitch near 69 (A4), got pitches {distinct_pitches}"
    assert notes_near_cs5, f"expected pitch near 73 (C#5), got pitches {distinct_pitches}"


@pytest.mark.slow
def test_basic_pitch_silence_returns_empty_or_minimal(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.base import TranscriptionResult
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    silence = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
    audio_path = tmp_path / "silence.wav"
    wavfile.write(audio_path, sample_rate, silence)

    transcriber = BasicPitchTranscriber()
    result = transcriber.transcribe(audio_path)

    assert isinstance(result, TranscriptionResult)
    assert result.transcriber == "basic_pitch"
    assert len(result.notes) == 0, f"expected no notes from silence, got {result.notes}"


@pytest.mark.slow
def test_basic_pitch_result_notes_are_sorted(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t) + 0.5 * np.sin(2.0 * np.pi * 554.37 * t)
    signal = (signal / np.max(np.abs(signal)) * 0.9).astype(np.float32)
    audio_path = tmp_path / "chord_sort.wav"
    wavfile.write(audio_path, sample_rate, signal)

    transcriber = BasicPitchTranscriber()
    result = transcriber.transcribe(audio_path)

    sort_keys = [(n["start_sec"], n["pitch"]) for n in result.notes]
    assert sort_keys == sorted(sort_keys), "notes are not sorted by (start_sec, pitch)"


@pytest.mark.slow
def test_basic_pitch_note_fields_are_valid_types(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "a440_types.wav"
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))

    transcriber = BasicPitchTranscriber()
    result = transcriber.transcribe(audio_path)

    assert len(result.notes) >= 1
    for n in result.notes:
        assert isinstance(n["pitch"], int), f"pitch must be int, got {type(n['pitch'])}"
        assert isinstance(n["velocity"], int), f"velocity must be int, got {type(n['velocity'])}"
        assert isinstance(n["start_sec"], float), f"start_sec must be float, got {type(n['start_sec'])}"
    assert_notes_satisfy_contract(result.notes)


# ── Basic Pitch configurable knobs + raw model outputs ───────────────

def test_basic_pitch_builder_forwards_new_knobs() -> None:
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    transcriber = make_transcriber(
        BasicPitchTranscriberConfig(
            melodia_trick=False,
            multiple_pitch_bends=True,
            save_raw_outputs=True,
        )
    )
    assert transcriber.melodia_trick is False
    assert transcriber.multiple_pitch_bends is True
    assert transcriber.save_raw_outputs is True


@pytest.mark.slow
def test_basic_pitch_passes_knobs_to_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Knobs reach their real destinations on the manual-loop call path.

    The rewrite routes audio through ``Model.predict`` window-by-window and
    note decoding through ``note_creation.model_output_to_notes``; this pins
    the forwarding contract instead of the old ``inference.predict`` wrapper.
    """
    pytest.importorskip("basic_pitch")
    import basic_pitch.inference as inference_module
    import basic_pitch.note_creation as note_creation_module
    import numpy as np
    from scipy.io import wavfile

    from basic_pitch.constants import AUDIO_N_SAMPLES
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 0.5
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    audio_path = tmp_path / "knobs.wav"
    wavfile.write(audio_path, sample_rate, signal)

    captured: dict = {}
    seen_batches: list[tuple[int, ...]] = []
    fake_model_output = {
        "onset": np.zeros((1, 60, 88), np.float32),
        "contour": np.zeros((1, 60, 264), np.float32),
        "note": np.zeros((1, 60, 88), np.float32),
    }

    class _FakeModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def predict(self, batch):
            seen_batches.append(tuple(np.asarray(batch).shape))
            return fake_model_output

    def fake_model_output_to_notes(*args, **kwargs):
        captured.update(kwargs)
        return None, []

    monkeypatch.setattr(inference_module, "Model", _FakeModel)
    monkeypatch.setattr(
        note_creation_module, "model_output_to_notes", fake_model_output_to_notes
    )

    transcriber = BasicPitchTranscriber(
        onset_threshold=0.42,
        frame_threshold=0.21,
        minimum_note_length_ms=250.0,
        minimum_frequency_hz=100.0,
        maximum_frequency_hz=1000.0,
        melodia_trick=False,
        multiple_pitch_bends=True,
    )
    result = transcriber.transcribe(audio_path)

    # Note-decoding knobs land on note_creation.model_output_to_notes.
    assert captured["onset_thresh"] == pytest.approx(0.42)
    assert captured["frame_thresh"] == pytest.approx(0.21)
    # 250 ms converted to frames at ANNOTATIONS_FPS (86.13).
    assert captured["min_note_len"] == 22
    assert captured["min_freq"] == pytest.approx(100.0)
    assert captured["max_freq"] == pytest.approx(1000.0)
    assert captured["melodia_trick"] is False
    assert captured["multiple_pitch_bends"] is True
    assert captured["midi_tempo"] == 120
    # Audio still arrives at Model.predict in the expected windowed shape.
    assert seen_batches == [(1, AUDIO_N_SAMPLES, 1)]
    assert result.metadata["device"] == "cpu"
    assert result.metadata["requested_device"] == "cpu"
    assert result.metadata["device_available"] is True


@pytest.mark.slow
def test_basic_pitch_raw_outputs_captured_when_enabled(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "a440_raw.wav"
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))

    transcriber = BasicPitchTranscriber(save_raw_outputs=True)
    result = transcriber.transcribe(audio_path)

    assert result.raw_outputs is not None
    assert set(result.raw_outputs) == {"onset", "contour", "note"}
    onset = result.raw_outputs["onset"]
    contour = result.raw_outputs["contour"]
    note = result.raw_outputs["note"]
    n = onset.shape[0]
    assert n > 0
    assert onset.shape == (n, 88)
    assert note.shape == (n, 88)
    assert contour.shape == (n, 264)
    for array in (onset, contour, note):
        assert np.all(array >= 0.0) and np.all(array <= 1.0)


@pytest.mark.slow
def test_basic_pitch_raw_outputs_none_when_disabled(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "a440_no_raw.wav"
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))

    transcriber = BasicPitchTranscriber()
    result = transcriber.transcribe(audio_path)

    assert result.raw_outputs is None


@pytest.mark.slow
def test_basic_pitch_gpu_unavailable_raises() -> None:
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        pytest.skip("GPU is available, cannot test unavailable path")
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    transcriber = BasicPitchTranscriber(device="cuda")
    with pytest.raises(TranscriptionError):
        transcriber.validate_device()


@pytest.mark.slow
def test_basic_pitch_model_loaded_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("basic_pitch")
    import basic_pitch.inference as inference_module
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    t = np.linspace(0.0, 0.5, int(sample_rate * 0.5), endpoint=False)
    signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    audio_paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for audio_path in audio_paths:
        wavfile.write(audio_path, sample_rate, signal)

    model_calls = {"n": 0}
    fake_model_output = {
        "onset": np.zeros((1, 60, 88), np.float32),
        "contour": np.zeros((1, 60, 264), np.float32),
        "note": np.zeros((1, 60, 88), np.float32),
    }

    class _CountingModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            model_calls["n"] += 1

        def predict(self, batch):
            return fake_model_output

    monkeypatch.setattr(inference_module, "Model", _CountingModel)

    transcriber = BasicPitchTranscriber()
    for audio_path in audio_paths:
        transcriber.transcribe(audio_path)
    assert model_calls["n"] == 1


@pytest.mark.slow
def test_basic_pitch_manual_loop_matches_inference_predict(tmp_path: Path) -> None:
    """The manual window loop is behavior-preserving vs inference.predict.

    Runs the library's own ``predict`` against the same loaded Model and
    compares its note events to this transcriber's public note output.
    """
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    audio_path = tmp_path / "a440_parity.wav"
    wavfile.write(audio_path, sample_rate, signal)

    model = Model(ICASSP_2022_MODEL_PATH)
    _, _, reference_events = predict(audio_path, model_or_model_path=model)
    reference_events = sorted(reference_events, key=lambda e: (e[0], e[2]))
    assert reference_events, "fixture must produce at least one reference note"

    result = BasicPitchTranscriber(device="cpu", batch_size=1).transcribe(audio_path)

    assert len(result.notes) == len(reference_events)
    assert result.metadata["note_events_total"] == len(reference_events)
    max_amp_diff = 0.0
    for note, (start, end, pitch, amplitude, _bends) in zip(result.notes, reference_events):
        assert note["pitch"] == int(pitch)
        assert note["start_sec"] == pytest.approx(float(start), abs=1e-9)
        assert note["start_sec"] + note["duration_sec"] == pytest.approx(
            float(end), abs=1e-9
        )
        max_amp_diff = max(max_amp_diff, abs(note["velocity"] / 127.0 - float(amplitude)))
    # velocity is amplitude quantised to 0..127; allow the rounding step.
    assert max_amp_diff <= 0.5 / 127.0 + 1e-9


@pytest.mark.slow
def test_basic_pitch_determinism(tmp_path: Path) -> None:
    """Same CPU input transcribes identically twice."""
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "a440_det.wav"
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))

    transcriber = BasicPitchTranscriber(device="cpu")
    r1 = transcriber.transcribe(audio_path)
    r2 = transcriber.transcribe(audio_path)
    assert r1.notes == r2.notes


def test_basic_pitch_builder_forwards_numeric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    monkeypatch.setenv("SONITRA_NUMERIC_MODE", "strict")
    monkeypatch.setenv("SONITRA_GPU_MEMORY_GROWTH", "1")
    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    assert transcriber.numeric_mode == "strict"
    assert transcriber.gpu_memory_growth is True


def test_basic_pitch_builder_forwards_batch_size() -> None:
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig

    assert make_transcriber(BasicPitchTranscriberConfig()).batch_size == 16
    assert make_transcriber(BasicPitchTranscriberConfig(batch_size=32)).batch_size == 32


@pytest.mark.slow
def test_basic_pitch_batched_matches_single(tmp_path: Path) -> None:
    """CPU stacking (batch_size>1) yields the same note set as batch_size=1."""
    pytest.importorskip("basic_pitch")
    import numpy as np
    from scipy.io import wavfile

    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    sample_rate = 22050
    duration = 4.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = sum(0.3 * np.sin(2.0 * np.pi * freq * t) for freq in (261.63, 329.63, 392.0))
    audio_path = tmp_path / "chord_batch.wav"
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))

    single = BasicPitchTranscriber(device="cpu", batch_size=1).transcribe(audio_path)
    batched = BasicPitchTranscriber(device="cpu", batch_size=2).transcribe(audio_path)
    assert batched.notes == single.notes
