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
