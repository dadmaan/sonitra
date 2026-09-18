"""TransKun backend — config schema, mapper helpers, and inference (slow)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sonitra.config import ConfigError, PipelineConfig


def _minimal_config_dict() -> dict:
    return {
        "render_pipeline": {
            "synth_backend": "dawdreamer_faust",
            "effects_chain": "none",
            "sample_rate": 44100,
            "bit_depth": 24,
            "channels": 2,
            "duration_padding_sec": 2.0,
            "overwrite": False,
            "resume": True,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": {
            "corpus_root": ".",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
    }


# ── Stage 6: config schema ─────────────────────────────────────────────

def test_transkun_config_defaults() -> None:
    from sonitra.transcribe.configs import TranskunTranscriberConfig

    cfg = TranskunTranscriberConfig()
    assert cfg.type == "transkun"
    assert cfg.device == "cpu"
    assert cfg.segment_size_sec is None
    assert cfg.segment_hop_sec is None
    assert cfg.weights_path is None
    assert cfg.conf_path is None
    assert cfg.enabled is True
    assert cfg.name is None


def test_transkun_config_rejects_unknown_key() -> None:
    bad = {
        **_minimal_config_dict(),
        "transcription": {"transcribers": [{"type": "transkun", "bogus_key": 1}]},
    }
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


def test_transkun_config_discriminator() -> None:
    cfg = PipelineConfig.model_validate(
        {
            **_minimal_config_dict(),
            "transcription": {"transcribers": [{"type": "transkun"}]},
        }
    )
    from sonitra.transcribe.configs import TranskunTranscriberConfig

    assert len(cfg.transcription.transcribers) == 1
    assert isinstance(cfg.transcription.transcribers[0], TranskunTranscriberConfig)
    assert cfg.transcription.transcribers[0].type == "transkun"


def test_transkun_config_device_variants() -> None:
    from sonitra.transcribe.configs import TranskunTranscriberConfig

    for dev in ("cpu", "cuda", "cuda:1", "mps", "GPU:0", "gpu"):
        cfg = TranskunTranscriberConfig(device=dev)
        assert cfg.device == dev


def test_transkun_config_paths_accept_str_and_path() -> None:
    from sonitra.transcribe.configs import TranskunTranscriberConfig

    cfg1 = TranskunTranscriberConfig(weights_path="/tmp/w.pt", conf_path="/tmp/c.conf")
    assert cfg1.weights_path == "/tmp/w.pt"
    cfg2 = TranskunTranscriberConfig(weights_path=Path("/tmp/w.pt"))
    assert str(cfg2.weights_path) == "/tmp/w.pt"


# ── Stage 8: mapper helpers ────────────────────────────────────────────

class _FakeNote:
    def __init__(self, pitch, start, end, velocity=80, hasOnset=True, hasOffset=True):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity
        self.hasOnset = hasOnset
        self.hasOffset = hasOffset


def test_resolve_device_gpu_variants() -> None:
    from sonitra.transcribe.transkun import _resolve_device

    assert _resolve_device("GPU:0") == "cuda:0"
    assert _resolve_device("gpu:0") == "cuda:0"
    assert _resolve_device("GPU:1") == "cuda:1"
    assert _resolve_device("gpu") == "cuda"
    assert _resolve_device("GPU") == "cuda"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("cuda:1") == "cuda:1"
    assert _resolve_device("mps") == "mps"
    assert _resolve_device("cpu") == "cpu"


def test_notes_to_dicts_basic_and_sorted() -> None:
    from sonitra.transcribe.transkun import _notes_to_dicts

    notes = [
        _FakeNote(pitch=60, start=1.0, end=1.5, velocity=90),
        _FakeNote(pitch=62, start=0.5, end=1.0, velocity=80),
        _FakeNote(pitch=60, start=0.5, end=1.0, velocity=70),
    ]
    dicts = _notes_to_dicts(notes)
    # sorted by (start_sec, pitch)
    assert [(d["start_sec"], d["pitch"]) for d in dicts] == sorted(
        [(d["start_sec"], d["pitch"]) for d in dicts]
    )
    assert len(dicts) == 3


def test_notes_to_dicts_filters_negative_pedal_before_make_note() -> None:
    from sonitra.transcribe.transkun import _notes_to_dicts

    notes = [
        _FakeNote(pitch=-64, start=0.0, end=0.5, velocity=100),
        _FakeNote(pitch=-67, start=0.1, end=0.6, velocity=100),
        _FakeNote(pitch=60, start=0.0, end=0.5, velocity=100),
    ]
    # must not raise ValueError, negatives are filtered before make_note
    dicts = _notes_to_dicts(notes)
    assert len(dicts) == 1
    assert dicts[0]["pitch"] == 60


def test_notes_to_dicts_no_pitch_offset() -> None:
    from sonitra.transcribe.transkun import _notes_to_dicts

    # transkun's targetMIDIPitch = [-64,-67] + range(21,109)
    # pitch is already real MIDI; adding +21 would shift
    notes = [_FakeNote(pitch=21, start=0.0, end=0.5), _FakeNote(pitch=108, start=0.5, end=1.0)]
    dicts = _notes_to_dicts(notes)
    pitches = {d["pitch"] for d in dicts}
    assert 21 in pitches
    assert 108 in pitches
    # ensure no offset bug: pitches should stay within 21-108
    for d in dicts:
        assert 21 <= d["pitch"] <= 108
    # offset bug would produce 42 and 129 (129 out of range -> ValueError or dropped)
    # explicitly check that 42 is not mistakenly produced from 21
    assert not any(d["pitch"] == 42 and len(dicts) == 2 and pitches == {42, 129} for d in dicts)


def test_notes_to_dicts_pitch_offset_would_fail() -> None:
    """Demonstrate that adding +21 would be caught: pitches would be 42-129."""
    # This test documents the most valuable trap: if implementation adds +21,
    # a pitch 21 would become 42 and pitch 108 would become 129 (>127 -> ValueError).
    # We assert the correct behaviour is NO offset.
    from sonitra.transcribe.transkun import _notes_to_dicts

    notes = [_FakeNote(pitch=21, start=0.0, end=1.0)]
    dicts = _notes_to_dicts(notes)
    assert dicts[0]["pitch"] == 21
    assert dicts[0]["pitch"] != 42


def test_notes_to_dicts_uses_make_note_contract() -> None:
    from sonitra.transcribe.transkun import _notes_to_dicts
    from tests.helpers import assert_notes_satisfy_contract

    # velocity clamping, start clamping, duration dropping, sortedness
    notes = [
        _FakeNote(pitch=60, start=-0.5, end=0.5, velocity=999),  # start clamped to 0, velocity clamped to 127
        _FakeNote(pitch=62, start=0.2, end=0.2, velocity=80),  # zero duration -> dropped
        _FakeNote(pitch=64, start=0.3, end=0.1, velocity=80),  # negative duration -> dropped
        _FakeNote(pitch=65, start=0.4, end=0.9, velocity=-5),  # velocity clamped to 1
    ]
    dicts = _notes_to_dicts(notes)
    # only 2 should survive
    assert len(dicts) == 2
    assert dicts[0]["pitch"] == 60
    assert dicts[0]["start_sec"] == 0.0
    assert dicts[0]["velocity"] == 127
    assert dicts[1]["velocity"] == 1
    assert_notes_satisfy_contract(dicts)


def test_notes_to_dicts_pitches_within_21_108_and_no_negatives() -> None:
    from sonitra.transcribe.transkun import _notes_to_dicts
    from tests.helpers import assert_notes_satisfy_contract

    notes = [_FakeNote(pitch=p, start=float(p) * 0.01, end=float(p) * 0.01 + 0.5) for p in [21, 60, 108]]
    notes.extend([_FakeNote(pitch=-64, start=0, end=1), _FakeNote(pitch=-67, start=0, end=1)])
    dicts = _notes_to_dicts(notes)
    assert all(21 <= d["pitch"] <= 108 for d in dicts)
    assert all(d["pitch"] >= 0 for d in dicts)
    assert_notes_satisfy_contract(dicts)
    assert len(dicts) == 3


def test_notes_to_dicts_keeps_boundary_notes() -> None:
    from sonitra.transcribe.transkun import _notes_to_dicts

    notes = [
        _FakeNote(pitch=60, start=0.0, end=0.5, hasOnset=False, hasOffset=True),
        _FakeNote(pitch=61, start=0.5, end=1.0, hasOnset=True, hasOffset=False),
        _FakeNote(pitch=62, start=1.0, end=1.5, hasOnset=False, hasOffset=False),
    ]
    dicts = _notes_to_dicts(notes)
    # upstream parity: boundary notes are kept
    assert len(dicts) == 3


def test_read_audio_resampled_wav_exact(tmp_path: Path) -> None:
    import numpy as np
    from sonitra.storage import read_audio_resampled, write_audio

    sr = 44100
    dur = 1.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.stack([sig, sig])
    path = tmp_path / "tone.wav"
    write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format="wav")
    out, out_sr = read_audio_resampled(path, target_sr=44100)
    assert out_sr == 44100
    assert out.shape == (2, int(sr * dur))
    assert out.dtype == np.float32


def test_read_audio_resampled_resamples_and_handles_flac(tmp_path: Path) -> None:
    import numpy as np
    from sonitra.storage import read_audio_resampled, write_audio

    # write at 22050, reading resampled to 44100 should double frames
    sr_in = 22050
    dur = 1.0
    t = np.linspace(0.0, dur, int(sr_in * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.stack([sig, sig])
    for fmt in ("wav", "flac"):
        path = tmp_path / f"tone_{fmt}.{fmt}"
        write_audio(audio, path, sample_rate=sr_in, bit_depth=24, output_format=fmt)
        out, out_sr = read_audio_resampled(path, target_sr=44100)
        assert out_sr == 44100
        # wav/flac exact after resampling
        assert out.shape[1] == int(44100 * dur), f"{fmt} frames {out.shape[1]} != {int(44100*dur)}"


def test_read_audio_resampled_mp3_padding(tmp_path: Path) -> None:
    import numpy as np
    from sonitra.storage import read_audio_resampled, write_audio

    # repro measurement from plan: 2s mono at 22.05kHz -> 91008 frames vs 88200
    sr_in = 22050
    dur = 2.0
    t = np.linspace(0.0, dur, int(sr_in * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.array([sig])  # mono
    path = tmp_path / "tone.mp3"
    write_audio(audio, path, sample_rate=sr_in, bit_depth=16, output_format="mp3")
    out, out_sr = read_audio_resampled(path, target_sr=44100)
    assert out_sr == 44100
    expected_exact = int(44100 * dur)
    # MP3 decodes longer (pedalboard 0.9.24 measured +2808 for stereo? check)
    assert out.shape[1] > expected_exact
    # but not wildly larger than expected + ~64ms
    assert out.shape[1] - expected_exact < 5000


def test_read_audio_resampled_missing_raises(tmp_path: Path) -> None:
    from sonitra.storage import read_audio_resampled

    with pytest.raises(FileNotFoundError):
        read_audio_resampled(tmp_path / "missing.wav")


# ── Stage 9 slow tests (require transkun installed) ────────────────────

@pytest.mark.slow
def test_transkun_transcribes_sine_tone(tmp_path: Path) -> None:
    pytest.importorskip("transkun")
    import numpy as np
    from sonitra.storage import write_audio
    from sonitra.transcribe.transkun import TranskunTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    sr = 44100
    dur = 4.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    # stack harmonics with decay envelope to help piano model
    signal = sum(np.sin(2 * np.pi * 440.0 * (2 ** (h * 0.0)) * t) * (0.5 ** h) for h in range(4))
    # Actually craft A4 with harmonics
    sig = 0.6 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t) + 0.1 * np.sin(2 * np.pi * 1320.0 * t)
    sig = sig * np.exp(-t * 0.5)
    audio = np.stack([sig, sig])
    path = tmp_path / "a440.wav"
    write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format="wav")

    transcriber = TranskunTranscriber()
    result = transcriber.transcribe(path)
    assert result.backend_type == "transkun"
    assert result.transcriber == "transkun"
    assert result.metadata["package_version"] != "unknown"
    assert "device" in result.metadata
    # notes may be empty on synthetic audio; but if present they must satisfy contract
    if result.notes:
        assert_notes_satisfy_contract(result.notes)
        assert all(21 <= n["pitch"] <= 108 for n in result.notes)


@pytest.mark.slow
def test_transkun_grad_mode_still_enabled_after_transcribe(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transkun")
    import torch
    import numpy as np
    from sonitra.storage import write_audio
    from sonitra.transcribe.transkun import TranskunTranscriber

    sr = 44100
    sig = np.zeros(int(sr * 2.0), dtype=np.float32)
    audio = np.stack([sig, sig])
    path = tmp_path / "silence.wav"
    write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format="wav")
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()
    transcriber = TranskunTranscriber()
    transcriber.transcribe(path)
    assert torch.is_grad_enabled(), "transcribe() must not leave grad disabled (use torch.no_grad, not set_grad_enabled(False))"


@pytest.mark.slow
def test_transkun_transcribes_flac_and_mp3(tmp_path: Path) -> None:
    pytest.importorskip("transkun")
    import numpy as np
    from sonitra.storage import write_audio
    from sonitra.transcribe.transkun import TranskunTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    sr = 44100
    dur = 2.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.stack([sig, sig])

    for fmt in ("flac", "mp3"):
        path = tmp_path / f"tone.{fmt}"
        write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format=fmt)
        transcriber = TranskunTranscriber()
        result = transcriber.transcribe(path)
        assert result.backend_type == "transkun"
        if result.notes:
            assert_notes_satisfy_contract(result.notes)
            assert all(21 <= n["pitch"] <= 108 for n in result.notes)


@pytest.mark.slow
def test_transkun_short_audio(tmp_path: Path) -> None:
    pytest.importorskip("transkun")
    import numpy as np
    from sonitra.storage import write_audio
    from sonitra.transcribe.transkun import TranskunTranscriber

    sr = 44100
    # shorter than one 16s segment
    dur = 0.5
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.stack([sig, sig])
    path = tmp_path / "short.wav"
    write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format="wav")
    transcriber = TranskunTranscriber()
    result = transcriber.transcribe(path)
    # should not crash; may return empty or notes
    assert isinstance(result.notes, list)
    assert result.backend_type == "transkun"


@pytest.mark.slow
def test_transkun_determinism(tmp_path: Path) -> None:
    pytest.importorskip("transkun")
    import numpy as np
    from sonitra.storage import write_audio
    from sonitra.transcribe.transkun import TranskunTranscriber

    sr = 44100
    dur = 2.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.stack([sig, sig])
    path = tmp_path / "det.wav"
    write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format="wav")
    transcriber = TranskunTranscriber()
    r1 = transcriber.transcribe(path)
    r2 = transcriber.transcribe(path)
    assert r1.notes == r2.notes


@pytest.mark.slow
def test_transkun_cuda_unavailable_raises(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transkun")
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, cannot test unavailable path")
    import numpy as np
    from sonitra.storage import write_audio
    from sonitra.transcribe.transkun import TranskunTranscriber
    from sonitra.transcribe.base import TranscriptionError

    sr = 44100
    sig = np.zeros(int(sr * 1.0), dtype=np.float32)
    audio = np.stack([sig, sig])
    path = tmp_path / "sil.wav"
    write_audio(audio, path, sample_rate=sr, bit_depth=24, output_format="wav")
    transcriber = TranskunTranscriber(device="cuda")
    with pytest.raises(TranscriptionError):
        transcriber.transcribe(path)
