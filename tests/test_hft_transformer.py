"""hFT-Transformer backend — config schema, weights resolution, mapper and inference (slow)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sonitra.config import ConfigError, PipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"

#: A FluidSynth piano render of a MAESTRO performance, used only by the opt-in test
#: that needs the released weights to actually detect something.
RENDERED_PIANO = (
    REPO_ROOT
    / "misc"
    / "hft_transformer_spike"
    / "out"
    / "audio"
    / "2008"
    / "MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.wav"
)

#: Every metadata key the backend must report on every result.
EXPECTED_METADATA_KEYS = (
    "package_version",
    "weights_sha256",
    "upstream_commit",
    "checkpoint",
    "device",
    "requested_device",
    "device_available",
    "numeric_mode",
    "output_head",
    "n_stride",
    "batch_size",
    "thresholds",
    "resampler",
    "sample_rate",
    "note_events_total",
    "notes_kept",
    "filtered_dropped",
    "offset_past_end",
)


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


# ── config schema ──────────────────────────────────────────────────────

def test_hft_config_defaults() -> None:
    from sonitra.transcribe.configs import HftTransformerTranscriberConfig

    cfg = HftTransformerTranscriberConfig()
    assert cfg.type == "hft_transformer"
    assert cfg.enabled is True
    assert cfg.name is None
    assert cfg.device == "cpu"
    assert cfg.checkpoint == "maestro"
    assert cfg.weights_path is None
    assert cfg.output == "second"
    assert cfg.n_stride == 0
    assert cfg.onset_threshold == 0.5
    assert cfg.offset_threshold == 0.5
    assert cfg.mpe_threshold == 0.5
    assert cfg.batch_size == 1
    assert not hasattr(cfg, "mode"), "mode is fixed to 'combination' upstream and must not be exposed"


def test_hft_config_rejects_unknown_key() -> None:
    bad = {
        **_minimal_config_dict(),
        "transcription": {"transcribers": [{"type": "hft_transformer", "bogus_key": 1}]},
    }
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


def test_hft_config_discriminator() -> None:
    cfg = PipelineConfig.model_validate(
        {
            **_minimal_config_dict(),
            "transcription": {"transcribers": [{"type": "hft_transformer"}]},
        }
    )
    from sonitra.transcribe.configs import HftTransformerTranscriberConfig

    assert len(cfg.transcription.transcribers) == 1
    assert isinstance(cfg.transcription.transcribers[0], HftTransformerTranscriberConfig)


def test_hft_config_rejects_unknown_checkpoint() -> None:
    bad = {
        **_minimal_config_dict(),
        "transcription": {"transcribers": [{"type": "hft_transformer", "checkpoint": "maps"}]},
    }
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("onset_threshold", 0.0),
        ("onset_threshold", 1.5),
        ("offset_threshold", 0.0),
        ("offset_threshold", -0.1),
        ("mpe_threshold", 0.0),
        ("mpe_threshold", 2.0),
        ("batch_size", 0),
        ("batch_size", -4),
        ("n_stride", -1),
        ("n_stride", 65),
    ],
)
def test_hft_config_rejects_out_of_range(field: str, value: float) -> None:
    from pydantic import ValidationError

    from sonitra.transcribe.configs import HftTransformerTranscriberConfig

    with pytest.raises(ValidationError):
        HftTransformerTranscriberConfig(**{field: value})


@pytest.mark.parametrize(
    "field,value", [("n_stride", 64), ("onset_threshold", 1.0), ("batch_size", 8)]
)
def test_hft_config_accepts_range_endpoints(field: str, value: float) -> None:
    from sonitra.transcribe.configs import HftTransformerTranscriberConfig

    assert getattr(HftTransformerTranscriberConfig(**{field: value}), field) == value


def test_hft_config_device_variants_and_weights_path_types() -> None:
    from sonitra.transcribe.configs import HftTransformerTranscriberConfig

    for dev in ("cpu", "cuda", "cuda:1", "mps", "GPU:0", "gpu"):
        assert HftTransformerTranscriberConfig(device=dev).device == dev
    cfg = HftTransformerTranscriberConfig(weights_path=Path("/tmp/model.pt"))
    assert str(cfg.weights_path) == "/tmp/model.pt"


def test_hft_unknown_numeric_mode_raises() -> None:
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    with pytest.raises(TranscriptionError, match="hft_transformer") as excinfo:
        HftTransformerTranscriber(numeric_mode="sometimes")
    assert "sometimes" in str(excinfo.value)
    for mode in ("off", "warn", "strict"):
        assert HftTransformerTranscriber(numeric_mode=mode).numeric_mode == mode


def test_hft_transformer_builder_forwards_numeric_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder, not the transcriber, resolves the numeric settings: it runs
    once per instance, while transcribe() runs per file."""
    from sonitra.transcribe.configs import HftTransformerTranscriberConfig
    from sonitra.transcribe.protocol import make_transcriber

    monkeypatch.setenv("SONITRA_NUMERIC_MODE", "strict")
    monkeypatch.setenv("SONITRA_GPU_MEMORY_GROWTH", "1")

    transcriber = make_transcriber(HftTransformerTranscriberConfig())

    assert transcriber.numeric_mode == "strict"
    assert transcriber.gpu_memory_growth is True


# ── import deferral ────────────────────────────────────────────────────

def _run_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_SRC), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_make_transcriber_defers_torch_import_hft() -> None:
    code = (
        "import sys\n"
        "from sonitra.transcribe.protocol import make_transcriber\n"
        "from sonitra.transcribe.configs import HftTransformerTranscriberConfig\n"
        "m = make_transcriber(HftTransformerTranscriberConfig())\n"
        "assert m.name == 'hft_transformer', m.name\n"
        "assert 'torch' not in sys.modules, 'torch imported by make_transcriber'\n"
        "assert 'torchaudio' not in sys.modules, 'torchaudio imported by make_transcriber'\n"
        "print('ok')\n"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_hft_cpu_validate_device_imports_no_torch() -> None:
    code = (
        "import sys\n"
        "from sonitra.transcribe.hft_transformer import HftTransformerTranscriber\n"
        "t = HftTransformerTranscriber(device='cpu')\n"
        "assert t.validate_device() == ('cpu', True)\n"
        "assert 'torch' not in sys.modules, 'cpu validate_device imported torch'\n"
        "assert 'torchaudio' not in sys.modules, 'cpu validate_device imported torchaudio'\n"
        "print('ok')\n"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_importing_hft_backend_does_not_import_torch() -> None:
    code = (
        "import sys\n"
        "import sonitra.transcribe.protocol\n"
        "import sonitra.transcribe.hft_transformer\n"
        "assert 'torch' not in sys.modules, 'torch imported at backend import'\n"
        "print('ok')\n"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


# ── weights resolution ─────────────────────────────────────────────────

def _audio_file(tmp_path: Path, name: str = "tone.wav", dur: float = 0.5) -> Path:
    import numpy as np
    from sonitra.storage import write_audio

    sr = 16000
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    sig = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    path = tmp_path / name
    write_audio(np.stack([sig, sig]), path, sample_rate=sr, bit_depth=24, output_format="wav")
    return path


def test_missing_weights_names_the_setup_script_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv("SONITRA_MODELS_DIR", str(models_dir))

    transcriber = HftTransformerTranscriber()
    with pytest.raises(TranscriptionError) as excinfo:
        transcriber.transcribe(_audio_file(tmp_path))
    message = str(excinfo.value)
    assert "scripts/setup_hft_transformer.py" in message
    assert str(models_dir) in message
    assert "maestro" in message


def test_weights_path_wins_over_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sonitra.transcribe._hft import checkpoints
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("SONITRA_MODELS_DIR", str(empty))

    explicit = tmp_path / "custom.pt"
    transcriber = HftTransformerTranscriber(weights_path=explicit)
    with pytest.raises(TranscriptionError) as excinfo:
        transcriber.transcribe(_audio_file(tmp_path))
    message = str(excinfo.value)
    assert str(explicit) in message
    # The default location must not be the file being complained about.
    default = checkpoints.checkpoint_path("maestro")
    assert str(default) != str(explicit)
    assert str(default) not in message


def test_weights_path_pointing_at_the_upstream_pickle_is_refused(tmp_path: Path) -> None:
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    pickle_path = tmp_path / "model_016_003.pkl"
    pickle_path.write_bytes(b"\x80\x04not-a-real-pickle")
    transcriber = HftTransformerTranscriber(weights_path=pickle_path)
    with pytest.raises(TranscriptionError) as excinfo:
        transcriber.transcribe(_audio_file(tmp_path))
    message = str(excinfo.value)
    assert "pickle" in message
    assert "scripts/setup_hft_transformer.py" in message


def test_unreadable_weights_file_raises(tmp_path: Path) -> None:
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    broken = tmp_path / "model.pt"
    broken.write_bytes(b"this is not a torch file")
    transcriber = HftTransformerTranscriber(weights_path=broken)
    with pytest.raises(TranscriptionError) as excinfo:
        transcriber.transcribe(_audio_file(tmp_path))
    assert str(broken) in str(excinfo.value)


# ── note mapping ───────────────────────────────────────────────────────

def _event(pitch: int, onset: float, offset: float, velocity: int = 80) -> dict:
    return {"pitch": pitch, "onset": onset, "offset": offset, "velocity": velocity}


def test_note_mapping_routes_through_make_note() -> None:
    from sonitra.transcribe.hft_transformer import notes_from_events

    mapped = notes_from_events(
        [_event(60, 1.0, 1.5, 90), _event(21, 0.5, 1.0), _event(108, 2.0, 2.25)],
        audio_duration_sec=10.0,
    )
    assert mapped.filtered_dropped == 0
    assert mapped.offset_past_end == 0
    assert [(n["start_sec"], n["pitch"]) for n in mapped.notes] == [
        (0.5, 21),
        (1.0, 60),
        (2.0, 108),
    ]
    assert mapped.notes[0]["duration_sec"] == pytest.approx(0.5)
    assert mapped.notes[0]["velocity"] == 80


def test_note_mapping_applies_no_pitch_offset() -> None:
    """The decoder already emits MIDI pitches; adding 21 would shift the whole piano."""
    from sonitra.transcribe.hft_transformer import notes_from_events
    from tests.helpers import assert_notes_satisfy_contract

    mapped = notes_from_events(
        [_event(21, 0.0, 0.5), _event(60, 0.5, 1.0), _event(108, 1.0, 1.5)], 5.0
    )
    pitches = [n["pitch"] for n in mapped.notes]
    assert pitches == [21, 60, 108]
    assert 42 not in pitches
    assert 129 not in pitches
    assert_notes_satisfy_contract(mapped.notes)


def test_note_mapping_drops_and_counts_non_positive_durations() -> None:
    from sonitra.transcribe.hft_transformer import notes_from_events

    mapped = notes_from_events(
        [
            _event(60, 0.0, 0.5),
            _event(61, 0.5, 0.5),  # zero duration
            _event(62, 0.6, 0.5),  # negative duration
            _event(63, 0.7, 1.2),
        ],
        audio_duration_sec=5.0,
    )
    assert [n["pitch"] for n in mapped.notes] == [60, 63]
    assert mapped.filtered_dropped == 2
    assert mapped.note_events_total == 4


def test_note_mapping_clamps_velocity_and_start() -> None:
    from sonitra.transcribe.hft_transformer import notes_from_events
    from tests.helpers import assert_notes_satisfy_contract

    mapped = notes_from_events(
        [_event(60, -0.5, 0.0, velocity=999), _event(64, 0.1, 0.6, velocity=-5)], 5.0
    )
    assert mapped.notes[0]["start_sec"] == 0.0
    assert mapped.notes[0]["velocity"] == 127
    assert mapped.notes[1]["velocity"] == 1
    assert_notes_satisfy_contract(mapped.notes)


def test_note_mapping_keeps_notes_ending_past_the_audio() -> None:
    """Tail padding is upstream behaviour, so the note is kept and counted, not clipped."""
    from sonitra.transcribe.hft_transformer import notes_from_events
    from tests.helpers import assert_notes_satisfy_contract

    mapped = notes_from_events(
        [_event(60, 3.0, 4.0), _event(62, 3.2, 5.5)], audio_duration_sec=4.0
    )
    assert [n["pitch"] for n in mapped.notes] == [60, 62]
    assert mapped.offset_past_end == 1
    assert mapped.notes[1]["duration_sec"] == pytest.approx(2.3)
    assert mapped.filtered_dropped == 0
    assert_notes_satisfy_contract(mapped.notes)


def test_note_mapping_empty_input() -> None:
    from sonitra.transcribe.hft_transformer import notes_from_events

    mapped = notes_from_events([], audio_duration_sec=4.0)
    assert mapped.notes == []
    assert mapped.filtered_dropped == 0
    assert mapped.offset_past_end == 0
    assert mapped.note_events_total == 0


# ── missing dependency ─────────────────────────────────────────────────

def test_hft_missing_torch_names_the_transkun_extra(tmp_path: Path) -> None:
    from unittest.mock import patch

    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    transcriber = HftTransformerTranscriber()
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(TranscriptionError) as excinfo:
            transcriber.transcribe(_audio_file(tmp_path))
    message = str(excinfo.value)
    assert "hft_transformer" in message
    assert "transkun" in message
    assert "torch" in message


# ── source.yaml documentation ──────────────────────────────────────────

def test_source_yaml_documents_every_hft_field() -> None:
    import re

    from sonitra.config import default_config_path

    text = default_config_path().read_text()
    entry = re.search(
        r"^[ \t]*# - type: hft_transformer$(?P<body>(?:\n[ \t]*#.*)*)", text, re.M
    )
    assert entry, "config/source.yaml has no commented hft_transformer entry"
    body = entry.group("body")
    for field in (
        "device",
        "checkpoint",
        "weights_path",
        "output",
        "n_stride",
        "onset_threshold",
        "offset_threshold",
        "mpe_threshold",
        "batch_size",
    ):
        assert re.search(rf"^[ \t]*#[ \t]+{field}:", body, re.M), field
    # The two facts that make the entry usable without reading the code.
    assert "GPU:N" in body
    assert "converted" in body
    assert "time-axis" in body


# ── slow tests (require torch) ─────────────────────────────────────────


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """Convert a fabricated tiny upstream pickle into a usable `model.pt`.

    The mel filter has to emit the tiny model's 8 bins rather than the released 256,
    otherwise the feature block does not match `n_bins` and the forward pass fails.
    """
    torch = pytest.importorskip("torch")
    from tests.test_hft_convert import TINY_HPARAMS, _FABRICATE

    from sonitra.transcribe._hft.convert import convert_checkpoint

    pkl_path = tmp_path / "tiny.pkl"
    reference_path = tmp_path / "tiny_reference.pt"
    source = REPO_ROOT / "misc" / "hft_transformer_spike" / "upstream" / "model" / "model_spec2midi.py"
    if not source.exists():
        pytest.skip(f"upstream module source not vendored at {source}")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_SRC), *sys.path])}
    result = subprocess.run(
        [sys.executable, "-c", _FABRICATE, str(pkl_path), str(reference_path), str(source), str(REPO_SRC)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    reference = torch.load(reference_path, weights_only=True)
    out_path = tmp_path / "tiny.pt"
    payload = convert_checkpoint(
        pkl_path, out_path, expected_state_digest=reference["state_digest"]
    )
    assert dict(payload["hparams"]) == TINY_HPARAMS
    payload["inference"]["mel_bins"] = int(payload["hparams"]["n_bins"])
    torch.save(payload, out_path)
    return out_path


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _tiny_checkpoint(tmp_path_factory.mktemp("tiny"))


@pytest.mark.slow
def test_hft_end_to_end_reports_every_metadata_key(tiny_weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    path = _audio_file(tmp_path, "tiny.wav", dur=1.0)
    transcriber = HftTransformerTranscriber(weights_path=tiny_weights)
    result = transcriber.transcribe(path)

    assert result.backend_type == "hft_transformer"
    assert result.transcriber == "hft_transformer"
    assert result.source_audio == path
    assert result.raw_outputs is None
    for key in EXPECTED_METADATA_KEYS:
        assert key in result.metadata, key
    assert set(result.metadata) == set(EXPECTED_METADATA_KEYS)
    assert result.metadata["package_version"] != "unknown"
    assert len(result.metadata["weights_sha256"]) == 64
    assert result.metadata["device"] == "cpu"
    assert result.metadata["requested_device"] == "cpu"
    assert result.metadata["device_available"] is True
    assert result.metadata["resampler"] == "pedalboard"
    assert result.metadata["sample_rate"] == 16000
    assert result.metadata["output_head"] == "second"
    assert result.metadata["n_stride"] == 0
    assert result.metadata["batch_size"] == 1
    assert result.metadata["thresholds"] == {
        "onset": 0.5,
        "offset": 0.5,
        "mpe": 0.5,
    }
    assert result.metadata["note_events_total"] == result.metadata["notes_kept"] + result.metadata[
        "filtered_dropped"
    ]
    assert result.metadata["notes_kept"] == len(result.notes)
    assert_notes_satisfy_contract(result.notes)


@pytest.mark.slow
def test_hft_first_head_and_stride_run(tiny_weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    path = _audio_file(tmp_path, "tiny.wav", dur=1.0)
    first = HftTransformerTranscriber(weights_path=tiny_weights, output="first").transcribe(path)
    strided = HftTransformerTranscriber(weights_path=tiny_weights, n_stride=4).transcribe(path)
    assert first.metadata["output_head"] == "first"
    assert strided.metadata["n_stride"] == 4
    assert_notes_satisfy_contract(first.notes)
    assert_notes_satisfy_contract(strided.notes)


@pytest.mark.slow
def test_hft_custom_name_is_reported_as_transcriber(tiny_weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    path = _audio_file(tmp_path, "tiny.wav", dur=0.5)
    result = HftTransformerTranscriber(weights_path=tiny_weights, name="maestro-run").transcribe(path)
    assert result.transcriber == "maestro-run"
    assert result.backend_type == "hft_transformer", "backend_type must stay the fixed discriminator"


@pytest.mark.slow
def test_hft_grad_mode_still_enabled_after_transcribe(
    tiny_weights: Path, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    path = _audio_file(tmp_path, "silence.wav", dur=1.0)
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()
    HftTransformerTranscriber(weights_path=tiny_weights).transcribe(path)
    assert torch.is_grad_enabled(), (
        "transcribe() must not leave grad disabled (use torch.no_grad, not set_grad_enabled(False))"
    )


@pytest.mark.slow
def test_hft_loads_the_model_once_for_two_threads(
    tiny_weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    import threading

    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    path = _audio_file(tmp_path, "tiny.wav", dur=1.0)
    transcriber = HftTransformerTranscriber(weights_path=tiny_weights)

    loads: list[str] = []
    real_load = torch.load

    def spy(*args: object, **kwargs: object):
        loads.append(str(args[0]) if args else "")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            transcriber.transcribe(path)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    monkeypatch.undo()

    assert not errors, errors
    assert loads, "the weights were never loaded"
    assert len(loads) == 1, f"the model was loaded {len(loads)} times: {loads}"


@pytest.mark.slow
def test_hft_short_audio_and_silence_produce_no_notes(tiny_weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import numpy as np
    from sonitra.storage import write_audio

    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    def write_silence(name: str, dur: float) -> Path:
        path = tmp_path / name
        write_audio(
            np.zeros((2, max(1, int(16000 * dur))), dtype=np.float32),
            path,
            sample_rate=16000,
            bit_depth=24,
            output_format="wav",
        )
        return path

    transcriber = HftTransformerTranscriber(weights_path=tiny_weights)
    for name, dur in (("shorter_than_one_hop.wav", 0.005), ("silence.wav", 0.5)):
        result = transcriber.transcribe(write_silence(name, dur))
        assert result.backend_type == "hft_transformer", name
        assert_notes_satisfy_contract(result.notes)
        assert result.metadata["notes_kept"] == len(result.notes), name

    # The tiny fixture's weights are random, so a constant feature block decodes to
    # activations around 0.73 whatever the audio is. Above that the zero-note claim is
    # about the plumbing — the block is built, windowed and decoded, and nothing fires.
    strict = HftTransformerTranscriber(
        weights_path=tiny_weights,
        onset_threshold=0.9,
        offset_threshold=0.9,
        mpe_threshold=0.9,
    )
    for name, dur in (("shorter_than_one_hop_strict.wav", 0.005), ("silence_strict.wav", 0.5)):
        result = strict.transcribe(write_silence(name, dur))
        assert result.notes == [], name
        assert result.metadata["note_events_total"] == 0, name
        assert result.metadata["notes_kept"] == 0, name
        assert result.metadata["thresholds"] == {"onset": 0.9, "offset": 0.9, "mpe": 0.9}


@pytest.mark.slow
def test_hft_reads_mp3_and_flac(tiny_weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import numpy as np
    from sonitra.storage import write_audio

    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    sig = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    for fmt in ("mp3", "flac"):
        path = tmp_path / f"tone.{fmt}"
        write_audio(
            np.stack([sig, sig]), path, sample_rate=sr, bit_depth=24, output_format=fmt
        )
        result = HftTransformerTranscriber(weights_path=tiny_weights).transcribe(path)
        assert result.backend_type == "hft_transformer", fmt
        assert_notes_satisfy_contract(result.notes)


@pytest.mark.slow
def test_hft_unreadable_audio_names_the_file(tiny_weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFFnot-audio")
    transcriber = HftTransformerTranscriber(weights_path=tiny_weights)
    with pytest.raises(TranscriptionError) as excinfo:
        transcriber.transcribe(broken)
    message = str(excinfo.value)
    assert str(broken) in message
    assert "hft_transformer" in message


@pytest.mark.slow
def test_hft_cuda_unavailable_raises(tiny_weights: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sonitra.transcribe.base import TranscriptionError
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber

    if torch.cuda.is_available():
        pytest.skip("CUDA is available, cannot test the unavailable path")
    path = _audio_file(tmp_path, "tiny.wav", dur=0.5)
    transcriber = HftTransformerTranscriber(device="cuda", weights_path=tiny_weights)
    with pytest.raises(TranscriptionError) as excinfo:
        transcriber.transcribe(path)
    assert "hft_transformer" in str(excinfo.value)


# ── opt-in: the released weights on real piano audio ───────────────────


def test_released_checkpoint_is_installed() -> None:
    """The opt-in parity test below needs the released weights; fail loudly if they went."""
    from sonitra.transcribe._hft import checkpoints

    assert checkpoints.checkpoint_path().exists(), (
        f"converted checkpoint missing at {checkpoints.checkpoint_path()}; "
        "the hft_transformer opt-in tests would silently skip"
    )


@pytest.mark.slow
def test_optin_real_piano_transcription_is_deterministic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sonitra.transcribe._hft import checkpoints

    if not checkpoints.checkpoint_path().exists():
        pytest.skip(f"converted checkpoint not installed at {checkpoints.checkpoint_path()}")
    if not RENDERED_PIANO.exists():
        pytest.skip(f"rendered piano fixture not available at {RENDERED_PIANO}")

    import numpy as np

    from sonitra.storage import read_audio_resampled, write_audio
    from sonitra.transcribe.hft_transformer import HftTransformerTranscriber
    from tests.helpers import assert_notes_satisfy_contract

    audio, sr = read_audio_resampled(RENDERED_PIANO, target_sr=16000)
    excerpt = audio[:, int(1.0 * sr) : int(9.0 * sr)]
    path = tmp_path / "excerpt.wav"
    write_audio(excerpt, path, sample_rate=sr, bit_depth=24, output_format="wav")

    transcriber = HftTransformerTranscriber()
    first = transcriber.transcribe(path)
    second = transcriber.transcribe(path)

    assert first.notes, "the released model detected nothing in 8 s of piano audio"
    assert first.notes == second.notes, "two calls on one instance must agree"
    assert all(21 <= n["pitch"] <= 108 for n in first.notes), "pitch offset regression"
    assert first.metadata["notes_kept"] == len(first.notes)
    assert first.metadata["upstream_commit"] == checkpoints.UPSTREAM_COMMIT
    assert first.metadata["checkpoint"] == "maestro"
    assert first.metadata["output_head"] == "second"
    assert_notes_satisfy_contract(first.notes)
    assert np.isfinite(np.array([n["start_sec"] for n in first.notes])).all()