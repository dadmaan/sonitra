"""Vendored hFT-Transformer — parity against upstream-recorded reference outputs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "hft"
REPO_SRC = Path(__file__).resolve().parents[1] / "src"

THRESHOLDS = {"onset_threshold": 0.5, "offset_threshold": 0.5, "mpe_threshold": 0.5}

NOTE_FIELDS = ("pitch", "onset", "offset", "velocity")

#: The checkpoint's `inference` block plus the architecture dimensions the
#: windowing and note decoding read from the same mapping.
INFERENCE = {
    "sr": 16000,
    "hop_sample": 256,
    "fft_bins": 2048,
    "window_length": 2048,
    "mel_bins": 256,
    "log_offset": 1e-08,
    "pad_mode": "constant",
    "window": "hann",
    "min_value": -18.42068099975586,
    "note_min": 21,
    "note_max": 108,
    "mel_norm": "slaney",
    "melfilter": "htk",
    "n_bins": 256,
    "margin_b": 32,
    "margin_f": 32,
    "num_frame": 128,
    "num_note": 88,
}

#: The `hparams` block the shipped weights were converted with.
HPARAMS = {
    "cnn_channel": 4,
    "cnn_kernel": 5,
    "dec_head": 4,
    "dec_layer": 3,
    "dropout": 0.1,
    "enc_head": 4,
    "enc_layer": 3,
    "hid_dim": 256,
    "margin_b": 32,
    "margin_f": 32,
    "n_bins": 256,
    "num_frame": 128,
    "num_note": 88,
    "num_velocity": 128,
    "pf_dim": 512,
}

TINY_HPARAMS = {
    "cnn_channel": 2,
    "cnn_kernel": 3,
    "dec_head": 2,
    "dec_layer": 1,
    "dropout": 0.1,
    "enc_head": 2,
    "enc_layer": 1,
    "hid_dim": 16,
    "margin_b": 4,
    "margin_f": 4,
    "n_bins": 8,
    "num_frame": 8,
    "num_note": 4,
    "num_velocity": 8,
    "pf_dim": 32,
}


def _manifest() -> dict:
    return json.loads((FIXTURES / "MANIFEST.json").read_text())


def _random_features() -> np.ndarray:
    """The fixture input block, regenerated from MANIFEST.json's generator constants."""
    generator = _manifest()["generator"]
    rng = np.random.default_rng(generator["random_seed"])
    values = rng.normal(
        generator["random_centre"],
        generator["random_spread"],
        size=(generator["random_frames"], generator["n_bins"]),
    ).astype(np.float32)
    return np.clip(values, generator["min_value"], None).astype(np.float32)


def _constant(frames: int, note_index: int, value: float, num_note: int = 88) -> np.ndarray:
    out = np.full((frames, num_note), 0.05, dtype=np.float32)
    out[:, note_index] = value
    return out


def _mpe2note_cases() -> dict[str, dict[str, np.ndarray]]:
    """The synthetic activation blocks behind each `mpe2note_*.npz` fixture.

    The archives store the resulting note lists rather than the inputs, so the
    inputs are rebuilt here from the same deterministic recipe: one
    `default_rng(7)` stream consumed in the order the cases are declared.
    """
    rng = np.random.default_rng(7)
    cases: dict[str, dict[str, np.ndarray]] = {}

    cases["plateau_above_threshold"] = {
        "onset": _constant(128, 40, 0.9),
        "offset": _constant(128, 40, 0.9),
        "mpe": _constant(128, 40, 0.05),
        "velocity": rng.integers(1, 128, size=(128, 88)).astype(np.int8),
    }

    cases["velocity_zero_ignored"] = {
        "onset": _constant(96, 55, 0.9),
        "offset": _constant(96, 55, 0.9),
        "mpe": _constant(96, 55, 0.05),
        "velocity": np.zeros((96, 88), dtype=np.int8),
    }

    at_zero = _constant(96, 30, 0.05)
    at_zero[0, 30] = 0.9
    cases["onset_at_frame_0"] = {
        "onset": at_zero,
        "offset": _constant(96, 30, 0.05),
        "mpe": _constant(96, 30, 0.9),
        "velocity": rng.integers(1, 128, size=(96, 88)).astype(np.int8),
    }

    retrigger = np.full((128, 88), 0.05, dtype=np.float32)
    retrigger[10, 60] = 0.9
    retrigger[70, 60] = 0.9
    retrigger[110, 60] = 0.95
    cases["same_pitch_overlap_trim"] = {
        "onset": retrigger,
        "offset": _constant(128, 60, 0.05),
        "mpe": _constant(128, 60, 0.05),
        "velocity": rng.integers(1, 128, size=(128, 88)).astype(np.int8),
    }

    at_end = _constant(96, 45, 0.05)
    at_end[-1, 45] = 0.9
    cases["note_to_last_frame"] = {
        "onset": at_end,
        "offset": _constant(96, 45, 0.05),
        "mpe": _constant(96, 45, 0.95),
        "velocity": rng.integers(1, 128, size=(96, 88)).astype(np.int8),
    }

    dense = rng.random((160, 88)).astype(np.float32)
    dense_velocity = rng.integers(0, 128, size=(160, 88)).astype(np.int8)
    dense_velocity[dense < 0.5] = 0
    cases["dense_random_onsets"] = {
        "onset": dense,
        "offset": (rng.random((160, 88)) * 0.9).astype(np.float32),
        "mpe": np.full((160, 88), 0.05, dtype=np.float32),
        "velocity": dense_velocity,
    }

    return cases


def _stored_notes(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / f"mpe2note_{name}.npz") as archive:
        return {field: archive[field] for field in NOTE_FIELDS}


def _notes_to_arrays(notes: list[dict]) -> dict[str, np.ndarray]:
    return {
        "pitch": np.array([note["pitch"] for note in notes], dtype=np.int16),
        "onset": np.array([note["onset"] for note in notes], dtype=np.float32),
        "offset": np.array([note["offset"] for note in notes], dtype=np.float32),
        "velocity": np.array([note["velocity"] for note in notes], dtype=np.int8),
    }


def _single_note_index(note_index: int, frames: int = 8) -> dict[str, np.ndarray]:
    """Activations with a single local maximum on one note column."""
    onset = _constant(frames, note_index, 0.05)
    onset[3, note_index] = 0.9
    return {
        "onset": onset,
        "offset": _constant(frames, note_index, 0.05),
        "mpe": _constant(frames, note_index, 0.05),
        "velocity": np.full((frames, 88), 64, dtype=np.int8),
    }


def _detect(case: dict[str, np.ndarray]) -> list[dict]:
    from sonitra.transcribe._hft.inference import mpe2note

    return mpe2note(case["onset"], case["offset"], case["mpe"], case["velocity"],
                    inference=INFERENCE, **THRESHOLDS)


# ── mpe2note parity ────────────────────────────────────────────────────

def test_mpe2note_matches_every_golden_case() -> None:
    manifest = _manifest()
    assert sorted(manifest["mpe2note_files"]) == sorted(
        f"mpe2note_{name}.npz" for name in _mpe2note_cases()
    )
    for name, case in _mpe2note_cases().items():
        expected = _stored_notes(name)
        notes = _detect(case)
        for note in notes:
            assert tuple(note) == NOTE_FIELDS, f"{name}: unexpected note keys {tuple(note)}"
        produced = _notes_to_arrays(notes)
        for field in NOTE_FIELDS:
            np.testing.assert_array_equal(
                produced[field], expected[field],
                err_msg=f"{name}: {field} differs from the recorded upstream output",
            )
        # Ordering is by onset, then pitch.
        keys = [(note["onset"], note["pitch"]) for note in notes]
        assert keys == sorted(keys), f"{name}: notes are not sorted by (onset, pitch)"


def test_mpe2note_pitch_index_0_is_midi_21() -> None:
    notes = _detect(_single_note_index(0))
    assert [note["pitch"] for note in notes] == [21]


def test_mpe2note_pitch_index_87_is_midi_108() -> None:
    notes = _detect(_single_note_index(87))
    assert [note["pitch"] for note in notes] == [108]


def test_mpe2note_pitch_offset_is_note_min() -> None:
    """Pitch is the note index plus `note_min`; neither +0 nor +42 may appear."""
    for note_index in (0, 21, 42, 66, 87):
        notes = _detect(_single_note_index(note_index))
        assert [note["pitch"] for note in notes] == [note_index + INFERENCE["note_min"]], (
            f"note index {note_index} must map to pitch {note_index + INFERENCE['note_min']}"
        )
    # A missing offset would map index 21 onto 21 and index 42 onto 42.
    assert [note["pitch"] for note in _detect(_single_note_index(21))] == [42]
    assert [note["pitch"] for note in _detect(_single_note_index(42))] == [63]
    # A doubled offset would put the top key two octaves too high; it never appears.
    assert max(note["pitch"] for note in _detect(_single_note_index(87))) == 108
    assert all(
        note["pitch"] <= INFERENCE["note_max"]
        for note in _detect(_single_note_index(87))
    )


def test_mpe2note_velocity_zero_dropped_before_overlap_trim() -> None:
    manifest = _manifest()["edge_cases"]

    zero_only = _mpe2note_cases()["velocity_zero_ignored"]
    assert (zero_only["velocity"] == 0).any(), "fixture must contain zero velocities"
    assert manifest["velocity_zero_ignored"]["zero_velocity_rows"] > 0
    # The branch is only meaningful if a peak really is detected and then dropped.
    assert (zero_only["onset"] >= 0.5).any()
    assert _detect(zero_only) == []

    dense = _mpe2note_cases()["dense_random_onsets"]
    assert int((dense["velocity"] == 0).any(axis=1).sum()) > 0
    notes = _detect(dense)
    assert notes, "the dense case must still produce notes"
    assert all(note["velocity"] > 0 for note in notes)
    # Dropping before the trim is observable: no velocity-0 note exists to be trimmed
    # into a velocity-0 note, and the note count matches the recorded output.
    assert len(notes) == len(_stored_notes("dense_random_onsets")["pitch"])


def test_mpe2note_times_are_frame_times_and_interpolation_is_bounded() -> None:
    hop = INFERENCE["hop_sample"] / INFERENCE["sr"]
    assert hop == pytest.approx(0.016)

    for name, case in _mpe2note_cases().items():
        frames = int(case["onset"].shape[0])
        notes = _detect(case)
        for note in notes:
            # Onsets and offsets are frame times, possibly pulled by at most half a hop
            # by the parabolic interpolation around the peak.
            for field in ("onset", "offset"):
                assert abs(note[field] / hop - round(note[field] / hop)) <= 0.5 + 1e-5, (
                    f"{name}: {field} {note[field]} is more than half a hop from a frame time"
                )
            # An offset can never point past the last frame the detector saw.
            assert note["offset"] <= (frames - 1) * hop + 1e-6, f"{name}: offset past the last frame"

    # The frame-0 branch needs no interpolation, so the time is exactly 0.
    assert _detect(_mpe2note_cases()["onset_at_frame_0"])[0]["onset"] == 0.0

    # A tie on every frame takes the equality branch, so every onset is an exact
    # multiple of the hop and the offsets advance by exactly one hop per note.
    plateau = _detect(_mpe2note_cases()["plateau_above_threshold"])
    np.testing.assert_array_equal(
        np.array([note["onset"] for note in plateau], dtype=np.float32),
        (np.arange(len(plateau)) * hop).astype(np.float32),
    )

    # A note that only ends because the arrays ran out offsets to the final frame.
    at_end = _detect(_mpe2note_cases()["note_to_last_frame"])
    assert at_end[0]["offset"] == pytest.approx(95 * hop)


def test_mpe2note_trims_same_pitch_overlaps_to_the_next_onset() -> None:
    """A note whose offset detection lands after the next same-pitch onset is cut short.

    The offset activation's parabolic interpolation can place the offset up to half a
    hop past the next onset, and only the overlap trim keeps the two from overlapping.
    A flat plateau cannot reach that state — the offset would land exactly on the next
    onset — so the offset peak needs a rising right neighbour.
    """
    frames = 32
    hop = INFERENCE["hop_sample"] / INFERENCE["sr"]
    note_index = 60

    onset = np.full((frames, 88), 0.05, dtype=np.float32)
    onset[10, note_index] = 0.9
    onset[20, note_index] = 0.9
    offset = np.full((frames, 88), 0.05, dtype=np.float32)
    offset[19, note_index] = 0.2
    offset[20, note_index] = 0.9
    offset[21, note_index] = 0.5
    mpe = np.full((frames, 88), 0.95, dtype=np.float32)
    velocity = np.full((frames, 88), 64, dtype=np.int8)

    trimmed = _detect({"onset": onset, "offset": offset, "mpe": mpe, "velocity": velocity})
    assert [note["onset"] for note in trimmed] == [10 * hop, 20 * hop]
    assert trimmed[0]["offset"] == trimmed[1]["onset"]

    # Without a following onset there is nothing to trim against, and the same offset
    # peak reports the interpolated time the trim above removed.
    lone = onset.copy()
    lone[20, note_index] = 0.05
    untrimmed = _detect({"onset": lone, "offset": offset, "mpe": mpe, "velocity": velocity})
    assert len(untrimmed) == 1
    assert untrimmed[0]["offset"] == pytest.approx(20 * hop + 0.5 * hop * 0.3 / 0.7)
    assert untrimmed[0]["offset"] > trimmed[0]["offset"] + 1e-6


def test_mpe2note_golden_retrigger_case_has_three_notes_on_one_pitch() -> None:
    notes = _detect(_mpe2note_cases()["same_pitch_overlap_trim"])
    assert [note["pitch"] for note in notes] == [81, 81, 81]
    hop = INFERENCE["hop_sample"] / INFERENCE["sr"]
    assert [note["onset"] for note in notes] == pytest.approx([10 * hop, 70 * hop, 110 * hop])


# ── import isolation ───────────────────────────────────────────────────

def test_inference_and_checkpoints_import_without_torch() -> None:
    code = (
        "import sys\n"
        "import sonitra.transcribe._hft\n"
        "import sonitra.transcribe._hft.inference\n"
        "import sonitra.transcribe._hft.checkpoints\n"
        # The converter lands in a separate change; only require the modules that exist.
        "try:\n"
        "    import sonitra.transcribe._hft.convert\n"
        "except ImportError:\n"
        "    pass\n"
        # mpe2note must stay free of torch on its own path too.
        "import numpy as np\n"
        "from sonitra.transcribe._hft.inference import mpe2note\n"
        "before = set(sys.modules)\n"
        "onset = np.full((8, 88), 0.05, dtype=np.float32)\n"
        "onset[3, 0] = 0.9\n"
        "config = {'num_note': 88, 'note_min': 21, 'hop_sample': 256, 'sr': 16000}\n"
        "notes = mpe2note(onset, onset, onset, np.full((8, 88), 64, dtype=np.int8),"
        " inference=config)\n"
        "assert notes[0]['pitch'] == 21, notes\n"
        "loaded = set(sys.modules) - before\n"
        "leaked = sorted(m for m in loaded if m.split('.')[0] in {'torch', 'torchaudio'})\n"
        "assert not leaked, leaked\n"
        "assert 'torch' not in sys.modules, 'inference pulled in torch'\n"
        "assert 'torchaudio' not in sys.modules, 'inference pulled in torchaudio'\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_SRC), *sys.path])
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


# ── module structure ───────────────────────────────────────────────────

@pytest.mark.slow
def test_tiny_model_forward_shapes() -> None:
    import torch

    from sonitra.transcribe._hft.model import build_model

    torch.manual_seed(0)
    model = build_model(TINY_HPARAMS, device="cpu").eval()
    length = TINY_HPARAMS["margin_b"] + TINY_HPARAMS["num_frame"] + TINY_HPARAMS["margin_f"]
    spec = torch.zeros(1, TINY_HPARAMS["n_bins"], length)
    with torch.no_grad():
        outputs = model(spec)

    assert len(outputs) == 9
    frame_shape = (1, TINY_HPARAMS["num_frame"], TINY_HPARAMS["num_note"])
    velocity_shape = frame_shape + (TINY_HPARAMS["num_velocity"],)
    for position in (0, 1, 2, 5, 6, 7):
        assert tuple(outputs[position].shape) == frame_shape, position
    for position in (3, 8):
        assert tuple(outputs[position].shape) == velocity_shape, position
    assert tuple(outputs[4].shape) == (
        1,
        TINY_HPARAMS["num_frame"],
        TINY_HPARAMS["dec_head"],
        TINY_HPARAMS["num_note"],
        TINY_HPARAMS["n_bins"],
    )


@pytest.mark.slow
def test_build_model_rejects_asymmetric_margins() -> None:
    from sonitra.transcribe._hft.model import build_model

    bad = dict(TINY_HPARAMS, margin_f=TINY_HPARAMS["margin_b"] + 1)
    with pytest.raises(ValueError, match="margin"):
        build_model(bad, device="cpu")


@pytest.mark.slow
def test_build_model_names_the_missing_hparam() -> None:
    from sonitra.transcribe._hft.model import build_model

    incomplete = dict(TINY_HPARAMS)
    del incomplete["pf_dim"]
    with pytest.raises(KeyError, match="pf_dim"):
        build_model(incomplete, device="cpu")


@pytest.mark.slow
def test_no_tensor_attributes_outside_parameters_and_buffers() -> None:
    import torch
    import torch.nn as nn

    from sonitra.transcribe._hft.model import build_model

    def offenders(module: nn.Module, prefix: str = "") -> list[str]:
        found: list[str] = []
        for name, value in vars(module).items():
            if name in ("_parameters", "_buffers"):
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, torch.Tensor):
                found.append(path)
            elif isinstance(value, nn.Module):
                found.extend(offenders(value, path))
            elif isinstance(value, (list, tuple)):
                found.extend(
                    f"{path}[{index}]" for index, item in enumerate(value)
                    if isinstance(item, torch.Tensor)
                )
        return found

    model = build_model(TINY_HPARAMS, device="cpu")
    # `.to(device)` only moves registered parameters and buffers, so a tensor parked in a
    # plain attribute would silently stay behind and break non-CPU inference.
    model = model.to("cpu")
    assert offenders(model) == []
    head_dim = TINY_HPARAMS["hid_dim"] // TINY_HPARAMS["enc_head"]
    assert model.encoder_spec2midi.scale_freq == pytest.approx(TINY_HPARAMS["hid_dim"] ** 0.5)
    assert model.decoder_spec2midi.scale_time == pytest.approx(TINY_HPARAMS["hid_dim"] ** 0.5)
    assert model.encoder_spec2midi.layers_freq[0].self_attention.scale == pytest.approx(
        head_dim**0.5
    )
    for attribute in ("scale_freq", "scale_time", "scale"):
        for module in model.modules():
            if hasattr(module, attribute):
                assert isinstance(getattr(module, attribute), float), (
                    f"{type(module).__name__}.{attribute} must be a plain float"
                )


@pytest.mark.slow
def test_forward_passes_device_to_every_torch_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from sonitra.transcribe._hft.model import build_model

    torch.manual_seed(0)
    model = build_model(TINY_HPARAMS, device="cpu").eval()
    length = TINY_HPARAMS["margin_b"] + TINY_HPARAMS["num_frame"] + TINY_HPARAMS["margin_f"]
    spec = torch.zeros(1, TINY_HPARAMS["n_bins"], length)

    devices: list[object] = []
    real_arange = torch.arange

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        assert "device" in kwargs, f"torch.arange({args!r}) was called without device="
        devices.append(kwargs["device"])
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", spy)
    with torch.no_grad():
        model(spec)
    assert devices, "the forward never called torch.arange, so nothing was verified"
    assert all(device == spec.device for device in devices)


@pytest.mark.slow
def test_build_model_state_dict_matches_the_converted_checkpoint() -> None:
    import torch

    from sonitra.transcribe._hft import checkpoints
    from sonitra.transcribe._hft.model import build_model

    payload = torch.load(checkpoints.checkpoint_path(), weights_only=False)
    assert payload["format_version"] == checkpoints.FORMAT_VERSION
    assert dict(payload["hparams"]) == HPARAMS
    assert payload["state_dict"].__len__() == 165

    model = build_model(payload["hparams"], device="cpu")
    keys = list(model.state_dict().keys())
    assert len(keys) == 165
    # Order matters too: the converter recorded the tensors in module order.
    assert keys == list(payload["state_dict"].keys())
    model.load_state_dict(payload["state_dict"], strict=True)


@pytest.mark.slow
def test_converted_checkpoint_is_installed() -> None:
    """The parity tests below need the shipped weights; fail loudly if they went missing."""
    from sonitra.transcribe._hft import checkpoints

    assert checkpoints.checkpoint_path().exists(), (
        f"converted checkpoint missing at {checkpoints.checkpoint_path()}; "
        "the upstream parity tests would silently skip"
    )


@pytest.fixture(scope="module")
def weights() -> dict:
    from sonitra.transcribe._hft import checkpoints

    path = checkpoints.checkpoint_path()
    if not path.exists():
        pytest.skip(f"converted checkpoint not installed at {path}")
    torch = pytest.importorskip("torch")

    payload = torch.load(path, weights_only=False)
    return {
        "hparams": payload["hparams"],
        "inference": payload["inference"],
        "state_dict": payload["state_dict"],
    }


@pytest.fixture(scope="module")
def loaded_model(weights: dict):
    import torch

    from sonitra.transcribe._hft.model import build_model

    model = build_model(weights["hparams"], device="cpu")
    model.load_state_dict(weights["state_dict"], strict=True)
    return model.eval()


def _runtime_config(weights: dict) -> dict:
    """The mapping the vendored functions take: the checkpoint's `inference` block
    extended with the architecture dimensions the windowing reads."""
    return {
        **weights["inference"],
        **{key: weights["hparams"][key] for key in ("n_bins", "margin_b", "margin_f", "num_frame", "num_note")},
    }


def test_wav2feature_matches_the_upstream_mel_chain(weights: dict) -> None:
    """Not marked slow: torchaudio's mel filter is cheap and runs in well under a second."""
    import torch
    import torchaudio

    from sonitra.transcribe._hft.inference import make_mel_transform, wav2feature

    inference = weights["inference"]
    rng = np.random.default_rng(12345)
    times = np.arange(inference["sr"], dtype=np.float64) / inference["sr"]
    wave = np.zeros(inference["sr"], dtype=np.float32)
    for frequency, decay, amplitude in ((220.0, 2.0, 0.5), (440.0, 4.0, 0.3), (997.0, 7.5, 0.2)):
        wave += amplitude * np.exp(-decay * times) * np.sin(2.0 * np.pi * frequency * times)
    wave = (wave + 0.01 * rng.standard_normal(inference["sr"])).astype(np.float32)

    produced = wav2feature(
        torch.from_numpy(wave), inference, mel=make_mel_transform(inference, "cpu")
    )

    reference_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=inference["sr"],
        n_fft=inference["fft_bins"],
        win_length=inference["window_length"],
        hop_length=inference["hop_sample"],
        pad_mode=inference["pad_mode"],
        n_mels=inference["mel_bins"],
        norm=inference["mel_norm"],
        mel_scale=inference["melfilter"],
    )
    expected = (
        torch.log(reference_transform(torch.from_numpy(wave)) + inference["log_offset"])
    ).T.numpy()

    assert produced.shape == expected.shape == (inference["sr"] // inference["hop_sample"] + 1, inference["mel_bins"])
    assert produced.dtype == np.float32
    np.testing.assert_allclose(produced, expected, atol=1e-6, rtol=0.0)


@pytest.mark.slow
def test_transcript_matches_golden_arrays(loaded_model, weights: dict) -> None:
    from sonitra.transcribe._hft.inference import transcript

    features = _random_features()
    assert list(features.shape) == [256, 256]
    arrays = transcript(features, loaded_model, inference=_runtime_config(weights), device="cpu")

    names = ("onset_A", "offset_A", "mpe_A", "velocity_A", "onset_B", "offset_B", "mpe_B", "velocity_B")
    assert len(arrays) == 8
    for name, array in zip(names, arrays):
        golden = np.load(FIXTURES / f"transcript_{name}.npy")
        assert array.dtype == golden.dtype, name
        assert array.shape == golden.shape == (256, 88), name
        np.testing.assert_array_equal(array, golden, err_msg=f"transcript_{name} differs")


@pytest.mark.slow
def test_transcript_stride_output_length_follows_the_half_window_rounding(
    loaded_model, weights: dict
) -> None:
    """Stride runs every half-window and rounds the length up to a whole number of them.

    The padded tail reaches the note detector, so an offset can land past the end of
    the audio; the tail is exactly zero when the input already holds a whole number of
    half-windows.
    """
    from sonitra.transcribe._hft.inference import transcript_stride

    inference = _runtime_config(weights)
    features = _random_features()
    hparams = weights["hparams"]
    half_frame = hparams["num_frame"] // 2
    names = ("onset_A", "offset_A", "mpe_A", "velocity_A", "onset_B", "offset_B", "mpe_B", "velocity_B")

    lengths = {}
    for frames in (256, 200, 129):
        padded = frames + hparams["margin_b"] + hparams["margin_f"] + half_frame
        tail = int(np.ceil(padded / half_frame) * half_frame) - padded
        arrays = transcript_stride(
            features[:frames], loaded_model, inference=inference, device="cpu", n_offset=32
        )
        lengths[frames] = frames + tail
        assert len(arrays) == 8
        for name, array in zip(names, arrays):
            assert array.shape == (frames + tail, 88), (frames, name)
            assert np.all(np.isfinite(array)), (frames, name)
        for name, array in zip(
            ("onset_A", "offset_A", "mpe_A", "onset_B", "offset_B", "mpe_B"),
            (arrays[0], arrays[1], arrays[2], arrays[4], arrays[5], arrays[6]),
        ):
            assert array.min() >= 0.0 and array.max() <= 1.0, (frames, name)
        assert arrays[3].dtype == np.int8 and arrays[7].dtype == np.int8

    # 256 frames is already a whole number of half-windows, so nothing is added.
    assert lengths == {256: 256, 200: 256, 129: 192}
    assert lengths[200] > 200 and lengths[129] > 129


@pytest.mark.slow
def test_batch_size_four_agrees_with_batch_size_one(loaded_model, weights: dict) -> None:
    """Grouping windows into one forward must not change the activations.

    The recorded random features drive the model far below every detection threshold,
    so the note-list comparison is run at a threshold low enough that both heads really
    do produce notes; otherwise it would compare two empty lists and prove nothing.
    """
    from sonitra.transcribe._hft.inference import mpe2note, transcript

    inference = _runtime_config(weights)
    features = _random_features()
    single = transcript(features, loaded_model, inference=_runtime_config(weights), device="cpu", batch_size=1)
    batched = transcript(features, loaded_model, inference=inference, device="cpu", batch_size=4)

    names = ("onset_A", "offset_A", "mpe_A", "velocity_A", "onset_B", "offset_B", "mpe_B", "velocity_B")
    for name, left, right in zip(names, single, batched):
        assert left.dtype == right.dtype, name
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=0.0, err_msg=f"{name} differs")

    # mode_velocity='org' keeps the velocity-0 notes this noise-like input produces.
    permissive = {
        "onset_threshold": 1e-06,
        "offset_threshold": 1e-06,
        "mpe_threshold": 1e-06,
        "mode_velocity": "org",
    }
    for base in (0, 4):
        left = mpe2note(single[base], single[base + 1], single[base + 2], single[base + 3],
                        inference=inference, **permissive)
        right = mpe2note(batched[base], batched[base + 1], batched[base + 2], batched[base + 3],
                         inference=inference, **permissive)
        assert left == right, f"head at +{base}: note lists differ between batch sizes"

    counts = [
        len(mpe2note(single[base], single[base + 1], single[base + 2], single[base + 3],
                     inference=inference, **permissive))
        for base in (0, 4)
    ]
    assert all(count > 0 for count in counts), f"note lists are empty, so this proves nothing: {counts}"