"""One-time conversion of the upstream pickled module into a plain state-dict file."""
from __future__ import annotations

import io
import json
import os
import pickle
import pickletools
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"

#: Vendored copy of the upstream module source and of the released checkpoint, so the
#: pickle the converter is built for can be exercised without a network fetch.
UPSTREAM_MODEL_SOURCE = (
    REPO_ROOT / "misc" / "hft_transformer_spike" / "upstream" / "model" / "model_spec2midi.py"
)
ENUMERATED_GLOBALS = (
    REPO_ROOT / "misc" / "hft_transformer_spike" / "out" / "a1_globals.json"
)
RELEASED_CHECKPOINT = (
    REPO_ROOT
    / "misc"
    / "hft_transformer_spike"
    / "download"
    / "extracted"
    / "checkpoint"
    / "MAESTRO-V3"
    / "model_016_003.pkl"
)
RELEASED_PARAMETER = RELEASED_CHECKPOINT.with_name("parameter.json")

TINY_HPARAMS: dict[str, object] = {
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

#: The hyperparameters the released checkpoint is pinned to.
RELEASED_HPARAMS: dict[str, object] = {
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

INFERENCE: dict[str, object] = {
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
}

#: Builds an upstream-shaped pickle in a child process: the upstream module source is
#: exec'd under the module name its pickle references, a tiny model is constructed from
#: it and dumped with protocol 4. The child's reference forward outputs and its own
#: state digest are saved next to the pickle so the parent can compare both.
_FABRICATE = '''
import pickle
import sys
import types
from pathlib import Path

import torch

pkl_path, reference_path, source_path, repo_src = sys.argv[1:5]
sys.path.insert(0, repo_src)
from sonitra.transcribe._hft import checkpoints

source = Path(source_path).read_text()
package = types.ModuleType("model")
package.__path__ = []
module = types.ModuleType("model.model_spec2midi")
module.__file__ = source_path
sys.modules["model"] = package
sys.modules["model.model_spec2midi"] = module
exec(compile(source, source_path, "exec"), module.__dict__)

hparams = {
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
margin = hparams["margin_b"]
encoder = module.Encoder_SPEC2MIDI(
    n_margin=margin,
    n_frame=hparams["num_frame"],
    n_bin=hparams["n_bins"],
    cnn_channel=hparams["cnn_channel"],
    cnn_kernel=hparams["cnn_kernel"],
    hid_dim=hparams["hid_dim"],
    n_layers=hparams["enc_layer"],
    n_heads=hparams["enc_head"],
    pf_dim=hparams["pf_dim"],
    dropout=hparams["dropout"],
    device="cpu",
)
decoder = module.Decoder_SPEC2MIDI(
    n_frame=hparams["num_frame"],
    n_bin=hparams["n_bins"],
    n_note=hparams["num_note"],
    n_velocity=hparams["num_velocity"],
    hid_dim=hparams["hid_dim"],
    n_layers=hparams["dec_layer"],
    n_heads=hparams["dec_head"],
    pf_dim=hparams["pf_dim"],
    dropout=hparams["dropout"],
    device="cpu",
)
torch.manual_seed(0)
model = module.Model_SPEC2MIDI(encoder, decoder)
model.eval()

with open(pkl_path, "wb") as handle:
    pickle.dump(model, handle, protocol=4)

generator = torch.Generator().manual_seed(11)
length = hparams["margin_b"] + hparams["num_frame"] + hparams["margin_f"]
spec = torch.randn(1, hparams["n_bins"], length, generator=generator)
with torch.no_grad():
    outputs = [value.clone() for value in model(spec)]

torch.save(
    {
        "hparams": hparams,
        "spec": spec,
        "outputs": outputs,
        "state_digest": checkpoints.state_digest(model.state_dict()),
    },
    reference_path,
)
print("ok")
'''


def _build_model(hparams: dict) -> object:
    try:
        from sonitra.transcribe._hft.model import build_model
    except ImportError as exc:  # pragma: no cover - depends on the sibling module
        pytest.skip(f"_hft.model is not importable yet: {exc}")
    return build_model(hparams, device="cpu")


@pytest.fixture(scope="module")
def upstream_pickle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """`(pickle, reference)` for a tiny model pickled by the real upstream module."""
    if not UPSTREAM_MODEL_SOURCE.exists():
        pytest.skip(f"upstream module source not vendored at {UPSTREAM_MODEL_SOURCE}")
    workdir = tmp_path_factory.mktemp("upstream_pickle")
    pkl_path = workdir / "tiny.pkl"
    reference_path = workdir / "tiny_reference.pt"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_SRC), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    result = subprocess.run(
        [sys.executable, "-c", _FABRICATE, str(pkl_path), str(reference_path),
         str(UPSTREAM_MODEL_SOURCE), str(REPO_SRC)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok", result.stdout
    return pkl_path, reference_path


def _reference(reference_path: Path) -> dict:
    torch = pytest.importorskip("torch")
    return torch.load(reference_path, weights_only=True)


# ── end-to-end conversion of an upstream-shaped pickle ─────────────────


@pytest.mark.slow
def test_upstream_pickle_converts_and_reproduces_the_forward_pass(
    upstream_pickle: tuple[Path, Path], tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")

    from sonitra.transcribe._hft import checkpoints
    from sonitra.transcribe._hft.convert import convert_checkpoint

    pkl_path, reference_path = upstream_pickle
    reference = _reference(reference_path)
    out_path = tmp_path / "model.pt"

    payload = convert_checkpoint(
        pkl_path, out_path, expected_state_digest=reference["state_digest"]
    )

    assert out_path.exists()
    assert payload["provenance"]["state_digest"] == reference["state_digest"]
    assert payload["provenance"]["state_digest"] == checkpoints.state_digest(payload["state_dict"])
    assert dict(payload["hparams"]) == TINY_HPARAMS
    assert payload["inference"] == INFERENCE
    assert payload["format_version"] == checkpoints.FORMAT_VERSION

    # The runtime path: no unpickling of the upstream module, weights_only is enough.
    loaded = torch.load(out_path, weights_only=True)
    assert set(loaded) == {
        "format_version",
        "state_dict",
        "hparams",
        "inference",
        "provenance",
    }

    model = _build_model(loaded["hparams"]).eval()
    missing, unexpected = model.load_state_dict(loaded["state_dict"], strict=True)
    assert missing == [] and unexpected == []

    with torch.no_grad():
        produced = model(reference["spec"])
    assert len(produced) == len(reference["outputs"])
    for index, (got, want) in enumerate(zip(produced, reference["outputs"])):
        assert got.dtype == want.dtype, index
        assert tuple(got.shape) == tuple(want.shape), index
        # The vendored module is upstream's own arithmetic on the same device, so the
        # conversion has to reproduce the pickled model's outputs bit for bit.
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"output {index} differs")


# ── the unpickling allowlist ────────────────────────────────────────────


def test_find_class_refuses_a_global_outside_the_allowlist(tmp_path: Path) -> None:
    """A pickle naming a dangerous callable must be refused without running anything."""
    from sonitra.transcribe._hft import convert

    sentinel = tmp_path / "executed"
    payload = tmp_path / "dangerous.pkl"

    class Shell:
        def __reduce__(self) -> tuple[object, tuple[str, ...]]:
            return (os.system, (f'touch "{sentinel}"',))

    payload.write_bytes(pickle.dumps(Shell(), protocol=2))
    # Sanity check that the fixture really does name the callable by reference.
    globals_named = {
        arg.replace(" ", ".")
        for op, arg, _pos in pickletools.genops(io.BytesIO(payload.read_bytes()))
        if op.name == "GLOBAL" and isinstance(arg, str) and " " in arg
    }
    assert any(name.endswith(".system") for name in globals_named), globals_named

    with pytest.raises(pickle.UnpicklingError) as excinfo:
        convert.load_stub(payload)

    message = str(excinfo.value)
    assert "allowlist" in message
    assert "system" in message
    assert not sentinel.exists(), "the refused global was still called"


def test_allowlist_matches_the_enumerated_globals() -> None:
    from sonitra.transcribe._hft import convert

    if not ENUMERATED_GLOBALS.exists():
        pytest.skip(f"enumerated global list not vendored at {ENUMERATED_GLOBALS}")
    enumerated = json.loads(ENUMERATED_GLOBALS.read_text())["allowlist"]
    expected = {tuple(entry.split(":", 1)) for entry in enumerated}
    assert len(expected) == len(enumerated) == 19
    assert set(convert.ALLOWED_GLOBALS) == expected


def test_every_allowlisted_global_resolves() -> None:
    from sonitra.transcribe._hft import convert

    for module, name in sorted(convert.ALLOWED_GLOBALS):
        unpickler = convert.AllowlistUnpickler(io.BytesIO(b""))
        resolved = unpickler.find_class(module, name)
        assert resolved is not None, f"{module}.{name} did not resolve"


# ── the CPU storage redirect ────────────────────────────────────────────


@pytest.mark.slow
def test_storage_redirect_forces_cpu_and_weights_only(
    upstream_pickle: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    from sonitra.transcribe._hft.convert import convert_checkpoint

    pkl_path, reference_path = upstream_pickle
    reference = _reference(reference_path)
    calls: list[tuple[tuple[object, ...], dict]] = []
    real_load = torch.load

    def spy(*args: object, **kwargs: object):
        calls.append((args, dict(kwargs)))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    out_path = tmp_path / "model.pt"
    payload = convert_checkpoint(
        pkl_path, out_path, expected_state_digest=reference["state_digest"]
    )
    monkeypatch.undo()

    assert calls, "the redirect never called torch.load, so nothing was verified"
    for args, kwargs in calls:
        # The inner storage payload is handed over as a byte string, never as a path.
        assert len(args) == 1 and isinstance(args[0], io.BytesIO), args
        assert kwargs.get("map_location") == "cpu", kwargs
        assert kwargs.get("weights_only") is True, kwargs
    assert payload["provenance"]["state_digest"] == reference["state_digest"]
    assert all(
        tensor.device.type == "cpu" and tensor.dtype is torch.float32
        for tensor in payload["state_dict"].values()
    )
    loaded = torch.load(out_path, weights_only=True)
    assert set(loaded["state_dict"]) == set(payload["state_dict"])


# ── failure paths ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_asymmetric_margins_are_refused(upstream_pickle: tuple[Path, Path],
                                        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe._hft import convert

    pkl_path, reference_path = upstream_pickle
    # The upstream encoder records one context-window width, so the two margins can only
    # disagree if the recovery is wrong; the guard is put out of reach on purpose.
    monkeypatch.setattr(
        convert, "extract_hparams", lambda root: dict(TINY_HPARAMS, margin_f=TINY_HPARAMS["margin_b"] + 1)
    )
    out_path = tmp_path / "model.pt"
    with pytest.raises(convert.ConversionError, match="margin"):
        convert.convert_checkpoint(
            pkl_path, out_path, expected_state_digest=_reference(reference_path)["state_digest"]
        )
    assert not out_path.exists()


@pytest.mark.slow
def test_wrong_expected_digest_writes_nothing(upstream_pickle: tuple[Path, Path],
                                              tmp_path: Path) -> None:
    from sonitra.transcribe._hft import convert

    pkl_path, _ = upstream_pickle
    out_path = tmp_path / "model.pt"
    with pytest.raises(convert.DigestMismatchError) as excinfo:
        convert.convert_checkpoint(pkl_path, out_path, expected_state_digest="00" * 32)
    message = str(excinfo.value)
    assert "00" * 32 in message
    assert "digest" in message
    assert not out_path.exists(), "a partial or mismatched checkpoint was left on disk"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_parameter_json_disagreement_is_an_error(upstream_pickle: tuple[Path, Path],
                                                 tmp_path: Path) -> None:
    from sonitra.transcribe._hft import convert

    pkl_path, reference_path = upstream_pickle
    wrong = tmp_path / "parameter.json"
    wrong.write_text(json.dumps({"transformer": {"hid_dim": 999}}))
    out_path = tmp_path / "model.pt"
    with pytest.raises(convert.ConversionError, match="hid_dim"):
        convert.convert_checkpoint(
            pkl_path,
            out_path,
            parameter_path=wrong,
            expected_state_digest=_reference(reference_path)["state_digest"],
        )
    assert not out_path.exists()


def test_cross_check_hparams_names_every_disagreement() -> None:
    from sonitra.transcribe._hft.convert import cross_check_hparams

    parameter = {
        "cnn": {"channel": 4, "kernel": 5},
        "transformer": {
            "hid_dim": 256,
            "pf_dim": 512,
            "encoder": {"n_layer": 3, "n_head": 4},
            "decoder": {"n_layer": 3, "n_head": 4},
        },
        "training": {"dropout": 0.1},
    }
    assert cross_check_hparams(RELEASED_HPARAMS, parameter) == []
    assert cross_check_hparams(dict(RELEASED_HPARAMS, hid_dim=128), parameter) == ["hid_dim"]
    assert cross_check_hparams(dict(RELEASED_HPARAMS, dropout=0.5), parameter) == ["dropout"]
    # A section the release did not ship is skipped, not reported as a disagreement.
    assert cross_check_hparams(RELEASED_HPARAMS, {}) == []


# ── plain-scalar payload (the TorchVersion trap) ────────────────────────


@pytest.mark.slow
def test_every_scalar_in_the_payload_is_plain(upstream_pickle: tuple[Path, Path],
                                             tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import numpy as np

    from sonitra.transcribe._hft.convert import convert_checkpoint

    pkl_path, reference_path = upstream_pickle
    out_path = tmp_path / "model.pt"
    convert_checkpoint(
        pkl_path, out_path, expected_state_digest=_reference(reference_path)["state_digest"]
    )

    payload = torch.load(out_path, weights_only=True)
    assert type(payload["format_version"]) is int
    # torch.__version__ is a str subclass; torch.save records the real class and
    # weights_only=True then refuses the whole file, so it must be a plain str.
    assert type(payload["provenance"]["torch_version_at_conversion"]) is str
    assert payload["provenance"]["torch_version_at_conversion"] == str(torch.__version__)
    for block in ("hparams", "inference", "provenance"):
        for key, value in payload[block].items():
            assert type(value) in (int, float, str), f"{block}.{key} is {type(value).__name__}"
    assert type(payload["inference"]["min_value"]) is float
    assert payload["inference"]["min_value"] == pytest.approx(
        float(np.log(np.float32(1e-8)))
    )
    for key, value in payload["provenance"].items():
        assert type(value) is str, key


@pytest.mark.slow
def test_torch_version_left_unconverted_would_break_weights_only(
    upstream_pickle: tuple[Path, Path], tmp_path: Path
) -> None:
    """The reason every scalar is coerced: the un-coerced class is not allowlisted."""
    torch = pytest.importorskip("torch")

    pkl_path, reference_path = upstream_pickle
    from sonitra.transcribe._hft.convert import convert_checkpoint

    out_path = tmp_path / "model.pt"
    payload = convert_checkpoint(
        pkl_path, out_path, expected_state_digest=_reference(reference_path)["state_digest"]
    )
    poisoned = dict(payload)
    poisoned["provenance"] = dict(
        payload["provenance"], torch_version_at_conversion=torch.__version__
    )
    assert type(poisoned["provenance"]["torch_version_at_conversion"]) is not str

    broken = tmp_path / "poisoned.pt"
    torch.save(poisoned, broken)
    with pytest.raises(Exception) as excinfo:
        torch.load(broken, weights_only=True)
    assert "TorchVersion" in str(excinfo.value)


# ── content digest ──────────────────────────────────────────────────────


@pytest.mark.slow
def test_state_digest_ignores_insertion_order_and_tracks_content() -> None:
    torch = pytest.importorskip("torch")

    from sonitra.transcribe._hft import checkpoints

    generator = torch.Generator().manual_seed(3)
    first = {
        "layer.weight": torch.randn(4, 4, generator=generator),
        "layer.bias": torch.randn(4, generator=generator),
        "other.weight": torch.randn(2, 2, generator=generator),
    }
    shuffled = {key: first[key] for key in ("other.weight", "layer.bias", "layer.weight")}
    assert list(first) != list(shuffled)
    assert checkpoints.state_digest(first) == checkpoints.state_digest(shuffled)

    perturbed = dict(shuffled)
    target = perturbed["layer.bias"].clone()
    target[0] += 1.0
    perturbed["layer.bias"] = target
    assert checkpoints.state_digest(perturbed) != checkpoints.state_digest(first)

    renamed = dict(first)
    renamed["layer.bias.renamed"] = renamed.pop("layer.bias")
    assert checkpoints.state_digest(renamed) != checkpoints.state_digest(first)

    assert checkpoints.state_digest({}) == checkpoints.state_digest({})


# ── the released checkpoint ─────────────────────────────────────────────


@pytest.mark.slow
def test_released_checkpoint_converts_to_the_pinned_digest(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from sonitra.transcribe._hft import checkpoints
    from sonitra.transcribe._hft.convert import convert_checkpoint

    if not RELEASED_CHECKPOINT.exists():
        pytest.skip(f"released checkpoint not vendored at {RELEASED_CHECKPOINT}")

    before = RELEASED_CHECKPOINT.stat()
    out_path = tmp_path / "model.pt"
    payload = convert_checkpoint(
        RELEASED_CHECKPOINT, out_path, parameter_path=RELEASED_PARAMETER
    )

    assert out_path.parent == tmp_path
    after = RELEASED_CHECKPOINT.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns), (
        "the released pickle is a read-only input and must not be rewritten"
    )

    assert payload["format_version"] == checkpoints.FORMAT_VERSION
    assert payload["inference"] == INFERENCE
    assert dict(payload["hparams"]) == RELEASED_HPARAMS

    state = payload["state_dict"]
    assert payload["provenance"]["state_digest"] == checkpoints.CHECKPOINTS["maestro"]["state_digest"]
    assert checkpoints.state_digest(state) == payload["provenance"]["state_digest"]
    assert len(state) == 165
    assert sum(tensor.numel() for tensor in state.values()) == 5_516_574
    for key, tensor in state.items():
        assert tensor.device.type == "cpu", key
        assert tensor.dtype is torch.float32, key
        assert tensor.is_contiguous(), key

    provenance = payload["provenance"]
    entry = checkpoints.checkpoint_entry("maestro")
    assert provenance["upstream_commit"] == checkpoints.UPSTREAM_COMMIT
    assert provenance["zip_sha256"] == entry["zip_sha256"]
    assert provenance["member"] == entry["member"]
    assert type(provenance["torch_version_at_conversion"]) is str

    reloaded = torch.load(out_path, weights_only=True)
    model = _build_model(reloaded["hparams"])
    missing, unexpected = model.load_state_dict(reloaded["state_dict"], strict=True)
    assert missing == [] and unexpected == []
    assert set(reloaded["state_dict"]) == set(model.state_dict())


# ── import isolation ────────────────────────────────────────────────────


def test_convert_imports_without_torch() -> None:
    code = (
        "import sys\n"
        "import sonitra.transcribe._hft.convert as convert\n"
        "assert 'torch' not in sys.modules, 'convert imported torch at module level'\n"
        "assert len(convert.ALLOWED_GLOBALS) == 19\n"
        "assert convert.inference_config()['sr'] == 16000\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_SRC), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"