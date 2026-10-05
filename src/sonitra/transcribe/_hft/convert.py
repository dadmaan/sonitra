"""One-time conversion of the upstream pickled module into a plain state-dict file.

The release ships a `pickle.dump` of a whole `nn.Module` whose storages are tagged
`cuda:0`, which no CPU host can load and which `weights_only=True` refuses because
it names upstream's module classes. This unpickles it once under a closed global
allowlist with every storage redirected to CPU, then writes a state dict that
`torch.load(..., weights_only=True)` accepts.

Torch is imported inside the functions that need it, so the allowlist, the errors
and the recovery helpers stay importable without the optional extra installed.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sonitra.transcribe._hft import checkpoints

__all__ = [
    "ALLOWED_GLOBALS",
    "AllowlistUnpickler",
    "ConversionError",
    "DigestMismatchError",
    "Stub",
    "build_state_dict",
    "convert_checkpoint",
    "cross_check_hparams",
    "extract_hparams",
    "inference_config",
    "load_stub",
]

#: The exact set of globals the pinned upstream pickle references, enumerated from the
#: released asset itself. Anything else is refused rather than imported, so a tampered
#: or unexpected pickle cannot make the converter import and call an arbitrary callable.
ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("model.model_spec2midi", "DecoderLayer"),
        ("model.model_spec2midi", "DecoderLayer_Zero"),
        ("model.model_spec2midi", "Decoder_SPEC2MIDI"),
        ("model.model_spec2midi", "EncoderLayer"),
        ("model.model_spec2midi", "Encoder_SPEC2MIDI"),
        ("model.model_spec2midi", "Model_SPEC2MIDI"),
        ("model.model_spec2midi", "MultiHeadAttentionLayer"),
        ("model.model_spec2midi", "PositionwiseFeedforwardLayer"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch.nn.modules.activation", "Sigmoid"),
        ("torch.nn.modules.container", "ModuleList"),
        ("torch.nn.modules.conv", "Conv2d"),
        ("torch.nn.modules.dropout", "Dropout"),
        ("torch.nn.modules.linear", "Linear"),
        ("torch.nn.modules.normalization", "LayerNorm"),
        ("torch.nn.modules.sparse", "Embedding"),
        ("torch.storage", "_load_from_bytes"),
    }
)

#: Prefixes whose classes are replaced by record-only stubs instead of the real ones, so
#: the conversion depends on neither upstream's source nor on `nn.Module` pickle support.
_STUBBED_PREFIXES = ("model.model_spec2midi", "torch.nn.modules")

_STORAGE_LOADER = ("torch.storage", "_load_from_bytes")


class ConversionError(Exception):
    """A released checkpoint could not be converted into a usable state-dict file."""


class DigestMismatchError(ConversionError):
    """The recovered weights are not the ones the registry pins, so nothing was written."""


class Stub:
    """Records only `__dict__`; never runs an upstream `__init__`, `__setstate__` or
    `forward`, so no released code is executed during the conversion."""

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"_stub_state": state})

    def __repr__(self) -> str:
        origin = getattr(type(self), "_stub_origin", ("?", "?"))
        return f"<Stub {origin[0]}.{origin[1]}>"


def _make_stub(module: str, name: str) -> type[Stub]:
    return type(name, (Stub,), {"_stub_origin": (module, name)})


class AllowlistUnpickler(pickle.Unpickler):
    """Unpickles into record-only stubs, with the legacy storage loader forced to CPU.

    Upstream's storages are tagged `cuda:0`, so the stock loader fails on a host without
    a matching GPU; `weights_only=True` is kept for the inner payloads because they
    only ever hold plain float tensors.
    """

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"global {module}.{name} is outside the allowlist; refusing to import it"
            )
        if key == _STORAGE_LOADER:
            return self._load_storage_cpu
        if module == _STUBBED_PREFIXES[0] or module.startswith(_STUBBED_PREFIXES[1] + "."):
            return _make_stub(module, name)
        return super().find_class(module, name)

    def _load_storage_cpu(self, payload: bytes) -> Any:
        import torch

        return torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )


def load_stub(pickle_path: Path | str) -> Any:
    """Unpickle one released checkpoint into record-only stubs, every storage on CPU."""
    with Path(pickle_path).open("rb") as handle:
        return AllowlistUnpickler(handle).load()


def walk_stub(root: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Depth-first `(dotted prefix, stub)` list over the recorded `_modules` tree."""
    found: list[tuple[str, Any]] = [(prefix, root)]
    for name, child in (getattr(root, "_modules", None) or {}).items():
        found.extend(walk_stub(child, f"{prefix}.{name}" if prefix else name))
    return found


def build_state_dict(root: Any) -> dict[str, Any]:
    """Dotted state dict of every parameter and buffer in the stub tree, in module order."""
    state: dict[str, Any] = {}
    for prefix, stub in walk_stub(root):
        for attribute in ("_parameters", "_buffers"):
            for name, tensor in (getattr(stub, attribute, None) or {}).items():
                if tensor is None:
                    continue
                state[f"{prefix}.{name}" if prefix else name] = tensor
    return state


def _attribute(stub: Any, name: str) -> Any:
    value = vars(stub).get(name)
    if value is None:
        raise ConversionError(
            f"the checkpoint does not record {name!r} on "
            f"{getattr(stub, '_stub_origin', ('<root>', ''))[1] or '<root>'}; "
            "it does not look like a released hFT-Transformer module"
        )
    return value


def _int_attribute(stub: Any, name: str) -> int:
    value = _attribute(stub, name)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConversionError(
            f"expected {name!r} to be an integer, got {type(value).__name__}"
        ) from exc


def _submodule(stub: Any, name: str) -> Any:
    child = (getattr(stub, "_modules", None) or {}).get(name)
    if child is None:
        raise ConversionError(f"the checkpoint has no submodule {name!r}")
    return child


def _head_dim(stub: Any) -> int | None:
    """Head width recovered from the attention scale, which is `sqrt(head_dim)`.

    The scale is a plain tensor attribute rather than a registered buffer, so it is the
    only place the head width survives the pickle; `sqrt` of a float32 is inexact, hence
    the rounding. Returns None for stubs that are not attention layers.
    """
    scale = vars(stub).get("scale")
    if scale is None or not hasattr(scale, "reshape"):
        return None
    value = float(scale.reshape(-1)[0]) ** 2
    head_dim = int(round(value))
    if head_dim <= 0 or abs(value - head_dim) > 1e-3:
        raise ConversionError(
            f"attention scale {value!r} does not square to an integer head width"
        )
    return head_dim


def _head_widths(tree: list[tuple[str, Any]], side: str) -> set[int]:
    """Head width of every attention layer on one side of the encoder/decoder pair."""
    widths: set[int] = set()
    for prefix, stub in tree:
        if not prefix.startswith(f"{side}_spec2midi."):
            continue
        head_dim = _head_dim(stub)
        if head_dim is not None:
            widths.add(head_dim)
    return widths


def _head_count(head_dims: set[int], hid_dim: int, side: str) -> int:
    if not head_dims:
        raise ConversionError(
            f"no attention layer was found on the {side} side, so its head count "
            "cannot be recovered from the checkpoint"
        )
    if len(head_dims) != 1:
        raise ConversionError(
            f"the {side} attention layers disagree on the head width: {sorted(head_dims)}"
        )
    head_dim = head_dims.pop()
    if hid_dim % head_dim:
        raise ConversionError(
            f"hid_dim {hid_dim} is not a multiple of the {side} head width {head_dim}"
        )
    return hid_dim // head_dim


def _layers(module_list: Any) -> list[tuple[str, Any]]:
    """`(_modules` name, stub)` pairs of a `ModuleList`."""
    return list((getattr(module_list, "_modules", None) or {}).items())


def extract_hparams(root: Any) -> dict[str, Any]:
    """Recover the constructor hyperparameters from the stub tree.

The attention head count is derived from the tensor geometry rather than read off a
recorded attribute: the attention scale is `sqrt(head_dim)` and
`hid_dim == n_heads * head_dim`, which leaves a derived value the released
`parameter.json` can confirm independently.
"""
    tree = walk_stub(root)
    encoder = next(
        (stub for prefix, stub in tree if prefix == "encoder_spec2midi"), None
    )
    decoder = next(
        (stub for prefix, stub in tree if prefix == "decoder_spec2midi"), None
    )
    if encoder is None or decoder is None:
        raise ConversionError(
            "the checkpoint has no encoder_spec2midi / decoder_spec2midi submodule pair"
        )

    hid_dim = _int_attribute(encoder, "hid_dim")
    margin = (_int_attribute(encoder, "n_proc") - 1) // 2
    encoder_layers = _layers(_submodule(encoder, "layers_freq"))
    feedforward = _submodule(
        _submodule(encoder_layers[0][1], "positionwise_feedforward"), "fc_1"
    )
    attention = {side: _head_widths(tree, side) for side in ("encoder", "decoder")}

    return {
        "hid_dim": hid_dim,
        "pf_dim": int(_attribute(feedforward, "_parameters")["weight"].shape[0]),
        "enc_layer": len(encoder_layers),
        "dec_layer": len(_layers(_submodule(decoder, "layers_time"))),
        "enc_head": _head_count(attention["encoder"], hid_dim, "encoder"),
        "dec_head": _head_count(attention["decoder"], hid_dim, "decoder"),
        "cnn_channel": _int_attribute(encoder, "cnn_channel"),
        "cnn_kernel": _int_attribute(encoder, "cnn_kernel"),
        "dropout": float(_attribute(_submodule(encoder, "dropout"), "p")),
        "margin_b": margin,
        "margin_f": margin,
        "num_frame": _int_attribute(encoder, "n_frame"),
        "n_bins": _int_attribute(encoder, "n_bin"),
        "num_note": _int_attribute(decoder, "n_note"),
        "num_velocity": _int_attribute(decoder, "n_velocity"),
    }


def _dig(mapping: Mapping[str, Any], *keys: str) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def cross_check_hparams(
    hparams: Mapping[str, Any], parameter: Mapping[str, Any]
) -> list[str]:
    """Names whose recovered value disagrees with the released `parameter.json`.

    Sections the release did not ship are skipped rather than reported, so a partial
    `parameter.json` still cross-checks whatever it does carry.
    """
    pairs = (
        ("hid_dim", hparams["hid_dim"], _dig(parameter, "transformer", "hid_dim")),
        ("pf_dim", hparams["pf_dim"], _dig(parameter, "transformer", "pf_dim")),
        ("enc_layer", hparams["enc_layer"],
         _dig(parameter, "transformer", "encoder", "n_layer")),
        ("dec_layer", hparams["dec_layer"],
         _dig(parameter, "transformer", "decoder", "n_layer")),
        ("enc_head", hparams["enc_head"],
         _dig(parameter, "transformer", "encoder", "n_head")),
        ("dec_head", hparams["dec_head"],
         _dig(parameter, "transformer", "decoder", "n_head")),
        ("cnn_channel", hparams["cnn_channel"], _dig(parameter, "cnn", "channel")),
        ("cnn_kernel", hparams["cnn_kernel"], _dig(parameter, "cnn", "kernel")),
        ("dropout", hparams["dropout"], _dig(parameter, "training", "dropout")),
    )
    return [
        name
        for name, recovered, expected in pairs
        if expected is not None and recovered != expected
    ]


def inference_config() -> dict[str, Any]:
    """The feature-extraction constants the released model was trained and decoded with."""
    return {
        "sr": 16000,
        "hop_sample": 256,
        "fft_bins": 2048,
        "window_length": 2048,
        "mel_bins": 256,
        "log_offset": 1e-08,
        "pad_mode": "constant",
        "window": "hann",
        # Upstream floors the log-mel at float32(log(1e-8)); float64 would differ in the
        # last bits and the clamp is applied to float32 activations at inference time.
        "min_value": -18.42068099975586,
        "note_min": 21,
        "note_max": 108,
        "mel_norm": "slaney",
        "melfilter": "htk",
    }


def _plain(value: Any) -> str | int | float:
    """Coerce one scalar to a plain `str`/`int`/`float` for `torch.save`.

    `torch.save` records the real class of every value, so a `str` subclass such as
    `torch.torch_version.TorchVersion` is stored as that class and `weights_only=True`
    then refuses the whole file. Coercion happens for every non-tensor value so the
    converted checkpoint carries no class `weights_only` has to be told about.
    """
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise ConversionError(
        f"cannot store a {type(value).__name__} in the converted checkpoint; "
        "every value must be a plain str, int or float"
    )


def _plain_block(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _plain(value) for key, value in mapping.items()}


def convert_checkpoint(
    pickle_path: Path | str,
    out_path: Path | str,
    *,
    checkpoint: str = checkpoints.DEFAULT_CHECKPOINT,
    parameter_path: Path | str | None = None,
    expected_state_digest: str | None = None,
) -> dict[str, Any]:
    """Convert a released pickle into a plain state-dict file and return the payload.

    `parameter_path`, when given, is the released `parameter.json` the recovered
    hyperparameters are cross-checked against. `expected_state_digest` defaults to the
    digest pinned for `checkpoint` in the registry; any digest mismatch, and any other
    inconsistency, is raised before a single byte is written.
    """
    entry = checkpoints.checkpoint_entry(checkpoint)
    root = load_stub(pickle_path)
    state = build_state_dict(root)
    hparams = extract_hparams(root)

    if hparams["margin_f"] != hparams["margin_b"]:
        raise ConversionError(
            "the encoder builds one context window per frame from a single width, so "
            f"margin_b and margin_f must match; recovered {hparams['margin_b']} and "
            f"{hparams['margin_f']}"
        )

    if parameter_path is not None:
        parameter = json.loads(Path(parameter_path).read_text())
        mismatches = cross_check_hparams(hparams, parameter)
        elements = sum(int(tensor.numel()) for tensor in state.values())
        expected_elements = parameter.get("parameters")
        if expected_elements is not None and int(expected_elements) != elements:
            mismatches.append("parameters")
        if mismatches:
            raise ConversionError(
                f"recovered hyperparameters disagree with {parameter_path} on: "
                f"{', '.join(mismatches)}"
            )

    digest = checkpoints.state_digest(state)
    expected_digest = (
        entry["state_digest"] if expected_state_digest is None else expected_state_digest
    )
    if digest != expected_digest:
        raise DigestMismatchError(
            f"state digest mismatch for {pickle_path}: recovered {digest}, expected "
            f"{expected_digest}; the registry pins {checkpoint!r} to that digest, so "
            "nothing was written"
        )

    import torch

    payload: dict[str, Any] = {
        "format_version": _plain(checkpoints.FORMAT_VERSION),
        "state_dict": {
            key: tensor.detach().cpu().contiguous() for key, tensor in state.items()
        },
        "hparams": _plain_block(hparams),
        "inference": _plain_block(inference_config()),
        "provenance": _plain_block(
            {
                "upstream_commit": checkpoints.UPSTREAM_COMMIT,
                "zip_sha256": entry["zip_sha256"],
                "member": entry["member"],
                "state_digest": digest,
                "torch_version_at_conversion": torch.__version__,
            }
        ),
    }

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f"{destination.name}.", suffix=".partial"
    )
    os.close(handle)
    try:
        torch.save(payload, staged)
        os.replace(staged, destination)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return payload