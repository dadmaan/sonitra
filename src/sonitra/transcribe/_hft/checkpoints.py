"""Pinned upstream checkpoint registry and models-directory resolution.

Standard library only: importing this module never loads torch, so it is usable
before the optional torch extra is installed. Weights are converted to a plain
state-dict file, never loaded from the upstream pickle at runtime.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

UPSTREAM_REPO = "https://github.com/sony/hFT-Transformer"
UPSTREAM_COMMIT = "71a2ee06e9ced1ea24673c95ee0acded2fc98d04"

#: Bumped whenever the on-disk layout changes; the backend refuses a mismatch.
FORMAT_VERSION = 1

DEFAULT_CHECKPOINT = "maestro"

#: The release publishes no digest, so these were measured from the asset itself.
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "maestro": {
        "url": (
            "https://github.com/sony/hFT-Transformer/releases/download/ismir2023/"
            "checkpoint.zip"
        ),
        "zip_sha256": "bb3c88758e4802e6b3881ccb6dc7927417fb43f76980df3739ab2bbc418d70c5",
        "zip_size": 20540269,
        "member": "checkpoint/MAESTRO-V3/model_016_003.pkl",
        "parameter_member": "checkpoint/MAESTRO-V3/parameter.json",
        "state_digest": "eaa7bf6ae2cf82f748b0efc267ca42e34a2a606a50b053803b3b28d4f15b943c",
    },
}


def checkpoint_entry(checkpoint: str) -> dict[str, Any]:
    """Registry entry for a checkpoint name, with the valid names on failure."""
    entry = CHECKPOINTS.get(checkpoint)
    if entry is None:
        raise KeyError(
            f"unknown hft_transformer checkpoint {checkpoint!r}; "
            f"use one of {sorted(CHECKPOINTS)}."
        )
    return entry


def _resolve_models_dir(override: Path | str | None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    configured = os.environ.get("SONITRA_MODELS_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "sonitra" / "models"


def models_dir() -> Path:
    """Directory holding downloaded weights.

    ``SONITRA_MODELS_DIR`` wins when set to a non-empty value, otherwise
    ``~/.cache/sonitra/models``. There is deliberately no repo-relative default:
    a checkout may be read-only or shared between users.
    """
    return _resolve_models_dir(None)


def checkpoint_dir(
    checkpoint: str = DEFAULT_CHECKPOINT, models_dir: Path | str | None = None
) -> Path:
    """Install directory for one checkpoint."""
    return _resolve_models_dir(models_dir) / "hft_transformer" / checkpoint


def checkpoint_path(
    checkpoint: str = DEFAULT_CHECKPOINT, models_dir: Path | str | None = None
) -> Path:
    """Path of the converted ``model.pt`` for one checkpoint."""
    return checkpoint_dir(checkpoint, models_dir) / "model.pt"


def manifest_path(
    checkpoint: str = DEFAULT_CHECKPOINT, models_dir: Path | str | None = None
) -> Path:
    """Path of the install manifest that records provenance next to the weights."""
    return checkpoint_dir(checkpoint, models_dir) / "manifest.json"


def state_digest(state: dict[str, Any]) -> str:
    """Content digest of a state dict: sha256 over key, dtype, shape and bytes.

    Digests tensor *content* rather than the ``.pt`` file, because ``torch.save``
    output bytes are not stable across torch versions while the weights are.
    Each field is NUL-terminated so no key or shape can be confused with the next.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(",".join(str(dim) for dim in tuple(tensor.shape)).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()
