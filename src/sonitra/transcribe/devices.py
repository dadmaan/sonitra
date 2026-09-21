"""Unified device strings across transcription backends.

Convention for every AMT backend, present and future: users write one of
``cpu``, ``cuda``, ``cuda:N`` or ``GPU:N`` in config. Each backend translates
that string to its framework's native form at its own boundary:

- torch consumers (transkun, demucs) use :func:`resolve_torch_device`
  (``GPU:N`` → ``cuda:N``);
- TensorFlow consumers (basic_pitch) use :func:`resolve_tf_device`
  (``cuda`` → ``GPU:0``, ``cuda:N`` → ``GPU:N``).

Both helpers are pure string mappings (no torch/TF import), so they are
cheap to unit-test. Anything outside the unified vocabulary raises
TranscriptionError naming the valid values, failing fast instead of
surfacing a framework parse error deep in inference.
"""
from __future__ import annotations

from sonitra.transcribe.base import TranscriptionError


def resolve_torch_device(device: str) -> str:
    """Map ``GPU:*`` style strings to torch ``cuda:*``; pass the rest through."""
    low = device.lower()
    if low == "gpu":
        return "cuda"
    if low.startswith("gpu:"):
        suffix = device.split(":", 1)[1]
        return f"cuda:{suffix}"
    return device


def resolve_tf_device(device: str, *, backend: str = "basic_pitch") -> str:
    """Map unified strings to TensorFlow device names (``cuda`` → ``GPU:0``).

    ``cpu``/``CPU:0`` and ``GPU:N`` pass through; explicit ``/...`` TF specs
    are left alone as a power-user escape hatch. Anything else raises
    TranscriptionError.
    """
    low = device.lower()
    if low in ("cpu", "cpu:0"):
        return device
    if low == "cuda" or low == "gpu":
        return "GPU:0"
    for prefix in ("cuda:", "gpu:"):
        if low.startswith(prefix):
            return f"GPU:{device.split(':', 1)[1]}"
    if device.startswith("/"):
        return device
    raise TranscriptionError(
        f"{backend} got unknown device {device!r}; "
        "use 'cpu', 'cuda', 'cuda:N' or 'GPU:N'."
    )
