"""Shared torch process settings and device validation for torch-based transcription backends.

Torch's numeric flags and its device state are process-global, so every torch
backend routes through here and one cache records what has been applied.
torch is imported lazily, so this module is safe to import without it.
"""
from __future__ import annotations

import logging
import os

from sonitra.transcribe.base import NumericSettingsError, TranscriptionError
from sonitra.transcribe.devices import resolve_torch_device

logger = logging.getLogger(__name__)

# What this process already applied to torch, as the requested settings plus the
# fallbacks they hit. Torch's flags are process-global, so every torch backend
# shares this one cache rather than keeping one per backend module.
_NUMERIC_STATE: tuple[tuple[str, bool], tuple[str, ...]] | None = None


def _is_cuda_device(device: str | None) -> bool:
    """Whether a resolved device string names a CUDA device."""
    return device is not None and device.lower().startswith("cuda")


def missing_dependency_error(module: str, *, backend: str) -> TranscriptionError:
    """TranscriptionError naming the module that failed and how to install it."""
    return TranscriptionError(
        f"{backend} backend unavailable: no module named '{module}'. "
        "In a repo checkout: `uv sync --locked --extra transkun --extra dev` "
        "(GPU: `--extra transkun-gpu`). Standalone: `pip install 'sonitra[transkun]'`."
    )


def apply_torch_numeric_settings(
    numeric_mode: str,
    gpu_memory_growth: bool,
    *,
    backend: str,
    device: str | None = None,
) -> tuple[str, ...]:
    """Apply process-global torch numeric settings ahead of model load.

    Returns the fallbacks that had to be accepted, empty when the requested
    settings were applied in full. ``warn`` maps to ``warn_only=True`` and falls
    back to unconstrained when the torch build lacks that keyword; ``strict``
    raises :class:`~sonitra.transcribe.base.NumericSettingsError` on every call
    and leaves the shared cache unset, so the next file is not told "already
    applied" about settings that never took effect. Any other fallback is
    returned, cached and logged once per process. TF32 is deterministic but less
    accurate (~10 vs 23 mantissa bits), so it is off in both modes.
    ``gpu_memory_growth`` is TF-only (torch already grows its cache); accepted for
    a uniform section schema.

    ``device`` is the resolved device, used only to decide the cuBLAS workspace
    below; ``None`` means not resolved yet and counts as non-CUDA.
    """
    global _NUMERIC_STATE
    settings = (numeric_mode, gpu_memory_growth)
    if _NUMERIC_STATE is not None and _NUMERIC_STATE[0] == settings:
        return _NUMERIC_STATE[1]
    if numeric_mode == "off":
        # Nothing to ask of torch, so it is never imported for the default path.
        _NUMERIC_STATE = (settings, ())
        return ()
    import torch

    failures: list[str] = []
    try:
        if numeric_mode == "warn":
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError as exc:
                # The keyword itself is missing, so the flag could not be set at
                # all: a fallback the caller has to record, not a hard failure.
                logger.warning(
                    "%s numeric_mode=warn: torch lacks warn_only; continuing unconstrained", backend
                )
                failures.append(f"warn_only unsupported: {exc}")
        else:
            if numeric_mode == "strict" and _is_cuda_device(device):
                # deterministic cuBLAS needs a fixed workspace size
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        failures.append(f"deterministic algorithms: {exc}")
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    except AttributeError as exc:
        failures.append(f"precision flags: {exc}")
    if failures and numeric_mode == "strict":
        raise NumericSettingsError(
            f"{backend} strict numeric_mode failed: {'; '.join(failures)}"
        )
    if failures:
        logger.warning("%s numeric settings fell back (%s)", backend, "; ".join(failures))
    fallbacks = tuple(failures)
    _NUMERIC_STATE = (settings, fallbacks)
    return fallbacks


def validate_torch_device(device: str, *, backend: str) -> tuple[str, bool]:
    """Resolve a unified device string and confirm it exists.

    Returns (resolved_name, available). Raises TranscriptionError when an
    accelerator was requested but is absent. CPU (and any non-cuda resolved
    device) short-circuits with no framework import.
    """
    resolved = resolve_torch_device(device)
    low = resolved.lower()
    if low in ("cpu", "cpu:0"):
        return resolved, True
    if not low.startswith("cuda"):
        return resolved, True
    try:
        import torch
    except ImportError as exc:
        raise missing_dependency_error("torch", backend=backend) from exc
    if not torch.cuda.is_available():
        if getattr(torch.version, "cuda", None) is None:
            raise TranscriptionError(
                f"{backend} device '{device}' resolved to '{resolved}' but the installed "
                f"torch ({torch.__version__}) is a CPU-only build. Reinstall the CUDA fork: "
                "`uv sync --locked --extra transkun-gpu --extra xla-ptx --extra dev "
                "--reinstall-package torch --reinstall-package torchaudio` "
                "(a plain sync will not swap the build: both forks pin the same version)."
            )
        raise TranscriptionError(
            f"{backend} device '{device}' resolved to '{resolved}' but CUDA is not available"
        )
    if ":" in resolved:
        try:
            idx = int(resolved.split(":", 1)[1])
        except ValueError as exc:
            raise TranscriptionError(
                f"{backend} got unknown device {device!r}; "
                "use 'cpu', 'cuda', 'cuda:N' or 'GPU:N'."
            ) from exc
        try:
            count = torch.cuda.device_count()
        except Exception as exc:
            raise TranscriptionError(
                f"{backend} device '{device}' resolved to '{resolved}' but CUDA device count is unavailable: {exc}"
            ) from exc
        if not 0 <= idx < count:
            raise TranscriptionError(
                f"{backend} device '{device}' resolved to '{resolved}' but only {count} CUDA device(s) are available"
            )
    return resolved, True
