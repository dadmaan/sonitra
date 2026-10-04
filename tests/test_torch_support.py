"""Shared torch support — process numeric settings and device validation."""
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest


def _run_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _fake_torch(
    calls: list[tuple[object, dict[str, object]]],
    *,
    version: str = "2.12.1",
    build_cuda: str | None = "13.0",
    cuda_available: bool = True,
    device_count: int = 1,
) -> types.ModuleType:
    """Build a stand-in torch module that records deterministic-algorithm calls."""

    def use_deterministic_algorithms(mode: bool, **kwargs: object) -> None:
        calls.append((mode, kwargs))

    fake = types.ModuleType("torch")
    fake.__version__ = version
    fake.use_deterministic_algorithms = use_deterministic_algorithms
    fake.version = types.SimpleNamespace(cuda=build_cuda)
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
    )
    fake.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=True, allow_tf32=True),
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
    )
    return fake


@pytest.fixture(autouse=True)
def _reset_numeric_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The applied-settings cache is process-global; start every test from empty."""
    from sonitra.transcribe import torch_support

    monkeypatch.setattr(torch_support, "_NUMERIC_APPLIED", None)


# ── numeric settings: no torch for 'off' ────────────────────────────────

def test_numeric_off_imports_no_torch() -> None:
    code = (
        "from sonitra.transcribe.torch_support import apply_torch_numeric_settings; "
        "import sys; "
        "apply_torch_numeric_settings('off', False, backend='x'); "
        "assert 'torch' not in sys.modules, 'torch unexpectedly in sys.modules'; "
        "print('ok')"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


# ── numeric settings: precision flags and determinism ───────────────────

def test_numeric_warn_sets_precision_flags_and_warn_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sonitra.transcribe import torch_support

    calls: list[tuple[object, dict[str, object]]] = []
    fake = _fake_torch(calls)
    monkeypatch.setitem(sys.modules, "torch", fake)

    torch_support.apply_torch_numeric_settings("warn", False, backend="x")

    assert calls == [(True, {"warn_only": True})]
    assert fake.backends.cudnn.benchmark is False
    assert fake.backends.cudnn.allow_tf32 is False
    assert fake.backends.cuda.matmul.allow_tf32 is False


def test_numeric_strict_sets_precision_flags_without_warn_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sonitra.transcribe import torch_support

    calls: list[tuple[object, dict[str, object]]] = []
    fake = _fake_torch(calls)
    monkeypatch.setitem(sys.modules, "torch", fake)

    torch_support.apply_torch_numeric_settings("strict", False, backend="x")

    assert calls == [(True, {})]
    assert fake.backends.cudnn.benchmark is False
    assert fake.backends.cudnn.allow_tf32 is False
    assert fake.backends.cuda.matmul.allow_tf32 is False


# ── numeric settings: the applied cache ─────────────────────────────────

def test_numeric_repeat_is_skipped_and_new_mode_reapplies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sonitra.transcribe import torch_support

    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))

    torch_support.apply_torch_numeric_settings("warn", False, backend="x")
    torch_support.apply_torch_numeric_settings("warn", False, backend="x")
    assert len(calls) == 1

    torch_support.apply_torch_numeric_settings("strict", False, backend="x")
    assert len(calls) == 2


def test_numeric_cache_is_shared_across_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support

    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))

    torch_support.apply_torch_numeric_settings("warn", False, backend="a")
    torch_support.apply_torch_numeric_settings("warn", False, backend="b")

    assert len(calls) == 1
    assert torch_support._NUMERIC_APPLIED == ("warn", False)


# ── numeric settings: failure reporting ─────────────────────────────────

def test_numeric_strict_failure_names_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    def boom(mode: bool, **kwargs: object) -> None:
        raise RuntimeError("det boom")

    fake = _fake_torch([])
    fake.use_deterministic_algorithms = boom
    monkeypatch.setitem(sys.modules, "torch", fake)

    with pytest.raises(TranscriptionError) as exc_info:
        torch_support.apply_torch_numeric_settings("strict", False, backend="transkun")
    assert str(exc_info.value) == "transkun strict numeric_mode failed: deterministic algorithms: det boom"

    monkeypatch.setattr(torch_support, "_NUMERIC_APPLIED", None)
    with pytest.raises(TranscriptionError) as other:
        torch_support.apply_torch_numeric_settings("strict", False, backend="torchy")
    assert str(other.value) == "torchy strict numeric_mode failed: deterministic algorithms: det boom"


def test_numeric_warn_failure_logs_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from sonitra.transcribe import torch_support

    def boom(mode: bool, **kwargs: object) -> None:
        raise RuntimeError("det boom")

    fake = _fake_torch([])
    fake.use_deterministic_algorithms = boom
    monkeypatch.setitem(sys.modules, "torch", fake)

    with caplog.at_level("WARNING", logger="sonitra.transcribe.torch_support"):
        torch_support.apply_torch_numeric_settings("warn", False, backend="transkun")

    assert "transkun numeric settings fell back (deterministic algorithms: det boom)" in caplog.text


def test_numeric_warn_without_warn_only_support_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from sonitra.transcribe import torch_support

    def no_warn_only(mode: bool) -> None:
        raise TypeError("unexpected keyword argument 'warn_only'")

    fake = _fake_torch([])
    fake.use_deterministic_algorithms = no_warn_only
    monkeypatch.setitem(sys.modules, "torch", fake)

    with caplog.at_level("WARNING", logger="sonitra.transcribe.torch_support"):
        torch_support.apply_torch_numeric_settings("warn", False, backend="transkun")

    assert (
        "transkun numeric_mode=warn: torch lacks warn_only; continuing unconstrained"
        in caplog.text
    )


# ── device validation: no framework import for cpu / non-cuda ───────────

@pytest.mark.parametrize("device", ["cpu", "cpu:0", "mps"])
def test_validate_device_short_circuit_imports_no_torch(device: str) -> None:
    code = (
        "from sonitra.transcribe.torch_support import validate_torch_device; "
        "import sys; "
        f"assert validate_torch_device({device!r}, backend='x') == ({device!r}, True); "
        "assert 'torch' not in sys.modules, 'torch unexpectedly in sys.modules'; "
        "print('ok')"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


# ── device validation: cuda paths ───────────────────────────────────────

def test_validate_device_cpu_only_build(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([], version="2.12.1+cpu", build_cuda=None, cuda_available=False),
    )

    with pytest.raises(TranscriptionError) as exc_info:
        torch_support.validate_torch_device("cuda", backend="transkun")
    message = str(exc_info.value)
    assert "2.12.1+cpu" in message
    assert "CPU-only build" in message
    assert "--reinstall-package torch --reinstall-package torchaudio" in message
    assert "CUDA is not available" not in message


def test_validate_device_cuda_build_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([], version="2.12.1", build_cuda="13.0", cuda_available=False),
    )

    with pytest.raises(TranscriptionError) as exc_info:
        torch_support.validate_torch_device("cuda", backend="transkun")
    message = str(exc_info.value)
    assert "CUDA is not available" in message
    assert "--reinstall-package" not in message
    assert "CPU-only build" not in message


def test_validate_device_index_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], device_count=1))

    with pytest.raises(TranscriptionError) as exc_info:
        torch_support.validate_torch_device("cuda:7", backend="transkun")
    message = str(exc_info.value)
    assert "cuda:7" in message
    assert "only 1 CUDA device(s) are available" in message


def test_validate_device_non_integer_index_uses_vocabulary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], device_count=1))

    with pytest.raises(TranscriptionError) as exc_info:
        torch_support.validate_torch_device("cuda:zz", backend="transkun")
    assert str(exc_info.value) == (
        "transkun got unknown device 'cuda:zz'; "
        "use 'cpu', 'cuda', 'cuda:N' or 'GPU:N'."
    )


def test_validate_device_device_count_error_names_device_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    def boom() -> int:
        raise RuntimeError("count boom")

    fake = _fake_torch([])
    fake.cuda.device_count = boom
    monkeypatch.setitem(sys.modules, "torch", fake)

    with pytest.raises(TranscriptionError) as exc_info:
        torch_support.validate_torch_device("cuda:0", backend="transkun")
    message = str(exc_info.value)
    assert "cuda:0" in message
    assert "count boom" in message


def test_validate_device_gpu_string_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support

    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], device_count=1))
    assert torch_support.validate_torch_device("GPU:0", backend="transkun") == ("cuda:0", True)


def test_validate_device_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import patch

    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(TranscriptionError) as exc_info:
            torch_support.validate_torch_device("cuda", backend="transkun")
    assert str(exc_info.value) == (
        "transkun backend unavailable: no module named 'torch'. "
        "In a repo checkout: `uv sync --locked --extra transkun --extra dev` "
        "(GPU: `--extra transkun-gpu`). Standalone: `pip install 'sonitra[transkun]'`."
    )


def test_validate_device_transkun_messages_are_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-for-byte parity with the messages the TransKun backend used to emit."""
    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import TranscriptionError

    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([], version="2.12.1+cpu", build_cuda=None, cuda_available=False),
    )
    with pytest.raises(TranscriptionError) as cpu_only:
        torch_support.validate_torch_device("cuda", backend="transkun")
    assert str(cpu_only.value) == (
        "transkun device 'cuda' resolved to 'cuda' but the installed torch (2.12.1+cpu) is a "
        "CPU-only build. Reinstall the CUDA fork: `uv sync --locked --extra transkun-gpu "
        "--extra xla-ptx --extra dev --reinstall-package torch --reinstall-package torchaudio` "
        "(a plain sync will not swap the build: both forks pin the same version)."
    )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([], version="2.12.1", build_cuda="13.0", cuda_available=False),
    )
    with pytest.raises(TranscriptionError) as no_cuda:
        torch_support.validate_torch_device("cuda", backend="transkun")
    assert str(no_cuda.value) == (
        "transkun device 'cuda' resolved to 'cuda' but CUDA is not available"
    )

    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], device_count=1))
    with pytest.raises(TranscriptionError) as out_of_range:
        torch_support.validate_torch_device("cuda:7", backend="transkun")
    assert str(out_of_range.value) == (
        "transkun device 'cuda:7' resolved to 'cuda:7' but only 1 CUDA device(s) are available"
    )

    def boom() -> int:
        raise RuntimeError("count boom")

    fake = _fake_torch([])
    fake.cuda.device_count = boom
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(TranscriptionError) as count_error:
        torch_support.validate_torch_device("cuda:0", backend="transkun")
    assert str(count_error.value) == (
        "transkun device 'cuda:0' resolved to 'cuda:0' but CUDA device count is unavailable: "
        "count boom"
    )


# ── cuBLAS workspace for strict cuda ────────────────────────────────────

def test_strict_cuda_sets_cublas_workspace_before_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sonitra.transcribe import torch_support

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seen: list[str | None] = []

    def use_deterministic_algorithms(mode: bool, **kwargs: object) -> None:
        seen.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    fake = _fake_torch([])
    fake.use_deterministic_algorithms = use_deterministic_algorithms
    monkeypatch.setitem(sys.modules, "torch", fake)

    torch_support.apply_torch_numeric_settings("strict", False, backend="x", device="cuda:0")

    assert seen == [":4096:8"]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_strict_cuda_preserves_existing_cublas_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([]))

    torch_support.apply_torch_numeric_settings("strict", False, backend="x", device="cuda")

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_strict_cpu_leaves_cublas_workspace_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([]))

    torch_support.apply_torch_numeric_settings("strict", False, backend="x", device="cpu")

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_warn_cuda_leaves_cublas_workspace_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from sonitra.transcribe import torch_support

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([]))

    torch_support.apply_torch_numeric_settings("warn", False, backend="x", device="cuda")

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


# ── the TransKun backend forwards its resolved device ───────────────────

def test_transkun_forwards_resolved_device_to_numeric_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict numeric mode on an accelerator only sets the cuBLAS workspace when
    the backend hands the shared helper its resolved device, so that handoff is
    pinned here rather than left to inspection."""
    from unittest.mock import patch

    from sonitra.transcribe import transkun as transkun_module
    from sonitra.transcribe.base import TranscriptionError

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        transkun_module,
        "validate_torch_device",
        lambda device, *, backend: ("cuda:0", True),
    )
    monkeypatch.setattr(
        transkun_module,
        "apply_torch_numeric_settings",
        lambda mode, growth, *, backend, device=None: seen.update(
            mode=mode, backend=backend, device=device
        ),
    )

    transcriber = transkun_module.TranskunTranscriber(device="GPU:0", numeric_mode="strict")
    # stopping at the first optional import keeps the test free of weights and audio
    with patch.dict(sys.modules, {"moduleconf": None}):
        with pytest.raises(TranscriptionError) as exc_info:
            transcriber.transcribe("unused.wav")
    assert "moduleconf" in str(exc_info.value)
    assert seen == {"mode": "strict", "backend": "transkun", "device": "cuda:0"}


# ── shared helpers ──────────────────────────────────────────────────────

def test_missing_dependency_error_message() -> None:
    from sonitra.transcribe.torch_support import missing_dependency_error

    error = missing_dependency_error("torch", backend="transkun")
    assert str(error) == (
        "transkun backend unavailable: no module named 'torch'. "
        "In a repo checkout: `uv sync --locked --extra transkun --extra dev` "
        "(GPU: `--extra transkun-gpu`). Standalone: `pip install 'sonitra[transkun]'`."
    )
    assert missing_dependency_error("moduleconf", backend="transkun").args[0].startswith(
        "transkun backend unavailable: no module named 'moduleconf'. "
    )
