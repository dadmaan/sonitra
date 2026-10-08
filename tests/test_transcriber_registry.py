"""Registry and import-conformance tests for transcribers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import pytest

from sonitra.transcribe.base import TranscriptionError
from sonitra.transcribe.configs import TranscriberConfig
from sonitra.transcribe.protocol import _TRANSCRIBER_REGISTRY, make_transcriber


def _union_types() -> set[str]:
    """Extract Literal type values from TranscriberConfig union."""
    # TranscriberConfig is Annotated[Union[...], Field(discriminator="type")]
    annotated_args = get_args(TranscriberConfig)
    union = annotated_args[0]
    types: set[str] = set()
    for cfg_cls in get_args(union):
        ann = cfg_cls.model_fields["type"].annotation  # type: ignore[attr-defined]
        lit_args = get_args(ann)
        assert lit_args, f"config {cfg_cls.__name__} type annotation has no Literal args: {ann!r}"
        assert len(lit_args) == 1, f"config {cfg_cls.__name__} expected single Literal"
        types.add(lit_args[0])
    return types


def _registry_types() -> set[str]:
    """Return populated registry keys, ensuring lazy imports ran."""
    # Trigger lazy registration via imports; make_transcriber also imports.
    import sonitra.transcribe.basic_pitch  # noqa: F401
    import sonitra.transcribe.external_command  # noqa: F401
    import sonitra.transcribe.hft_transformer  # noqa: F401
    import sonitra.transcribe.precomputed  # noqa: F401

    # Also call make_transcriber once to cover protocol's inside import path
    from sonitra.transcribe.configs import PrecomputedTranscriberConfig

    try:
        make_transcriber(PrecomputedTranscriberConfig(midi_dir=Path("/tmp")))
    except Exception:
        pass
    return set(_TRANSCRIBER_REGISTRY.keys())


def _assert_registry_matches_union(registry: set[str], union: set[str]) -> None:
    if registry != union:
        extra = registry - union
        missing = union - registry
        raise AssertionError(
            f"registry↔union mismatch: registry={registry} union={union} "
            f"extra_in_registry={extra} missing_from_registry={missing}"
        )


# ── registry ↔ union consistency ──────────────────────────────────────────

def test_registry_matches_union() -> None:
    registry = _registry_types()
    union = _union_types()
    _assert_registry_matches_union(registry, union)


def test_registry_consistency_detects_extra_key() -> None:
    union = _union_types()
    registry = _registry_types()
    fake_registry = set(registry) | {"dummy_backend"}
    with pytest.raises(AssertionError, match="dummy_backend"):
        _assert_registry_matches_union(fake_registry, union)
    # also missing registration (union has ghost)
    fake_union = set(union) | {"ghost_backend"}
    with pytest.raises(AssertionError, match="ghost_backend"):
        _assert_registry_matches_union(registry, fake_union)


def test_registry_consistency_detects_missing_config() -> None:
    registry = _registry_types()
    union = _union_types()
    # simulate config without registry: remove one
    if registry:
        one = next(iter(registry))
        reduced = set(registry) - {one}
        with pytest.raises(AssertionError, match=one):
            _assert_registry_matches_union(reduced, union)


# ── missing-dependency message via patch.dict ─────────────────────────────

def test_basic_pitch_missing_dependency_message_via_patch(tmp_path: Path) -> None:
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    transcriber = BasicPitchTranscriber()
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFF....WAVE")
    # Patch heavy deps to None; transcribe should raise TranscriptionError with helpful hint
    # Pattern mirrors tests/test_separation.py:38-45
    with patch.dict(sys.modules, {"tensorflow": None, "basic_pitch": None, "basic_pitch.inference": None}):
        with pytest.raises(TranscriptionError, match="not installed"):
            transcriber.transcribe(dummy_wav)


def test_precomputed_does_not_require_heavy_deps(tmp_path: Path) -> None:
    # Control: precomputed should still work even when heavy deps are patched away
    from sonitra.transcribe.configs import PrecomputedTranscriberConfig

    cfg = PrecomputedTranscriberConfig(midi_dir=tmp_path)
    with patch.dict(sys.modules, {"tensorflow": None, "basic_pitch": None, "torch": None}):
        t = make_transcriber(cfg)
        assert t.name == "precomputed"


# ── import deferral via subprocess (not in-process) ───────────────────────

def _run_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_make_transcriber_defers_torch_import_precomputed() -> None:
    code = (
        "from sonitra.transcribe.protocol import make_transcriber; "
        "from sonitra.transcribe.configs import PrecomputedTranscriberConfig; "
        "import sys; "
        "m=make_transcriber(PrecomputedTranscriberConfig(midi_dir='/tmp')); "
        "assert 'torch' not in sys.modules, f\"torch unexpectedly in sys.modules: {[k for k in sys.modules if k=='torch']}\"; "
        "assert 'tensorflow' not in sys.modules, f\"tensorflow unexpectedly in sys.modules\"; "
        "print('ok')"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, f"deferral precomputed failed: stdout={result.stdout!r} stderr={result.stderr!r}"


def test_make_transcriber_defers_torch_import_basic_pitch() -> None:
    code = (
        "from sonitra.transcribe.protocol import make_transcriber; "
        "from sonitra.transcribe.configs import BasicPitchTranscriberConfig; "
        "import sys; "
        "m=make_transcriber(BasicPitchTranscriberConfig()); "
        "assert 'torch' not in sys.modules, f\"torch unexpectedly in sys.modules\"; "
        "print('ok')"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, f"deferral basic_pitch failed: stdout={result.stdout!r} stderr={result.stderr!r}"


def test_importing_protocol_does_not_import_heavy_deps() -> None:
    code = (
        "import sys; "
        "import sonitra.transcribe.protocol; "
        "assert 'torch' not in sys.modules, 'torch imported at protocol import'; "
        "assert 'tensorflow' not in sys.modules, 'tensorflow imported at protocol import'; "
        "assert 'basic_pitch' not in sys.modules, 'basic_pitch imported at protocol import'; "
        "print('ok')"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, f"protocol heavy import failed: stdout={result.stdout!r} stderr={result.stderr!r}"


def test_protocol_top_level_is_light() -> None:
    text = Path("src/sonitra/transcribe/protocol.py").read_text()
    # Heavy backends must be lazy inside functions, not at top level
    assert "from sonitra.transcribe import basic_pitch" in text
    lines = text.splitlines()
    for line in lines:
        if "import torch" in line or "import tensorflow" in line:
            assert line.lstrip() != line or line.startswith("    ") or line.startswith("\t"), (
                f"heavy import must be lazy, found top-level: {line!r}"
            )
        if "import basic_pitch" in line:
            # lazy inside make_transcriber is indented
            assert line.startswith("    ") or line.startswith("\t"), (
                f"basic_pitch import must be inside function, found top-level: {line!r}"
            )


def test_validate_device_cpu_imports_no_framework() -> None:
    code = (
        "from sonitra.transcribe.basic_pitch import BasicPitchTranscriber; "
        "from sonitra.transcribe.transkun import TranskunTranscriber; "
        "import sys; "
        "b=BasicPitchTranscriber(device='cpu'); b.validate_device(); "
        "t=TranskunTranscriber(device='cpu'); t.validate_device(); "
        "assert 'torch' not in sys.modules, 'torch imported by cpu validate_device'; "
        "assert 'tensorflow' not in sys.modules, 'tensorflow imported by cpu validate_device'; "
        "print('ok')"
    )
    result = _run_subprocess(code)
    assert result.returncode == 0, f"cpu validate_device imported framework: stdout={result.stdout!r} stderr={result.stderr!r}"
