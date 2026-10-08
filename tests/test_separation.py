from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sonitra.config import SeparationSection
from sonitra.separation.passthrough import PassthroughSeparator
from sonitra.separation.protocol import SeparationError, make_separator


def test_passthrough_returns_input_as_mix(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    stems = PassthroughSeparator().separate(audio, tmp_path / "stems")
    assert stems == {"mix": audio}


def test_make_separator_default_is_passthrough() -> None:
    separator = make_separator(SeparationSection())
    assert isinstance(separator, PassthroughSeparator)


def test_make_separator_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown separation backend"):
        make_separator(SeparationSection(backend="spleeter"))


def test_demucs_backend_is_registered() -> None:
    from sonitra.separation.demucs_separator import DemucsSeparator

    separator = make_separator(SeparationSection(backend="demucs", model="htdemucs"))
    assert isinstance(separator, DemucsSeparator)
    assert separator.model == "htdemucs"


def test_demucs_separator_hints_at_extra(tmp_path: Path) -> None:
    from sonitra.separation.demucs_separator import DemucsSeparator

    separator = DemucsSeparator()
    with patch.dict(sys.modules, {"demucs": None, "demucs.api": None}):
        with pytest.raises(SeparationError, match=r"pip install 'sonitra\[demucs\]'"):
            separator.separate(tmp_path / "song.wav", tmp_path / "stems")


def test_demucs_device_translated(tmp_path: Path) -> None:
    import types

    from sonitra.separation.demucs_separator import DemucsSeparator

    captured: dict[str, object] = {}

    class _FakeSeparator:
        def __init__(self, model: str, device: str) -> None:
            captured["device"] = device
            self.samplerate = 44100

        def separate_audio_file(self, path: str) -> tuple[None, dict]:
            return None, {}

    fake_api = types.ModuleType("demucs.api")
    fake_api.Separator = _FakeSeparator  # type: ignore[attr-defined]
    fake_api.save_audio = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake_demucs = types.ModuleType("demucs")
    fake_demucs.api = fake_api  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"demucs": fake_demucs, "demucs.api": fake_api}):
        separator = DemucsSeparator(model="htdemucs", device="GPU:0")
        result = separator.separate(tmp_path / "song.wav", tmp_path / "stems")
    assert captured["device"] == "cuda:0"
    assert result == {}
