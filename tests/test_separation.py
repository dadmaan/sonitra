from __future__ import annotations

from pathlib import Path

import pytest

from sonitra.config import SeparationSection
from sonitra.separation.passthrough import PassthroughSeparator
from sonitra.separation.protocol import make_separator


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
