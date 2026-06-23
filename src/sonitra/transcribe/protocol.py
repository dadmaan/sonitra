from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from sonitra.transcribe.base import TranscriptionResult
from sonitra.transcribe.configs import TranscriberConfig


@runtime_checkable
class TranscriberProtocol(Protocol):
    name: str

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult: ...


TranscriberBuilder = Callable[[TranscriberConfig], TranscriberProtocol]

_TRANSCRIBER_REGISTRY: dict[str, TranscriberBuilder] = {}


def register_transcriber(type_name: str) -> Callable[[TranscriberBuilder], TranscriberBuilder]:
    """Register a builder for a transcriber config `type` discriminator."""

    def decorator(builder: TranscriberBuilder) -> TranscriberBuilder:
        _TRANSCRIBER_REGISTRY[type_name] = builder
        return builder

    return decorator


def make_transcriber(cfg: TranscriberConfig) -> TranscriberProtocol:
    # Import backends lazily so registration happens on first use without
    # forcing optional dependencies at package import time.
    from sonitra.transcribe import basic_pitch, external_command, precomputed  # noqa: F401

    builder = _TRANSCRIBER_REGISTRY.get(cfg.type)
    if builder is None:
        raise ValueError(f"Unknown transcriber type: {cfg.type}")
    return builder(cfg)
