from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sonitra.config import SeparationSection


@runtime_checkable
class StemSeparatorProtocol(Protocol):
    name: str

    def separate(self, audio_path: Path | str, output_dir: Path | str) -> dict[str, Path]:
        """Split audio into stems; returns a mapping of stem name to file path."""
        ...


SeparatorBuilder = Callable[["SeparationSection"], StemSeparatorProtocol]

_SEPARATOR_REGISTRY: dict[str, SeparatorBuilder] = {}


def register_separator(backend: str) -> Callable[[SeparatorBuilder], SeparatorBuilder]:
    def decorator(builder: SeparatorBuilder) -> SeparatorBuilder:
        _SEPARATOR_REGISTRY[backend] = builder
        return builder

    return decorator


def make_separator(cfg: "SeparationSection") -> StemSeparatorProtocol:
    from sonitra.separation import demucs_separator, passthrough  # noqa: F401

    builder = _SEPARATOR_REGISTRY.get(cfg.backend)
    if builder is None:
        raise ValueError(f"Unknown separation backend: {cfg.backend}")
    return builder(cfg)
