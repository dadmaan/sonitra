from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sonitra.separation.protocol import register_separator

if TYPE_CHECKING:
    from sonitra.config import SeparationSection


class PassthroughSeparator:
    """No-op separator: returns the input audio as a single 'mix' stem."""

    name = "passthrough"

    def separate(self, audio_path: Path | str, output_dir: Path | str) -> dict[str, Path]:
        return {"mix": Path(audio_path)}


@register_separator("passthrough")
def _build(cfg: "SeparationSection") -> PassthroughSeparator:
    return PassthroughSeparator()
