from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    pass


@dataclass
class TranscriptionResult:
    """Output of a transcriber: note dicts in the midi_reader schema."""

    notes: list[dict[str, Any]]
    transcriber: str
    source_audio: Path | None = None
    midi_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_outputs: dict[str, Any] | None = None
