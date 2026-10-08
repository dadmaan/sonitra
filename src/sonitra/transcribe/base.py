from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def checkpoint_identity(
    package_version: str, weights_path: Path | str | None = None
) -> dict[str, str]:
    """Checkpoint identity for provenance metadata.

    Always records the package version; computes a SHA-256 of the
    weights file only when a custom weights_path is set. Zero I/O
    when weights_path is None (default path).
    """

    identity: dict[str, str] = {"package_version": package_version}
    if weights_path is not None:
        data = Path(weights_path).read_bytes()
        identity["weights_sha256"] = hashlib.sha256(data).hexdigest()
    return identity


class TranscriptionError(RuntimeError):
    pass


class NumericSettingsError(TranscriptionError):
    """Requested numeric settings could not be applied under numeric_mode: strict; fatal for the whole run."""


@dataclass
class TranscriptionResult:
    """Output of a transcriber: note dicts in the midi_reader schema."""

    notes: list[dict[str, Any]]
    transcriber: str
    source_audio: Path | None = None
    midi_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_outputs: dict[str, Any] | None = None
    backend_type: str | None = None
    """Fixed discriminator for registry lookup (e.g. "basic_pitch").

    Unlike ``transcriber`` (which is user-overridable via config ``name``),
    ``backend_type`` is constant per backend and keys the raw-output writer
    registry. Defaults to None for backward compatibility.
    """
