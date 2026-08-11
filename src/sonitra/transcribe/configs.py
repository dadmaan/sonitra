from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _TranscriberBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    name: str | None = None


class BasicPitchTranscriberConfig(_TranscriberBase):
    """Spotify Basic Pitch (installed by default with `pip install sonitra`)."""

    type: Literal["basic_pitch"] = "basic_pitch"
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 127.7
    minimum_frequency_hz: float | None = None
    maximum_frequency_hz: float | None = None
    device: str = "cpu"
    melodia_trick: bool = True            # HMM/melodia post-processing smoothing
    multiple_pitch_bends: bool = False    # allow overlapping same-pitch notes w/ glissando
    save_raw_outputs: bool = False        # Feature 2 gate: persist raw model outputs as CSV sidecar


class ExternalCommandTranscriberConfig(_TranscriberBase):
    """Any CLI transcription tool invoked as `command` with {input}/{output} placeholders."""

    type: Literal["external_command"] = "external_command"
    command: str
    output_extension: str = ".mid"
    timeout_sec: float = 600.0


class PrecomputedTranscriberConfig(_TranscriberBase):
    """Pre-existing MIDI transcriptions looked up by audio file stem.

    Adapter for black-box commercial tools (klang.io, Moises, AnthemScore, ...)
    whose output was exported manually into `midi_dir`.
    """

    type: Literal["precomputed"] = "precomputed"
    midi_dir: Path | str
    extensions: list[str] = Field(default_factory=lambda: [".mid", ".midi"])


TranscriberConfig = Annotated[
    Union[
        BasicPitchTranscriberConfig,
        ExternalCommandTranscriberConfig,
        PrecomputedTranscriberConfig,
    ],
    Field(discriminator="type"),
]
