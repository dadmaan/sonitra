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
    minimum_note_length_ms: float = Field(default=127.7, ge=0)
    minimum_frequency_hz: float | None = None
    maximum_frequency_hz: float | None = None
    device: str = "cpu"
    melodia_trick: bool = True            # HMM/melodia post-processing smoothing
    multiple_pitch_bends: bool = False    # allow overlapping same-pitch notes w/ glissando
    save_raw_outputs: bool = False        # Feature 2 gate: persist raw model outputs as CSV sidecar
    # Windows per inference call. Unset means 1 on a GPU and 16 on the CPU; any
    # set value applies on every device, because batches above 8 changed notes
    # slightly in measurement on CUDA while batch 1 reproduces upstream
    # per-window inference exactly.
    batch_size: int | None = Field(default=None, ge=1)


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


class TranskunTranscriberConfig(_TranscriberBase):
    """TransKun piano transcription (requires `pip install sonitra[transkun]`)."""

    type: Literal["transkun"] = "transkun"
    device: str = "cpu"
    segment_size_sec: float | None = None
    segment_hop_sec: float | None = None
    weights_path: Path | str | None = None
    conf_path: Path | str | None = None


class HftTransformerTranscriberConfig(_TranscriberBase):
    """Sony hFT-Transformer piano transcription (requires the transkun extra + setup script)."""

    type: Literal["hft_transformer"] = "hft_transformer"
    device: str = "cpu"
    checkpoint: Literal["maestro"] = "maestro"
    weights_path: Path | str | None = None
    # The decoder has two head sets: a frequency axis and a time axis. "second" is the
    # time-axis set the model reports in its paper, so it is the default.
    output: Literal["first", "second"] = "second"
    # Window stride, in frames of 16 ms. Overlap sharpens onsets but costs compute in
    # proportion. 0 keeps the plain whole-window pass; the bound is num_frame / 2 (64),
    # because the strided pass reads num_frame // 2 frames starting at n_stride, and above
    # that bound the slice runs off the end of the window and acceptance stops being
    # monotonic in the input length.
    n_stride: int = Field(default=0, ge=0, le=64)
    onset_threshold: float = Field(default=0.5, gt=0, le=1)
    offset_threshold: float = Field(default=0.5, gt=0, le=1)
    mpe_threshold: float = Field(default=0.5, gt=0, le=1)
    batch_size: int = Field(default=1, ge=1)


TranscriberConfig = Annotated[
    Union[
        BasicPitchTranscriberConfig,
        ExternalCommandTranscriberConfig,
        HftTransformerTranscriberConfig,
        PrecomputedTranscriberConfig,
        TranskunTranscriberConfig,
    ],
    Field(discriminator="type"),
]
