"""Input sources that load audio for the render pipeline.

``make_source`` returns a MIDI source (synthesised audio) or an audio source
(recordings read from disk) per ``render_pipeline.input_type``. ``load()``
always returns the audio's real sample rate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np

from sonitra.config import InputType, PipelineConfig, SynthBackend
from sonitra.midi_reader import parse_midi
from sonitra.storage import read_audio
from sonitra.synth.protocol import make_synth

logger = logging.getLogger(__name__)


@runtime_checkable
class SourceProtocol(Protocol):
    def load(self, path: Path) -> tuple[np.ndarray, int]: ...

    def expected_duration_sec(self, path: Path) -> float | None: ...


def _compute_duration(notes: List[Dict[str, Any]], padding_sec: float) -> float:
    if not notes:
        return max(0.0, float(padding_sec))
    last = max(float(note["start_sec"]) + float(note["duration_sec"]) for note in notes)
    return max(0.0, last + float(padding_sec))


class MidiSource:
    """Synthesises a MIDI file to audio via the configured synth backend.

    Returns audio at ``cfg.render_pipeline.sample_rate`` — MIDI has no native
    sample rate of its own, so the config rate is authoritative, exactly as
    before this abstraction existed.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._synth = make_synth(cfg)
        self._warned_multi_program = False

    def _uses_pedalboard_synth(self) -> bool:
        """Whether ``make_synth`` would build ``PedalboardSynth`` for this config.

        Mirrors the fallback in ``make_synth``: the pedalboard backend falls
        back to FluidSynth when no plugin path is set but a SoundFont is.
        """
        cfg = self._cfg
        if cfg.render_pipeline.synth_backend != SynthBackend.PEDALBOARD_INSTRUMENT:
            return False
        instrument = cfg.pedalboard.instrument
        return not (
            instrument.plugin_path is None and cfg.fluidsynth.soundfont_path is not None
        )

    def expected_duration_sec(self, path: Path) -> float | None:
        """Return the audio duration this render is planned to produce.

        The synth trims or pads its output to the requested duration, so the
        planned duration is the last note end plus ``duration_padding_sec``.
        A pedalboard instrument also receives ``silence_flush_sec`` of extra
        render time (see ``PedalboardSynth.render``).
        """
        cfg = self._cfg
        meta = parse_midi(path, return_meta=True)
        notes: List[Dict[str, Any]] = meta["notes"]
        duration = _compute_duration(notes, cfg.render_pipeline.duration_padding_sec)
        if self._uses_pedalboard_synth():
            duration += max(0.0, float(cfg.pedalboard.instrument.silence_flush_sec))
        return duration

    def load(self, path: Path) -> tuple[np.ndarray, int]:
        cfg = self._cfg
        meta = parse_midi(path, return_meta=True)
        notes: List[Dict[str, Any]] = meta["notes"]
        duration = _compute_duration(notes, cfg.render_pipeline.duration_padding_sec)
        programs = meta.get("programs", [])
        if len(programs) == 1:
            file_program = programs[0]
        elif len(programs) == 0:
            file_program = None
        else:
            file_program = None
            if not self._warned_multi_program:
                logger.warning(
                    "MIDI file %s contains multiple programs %s; "
                    "using SoundFont default "
                    "(set fluidsynth.program to choose one).",
                    path,
                    list(programs),
                )
                self._warned_multi_program = True
        audio = self._synth.render(notes, duration_sec=duration, program=file_program)
        return audio, cfg.render_pipeline.sample_rate


class AudioSource:
    """Reads a pre-rendered audio recording directly, at its own sample rate.

    Unlike :class:`MidiSource`, the returned sample rate comes from the
    source file itself (:func:`sonitra.storage.read_audio`), never from
    ``cfg.render_pipeline.sample_rate`` — the config rate has no bearing on a
    recording that was already rendered elsewhere at its own rate.
    """

    def load(self, path: Path) -> tuple[np.ndarray, int]:
        return read_audio(path)

    def expected_duration_sec(self, path: Path) -> float | None:
        """Audio recordings have no planned duration; the duration gate is skipped."""
        return None


def make_source(cfg: PipelineConfig) -> SourceProtocol:
    if cfg.render_pipeline.input_type == InputType.AUDIO:
        return AudioSource()
    return MidiSource(cfg)
