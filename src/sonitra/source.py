"""Pluggable input-source abstraction for the render pipeline.

Mirrors the shape of :mod:`sonitra.synth.protocol`'s ``make_synth``: a
``runtime_checkable`` protocol, one implementation per
``pipeline.input_type`` value, and a ``make_source`` factory that dispatches
on config. This is the single code path both MIDI-mode and audio-mode
renders go through in :mod:`sonitra.pipeline`.

The load-bearing property of this abstraction is that ``load()`` always
returns the *real* sample rate of the audio it produced: the config's
``pipeline.sample_rate`` for synthesised MIDI (which has no native rate of
its own), and the source file's own rate — as reported by
:func:`sonitra.storage.read_audio` — for audio-mode input. Threading that
returned rate through normalisation/effects/quality-gate/write is therefore
automatic for both modes, rather than something an implementer has to
remember to do at each call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np

from sonitra.config import InputType, PipelineConfig
from sonitra.midi_reader import parse_midi
from sonitra.storage import read_audio
from sonitra.synth.protocol import make_synth


@runtime_checkable
class SourceProtocol(Protocol):
    def load(self, path: Path) -> tuple[np.ndarray, int]: ...


def _scale_note_timings(notes: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    if abs(scale - 1.0) <= 1e-6:
        return notes
    return [
        {**n, "start_sec": n["start_sec"] * scale, "duration_sec": n["duration_sec"] * scale}
        for n in notes
    ]


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

    def load(self, path: Path) -> tuple[np.ndarray, int]:
        cfg = self._cfg
        meta = parse_midi(path, return_meta=True)
        notes: List[Dict[str, Any]] = meta["notes"]
        native_bpm: float = meta["bpm"]
        if native_bpm > 0:
            notes = _scale_note_timings(notes, native_bpm / cfg.render_pipeline.bpm)
        duration = _compute_duration(notes, cfg.render_pipeline.duration_padding_sec)
        audio = self._synth.render(notes, duration_sec=duration)
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


def make_source(cfg: PipelineConfig) -> SourceProtocol:
    if cfg.render_pipeline.input_type == InputType.AUDIO:
        return AudioSource()
    return MidiSource(cfg)
