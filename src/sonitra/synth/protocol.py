from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from sonitra.config import PipelineConfig, SynthBackend
from sonitra.synth.dawdreamer_synth import DawDreamerSynth
from sonitra.synth.pedalboard_synth import PedalboardSynth

logger = logging.getLogger(__name__)


@runtime_checkable
class SynthesiserProtocol(Protocol):
    def render(self, notes: list[dict], duration_sec: float) -> np.ndarray: ...


def make_synth(cfg: PipelineConfig) -> SynthesiserProtocol:
    backend = cfg.pipeline.synth_backend

    if backend == SynthBackend.PEDALBOARD_INSTRUMENT:
        instrument = cfg.pedalboard.instrument
        if instrument.plugin_path is None and cfg.fluidsynth.soundfont_path is not None:
            logger.warning(
                "No VST instrument plugin path configured for synth_backend=pedalboard_instrument; "
                "falling back to FluidSynth (soundfont_path=%s).",
                cfg.fluidsynth.soundfont_path,
            )
            from sonitra.synth.fluid_synth import FluidSynth

            return FluidSynth(
                sample_rate=cfg.pipeline.sample_rate,
                channels=cfg.pipeline.channels,
                soundfont_path=cfg.fluidsynth.soundfont_path,
                bpm=cfg.pipeline.bpm,
            )
        return PedalboardSynth(
            sample_rate=cfg.pipeline.sample_rate,
            channels=cfg.pipeline.channels,
            plugin_path=instrument.plugin_path,
            preset_path=instrument.preset_path,
            reload_plugin_per_file=instrument.reload_plugin_per_file,
            silence_flush_sec=instrument.silence_flush_sec,
            bpm=cfg.pipeline.bpm,
        )

    if backend == SynthBackend.FLUIDSYNTH:
        from sonitra.synth.fluid_synth import FluidSynth

        return FluidSynth(
            sample_rate=cfg.pipeline.sample_rate,
            channels=cfg.pipeline.channels,
            soundfont_path=cfg.fluidsynth.soundfont_path,
            bpm=cfg.pipeline.bpm,
        )

    return DawDreamerSynth(
        sample_rate=cfg.pipeline.sample_rate,
        block_size=cfg.dawdreamer.block_size,
        plugin_path=cfg.dawdreamer.plugin_path,
        preset_path=cfg.dawdreamer.preset_path,
        bpm=cfg.pipeline.bpm,
        faust_code=cfg.dawdreamer.faust_code,
        clear_midi_between_renders=cfg.dawdreamer.clear_midi_between_renders,
    )
