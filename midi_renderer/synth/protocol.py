from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from midi_renderer.config import PipelineConfig, RenderingMode
from midi_renderer.synth.dawdreamer_synth import DawDreamerSynth
from midi_renderer.synth.pedalboard_synth import PedalboardSynth


@runtime_checkable
class SynthesiserProtocol(Protocol):
    def render(self, notes: list[dict], duration_sec: float) -> np.ndarray: ...


def make_synth(cfg: PipelineConfig) -> SynthesiserProtocol:
    if cfg.pipeline.rendering_mode == RenderingMode.PEDALBOARD_ONLY:
        instrument = cfg.pedalboard.instrument
        return PedalboardSynth(
            sample_rate=cfg.pipeline.sample_rate,
            channels=cfg.pipeline.channels,
            plugin_path=instrument.plugin_path,
            preset_path=instrument.preset_path,
            reload_plugin_per_file=instrument.reload_plugin_per_file,
            silence_flush_sec=instrument.silence_flush_sec,
        )
    return DawDreamerSynth(
        sample_rate=cfg.pipeline.sample_rate,
        block_size=cfg.dawdreamer.block_size,
        plugin_path=cfg.dawdreamer.plugin_path,
        preset_path=cfg.dawdreamer.preset_path,
        bpm=cfg.dawdreamer.bpm,
        faust_code=cfg.dawdreamer.faust_code,
        clear_midi_between_renders=cfg.dawdreamer.clear_midi_between_renders,
    )
