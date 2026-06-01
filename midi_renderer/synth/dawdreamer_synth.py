from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from midi_renderer.engine import RendererEngine
from midi_renderer.renderer import render_notes_faust, render_notes_vst


class DawDreamerSynth:
    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int = 512,
        plugin_path: Path | str | None = None,
        preset_path: Path | str | None = None,
        bpm: int = 120,
        faust_code: str | None = None,
        clear_midi_between_renders: bool = True,
    ) -> None:
        self.engine = RendererEngine(sample_rate=sample_rate, block_size=block_size)
        self.sample_rate = sample_rate
        self.plugin_path = Path(plugin_path) if plugin_path else None
        self.preset_path = Path(preset_path) if preset_path else None
        self.bpm = bpm
        self.faust_code = faust_code
        self.clear_midi_between_renders = clear_midi_between_renders

    def render(self, notes: Iterable[dict], duration_sec: float) -> np.ndarray:
        if self.plugin_path is None:
            return render_notes_faust(list(notes), engine=self.engine, duration_sec=duration_sec)
        return render_notes_vst(
            list(notes),
            engine=self.engine,
            plugin_path=self.plugin_path,
            duration_sec=duration_sec,
        )
