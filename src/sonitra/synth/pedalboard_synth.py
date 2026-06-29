from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mido
import numpy as np
import pedalboard


class PedalboardSynth:
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int = 2,
        plugin_path: Path | str | None = None,
        preset_path: Path | str | None = None,
        reload_plugin_per_file: bool = False,
        silence_flush_sec: float = 0.0,
        bpm: int = 120,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.plugin_path = Path(plugin_path) if plugin_path else None
        self.preset_path = Path(preset_path) if preset_path else None
        self.reload_plugin_per_file = reload_plugin_per_file
        self.silence_flush_sec = float(silence_flush_sec)
        self.bpm = int(bpm)
        self._plugin = None

    def render(self, notes: Iterable[dict], duration_sec: float) -> np.ndarray:
        duration = max(0.0, float(duration_sec))
        if self.plugin_path is None:
            raise ValueError(
                "PedalboardSynth requires a VST instrument plugin path. "
                "Set pedalboard.instrument.plugin_path in your config, or use "
                "a different synth_backend (fluidsynth, dawdreamer_faust, etc.)."
            )

        plugin = self._load_plugin() if self.reload_plugin_per_file or self._plugin is None else self._plugin
        if self._plugin is None and not self.reload_plugin_per_file:
            self._plugin = plugin

        messages = _notes_to_messages(notes)
        total_duration = duration + max(0.0, self.silence_flush_sec)
        audio = plugin.process(
            messages,
            duration=total_duration,
            sample_rate=float(self.sample_rate),
            num_channels=self.channels,
            reset=True,
        )
        if audio is None:
            return np.zeros((self.channels, int(self.sample_rate * duration)))
        audio_arr = np.asarray(audio)
        if audio_arr.ndim == 2 and audio_arr.shape[0] != self.channels and audio_arr.shape[1] == self.channels:
            audio_arr = audio_arr.T
        return audio_arr

    def _load_plugin(self):
        if self.plugin_path is None:
            return None
        plugin = pedalboard.load_plugin(str(self.plugin_path))
        if not getattr(plugin, "is_instrument", False):
            raise ValueError("VST3 plugin is not an instrument")
        if self.preset_path and hasattr(plugin, "load_preset"):
            plugin.load_preset(str(self.preset_path))
        return plugin


def _notes_to_messages(notes: Iterable[dict]) -> list[mido.Message]:
    messages: list[mido.Message] = []
    for note in notes:
        velocity = int(note.get("velocity", 0))
        if velocity <= 0:
            continue
        pitch = int(note["pitch"])
        start = max(0.0, float(note["start_sec"]))
        duration = max(0.0, float(note["duration_sec"]))
        if duration <= 0.0:
            continue
        messages.append(mido.Message("note_on", note=pitch, velocity=velocity, time=start))
        messages.append(mido.Message("note_off", note=pitch, velocity=0, time=start + duration))
    messages.sort(key=lambda msg: (msg.time, 0 if msg.type == "note_on" else 1))
    return messages
