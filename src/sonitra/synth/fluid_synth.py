from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import mido
import numpy as np
from scipy.io import wavfile

_PPQN = 480
"""Pulses per quarter note used for intermediate MIDI files."""


class FluidSynth:
    """SoundFont-based synthesiser using the fluidsynth CLI.

    Renders symbolic note dictionaries by writing them to a temporary MIDI file
    and invoking ``fluidsynth`` to synthesise audio via a SoundFont. The output
    is normalised to float32 in ``[-1, 1]`` and returned as ``(channels, samples)``.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int = 2,
        soundfont_path: Path | str,
        bpm: int = 120,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.soundfont_path = Path(soundfont_path)
        self.bpm = int(bpm)
        if not self.soundfont_path.exists():
            raise FileNotFoundError(f"SoundFont not found: {self.soundfont_path}")

    def render(self, notes: Iterable[dict], duration_sec: float) -> np.ndarray:
        """Render ``notes`` to audio using the configured SoundFont.

        Args:
            notes: Iterable of note dictionaries with keys ``pitch``,
                ``velocity``, ``start_sec``, and ``duration_sec``.
            duration_sec: Target duration of the rendered audio in seconds.

        Returns:
            Audio array of shape ``(channels, samples)`` normalised to
            ``[-1, 1]`` and trimmed/padded to exactly
            ``int(duration_sec * sample_rate)`` samples.
        """
        duration = max(0.0, float(duration_sec))
        notes_list = list(notes)

        with tempfile.TemporaryDirectory(prefix="sonitra_fluid_") as tmpdir:
            tmp_path = Path(tmpdir)
            midi_path = tmp_path / "render.mid"
            wav_path = tmp_path / "render.wav"

            _write_notes_to_midi(midi_path, notes_list, bpm=self.bpm)
            _run_fluidsynth(
                soundfont_path=self.soundfont_path,
                midi_path=midi_path,
                wav_path=wav_path,
                sample_rate=self.sample_rate,
            )

            _, raw_audio = wavfile.read(str(wav_path))

        audio = _normalise_wav(raw_audio)
        audio = _ensure_channels(audio, self.channels)
        return _trim_or_pad(audio, int(duration * self.sample_rate))


def _write_notes_to_midi(path: Path, notes: list[dict], bpm: int = 120) -> None:
    """Write note dictionaries to a type-0 MIDI file at ``path``.

    Args:
        path: Output MIDI file path.
        notes: List of note dicts with keys ``pitch``, ``velocity``,
            ``start_sec``, and ``duration_sec``.
        bpm: Beats per minute for tempo meta message and tick computation.
    """
    bpm = int(bpm)
    tempo_us = round(60_000_000 / bpm)
    ticks_per_sec = _PPQN * bpm / 60.0
    midi = mido.MidiFile(type=0, ticks_per_beat=_PPQN)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    events: list[tuple[int, str, int, int]] = []
    for note in notes:
        velocity = int(note.get("velocity", 0))
        if velocity <= 0:
            continue
        duration = max(0.0, float(note.get("duration_sec", 0.0)))
        if duration <= 0.0:
            continue
        start = max(0.0, float(note.get("start_sec", 0.0)))
        pitch = int(note["pitch"])
        start_tick = int(round(start * ticks_per_sec))
        end_tick = int(round((start + duration) * ticks_per_sec))
        events.append((start_tick, "note_on", pitch, velocity))
        events.append((end_tick, "note_off", pitch, 0))

    if not events:
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.save(str(path))
        return

    events.sort(key=lambda ev: (ev[0], 0 if ev[1] == "note_on" else 1))

    previous_tick = 0
    for tick, msg_type, pitch, velocity in events:
        delta = max(0, tick - previous_tick)
        track.append(mido.Message(msg_type, note=pitch, velocity=velocity, time=delta))
        previous_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(str(path))


def _run_fluidsynth(
    *,
    soundfont_path: Path,
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
) -> None:
    """Invoke ``fluidsynth`` to render ``midi_path`` to ``wav_path``."""
    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            "-r",
            str(sample_rate),
            "-o",
            f"synth.sample-rate={sample_rate}",
            "-F",
            str(wav_path),
            str(soundfont_path),
            str(midi_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _normalise_wav(raw_audio: np.ndarray) -> np.ndarray:
    """Convert WAV data to float32 in ``[-1, 1]`` with shape ``(channels, samples)``."""
    audio = np.asarray(raw_audio)
    if audio.dtype.kind in "ui":
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    elif audio.ndim == 2 and audio.shape[1] < audio.shape[0]:
        # scipy yields (samples, channels); convert to (channels, samples).
        audio = audio.T
    return audio


def _ensure_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Expand mono to stereo if requested, otherwise leave channel count as-is."""
    if channels == 2 and audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    return audio


def _trim_or_pad(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Trim or zero-pad ``audio`` to ``target_samples`` along the time axis."""
    current = audio.shape[1]
    if current == target_samples:
        return audio
    if current > target_samples:
        return audio[:, :target_samples]
    pad_width = ((0, 0), (0, target_samples - current))
    return np.pad(audio, pad_width, mode="constant")
