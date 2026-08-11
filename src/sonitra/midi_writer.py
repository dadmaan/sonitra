from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import mido
import numpy as np

if TYPE_CHECKING:
    from sonitra.transcribe.base import TranscriptionResult

logger = logging.getLogger(__name__)

DEFAULT_TICKS_PER_BEAT = 480

_MIN_MIDI_PITCH = 21
_MAX_MIDI_PITCH = 108
_CONTOUR_BINS = 264
_TOTAL_COLUMNS = 1 + 88 + _CONTOUR_BINS + 88  # time + onset + contour + note = 441


def write_midi(
    notes: Iterable[dict[str, Any]],
    path: Path | str,
    *,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    tempo_bpm: float = 120.0,
) -> Path:
    """Write midi_reader-style note dicts to a single-track MIDI file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tempo = mido.bpm2tempo(tempo_bpm)
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[float, int, mido.Message]] = []
    for note in notes:
        pitch = int(note["pitch"])
        velocity = max(1, min(127, int(note.get("velocity", 64))))
        start = max(0.0, float(note["start_sec"]))
        duration = float(note["duration_sec"])
        if duration <= 0.0:
            continue
        # note_off sorts before note_on at the same instant so retriggers of
        # the same pitch survive the round trip through midi_reader
        events.append((start, 1, mido.Message("note_on", note=pitch, velocity=velocity, time=0)))
        events.append((start + duration, 0, mido.Message("note_off", note=pitch, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for time_sec, _, message in events:
        tick = int(round(mido.second2tick(time_sec, ticks_per_beat, tempo)))
        track.append(message.copy(time=max(0, tick - previous_tick)))
        previous_tick = tick

    midi.save(output_path)
    return output_path


def write_raw_outputs(
    raw_outputs: dict[str, Any],
    midi_path: Path | str,
) -> Path:
    """Persist raw model-output piano-roll arrays as a wide CSV sidecar.

    The sidecar is written next to the MIDI as ``<stem>.model_outputs.csv``
    with one row per model frame and 441 columns: ``time_sec``, the 88 onset
    probabilities (MIDI pitches 21-108), the 264 contour-bin probabilities
    (positional index into the 264-bin contour head, not MIDI-pitch naming),
    and the 88 note probabilities (MIDI pitches 21-108). ``time_sec`` uses
    ``basic_pitch.note_creation.model_frames_to_time`` so the rows share the
    sibling MIDI's frame-to-time mapping exactly.

    Args:
        raw_outputs: Mapping of ``onset`` (n, 88), ``contour`` (n, 264) and
            ``note`` (n, 88) probability arrays as returned by
            ``basic_pitch.inference.predict``.
        midi_path: Path of the sibling MIDI file; the CSV is written to the
            same directory using its stem.

    Returns:
        Path of the written CSV sidecar.

    Raises:
        ValueError: If ``onset``, ``contour`` and ``note`` do not share the
            same frame count (axis 0). Differing axis-1 widths (88 / 264 /
            88) are by design and are not an error.
    """
    onset = np.asarray(raw_outputs["onset"], dtype=np.float64)
    contour = np.asarray(raw_outputs["contour"], dtype=np.float64)
    note = np.asarray(raw_outputs["note"], dtype=np.float64)

    n_frames = onset.shape[0]
    # Defensive-only guard: basic-pitch's unwrap_output already guarantees a
    # shared frame count; the arrays' second axis legitimately differs.
    if not (contour.shape[0] == n_frames and note.shape[0] == n_frames):
        raise ValueError(
            f"raw output frame counts differ: onset {onset.shape[0]}, "
            f"contour {contour.shape[0]}, note {note.shape[0]}"
        )

    # Lazy import: pulling basic_pitch at module top level would load
    # TensorFlow; midi_writer must stay import-light.
    from basic_pitch.note_creation import model_frames_to_time

    times = model_frames_to_time(n_frames)

    columns = np.empty((n_frames, _TOTAL_COLUMNS), dtype=np.float64)
    columns[:, 0] = times
    columns[:, 1:89] = onset
    columns[:, 89:353] = contour
    columns[:, 353:441] = note

    pitch_columns = ",".join(
        f"onset_{pitch}" for pitch in range(_MIN_MIDI_PITCH, _MAX_MIDI_PITCH + 1)
    )
    contour_columns = ",".join(
        f"contour_bin_{bin_index:03d}" for bin_index in range(_CONTOUR_BINS)
    )
    note_columns = ",".join(
        f"note_{pitch}" for pitch in range(_MIN_MIDI_PITCH, _MAX_MIDI_PITCH + 1)
    )
    header = f"time_sec,{pitch_columns},{contour_columns},{note_columns}"

    midi_output_path = Path(midi_path)
    sidecar = midi_output_path.with_name(midi_output_path.stem + ".model_outputs.csv")
    np.savetxt(sidecar, columns, delimiter=",", fmt="%.6f", header=header)
    return sidecar


def write_transcription_outputs(
    result: TranscriptionResult,
    midi_path: Path | str,
) -> Path:
    """Write a transcription's MIDI and (when present) its raw-outputs sidecar.

    The MIDI write is failure-critical and propagates to the caller. The
    raw-outputs CSV write is best-effort and isolated in its own try/except:
    a sidecar failure is logged as a warning and never fails the
    transcription it exists to diagnose.

    Args:
        result: Transcription result whose ``notes`` are written to MIDI and
            whose ``raw_outputs`` (when not None) are written as a CSV sidecar.
        midi_path: Output path for the MIDI file.

    Returns:
        Path of the written MIDI file.
    """
    write_midi(result.notes, midi_path)
    if result.raw_outputs is not None:
        try:
            write_raw_outputs(result.raw_outputs, midi_path)
        except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
            logger.warning("Failed to write raw outputs CSV for %s: %s", midi_path, exc)
    return Path(midi_path)
