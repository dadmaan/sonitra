from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from pedalboard.io import AudioFile


def write_wav(
    audio: np.ndarray,
    path: Path | str,
    *,
    sample_rate: int,
    normalize: bool = True,
    overwrite: bool = True,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_to_write = np.array(audio, copy=True)
    if normalize:
        peak = float(np.max(np.abs(audio_to_write))) if audio_to_write.size else 0.0
        if peak > 0.0:
            audio_to_write = audio_to_write / peak
    audio_to_write = np.clip(audio_to_write, -1.0, 1.0)
    audio_to_write = (audio_to_write.T * 32767.0).astype(np.int16)

    wavfile.write(output_path, int(sample_rate), audio_to_write)
    return output_path


def read_audio(path: Path | str) -> tuple[np.ndarray, int]:
    """Read an audio file as a (channels, samples) float32 array and its sample rate."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    with AudioFile(str(input_path)) as f:
        audio = f.read(f.frames)
        sample_rate = int(f.samplerate)
    return np.asarray(audio, dtype=np.float32), sample_rate


def derive_output_path(midi_path: Path | str, *, out_dir: Path | str, ext: str = ".wav") -> Path:
    midi = Path(midi_path)
    out_dir = Path(out_dir)
    return out_dir / f"{midi.stem}{ext}"


def write_audio(
    audio: np.ndarray,
    path: Path | str,
    *,
    sample_rate: int,
    bit_depth: int,
    output_format: str,
    mp3_bitrate_kbps: int | None = None,
    overwrite: bool = True,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.ndim == 1:
        audio_arr = np.expand_dims(audio_arr, axis=0)
    if audio_arr.ndim == 2 and audio_arr.shape[0] > audio_arr.shape[1]:
        audio_arr = audio_arr.T
    audio_arr = np.clip(audio_arr, -1.0, 1.0)

    quality = mp3_bitrate_kbps if output_format == "mp3" else None
    with AudioFile(
        str(output_path),
        "w",
        samplerate=float(sample_rate),
        num_channels=int(audio_arr.shape[0]),
        bit_depth=int(bit_depth),
        quality=quality,
    ) as f:
        f.write(audio_arr)
    return output_path