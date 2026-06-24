from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

import numpy as np

from sonitra.engine import RendererEngine


# Vital's VST3 component class ID, derived from its VST2 unique ID ("Vita")
# and plugin name ("vital") using Steinberg's convertVST2UID_To_FUID.
# This is required to wrap Vital's native .vital JSON presets in the
# standard .vstpreset container that DawDreamer's load_vst3_preset understands.
_VITAL_VST3_COMPONENT_CLASS_ID = "56535456697461766974616C00000000"


def render_notes_faust(
    notes: Iterable[Dict[str, Any]],
    *,
    engine: RendererEngine,
    duration_sec: float | None,
) -> np.ndarray:
    engine.assert_thread()
    notes_list = list(notes)
    duration = _resolve_duration(notes_list, duration_sec, engine.sample_rate)
    if not notes_list:
        return np.zeros((2, int(duration * engine.sample_rate)))

    processor = _get_faust_note_processor(engine)
    processor.clear_midi()
    _feed_notes(processor, notes_list, duration)
    engine.engine.load_graph([(processor, [])])
    engine.engine.render(float(duration))
    return _normalise_audio(engine.engine.get_audio())


def render_notes_vst(
    notes: Iterable[Dict[str, Any]],
    *,
    engine: RendererEngine,
    plugin_path: Path | str,
    duration_sec: float | None,
    preset_path: Path | str | None = None,
) -> np.ndarray:
    engine.assert_thread()
    notes_list = list(notes)
    duration = _resolve_duration(notes_list, duration_sec, engine.sample_rate)
    if not notes_list:
        return np.zeros((2, int(duration * engine.sample_rate)))

    processor = _get_vst_processor(engine, plugin_path, preset_path=preset_path)
    processor.clear_midi()
    _feed_notes(processor, notes_list, duration)
    engine.engine.load_graph([(processor, [])])
    engine.engine.render(float(duration))
    return _normalise_audio(engine.engine.get_audio())


def _resolve_duration(notes: List[Dict[str, Any]], duration_sec: float | None, sample_rate: int) -> float:
    if duration_sec is not None:
        return max(0.0, float(duration_sec))
    if not notes:
        return 0.0
    last = max(float(note["start_sec"]) + float(note["duration_sec"]) for note in notes)
    return max(0.0, last)


def _feed_notes(processor, notes: List[Dict[str, Any]], duration_sec: float) -> None:
    for note in notes:
        pitch = int(note["pitch"])
        velocity = int(note.get("velocity", 0))
        if velocity <= 0:
            continue
        start = max(0.0, float(note["start_sec"]))
        dur = max(0.0, float(note["duration_sec"]))
        if start >= duration_sec:
            continue
        if start + dur > duration_sec:
            dur = duration_sec - start
        if dur <= 0:
            continue
        processor.add_midi_note(pitch, velocity, start, dur)


def _get_faust_note_processor(engine: RendererEngine):
    processor = getattr(engine, "_note_processor", None)
    if processor is None:
        processor = engine.engine.make_faust_processor("faust_notes")
        processor.set_dsp_string("process = os.osc(440), os.osc(440);")
        if not processor.compile():
            raise RuntimeError("Failed to compile Faust note processor.")
        setattr(engine, "_note_processor", processor)
    return processor


def _get_vst_processor(
    engine: RendererEngine,
    plugin_path: Path | str,
    preset_path: Path | str | None = None,
):
    cache = getattr(engine, "_vst_processors", None)
    if cache is None:
        cache = {}
        setattr(engine, "_vst_processors", cache)
    path = Path(plugin_path).resolve()
    cache_key = (path, Path(preset_path).resolve() if preset_path else None)
    if cache_key not in cache:
        processor = engine.load_plugin(path)
        if preset_path:
            preset = Path(preset_path)
            if not preset.exists():
                raise FileNotFoundError(f"Preset not found: {preset}")
            _load_vst_preset(processor, preset)
        cache[cache_key] = processor
    return cache[cache_key]


def _load_vst_preset(processor, preset: Path) -> None:
    suffix = preset.suffix.lower()
    if suffix == ".vstpreset":
        processor.load_vst3_preset(str(preset))
        return
    if suffix == ".vital":
        # DawDreamer's load_state passes raw bytes through JUCE's VST3
        # setStateInformation wrapper, which does not understand Vital's
        # native .vital JSON. Converting to a standard .vstpreset and using
        # load_vst3_preset routes the state through the VST3 preset mechanism
        # (IComponent::setState) and actually applies the preset.
        processor.load_vst3_preset(str(_get_cached_vstpreset(preset)))
        return
    processor.load_preset(str(preset))


def _get_cached_vstpreset(vital_path: Path) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "sonitra" / "vital_vstpreset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{vital_path.name}.vstpreset"

    if not cache_path.exists() or cache_path.stat().st_mtime < vital_path.stat().st_mtime:
        # Write to a temporary file and rename atomically to avoid races
        # between parallel renders reading a partially-written preset.
        tmp_path = cache_dir / f".{vital_path.name}.vstpreset.tmp"
        _write_vstpreset(tmp_path, _VITAL_VST3_COMPONENT_CLASS_ID, {"Comp": vital_path.read_bytes()})
        tmp_path.replace(cache_path)

    return cache_path


def _write_vstpreset(path: Path, class_id: str, chunks: Dict[str, bytes]) -> None:
    """Write a minimal VST3 preset file (.vstpreset).

    Format (little-endian):
        HEADER (48 bytes):
            - header id "VST3" (4 bytes)
            - version (int32, 4 bytes)
            - ASCII class id (32 bytes)
            - offset to chunk list (int64, 8 bytes)
        CHUNKS:
            - concatenated chunk data
        CHUNK LIST:
            - "List" (4 bytes)
            - entry count (int32, 4 bytes)
            - for each chunk: id (4 bytes), offset (int64), size (int64)
    """
    header_fmt = "<4si32sq"
    header_size = struct.calcsize(header_fmt)
    chunklist_header_fmt = "<4si"
    chunklist_entry_fmt = "<4sqq"

    chunk_data = b"".join(chunks.values())
    chunklist_offset = header_size + len(chunk_data)

    data = struct.pack(
        header_fmt,
        b"VST3",
        1,
        class_id.encode("ascii"),
        chunklist_offset,
    )

    offsets = {}
    offset = header_size
    for chunk_id, chunk_bytes in chunks.items():
        offsets[chunk_id] = offset
        offset += len(chunk_bytes)

    data += chunk_data
    data += struct.pack(chunklist_header_fmt, b"List", len(chunks))
    for chunk_id, chunk_bytes in chunks.items():
        data += struct.pack(
            chunklist_entry_fmt,
            chunk_id.encode("ascii"),
            offsets[chunk_id],
            len(chunk_bytes),
        )

    path.write_bytes(data)


def _normalise_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        return audio / peak
    return audio