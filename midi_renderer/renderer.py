from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

import numpy as np

from midi_renderer.engine import RendererEngine


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
) -> np.ndarray:
    engine.assert_thread()
    notes_list = list(notes)
    duration = _resolve_duration(notes_list, duration_sec, engine.sample_rate)
    if not notes_list:
        return np.zeros((2, int(duration * engine.sample_rate)))

    processor = _get_vst_processor(engine, plugin_path)
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


def _get_vst_processor(engine: RendererEngine, plugin_path: Path | str):
    cache = getattr(engine, "_vst_processors", None)
    if cache is None:
        cache = {}
        setattr(engine, "_vst_processors", cache)
    path = Path(plugin_path).resolve()
    if path not in cache:
        cache[path] = engine.load_plugin(path)
    return cache[path]


def _normalise_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        return audio / peak
    return audio