from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from sonitra.notes import make_note
from sonitra.transcribe.base import (
    TranscriptionError,
    TranscriptionResult,
    checkpoint_identity,
)
from sonitra.transcribe.configs import BasicPitchTranscriberConfig
from sonitra.transcribe.devices import resolve_tf_device
from sonitra.transcribe.protocol import register_transcriber


def _basic_pitch_package_version() -> str:
    """Best-effort package version for provenance metadata."""

    try:
        from importlib.metadata import version

        try:
            return version("basic-pitch")
        except Exception:
            try:
                return version("sonitra")
            except Exception:
                return "unknown"
    except Exception:
        return "unknown"


class BasicPitchTranscriber:
    """Spotify Basic Pitch backend (lightweight multi-pitch baseline).

    The `basic-pitch` dependency is installed by default with
    `pip install sonitra`; it is loaded lazily when `transcribe()` is called.

    Note: `multiple_pitch_bends=True` changes note eventing only — the
    `bends` tuple is still dropped and `midi_writer.py` writes no pitch-wheel
    messages, so glissando curves are not represented in the output MIDI
    (documented limitation).
    """

    def __init__(
        self,
        *,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 127.7,
        minimum_frequency_hz: float | None = None,
        maximum_frequency_hz: float | None = None,
        device: str = "cpu",
        melodia_trick: bool = True,
        multiple_pitch_bends: bool = False,
        save_raw_outputs: bool = False,
        name: str = "basic_pitch",
    ) -> None:
        self.onset_threshold = float(onset_threshold)
        self.frame_threshold = float(frame_threshold)
        self.minimum_note_length_ms = float(minimum_note_length_ms)
        self.minimum_frequency_hz = minimum_frequency_hz
        self.maximum_frequency_hz = maximum_frequency_hz
        self.device = device
        self.melodia_trick = melodia_trick
        self.multiple_pitch_bends = multiple_pitch_bends
        self.save_raw_outputs = save_raw_outputs
        self.name = name
        self._model: Any | None = None
        self._lock = threading.RLock()

    def validate_device(self) -> tuple[str, bool]:
        """Resolve the configured device and confirm it exists.

        Returns (resolved_name, available). Raises TranscriptionError when an
        accelerator was requested but is absent. CPU short-circuits with no
        framework import.
        """
        tf_device = resolve_tf_device(self.device, backend="basic_pitch")
        low = tf_device.lower()
        if low in ("cpu", "cpu:0"):
            return tf_device, True
        if tf_device.startswith("/"):
            return tf_device, True
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise TranscriptionError(
                "basic-pitch is not installed; it should be present after `pip install sonitra`."
            ) from exc
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception as exc:
            raise TranscriptionError(
                f"basic_pitch device '{self.device}' resolved to '{tf_device}' but GPU availability check failed: {exc}"
            ) from exc
        if not gpus:
            raise TranscriptionError(
                f"basic_pitch device '{self.device}' resolved to '{tf_device}' but no GPU is available"
            )
        if ":" in tf_device:
            try:
                idx = int(tf_device.split(":", 1)[1])
            except ValueError as exc:
                raise TranscriptionError(
                    f"basic_pitch got unknown device {self.device!r}; "
                    "use 'cpu', 'cuda', 'cuda:N' or 'GPU:N'."
                ) from exc
            if not 0 <= idx < len(gpus):
                raise TranscriptionError(
                    f"basic_pitch device '{self.device}' resolved to '{tf_device}' but only {len(gpus)} GPU(s) are available"
                )
        return tf_device, True

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        import logging as _logging

        # Suppress framework chatter only when the user did not ask for debug
        # logs; the root level is already set from config by set_log_level().
        if _logging.getLogger().getEffectiveLevel() > _logging.DEBUG:
            for _name in ("tensorflow", "absl", "basic_pitch"):
                _logging.getLogger(_name).setLevel(_logging.ERROR)

        # Fail fast on a missing accelerator; also the backstop for worker
        # subprocesses and direct library use.
        tf_device, _available = self.validate_device()

        try:
            import tensorflow as tf
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model, predict
        except ImportError as exc:
            raise TranscriptionError(
                "basic-pitch is not installed; it should be present after `pip install sonitra`."
            ) from exc

        audio_path = Path(audio_path)
        # Load the SavedModel once per instance; the lock matters because
        # `sonitra transcribe` shares one instance across a ThreadPool.
        # Built inside the device scope so placement matches the old per-call
        # path load.
        with tf.device(tf_device):
            with self._lock:
                if self._model is None:
                    self._model = Model(ICASSP_2022_MODEL_PATH)
                model = self._model
                model_output, _, note_events = predict(
                    str(audio_path),
                    model_or_model_path=model,
                    onset_threshold=self.onset_threshold,
                    frame_threshold=self.frame_threshold,
                    minimum_note_length=self.minimum_note_length_ms,
                    minimum_frequency=self.minimum_frequency_hz,
                    maximum_frequency=self.maximum_frequency_hz,
                    melodia_trick=self.melodia_trick,
                    multiple_pitch_bends=self.multiple_pitch_bends,
                )
        notes: list[dict[str, object]] = []
        for start, end, pitch, amplitude, _bends in note_events:
            note = make_note(
                pitch=int(pitch),
                velocity=round(float(amplitude) * 127),
                start_sec=float(start),
                duration_sec=float(end) - float(start),
            )
            if note is not None:
                notes.append(note)
        notes.sort(key=lambda n: (n["start_sec"], n["pitch"]))
        # Provenance metadata: package version, checkpoint identity (no I/O on
        # default path), device actually used, and filtered-note counts.
        pkg_version = _basic_pitch_package_version()
        checkpoint = checkpoint_identity(pkg_version)
        filtered_dropped = len(note_events) - len(notes)
        # device_available is what validate_device() reported (True here —
        # a missing accelerator raises before any metadata is built).
        metadata: dict[str, object] = {
            **checkpoint,
            "device": tf_device,
            "requested_device": self.device,
            "device_available": _available,
            "filtered_dropped": filtered_dropped,
            "note_events_total": len(note_events),
            "notes_kept": len(notes),
        }
        return TranscriptionResult(
            notes=notes,
            transcriber=self.name,
            source_audio=audio_path,
            raw_outputs=model_output if self.save_raw_outputs else None,
            backend_type="basic_pitch",
            metadata=metadata,
        )


@register_transcriber("basic_pitch")
def _build(cfg: BasicPitchTranscriberConfig) -> BasicPitchTranscriber:
    return BasicPitchTranscriber(
        onset_threshold=cfg.onset_threshold,
        frame_threshold=cfg.frame_threshold,
        minimum_note_length_ms=cfg.minimum_note_length_ms,
        minimum_frequency_hz=cfg.minimum_frequency_hz,
        maximum_frequency_hz=cfg.maximum_frequency_hz,
        device=cfg.device,
        melodia_trick=cfg.melodia_trick,
        multiple_pitch_bends=cfg.multiple_pitch_bends,
        save_raw_outputs=cfg.save_raw_outputs,
        name=cfg.name or "basic_pitch",
    )
