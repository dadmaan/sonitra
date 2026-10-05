from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from sonitra.notes import make_note
from sonitra.transcribe.base import (
    NumericSettingsError,
    TranscriptionError,
    TranscriptionResult,
    checkpoint_identity,
)
from sonitra.transcribe.configs import BasicPitchTranscriberConfig
from sonitra.transcribe.devices import resolve_tf_device
from sonitra.transcribe.numerics import read_numeric_env
from sonitra.transcribe.protocol import register_transcriber

logger = logging.getLogger(__name__)


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


#: What this process already applied to TensorFlow, as the requested settings
#: plus the fallbacks they hit. TF refuses repeated setup once the device is
#: initialised, so the answer is cached per process rather than per file.
_NUMERIC_STATE: tuple[tuple[str, bool], tuple[str, ...]] | None = None


def _apply_numeric_settings(
    numeric_mode: str, gpu_memory_growth: bool
) -> tuple[str, ...]:
    """Apply process-global TF numeric settings ahead of model load.

    Returns the fallbacks that had to be accepted, empty when the requested
    settings were applied in full. A strict failure raises
    :class:`~sonitra.transcribe.base.NumericSettingsError` on every call and
    leaves the cache unset, so the next file is not told "already applied" about
    settings that never took effect; every other fallback is returned, cached and
    logged once per process.
    """
    global _NUMERIC_STATE
    settings = (numeric_mode, gpu_memory_growth)
    if _NUMERIC_STATE is not None and _NUMERIC_STATE[0] == settings:
        return _NUMERIC_STATE[1]
    if numeric_mode == "off" and not gpu_memory_growth:
        # The default CPU path asks for nothing, so TensorFlow is never imported.
        _NUMERIC_STATE = (settings, ())
        return ()
    import tensorflow as tf

    failures: list[str] = []
    if gpu_memory_growth:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            failures.append(f"memory growth: {exc}")
    if numeric_mode != "off":
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError) as exc:
            failures.append(f"op determinism: {exc}")
        try:
            # TF32 is deterministic but less accurate than float32.
            tf.config.experimental.enable_tensor_float_32_execution(False)
        except (AttributeError, RuntimeError) as exc:
            failures.append(f"TF32 disable: {exc}")
    if failures and numeric_mode == "strict":
        raise NumericSettingsError(
            f"basic_pitch strict numeric_mode failed: {'; '.join(failures)}"
        )
    if failures:
        logger.warning(
            "basic_pitch numeric settings fell back (%s)", "; ".join(failures)
        )
    fallbacks = tuple(failures)
    _NUMERIC_STATE = (settings, fallbacks)
    return fallbacks


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
        numeric_mode: str = "off",
        gpu_memory_growth: bool = False,
        batch_size: int = 16,
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
        mode = numeric_mode.lower() if isinstance(numeric_mode, str) else numeric_mode
        if mode not in ("off", "warn", "strict"):
            raise TranscriptionError(
                f"basic_pitch got unknown numeric_mode {numeric_mode!r}; "
                "use 'off', 'warn' or 'strict'."
            )
        self.numeric_mode = mode
        self.gpu_memory_growth = bool(gpu_memory_growth)
        self.batch_size = int(batch_size)
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

    def apply_numeric_settings(self) -> tuple[str, ...]:
        """Process-global numeric settings for this backend, with the fallbacks it hit."""
        return _apply_numeric_settings(self.numeric_mode, self.gpu_memory_growth)

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
            from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
            import basic_pitch.note_creation as infer
            from basic_pitch.inference import Model, unwrap_output
        except ImportError as exc:
            raise TranscriptionError(
                "basic-pitch is not installed; it should be present after `pip install sonitra`."
            ) from exc

        numeric_fallbacks = self.apply_numeric_settings()

        from sonitra.storage import read_audio_basic_pitch

        audio_path = Path(audio_path)
        # Replicates basic_pitch.inference.predict internals (run_inference):
        # same 30-frame overlap / hop geometry, same front zero-pad, same
        # unwrap. Audio comes from read_audio_basic_pitch, which is
        # bit-identical to the librosa.load call predict() would make.
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len
        # CPU-only batching. GPU stays at batch 1: batched-GPU throughput is
        # unmeasured; batch 1 reproduces upstream per-window inference exactly.
        effective_batch = self.batch_size if "cpu" in tf_device.lower() else 1

        audio, _sample_rate = read_audio_basic_pitch(str(audio_path))
        original_length = int(audio.shape[0])
        padded = np.concatenate(
            [np.zeros(overlap_len // 2, dtype=np.float32), np.asarray(audio, dtype=np.float32)]
        )
        windows = []
        for i in range(0, padded.shape[0], hop_size):
            window = padded[i : i + AUDIO_N_SAMPLES]
            if window.shape[0] < AUDIO_N_SAMPLES:
                window = np.pad(window, (0, AUDIO_N_SAMPLES - window.shape[0]))
            windows.append(np.expand_dims(window, axis=-1))

        # Load the SavedModel once per instance; the lock matters because
        # `sonitra transcribe` shares one instance across a ThreadPool.
        # Built inside the device scope so placement matches the old per-call
        # path load.
        with tf.device(tf_device):
            with self._lock:
                if self._model is None:
                    self._model = Model(ICASSP_2022_MODEL_PATH)
                model = self._model
                output: dict[str, list[Any]] = {"note": [], "onset": [], "contour": []}
                for start in range(0, len(windows), effective_batch):
                    batch = np.stack(windows[start : start + effective_batch], axis=0)
                    for key, value in model.predict(batch).items():
                        output[key].append(value)
                model_output = {
                    key: unwrap_output(
                        np.concatenate(output[key]), original_length, n_overlapping_frames
                    )
                    for key in output
                }
                min_note_len = int(
                    round(self.minimum_note_length_ms / 1000.0 * (AUDIO_SAMPLE_RATE / FFT_HOP))
                )
                _, note_events = infer.model_output_to_notes(
                    model_output,
                    onset_thresh=self.onset_threshold,
                    frame_thresh=self.frame_threshold,
                    min_note_len=min_note_len,
                    min_freq=self.minimum_frequency_hz,
                    max_freq=self.maximum_frequency_hz,
                    multiple_pitch_bends=self.multiple_pitch_bends,
                    melodia_trick=self.melodia_trick,
                    midi_tempo=120,
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
            "numeric_mode": self.numeric_mode,
            "gpu_memory_growth": self.gpu_memory_growth,
            "numeric_fallbacks": list(numeric_fallbacks),
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
    numeric_mode, gpu_memory_growth = read_numeric_env()
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
        batch_size=cfg.batch_size,
        numeric_mode=numeric_mode,
        gpu_memory_growth=gpu_memory_growth,
        name=cfg.name or "basic_pitch",
    )
