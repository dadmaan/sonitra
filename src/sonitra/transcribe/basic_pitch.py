from __future__ import annotations

from pathlib import Path

from sonitra.notes import make_note
from sonitra.transcribe.base import (
    TranscriptionError,
    TranscriptionResult,
    checkpoint_identity,
)
from sonitra.transcribe.configs import BasicPitchTranscriberConfig
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

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        import logging as _logging

        for _name in ("tensorflow", "absl", "basic_pitch"):
            _logging.getLogger(_name).setLevel(_logging.ERROR)

        try:
            import tensorflow as tf
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise TranscriptionError(
                "basic-pitch is not installed; it should be present after `pip install sonitra`."
            ) from exc

        audio_path = Path(audio_path)
        with tf.device(self.device):
            model_output, _, note_events = predict(
                str(audio_path),
                model_or_model_path=ICASSP_2022_MODEL_PATH,
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
        metadata: dict[str, object] = {
            **checkpoint,
            "device": self.device,
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
