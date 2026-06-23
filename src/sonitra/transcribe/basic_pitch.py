from __future__ import annotations

from pathlib import Path

from sonitra.transcribe.base import TranscriptionError, TranscriptionResult
from sonitra.transcribe.configs import BasicPitchTranscriberConfig
from sonitra.transcribe.protocol import register_transcriber


class BasicPitchTranscriber:
    """Spotify Basic Pitch backend (lightweight multi-pitch baseline).

    Requires the optional `basic-pitch` dependency; install with
    `pip install sonitra[basicpitch]`.
    """

    def __init__(
        self,
        *,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 127.7,
        minimum_frequency_hz: float | None = None,
        maximum_frequency_hz: float | None = None,
        name: str = "basic_pitch",
    ) -> None:
        self.onset_threshold = float(onset_threshold)
        self.frame_threshold = float(frame_threshold)
        self.minimum_note_length_ms = float(minimum_note_length_ms)
        self.minimum_frequency_hz = minimum_frequency_hz
        self.maximum_frequency_hz = maximum_frequency_hz
        self.name = name

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise TranscriptionError(
                "basic-pitch is not installed; install with `pip install sonitra[basicpitch]`"
            ) from exc

        audio_path = Path(audio_path)
        _, _, note_events = predict(
            str(audio_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length_ms,
            minimum_frequency=self.minimum_frequency_hz,
            maximum_frequency=self.maximum_frequency_hz,
        )
        notes = [
            {
                "pitch": int(pitch),
                "velocity": max(1, min(127, round(float(amplitude) * 127))),
                "start_sec": float(start),
                "duration_sec": max(0.0, float(end) - float(start)),
            }
            for start, end, pitch, amplitude, _bends in note_events
        ]
        notes.sort(key=lambda note: (note["start_sec"], note["pitch"]))
        return TranscriptionResult(
            notes=notes,
            transcriber=self.name,
            source_audio=audio_path,
        )


@register_transcriber("basic_pitch")
def _build(cfg: BasicPitchTranscriberConfig) -> BasicPitchTranscriber:
    return BasicPitchTranscriber(
        onset_threshold=cfg.onset_threshold,
        frame_threshold=cfg.frame_threshold,
        minimum_note_length_ms=cfg.minimum_note_length_ms,
        minimum_frequency_hz=cfg.minimum_frequency_hz,
        maximum_frequency_hz=cfg.maximum_frequency_hz,
        name=cfg.name or "basic_pitch",
    )
