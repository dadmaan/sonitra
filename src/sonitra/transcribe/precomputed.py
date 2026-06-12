from __future__ import annotations

from pathlib import Path

from sonitra.midi_reader import parse_midi
from sonitra.transcribe.base import TranscriptionError, TranscriptionResult
from sonitra.transcribe.configs import PrecomputedTranscriberConfig
from sonitra.transcribe.protocol import register_transcriber


class PrecomputedTranscriber:
    """Looks up pre-existing MIDI transcriptions by audio file stem."""

    def __init__(
        self,
        *,
        midi_dir: Path | str,
        extensions: list[str] | None = None,
        name: str = "precomputed",
    ) -> None:
        self.midi_dir = Path(midi_dir)
        self.extensions = extensions or [".mid", ".midi"]
        self.name = name

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        audio_path = Path(audio_path)
        for extension in self.extensions:
            candidate = self.midi_dir / f"{audio_path.stem}{extension}"
            if candidate.exists():
                return TranscriptionResult(
                    notes=parse_midi(candidate),
                    transcriber=self.name,
                    source_audio=audio_path,
                    midi_path=candidate,
                )
        raise TranscriptionError(
            f"No precomputed MIDI for '{audio_path.stem}' in {self.midi_dir}"
        )


@register_transcriber("precomputed")
def _build(cfg: PrecomputedTranscriberConfig) -> PrecomputedTranscriber:
    return PrecomputedTranscriber(
        midi_dir=cfg.midi_dir,
        extensions=cfg.extensions,
        name=cfg.name or "precomputed",
    )
