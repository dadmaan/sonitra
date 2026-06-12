from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

from sonitra.midi_reader import parse_midi
from sonitra.transcribe.base import TranscriptionError, TranscriptionResult
from sonitra.transcribe.configs import ExternalCommandTranscriberConfig
from sonitra.transcribe.protocol import register_transcriber


class ExternalCommandTranscriber:
    """Runs an arbitrary CLI transcription tool.

    The command template must contain `{input}` and `{output}` placeholders;
    the tool is expected to write a MIDI file to the `{output}` path.
    """

    def __init__(
        self,
        *,
        command: str,
        output_extension: str = ".mid",
        timeout_sec: float = 600.0,
        name: str = "external_command",
    ) -> None:
        if "{input}" not in command or "{output}" not in command:
            raise ValueError("command must contain {input} and {output} placeholders")
        self.command = command
        self.output_extension = output_extension
        self.timeout_sec = float(timeout_sec)
        self.name = name

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        audio_path = Path(audio_path)
        with tempfile.TemporaryDirectory(prefix="sonitra-transcribe-") as tmp_dir:
            output_path = Path(tmp_dir) / f"{audio_path.stem}{self.output_extension}"
            argv = [
                part.replace("{input}", str(audio_path)).replace("{output}", str(output_path))
                for part in shlex.split(self.command)
            ]
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                raise TranscriptionError(
                    f"Transcription command timed out after {self.timeout_sec}s"
                ) from exc
            if completed.returncode != 0:
                raise TranscriptionError(
                    f"Transcription command failed ({completed.returncode}): "
                    f"{completed.stderr.strip()}"
                )
            if not output_path.exists():
                raise TranscriptionError(
                    f"Transcription command produced no output at {output_path}"
                )
            return TranscriptionResult(
                notes=parse_midi(output_path),
                transcriber=self.name,
                source_audio=audio_path,
                metadata={"command": self.command},
            )


@register_transcriber("external_command")
def _build(cfg: ExternalCommandTranscriberConfig) -> ExternalCommandTranscriber:
    return ExternalCommandTranscriber(
        command=cfg.command,
        output_extension=cfg.output_extension,
        timeout_sec=cfg.timeout_sec,
        name=cfg.name or "external_command",
    )
