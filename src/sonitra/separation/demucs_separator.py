from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sonitra.separation.protocol import register_separator

if TYPE_CHECKING:
    from sonitra.config import SeparationSection


class DemucsSeparator:
    """Demucs stem separation backend.

    Requires the optional `demucs` dependency; install with
    `pip install sonitra[demucs]`.
    """

    name = "demucs"

    def __init__(self, *, model: str = "htdemucs", device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._separator = None

    def separate(self, audio_path: Path | str, output_dir: Path | str) -> dict[str, Path]:
        try:
            import demucs.api
        except ImportError as exc:
            raise RuntimeError(
                "demucs is not installed; install with `pip install sonitra[demucs]`"
            ) from exc

        if self._separator is None:
            self._separator = demucs.api.Separator(model=self.model, device=self.device)

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _, stems = self._separator.separate_audio_file(str(audio_path))
        result: dict[str, Path] = {}
        for stem_name, waveform in stems.items():
            stem_path = output_dir / f"{audio_path.stem}.{stem_name}.wav"
            demucs.api.save_audio(
                waveform, str(stem_path), samplerate=self._separator.samplerate
            )
            result[stem_name] = stem_path
        return result


@register_separator("demucs")
def _build(cfg: "SeparationSection") -> DemucsSeparator:
    return DemucsSeparator(model=cfg.model, device=cfg.device)
