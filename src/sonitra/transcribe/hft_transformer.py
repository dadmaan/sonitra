"""Sony hFT-Transformer piano transcription. Needs the transkun extra and converted weights from scripts/setup_hft_transformer.py."""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, NamedTuple

from sonitra.notes import make_note
from sonitra.transcribe.base import TranscriptionError, TranscriptionResult
from sonitra.transcribe.configs import HftTransformerTranscriberConfig
from sonitra.transcribe.numerics import read_numeric_env
from sonitra.transcribe.protocol import register_transcriber
from sonitra.transcribe.torch_support import (
    apply_torch_numeric_settings,
    missing_dependency_error,
    validate_torch_device,
)

logger = logging.getLogger(__name__)

#: The feature extractor is pinned to 16 kHz: the mel analysis resolution and the
#: frame rate the released model was trained and decoded with.
SAMPLE_RATE = 16000

BACKEND_TYPE = "hft_transformer"

_SETUP_SCRIPT = "python scripts/setup_hft_transformer.py"


class MappedNotes(NamedTuple):
    """Result of mapping decoder events onto Sonitra note dicts."""

    notes: list[dict[str, Any]]
    filtered_dropped: int
    offset_past_end: int
    note_events_total: int


def notes_from_events(
    events: list[dict[str, Any]], audio_duration_sec: float
) -> MappedNotes:
    """Map ``mpe2note`` events onto Sonitra note dicts, sorted by (start_sec, pitch).

    A pure function of its arguments, so the note contract can be checked without
    weights or torch. ``events`` carry absolute ``pitch`` in 21..108, ``onset`` and
    ``offset`` in seconds and a ``velocity`` in 0..127; the offset becomes a duration.

    Notes whose offset lands past the audio end are kept rather than clipped: the
    decoder pads the tail so the final note can terminate, and truncating it would
    disagree with the released model. The count is reported so the effect is visible.
    """
    notes: list[dict[str, Any]] = []
    filtered_dropped = 0
    offset_past_end = 0
    for event in events:
        onset = float(event["onset"])
        offset = float(event["offset"])
        # make_note raises on an out-of-range pitch, which is the point: a note outside
        # 0..127 means the pitch mapping is wrong, not that the note is unusable.
        note = make_note(
            pitch=int(event["pitch"]),
            velocity=int(event["velocity"]),
            start_sec=onset,
            duration_sec=offset - onset,
        )
        if note is None:
            filtered_dropped += 1
            continue
        if offset > audio_duration_sec:
            offset_past_end += 1
        notes.append(note)
    notes.sort(key=lambda n: (n["start_sec"], n["pitch"]))
    return MappedNotes(
        notes=notes,
        filtered_dropped=filtered_dropped,
        offset_past_end=offset_past_end,
        note_events_total=len(events),
    )


def _sonitra_version() -> str:
    try:
        from importlib.metadata import version

        return version("sonitra")
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_dependency(module: str) -> TranscriptionError:
    """TranscriptionError naming the module that failed and how to install it."""
    return missing_dependency_error(module, backend=BACKEND_TYPE)


def _weights_error(path: Path, detail: str) -> TranscriptionError:
    return TranscriptionError(
        f"{BACKEND_TYPE} could not use weights at {path}: {detail}. "
        f"Run `{_SETUP_SCRIPT}` to fetch and convert the released checkpoint, or set "
        "weights_path to an already converted model.pt."
    )


class HftTransformerTranscriber:
    """hFT-Transformer piano transcription.

    Requires ``pip install sonitra[transkun]`` and the converted checkpoint that
    ``scripts/setup_hft_transformer.py`` installs.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        checkpoint: str = "maestro",
        weights_path: Path | str | None = None,
        output: str = "second",
        n_stride: int = 0,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
        mpe_threshold: float = 0.5,
        batch_size: int = 1,
        numeric_mode: str = "off",
        gpu_memory_growth: bool = False,
        name: str = BACKEND_TYPE,
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.output = output
        self.n_stride = int(n_stride)
        self.onset_threshold = float(onset_threshold)
        self.offset_threshold = float(offset_threshold)
        self.mpe_threshold = float(mpe_threshold)
        self.batch_size = int(batch_size)
        mode = numeric_mode.lower() if isinstance(numeric_mode, str) else numeric_mode
        if mode not in ("off", "warn", "strict"):
            raise TranscriptionError(
                f"{BACKEND_TYPE} got unknown numeric_mode {numeric_mode!r}; "
                "use 'off', 'warn' or 'strict'."
            )
        self.numeric_mode = mode
        self.gpu_memory_growth = bool(gpu_memory_growth)
        self.name = name
        self._model: Any | None = None
        self._runtime: dict[str, Any] | None = None
        self._provenance: dict[str, Any] = {}
        self._weights_sha256 = ""
        self._mel: dict[str, Any] = {}
        self._lock = threading.RLock()

    def validate_device(self) -> tuple[str, bool]:
        """Resolve the configured device and confirm it exists.

        Returns (resolved_name, available). Raises TranscriptionError when an
        accelerator was requested but is absent. CPU short-circuits with no
        framework import.
        """
        return validate_torch_device(self.device, backend=BACKEND_TYPE)

    def _weights_file(self) -> Path:
        """The converted checkpoint to load, with the two ways that can be wrong named."""
        if self.weights_path is None:
            from sonitra.transcribe._hft import checkpoints

            try:
                return checkpoints.checkpoint_path(self.checkpoint)
            except KeyError as exc:
                raise TranscriptionError(f"{BACKEND_TYPE}: {exc}") from exc
        path = Path(self.weights_path)
        if path.suffix.lower() == ".pkl":
            raise TranscriptionError(
                f"{BACKEND_TYPE} weights_path {path} is the unconverted upstream pickle. "
                "Upstream ships a pickled module and Sonitra loads a plain state dict, "
                f"so convert it first with `{_SETUP_SCRIPT}`, then point weights_path at "
                "the resulting model.pt."
            )
        return path

    def _load_runtime(self, path: Path, resolved: str) -> tuple[Any, dict[str, Any]]:
        """Load and cache the model, its runtime mapping and its provenance."""
        import torch

        from sonitra.transcribe._hft import checkpoints

        try:
            blob = torch.load(path, weights_only=True, map_location=resolved)
        except Exception as exc:
            raise _weights_error(path, f"torch.load failed ({exc})") from exc

        version = blob.get("format_version")
        if version != checkpoints.FORMAT_VERSION:
            raise _weights_error(
                path,
                f"format_version {version!r}, this build reads {checkpoints.FORMAT_VERSION}",
            )

        try:
            from sonitra.transcribe._hft.model import build_model

            model = build_model(blob["hparams"], device=resolved)
            model.load_state_dict(blob["state_dict"], strict=True)
        except Exception as exc:
            raise _weights_error(path, f"the state dict does not fit the model ({exc})") from exc
        model = model.to(resolved).eval()

        # The windowing and note decoding read the architecture dimensions from the same
        # mapping as the feature constants, so the two converted blocks are merged once.
        runtime = {**blob["inference"], **blob["hparams"]}
        self._weights_sha256 = _file_sha256(path)
        self._provenance = dict(blob.get("provenance", {}))
        return model, runtime

    def _mel_transform(self, runtime: dict[str, Any], resolved: str) -> Any:
        """Mel transform for one device, cached because it holds device-resident buffers."""
        from sonitra.transcribe._hft.inference import make_mel_transform

        mel = self._mel.get(resolved)
        if mel is None:
            mel = make_mel_transform(runtime, resolved)
            self._mel[resolved] = mel
        return mel

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        # Fail fast on a missing accelerator; also the backstop for worker
        # subprocesses and direct library use.
        resolved, available = self.validate_device()
        # lazy imports so torch is never touched unless this backend is used
        try:
            import torch
        except ImportError as exc:
            raise _missing_dependency("torch") from exc

        apply_torch_numeric_settings(
            self.numeric_mode,
            self.gpu_memory_growth,
            backend=BACKEND_TYPE,
            device=resolved,
        )

        path = self._weights_file()
        if not path.is_file():
            raise TranscriptionError(
                f"{BACKEND_TYPE} has no weights at {path}. "
                f"Run `{_SETUP_SCRIPT}`, or set weights_path to an already converted model.pt. "
                f"The models directory is {self._models_dir()}; override it with "
                "SONITRA_MODELS_DIR."
            )

        # thread-safe lazy init + inference: one load, one model, shared by every worker
        with self._lock:
            if self._model is None or self._runtime is None:
                model, runtime = self._load_runtime(path, resolved)
                self._model = model
                self._runtime = runtime
            model = self._model
            runtime = self._runtime

            from sonitra.transcribe._hft.inference import (
                mpe2note,
                transcript,
                transcript_stride,
                wav2feature,
            )

            audio_path_p = Path(audio_path)
            try:
                from sonitra.storage import read_audio_resampled

                audio, sample_rate = read_audio_resampled(
                    audio_path_p, target_sr=SAMPLE_RATE
                )
            except Exception as exc:
                raise TranscriptionError(
                    f"{BACKEND_TYPE} failed to read audio {audio_path_p}: {exc}"
                ) from exc

            import numpy as np

            # (channels, samples) -> mono 1-D at the pinned feature rate
            mono = (
                np.mean(audio, axis=0)
                if audio.ndim > 1
                else np.asarray(audio, dtype=np.float32)
            )
            mono = np.ascontiguousarray(mono, dtype=np.float32)
            duration_sec = float(mono.shape[0]) / float(sample_rate or SAMPLE_RATE)
            mel = self._mel_transform(runtime, resolved)

            with torch.no_grad():
                wave = torch.from_numpy(mono).to(resolved)
                features = wav2feature(wave, runtime, mel=mel)
                if self.n_stride > 0:
                    arrays = transcript_stride(
                        features,
                        model,
                        inference=runtime,
                        device=resolved,
                        n_offset=self.n_stride,
                        batch_size=self.batch_size,
                    )
                else:
                    arrays = transcript(
                        features,
                        model,
                        inference=runtime,
                        device=resolved,
                        batch_size=self.batch_size,
                    )
                # "second" is the time-axis head set, which the model reports.
                head = arrays[4:8] if self.output == "second" else arrays[0:4]
                events = mpe2note(
                    head[0],
                    head[1],
                    head[2],
                    head[3],
                    inference=runtime,
                    onset_threshold=self.onset_threshold,
                    offset_threshold=self.offset_threshold,
                    mpe_threshold=self.mpe_threshold,
                    mode_velocity="ignore_zero",
                    mode_offset="shorter",
                )

        mapped = notes_from_events(events, duration_sec)

        metadata: dict[str, Any] = {
            "package_version": _sonitra_version(),
            "weights_sha256": self._weights_sha256,
            "upstream_commit": self._provenance.get("upstream_commit", "unknown"),
            "checkpoint": self.checkpoint,
            "device": resolved,
            "requested_device": self.device,
            "device_available": available,
            "numeric_mode": self.numeric_mode,
            "output_head": self.output,
            "n_stride": self.n_stride,
            "batch_size": self.batch_size,
            "thresholds": {
                "onset": self.onset_threshold,
                "offset": self.offset_threshold,
                "mpe": self.mpe_threshold,
            },
            "resampler": "pedalboard",
            "sample_rate": SAMPLE_RATE,
            "note_events_total": mapped.note_events_total,
            "notes_kept": len(mapped.notes),
            "filtered_dropped": mapped.filtered_dropped,
            "offset_past_end": mapped.offset_past_end,
        }

        return TranscriptionResult(
            notes=mapped.notes,
            transcriber=self.name,
            source_audio=audio_path_p,
            raw_outputs=None,
            backend_type=BACKEND_TYPE,
            metadata=metadata,
        )

    def _models_dir(self) -> Path:
        from sonitra.transcribe._hft import checkpoints

        return checkpoints.models_dir()


@register_transcriber(BACKEND_TYPE)
def _build(cfg: HftTransformerTranscriberConfig) -> HftTransformerTranscriber:
    numeric_mode, gpu_memory_growth = read_numeric_env()
    return HftTransformerTranscriber(
        device=cfg.device,
        checkpoint=cfg.checkpoint,
        weights_path=cfg.weights_path,
        output=cfg.output,
        n_stride=cfg.n_stride,
        onset_threshold=cfg.onset_threshold,
        offset_threshold=cfg.offset_threshold,
        mpe_threshold=cfg.mpe_threshold,
        batch_size=cfg.batch_size,
        numeric_mode=numeric_mode,
        gpu_memory_growth=gpu_memory_growth,
        name=cfg.name or BACKEND_TYPE,
    )